#!/usr/bin/env python3
"""M9: fuse the manual attention clusters of the flow AR graph into ONNX
opset-23 `Attention` nodes.

Each flow layer attention is exported as: Transpose(q) -> Transpose(cached k)
-> MatMul(QK) -> Div(scale) -> Where(mask) -> Softmax -> MatMul(AV) ->
Transpose(out), plus a per-layer position/mask chain. For the AR step
(seq=1, full cache, offset == past) every cached key is visible, so the whole
mask chain is dead once the core becomes one fused Attention node.

Input : models/subgraphs/flow/flow_ar_step.onnx
Output: models/subgraphs/flow/flow_ar_step_fused.onnx

Usage:
    python python/fuse_attention.py
"""
from __future__ import annotations

import argparse
import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TMPDIR", os.path.join(PROJ, ".work_tmp", "tmp"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

SRC = os.path.join(PROJ, "models", "subgraphs", "flow", "flow_ar_step.onnx")
DST = os.path.join(PROJ, "models", "subgraphs", "flow", "flow_ar_step_fused.onnx")

NUM_HEADS = 16
HEAD_DIM = 64
SCALE = 1.0 / (HEAD_DIM ** 0.5)


def producer_map(model):
    return {o: n for n in model.graph.node for o in n.output}


def consumers_map(model):
    cons = {}
    for n in model.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n.name)
    return cons


def prune_and_sort(model: onnx.ModelProto) -> None:
    """Dead-code elimination + topological sort (mask chains become dead)."""
    nodes = list(model.graph.node)
    producer = {o: n for n in nodes for o in n.output}
    needed: set[str] = set()
    stack = [o.name for o in model.graph.output]
    while stack:
        t = stack.pop()
        n = producer.get(t)
        if n is None or n.name in needed:
            continue
        needed.add(n.name)
        stack.extend(n.input)
    kept = [n for n in nodes if n.name in needed]
    by_out = {}
    for n in kept:
        for o in n.output:
            by_out[o] = n.name
    deps = {n.name: {by_out[t] for t in n.input if t in by_out} for n in kept}
    consumers: dict[str, set[str]] = {}
    for n in kept:
        for t in n.input:
            if t in by_out:
                consumers.setdefault(by_out[t], set()).add(n.name)
    node_by_name = {n.name: n for n in kept}
    ready = [n.name for n in kept if not deps[n.name]]
    order = []
    while ready:
        name = ready.pop()
        order.append(node_by_name[name])
        for c in sorted(consumers.get(name, ())):
            deps[c].discard(name)
            if not deps[c]:
                ready.append(c)
    if len(order) != len(kept):
        raise RuntimeError("cycle after prune")
    del model.graph.node[:]
    for n in order:
        model.graph.node.append(n)
    used = {t for n in order for t in n.input}
    kept_inits = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inits)


def rewrite(model: onnx.ModelProto) -> dict:
    producer = producer_map(model)
    replaced = []
    new_nodes = []
    removed = set()

    out_projs = [n for n in model.graph.node
                 if n.op_type == "MatMul" and n.name.startswith("/out_proj")]
    if len(out_projs) != 6:
        raise RuntimeError(f"expected 6 flow out_proj MatMuls, found {len(out_projs)}")

    for idx, out_proj in enumerate(out_projs):
        head = producer[out_proj.input[0]]
        reshape_out = None
        if head.op_type == "Reshape":
            reshape_out = head
            head = producer[head.input[0]]
        t_out = head
        if t_out.op_type != "Transpose":
            raise RuntimeError(f"{out_proj.name}: expected Transpose before out_proj")
        m_av = producer[t_out.input[0]]
        if m_av.op_type != "MatMul":
            raise RuntimeError(f"{out_proj.name}: expected AV MatMul")
        t_v = producer[m_av.input[1]]
        softmax = producer[m_av.input[0]]
        if t_v.op_type != "Transpose" or softmax.op_type != "Softmax":
            raise RuntimeError(f"{out_proj.name}: unexpected AV inputs")
        where = producer[softmax.input[0]]
        if where.op_type != "Where":
            raise RuntimeError(f"{out_proj.name}: expected Where before Softmax")
        div = producer[where.input[1]]
        if div.op_type != "Div":
            raise RuntimeError(f"{out_proj.name}: expected Div inside Where")
        m_qk = producer[div.input[0]]
        if m_qk.op_type != "MatMul":
            raise RuntimeError(f"{out_proj.name}: expected QK MatMul")
        t_q = producer[m_qk.input[0]]
        t_k = producer[m_qk.input[1]]
        if t_q.op_type != "Transpose" or t_k.op_type != "Transpose":
            raise RuntimeError(f"{out_proj.name}: expected Transposes on QK inputs")

        q_src = t_q.input[0]          # [1, 1, H, D]
        k_src = t_k.input[0]          # [1, S, H, D]
        v_src = t_v.input[0]          # [1, S, H, D]

        base = f"/fused_attn{idx}"
        shape_2d = numpy_helper.from_array(np.asarray([0, 0, -1], dtype=np.int64),
                                           name=base + "_shape")
        new_nodes.append(helper.make_node("Constant", [], [base + "_shape_c"],
                                          value=shape_2d, name=base + "_shape_node"))
        new_nodes.append(helper.make_node("Reshape", [q_src, base + "_shape_c"],
                                          [base + "_q"], name=base + "_reshape_q"))
        new_nodes.append(helper.make_node("Reshape", [k_src, base + "_shape_c"],
                                          [base + "_k"], name=base + "_reshape_k"))
        new_nodes.append(helper.make_node("Reshape", [v_src, base + "_shape_c"],
                                          [base + "_v"], name=base + "_reshape_v"))
        attn = helper.make_node(
            "Attention", [base + "_q", base + "_k", base + "_v"], [base + "_out"],
            name=base, domain="", q_num_heads=NUM_HEADS, kv_num_heads=NUM_HEADS,
            scale=SCALE, is_causal=0,
        )
        new_nodes.append(attn)
        out_proj.input[0] = base + "_out"

        removed.update([t_out.name, m_av.name, t_v.name, softmax.name, where.name,
                        div.name, m_qk.name, t_q.name, t_k.name])
        if reshape_out is not None:
            removed.add(reshape_out.name)
        replaced.append(out_proj.name)

    kept = [n for n in model.graph.node if n.name not in removed]
    kept.extend(new_nodes)
    del model.graph.node[:]
    for n in kept:
        model.graph.node.append(n)

    # bump the default-domain opset to 23 (Attention is standard since 23)
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            imp.domain = ""
            imp.version = max(imp.version, 23)
    return {"replaced_out_projs": replaced, "removed_nodes": sorted(removed)}


def main() -> None:
    ap = argparse.ArgumentParser(description="fuse flow AR attention")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--no-simplify", action="store_true")
    args = ap.parse_args()

    model = onnx.load(args.src, load_external_data=False)
    info = rewrite(model)
    prune_and_sort(model)
    if not args.no_simplify:
        import onnxsim

        model, ok = onnxsim.simplify(model)
        if not ok:
            raise RuntimeError("onnxsim failed")
        onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    info["dst"] = os.path.relpath(args.dst, PROJ)
    info["nodes_before"] = len(onnx.load(args.src, load_external_data=False).graph.node)
    info["nodes_after"] = len(onnx.load(args.dst, load_external_data=False).graph.node)
    info["size_mb"] = round(os.path.getsize(args.dst) / 1e6, 1)
    with open(os.path.join(os.path.dirname(args.dst), "ar_fused_info.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
