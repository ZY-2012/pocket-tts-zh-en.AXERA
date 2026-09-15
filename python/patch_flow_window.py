#!/usr/bin/env python3
"""M1: patch the flow_step subgraph so a windowed flow KV cache is exact.

The fused graph labels key positions as pos_k = arange(0, past+seq), i.e. it
assumes the cache holds the full history (offset == cache length). For a
sliding window we relabel them to absolute positions:

    pos_k = (flow_offset - past) + arange(0, past+seq),  past = total - seq

Only the attention mask labels change; stored keys are already RoPE-rotated at
their absolute write positions, and query RoPE still uses flow_offset. With the
full cache this is a no-op (shift == 0).

The patch touches the six per-layer "key position" Add nodes (one per flow
layer) that match: Add(0-constant, Range(0, Cast(Gather(Shape(kv_concat))))).

Usage:
    python python/patch_flow_window.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORK_TMP = os.path.join(PROJ, ".work_tmp")
os.makedirs(os.path.join(WORK_TMP, "tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.join(WORK_TMP, "tmp")

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import helper, numpy_helper  # noqa: E402

PREFIX = "/m1_patch"


def const_scalar(name: str, value: int, dtype=onnx.TensorProto.INT64):
    return helper.make_node(
        "Constant", [], [name],
        value=numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name + "_t"),
        name=name + "_node",
    )


def find_key_pos_adds(model: onnx.ModelProto):
    """Return [(add_node, total_cast_output)] for the six flow key-position adds."""
    producer = {}
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node
    init_vals = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    def const_val(name):
        if name in init_vals:
            return init_vals[name]
        node = producer.get(name)
        if node is not None and node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)
        return None

    found = []
    for node in model.graph.node:
        if node.op_type != "Add" or len(node.input) != 2:
            continue
        v0 = const_val(node.input[0])
        rng = producer.get(node.input[1])
        if v0 is None or v0.size != 1 or int(v0.ravel()[0]) != 0:
            continue
        if rng is None or rng.op_type != "Range":
            continue
        total = producer.get(rng.input[1])
        if total is None or total.op_type != "Cast":
            continue
        gather = producer.get(total.input[0])
        if gather is None or gather.op_type != "Gather":
            continue
        shape = producer.get(gather.input[0])
        if shape is None or shape.op_type != "Shape":
            continue
        concat = producer.get(shape.input[0])
        if concat is None or concat.op_type != "Concat":
            continue
        found.append((node, total.output[0]))
    return found


def patch(model_path: str, output_path: str) -> dict:
    model = onnx.load(model_path)
    pairs = find_key_pos_adds(model)
    if len(pairs) != 6:
        raise RuntimeError(f"expected 6 flow key-position Adds, found {len(pairs)}")

    shared_nodes = [
        helper.make_node("Shape", ["tokens"], [f"{PREFIX}/tokens_shape"], name=f"{PREFIX}/Shape_tokens"),
        const_scalar(f"{PREFIX}/one", 1),
        helper.make_node(
            "Gather", [f"{PREFIX}/tokens_shape", f"{PREFIX}/one"], [f"{PREFIX}/seq"],
            axis=0, name=f"{PREFIX}/Gather_seq",
        ),
    ]
    before: dict[str, list] = {}
    patched = []
    for idx, (add_node, total_out) in enumerate(pairs):
        past = f"{PREFIX}/past_{idx}"
        shift = f"{PREFIX}/shift_{idx}"
        before[add_node.name] = [
            helper.make_node("Sub", [total_out, f"{PREFIX}/seq"], [past],
                             name=f"{PREFIX}/Sub_past_{idx}"),
            helper.make_node("Sub", ["flow_offset", past], [shift],
                             name=f"{PREFIX}/Sub_shift_{idx}"),
        ]
        add_node.input[0] = shift
        patched.append(add_node.name)

    ordered = list(shared_nodes)
    for node in model.graph.node:
        ordered.extend(before.get(node.name, []))
        ordered.append(node)
    del model.graph.node[:]
    for node in ordered:
        model.graph.node.append(node)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save(model, output_path)
    return {"patched_adds": patched, "output": output_path}


def main() -> None:
    ap = argparse.ArgumentParser(description="patch flow window positions")
    ap.add_argument("--manifest", default=os.path.join(PROJ, "models", "subgraphs", "manifest.json"))
    ap.add_argument("--variant", default="raw", choices=["raw", "onnxsim"])
    ap.add_argument("--out-dir", default=os.path.join(PROJ, "models", "subgraphs", "windowed"))
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    src = os.path.join(PROJ, manifest["subgraphs"]["flow_step"][args.variant])
    dst = os.path.join(args.out_dir, "flow_step.onnx")
    info = patch(src, dst)
    info.update({"source": src, "variant": args.variant})
    with open(os.path.join(args.out_dir, "patch_info.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"patched {len(info['patched_adds'])} adds:")
    for name in info["patched_adds"]:
        print(f"  {name}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
