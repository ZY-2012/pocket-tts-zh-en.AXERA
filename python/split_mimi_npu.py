#!/usr/bin/env python3
"""M2: split the mimi decoder into transformer (CPU) + conv decoder (NPU).

transformer part (ONNX, CPU, fp32, keeps mimi_offset semantics):
    next_latent, mimi_kv, mimi_conv, mimi_offset
      -> decoder_embedding, mimi_kv_new

conv decoder part (axmodel, NPU, no int64 / no mask / no RoPE):
    decoder_embedding, mimi_conv -> audio, mimi_conv_out

The transformer part reads the positional-embedding slice of mimi_conv
(read-only); only the conv part writes the updated state.

Usage:
    python python/split_mimi_npu.py
"""
from __future__ import annotations

import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TMPDIR", os.path.join(PROJ, ".work_tmp", "tmp"))

import onnx  # noqa: E402
from onnx import helper  # noqa: E402

SRC = os.path.join(PROJ, "models", "subgraphs", "onnxsim", "mimi_decoder_step.onnx")
OUT_DIR = os.path.join(PROJ, "models", "subgraphs", "mimi_split")

SPLIT_TENSOR = "/Transpose_66_output_0"
UPSAMPLE_STATE = "/Reshape_43_output_0"
RENAME = {SPLIT_TENSOR: "decoder_embedding", UPSAMPLE_STATE: "upsample_state"}

TRANSFORMER_INPUTS = ["next_latent", "mimi_kv", "mimi_conv", "mimi_offset"]
TRANSFORMER_OUTPUTS = [SPLIT_TENSOR, "mimi_kv_new", UPSAMPLE_STATE]
CONV_INPUTS = [SPLIT_TENSOR, "mimi_conv", UPSAMPLE_STATE]
CONV_OUTPUTS = ["audio", "mimi_conv_out"]


def value_info_map(model: onnx.ModelProto) -> dict:
    result = {}
    for coll in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in coll:
            result[vi.name] = vi
    return result


def extract_subgraph(model: onnx.ModelProto, input_names: list[str],
                     output_names: list[str], name: str) -> onnx.ModelProto:
    """DFS-based extraction with a proper topological sort of the node subset."""
    producer = {o: n for n in model.graph.node for o in n.output}

    needed_nodes: dict[str, onnx.NodeProto] = {}
    stop = set(input_names)
    stack = [t for t in output_names if t in producer and t not in stop]
    while stack:
        tensor = stack.pop()
        node = producer.get(tensor)
        if node is None or node.name in needed_nodes:
            continue
        needed_nodes[node.name] = node
        stack.extend(t for t in node.input if t in producer and t not in stop)

    # Kahn topological sort over the node subset
    deps: dict[str, set[str]] = {}
    consumers: dict[str, set[str]] = {}
    node_by_output = {}
    for node in needed_nodes.values():
        for out in node.output:
            node_by_output[out] = node.name
    for node in needed_nodes.values():
        d = {node_by_output[t] for t in node.input if t in node_by_output}
        deps[node.name] = d
        for t in node.input:
            if t in node_by_output:
                consumers.setdefault(node_by_output[t], set()).add(node.name)
    ready = [n for n, d in deps.items() if not d]
    ordered = []
    while ready:
        current = ready.pop()
        ordered.append(current)
        for downstream in sorted(consumers.get(current, ())):
            deps[downstream].discard(current)
            if not deps[downstream]:
                ready.append(downstream)
    if len(ordered) != len(needed_nodes):
        raise RuntimeError(f"cycle detected while extracting {name}")
    ordered_nodes = [needed_nodes[n] for n in ordered]

    vi_map = value_info_map(model)
    graph_inputs = [vi_map[t] for t in input_names]
    graph_outputs = [vi_map[t] for t in output_names]

    used_inits = []
    inits = {i.name: i for i in model.graph.initializer}
    for node in ordered_nodes:
        for t in node.input:
            if t in inits:
                used_inits.append(inits[t])
    dedup = {i.name: i for i in used_inits}

    graph = helper.make_graph(
        ordered_nodes, name, graph_inputs, graph_outputs,
        initializer=list(dedup.values()),
        value_info=[vi for t, vi in vi_map.items()
                    if t not in set(input_names) | set(output_names)
                    and any(t in n.output for n in ordered_nodes)],
    )
    extracted = helper.make_model(
        graph, opset_imports=model.opset_import, ir_version=model.ir_version,
        producer_name="pocket-tts-zh-en split")
    return extracted


def rename_io(model: onnx.ModelProto, mapping: dict[str, str]) -> None:
    for coll in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in coll:
            if vi.name in mapping:
                vi.name = mapping[vi.name]
    for node in model.graph.node:
        node.input[:] = [mapping.get(x, x) for x in node.input]
        node.output[:] = [mapping.get(x, x) for x in node.output]
    for init in model.graph.initializer:
        if init.name in mapping:
            init.name = mapping[init.name]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    info = {"source": os.path.relpath(SRC), "split_tensor": SPLIT_TENSOR, "parts": {}}

    for name, inputs, outputs in (
        ("mimi_transformer_step", TRANSFORMER_INPUTS, TRANSFORMER_OUTPUTS),
        ("mimi_conv_step", CONV_INPUTS, CONV_OUTPUTS),
    ):
        model = onnx.load(SRC, load_external_data=False)
        model = extract_subgraph(model, inputs, outputs, name)
        rename_io(model, RENAME)
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        onnx.checker.check_model(model)
        import onnxsim

        model, ok = onnxsim.simplify(model)
        if not ok:
            raise RuntimeError(f"onnxsim failed for {name}")
        out = os.path.join(OUT_DIR, f"{name}.onnx")
        onnx.save(model, out)
        io_info = {
            "inputs": [f"{i.name}:{list(i.type.tensor_type.shape.dim).__len__()}d"
                       for i in model.graph.input],
            "outputs": [o.name for o in model.graph.output],
            "size_mb": round(os.path.getsize(out) / 1e6, 1),
        }
        info["parts"][name] = io_info
        print(f"{name}: in={[i.name for i in model.graph.input]} "
              f"out={[o.name for o in model.graph.output]} ({io_info['size_mb']} MB)")

    with open(os.path.join(OUT_DIR, "split_info.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"wrote {os.path.join(OUT_DIR, 'split_info.json')}")


if __name__ == "__main__":
    main()
