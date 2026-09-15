#!/usr/bin/env python3
"""M1: extract minimal subgraphs from the fused vendor step_model.onnx.

Boundaries are the already-explicit state I/O of the fused graph:

  flow_step       tokens/latent/is_bos/cond/gates/flow_kv/flow_offset
                    -> conditioning (flow LM final hidden), eos_logit, flow_kv_new
  flow_net_step   conditioning/noise/decode_steps -> next_latent
  mimi_decoder    next_latent/mimi_kv/mimi_conv/mimi_offset
                    -> audio, mimi_kv_new, mimi_conv_out, mimi_offset_out

Outputs land in models/subgraphs/{raw,onnxsim}/ + manifest.json.
Nothing is written to /tmp.

Usage:
    python python/extract_subgraphs.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.environ.get("POCKET_TTS_ROOT", "/data/shared/huyuan/TTS/pocket-tts-zh-en")

WORK_TMP = os.path.join(PROJ, ".work_tmp")
os.makedirs(os.path.join(WORK_TMP, "tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.join(WORK_TMP, "tmp")

import onnx  # noqa: E402
from onnx.utils import extract_model  # noqa: E402

SPEC = {
    "flow_step": {
        "inputs": ["tokens", "latent", "is_bos", "cond", "gates", "flow_kv", "flow_offset"],
        "outputs": ["/Gather_94_output_0", "eos_logit", "flow_kv_new"],
    },
    "flow_net_step": {
        "inputs": ["/Gather_94_output_0", "noise", "decode_steps"],
        "outputs": ["next_latent"],
    },
    "mimi_decoder_step": {
        "inputs": ["next_latent", "mimi_kv", "mimi_conv", "mimi_offset"],
        "outputs": ["audio", "mimi_kv_new", "mimi_conv_out", "mimi_offset_out"],
    },
}

RENAMES = {"/Gather_94_output_0": "conditioning"}


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


def io_names(model: onnx.ModelProto) -> tuple[list[str], list[str]]:
    return [i.name for i in model.graph.input], [o.name for o in model.graph.output]


def main() -> None:
    ap = argparse.ArgumentParser(description="extract pocket-tts subgraphs")
    ap.add_argument("--model-dir", default=os.path.join(MODEL_ROOT, "step_onnx"))
    ap.add_argument("--out-dir", default=os.path.join(PROJ, "models", "subgraphs"))
    ap.add_argument("--no-simplify", action="store_true")
    args = ap.parse_args()

    src = os.path.join(args.model_dir, "step_model.onnx")
    raw_dir = os.path.join(args.out_dir, "raw")
    sim_dir = os.path.join(args.out_dir, "onnxsim")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(sim_dir, exist_ok=True)

    manifest = {"source": src, "subgraphs": {}}
    for name, spec in SPEC.items():
        raw_path = os.path.join(raw_dir, f"{name}.onnx")
        extract_model(src, raw_path, spec["inputs"], spec["outputs"])
        model = onnx.load(raw_path)
        rename_io(model, RENAMES)
        onnx.save(model, raw_path)
        onnx.checker.check_model(raw_path)
        ins, outs = io_names(model)
        entry = {
            "raw": os.path.relpath(raw_path, PROJ),
            "inputs": ins,
            "outputs": outs,
            "source_inputs": spec["inputs"],
            "source_outputs": spec["outputs"],
        }
        print(f"{name}: {ins} -> {outs}  ({os.path.getsize(raw_path) / 1e6:.1f} MB)")

        if not args.no_simplify:
            import onnxsim

            sim_model, ok = onnxsim.simplify(model)
            if not ok:
                raise RuntimeError(f"onnxsim failed for {name}")
            sim_path = os.path.join(sim_dir, f"{name}.onnx")
            onnx.save(sim_model, sim_path)
            onnx.checker.check_model(sim_path)
            entry["onnxsim"] = os.path.relpath(sim_path, PROJ)
            print(f"  onnxsim -> {os.path.getsize(sim_path) / 1e6:.1f} MB")

        manifest["subgraphs"][name] = entry

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
