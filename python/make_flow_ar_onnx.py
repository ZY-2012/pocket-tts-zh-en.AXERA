#!/usr/bin/env python3
"""M3: build a lean flow-LM autoregressive step graph (latent gate only).

The windowed flow_step graph handles text/voice prefill and AR steps via the
3-way `gates` one-hot selector. AR steps always use gates=[0,1,0], tokens=0,
cond=0; baking those constants lets onnxsim drop the 25055x1024 embedding and
the cond branch, shrinking the per-frame graph substantially.

Output: models/subgraphs/flow/flow_ar_step.onnx (fp32) and optionally an
ORT dynamic-int8 copy.

Usage:
    python python/make_flow_ar_onnx.py [--int8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TMPDIR", os.path.join(PROJ, ".work_tmp", "tmp"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import helper, numpy_helper  # noqa: E402

SRC = os.path.join(PROJ, "models", "subgraphs", "windowed", "flow_step.onnx")
OUT_DIR = os.path.join(PROJ, "models", "subgraphs", "flow")


def bake(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    const_name = f"{name}_baked"
    node = helper.make_node(
        "Constant", [], [const_name],
        value=numpy_helper.from_array(value, name=const_name),
        name=f"bake_{name}")
    kept = [i for i in model.graph.input if i.name != name]
    del model.graph.input[:]
    model.graph.input.extend(kept)
    model.graph.node.insert(0, node)
    for n in model.graph.node:
        for i in range(len(n.input)):
            if n.input[i] == name:
                n.input[i] = const_name


def main() -> None:
    ap = argparse.ArgumentParser(description="lean flow AR step graph")
    ap.add_argument("--int8", action="store_true", help="also produce ORT dynamic-int8 model")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    model = onnx.load(SRC, load_external_data=False)
    bake(model, "tokens", np.zeros((1, 1), dtype=np.int64))
    bake(model, "cond", np.zeros((1, 1, 1024), dtype=np.float32))
    bake(model, "gates", np.array([0.0, 1.0, 0.0], dtype=np.float32))
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    onnx.checker.check_model(model)
    import onnxsim

    model, ok = onnxsim.simplify(model)
    if not ok:
        raise RuntimeError("onnxsim failed")
    onnx.checker.check_model(model)
    out = os.path.join(OUT_DIR, "flow_ar_step.onnx")
    onnx.save(model, out)
    ins = [i.name for i in model.graph.input]
    outs = [o.name for o in model.graph.output]
    print(f"flow_ar_step: in={ins} out={outs} ({os.path.getsize(out) / 1e6:.1f} MB)")

    info = {"source": os.path.relpath(SRC), "fp32": os.path.relpath(out, PROJ),
            "inputs": ins, "outputs": outs}
    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = os.path.join(OUT_DIR, "flow_ar_step_int8.onnx")
        quantize_dynamic(out, int8_path, weight_type=QuantType.QInt8,
                         op_types_to_quantize=None)
        onnx.checker.check_model(int8_path)
        info["int8"] = os.path.relpath(int8_path, PROJ)
        info["int8_size_mb"] = round(os.path.getsize(int8_path) / 1e6, 1)
        print(f"flow_ar_step_int8: {info['int8_size_mb']} MB")

    with open(os.path.join(OUT_DIR, "flow_ar_info.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
