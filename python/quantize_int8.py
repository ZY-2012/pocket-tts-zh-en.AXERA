#!/usr/bin/env python3
"""M6: ORT dynamic-int8 for additional runtime graphs.

Targets (written next to their fp32 source, *_int8.onnx):
  - models/subgraphs/mimi_split/mimi_transformer_step_int8.onnx
  - models/subgraphs/windowed/flow_step_windowed_int8.onnx   (prefill graph)

Usage:
    python python/quantize_int8.py
    python python/quantize_int8.py --only mimi
"""
from __future__ import annotations

import argparse
import json
import os
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TMPDIR", os.path.join(PROJ, ".work_tmp", "tmp"))

import onnx  # noqa: E402
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402

TARGETS = {
    "mimi_tf": os.path.join(PROJ, "models", "subgraphs", "mimi_split",
                            "mimi_transformer_step.onnx"),
    "flow_prefill": os.path.join(PROJ, "models", "subgraphs", "windowed",
                                 "flow_step.onnx"),
}


def quantize(src: str) -> tuple[str, float]:
    dst = src.replace(".onnx", "_int8.onnx")
    t0 = time.perf_counter()
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    onnx.checker.check_model(dst)
    mb = os.path.getsize(dst) / 1e6
    print(f"  {os.path.basename(src)} -> {os.path.basename(dst)} "
          f"({mb:.1f} MB, {time.perf_counter() - t0:.0f}s)")
    return dst, mb


def main() -> None:
    ap = argparse.ArgumentParser(description="ORT dynamic int8 for extra graphs")
    ap.add_argument("--only", choices=["mimi_tf", "flow_prefill"], default=None)
    args = ap.parse_args()

    info = {}
    for key, src in TARGETS.items():
        if args.only and key != args.only:
            continue
        print(f"== {key}: {src}")
        dst, mb = quantize(src)
        model = onnx.load(dst, load_external_data=False)
        info[key] = {
            "source": os.path.relpath(src, PROJ),
            "int8": os.path.relpath(dst, PROJ),
            "size_mb": round(mb, 1),
            "inputs": [i.name for i in model.graph.input],
            "outputs": [o.name for o in model.graph.output],
        }
    out = os.path.join(PROJ, "models", "subgraphs", "int8_extra.json")
    with open(out, "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
