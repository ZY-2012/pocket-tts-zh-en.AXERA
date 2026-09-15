#!/usr/bin/env python3
"""M2.2: generate Pulsar2 quantization configs for the static subgraphs.

Inputs:  models/subgraphs/{variant}/<model>.onnx
Calib:   model_convert/calib_data/subgraphs/<model>/<input>.tar.gz
Outputs: model_convert/pulsar2_<run>/config_<model>_check<N>.json + manifest

Usage:
    python python/make_quant_configs.py --models flow_net_step mimi_decoder_step --check-level 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("TMPDIR", os.path.join(PROJ, ".work_tmp", "tmp"))

import onnx  # noqa: E402


def graph_inputs(model: onnx.ModelProto) -> dict[str, list[int]]:
    init_names = {init.name for init in model.graph.initializer}
    shapes = {}
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        dims = []
        for dim in inp.type.tensor_type.shape.dim:
            value = dim.dim_value
            dims.append(int(value) if value > 0 else -1)
        shapes[inp.name] = dims
    return shapes


def staticize_inputs(onnx_path: str, out_path: str) -> None:
    """Replace dim_param on graph inputs with dim_value=1 (Pulsar2 rejects params)."""
    model = onnx.load(onnx_path, load_external_data=False)
    changed = False
    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param:
                dim.ClearField("dim_param")
                dim.dim_value = 1
                changed = True
    onnx.save(model, out_path)
    if changed:
        print(f"  staticized inputs -> {out_path}")


def shape_str(dims: list[int]) -> str:
    if not dims:
        return "1"
    return "x".join(str(d) if d > 0 else "1" for d in dims)


def safe_input_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def main() -> None:
    ap = argparse.ArgumentParser(description="generate Pulsar2 configs")
    ap.add_argument("--models", nargs="+", default=["flow_net_step", "mimi_conv_step"])
    ap.add_argument("--variant", default="npu", choices=["raw", "onnxsim", "npu"])
    ap.add_argument("--onnx-dir", default=None)
    ap.add_argument("--calib-dir", default=os.path.join(PROJ, "model_convert", "calib_data", "subgraphs"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--check-level", type=int, default=0)
    ap.add_argument("--target-hardware", default="AX650")
    ap.add_argument("--npu-mode", default="NPU3")
    ap.add_argument("--default-data-type", default="U16")
    ap.add_argument("--precision-analysis", action="store_true")
    args = ap.parse_args()

    onnx_dir = args.onnx_dir or os.path.join(PROJ, "models", "subgraphs", args.variant)
    search_dirs = [onnx_dir,
                   os.path.join(PROJ, "models", "subgraphs", "encoder"),
                   os.path.join(PROJ, "models", "subgraphs", "mimi_split"),
                   os.path.join(PROJ, "models", "subgraphs", "npu")]
    run_name = args.run_name or f"pulsar2_zh_en_check{args.check_level}"
    out_dir = args.output_dir or os.path.join(PROJ, "model_convert", run_name)
    os.makedirs(out_dir, exist_ok=True)

    layer_configs = [
        {"op_type": "Pow", "data_type": "U8"},
        {"op_types": ["Softmax", "ReduceMean"], "data_type": "FP32"},
        {"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"],
         "data_type": args.default_data_type},
    ]

    manifest = []
    static_dir = os.path.join(out_dir, "static")
    os.makedirs(static_dir, exist_ok=True)
    for model in args.models:
        onnx_path = None
        for directory in search_dirs:
            candidate = os.path.join(directory, f"{model}.onnx")
            if os.path.exists(candidate):
                onnx_path = candidate
                break
        if onnx_path is None:
            raise FileNotFoundError(f"{model}.onnx not found in {search_dirs}")
        static_path = os.path.join(static_dir, f"{model}.onnx")
        staticize_inputs(onnx_path, static_path)
        shapes = graph_inputs(onnx.load(static_path, load_external_data=False))
        input_configs = []
        for name, dims in shapes.items():
            tar_path = os.path.join(args.calib_dir, model, f"{safe_input_name(name)}.tar.gz")
            if not os.path.exists(tar_path):
                print(f"[warn] missing calibration for {model}/{name}: {tar_path}")
            input_configs.append({
                "tensor_name": name,
                "calibration_dataset": tar_path,
                "calibration_format": "Numpy",
                "calibration_size": -1,
            })
        config = {
            "model_type": "ONNX",
            "npu_mode": args.npu_mode,
            "input": static_path,
            "output_name": f"{model}.axmodel",
            "output_dir": os.path.abspath(os.path.join(out_dir, f"build-{model}-{run_name}")),
            "target_hardware": args.target_hardware,
            "onnx_opt": {"disable_onnx_optimization": False, "enable_onnxsim": True},
            "quant": {
                "input_configs": input_configs,
                "layer_configs": layer_configs,
                "calibration_method": "MinMax",
                "enable_smooth_quant": True,
                "conv_bias_data_type": "FP32",
                "precision_analysis": bool(args.precision_analysis),
                "precision_analysis_method": "EndToEnd",
                "disable_auto_refine_scale": True,
                "transformer_opt_level": 0,
            },
            "input_processors": [{"tensor_name": "DEFAULT"}],
            "compiler": {"check": args.check_level, "enable_slice_mode": False},
        }
        config_path = os.path.join(out_dir, f"config_{model}_check{args.check_level}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        manifest.append({
            "model": model,
            "config": config_path,
            "build_dir": os.path.abspath(os.path.join(out_dir, f"build-{model}-{run_name}")),
            "output_name": f"{model}.axmodel",
            "input_shapes": ";".join(f"{name}:{shape_str(dims)}" for name, dims in shapes.items()),
        })
        print(f"{model}: {len(input_configs)} inputs, config {config_path}")
        print(f"  input_shapes: {manifest[-1]['input_shapes']}")

    manifest_path = os.path.join(out_dir, f"quant_manifest_check{args.check_level}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
