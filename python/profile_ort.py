#!/usr/bin/env python3
"""ORT node-level profile for pocket-tts-zh-en step_model.onnx.

Runs a short generation (prefix + N frames) with profiling enabled, then
aggregates node latencies into coarse module buckets to guide the NPU/CPU
split. Results are written under results/ (never /tmp).

Usage:
    python python/profile_ort.py --model step_onnx_int8 --frames 16
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import platform
import re
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.environ.get("POCKET_TTS_ROOT", "/data/shared/huyuan/TTS/pocket-tts-zh-en")

WORK_TMP = os.path.join(PROJ, ".work_tmp")
os.makedirs(os.path.join(WORK_TMP, "tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.join(WORK_TMP, "tmp")
os.environ["MPLCONFIGDIR"] = os.path.join(WORK_TMP, "matplotlib")
os.environ["XDG_CACHE_HOME"] = os.path.join(WORK_TMP, "xdg_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

sys.path.insert(0, MODEL_ROOT)
sys.path.insert(0, PROJ)

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

import demo  # noqa: E402
from step_runtime import StepRuntime, resample_24k  # noqa: E402

SPM_MODEL = os.path.join(MODEL_ROOT, demo.SPM_MODEL)

_MIMI_LAYER_RE = re.compile(r"_(6|7)(/|_|$)")


def bucket_of(name: str) -> str:
    if "flow_net" in name:
        return "flow_net"
    if _MIMI_LAYER_RE.search(name):
        return "mimi_transformer_2l"
    if any(k in name for k in ("/conv", "convtr")) or name.startswith("/model.") \
            or "/block." in name or "/conv_7/" in name:
        return "mimi_conv_decoder"
    if "output_proj/Conv" in name or "Transpose_48" in name:
        return "quantizer_conv"
    if name.startswith("/Gather") or name.startswith("/Where"):
        return "input_switch_embed"
    if "flow_kv_new" in name or re.search(r"/Concat_(66|86)\b", name):
        return "kv_cache_concat"
    if any(k in name for k in ("/in_proj", "/out_proj", "/linear1", "/linear2",
                               "/norm1", "/norm2", "input_linear", "layer_scale")):
        return "flow_lm_transformer_6l"
    if any(k in name for k in ("/Range", "/GreaterOrEqual", "/And", "/Sin", "/Cos",
                               "/Sub", "/Where", "/Slice", "/Transpose")):
        return "shapes_attn_aux"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description="pocket-tts-zh-en ORT profile")
    ap.add_argument("--model", default="step_onnx_int8", choices=["step_onnx", "step_onnx_int8"])
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--text", default="The train arrived before sunset, and the station became quiet.")
    ap.add_argument("--out-dir", default=os.path.join(PROJ, "results", "profiles"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    profile_prefix = os.path.join(args.out_dir, f"ortprof_{args.model}")

    model_dir = os.path.join(MODEL_ROOT, args.model)
    rt = StepRuntime(model_dir, intra_op_num_threads=args.threads)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.enable_profiling = True
    opts.profile_file_prefix = profile_prefix
    rt.session = ort.InferenceSession(
        os.path.join(model_dir, "step_model.onnx"), opts, providers=["CPUExecutionProvider"]
    )
    rt._output_names = [o.name for o in rt.session.get_outputs()]

    ref, sr = None, None
    import soundfile as sf  # local import to keep header light

    audio, sr = sf.read(os.path.join(MODEL_ROOT, "Vivian.wav"), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    ref = resample_24k(audio, sr).astype(np.float32)
    ids = demo.text2ids(args.text, SPM_MODEL)

    n = 0
    for _ in rt.stream(ids, ref, temp=0.0, max_frames=args.frames, seed=0):
        n += 1
    profile_path = rt.session.end_profiling()
    print(f"ran {n} frames, profile: {profile_path}")

    with open(profile_path) as f:
        prof = json.load(f)

    node_events = [e for e in prof if e.get("cat") == "Node" and "dur" in e]
    total_us = sum(e["dur"] for e in node_events)

    by_bucket: dict[str, float] = collections.defaultdict(float)
    by_op: dict[str, float] = collections.defaultdict(float)
    by_node: dict[str, float] = collections.defaultdict(float)
    for e in node_events:
        d = e["dur"]
        by_bucket[bucket_of(e["name"])] += d
        by_op[e.get("args", {}).get("op_name", "?")] += d
        by_node[e["name"]] += d

    def top(d: dict, k: int = 30):
        return [
            {"name": name, "ms": round(us / 1000.0, 3), "pct": round(100.0 * us / total_us, 2)}
            for name, us in sorted(d.items(), key=lambda kv: -kv[1])[:k]
        ]

    report = {
        "meta": {
            "model": args.model,
            "text": args.text,
            "frames": n,
            "threads": args.threads,
            "host": platform.node(),
            "profile_file": os.path.basename(profile_path),
            "note": "single-session aggregate over prefill + frames; host CPU only",
        },
        "total_node_ms": round(total_us / 1000.0, 3),
        "total_node_ms_per_frame": round(total_us / 1000.0 / max(n, 1), 3),
        "buckets": [
            {"name": k, "ms": round(v / 1000.0, 3), "pct": round(100.0 * v / total_us, 2)}
            for k, v in sorted(by_bucket.items(), key=lambda kv: -kv[1])
        ],
        "op_types": top(by_op, 20),
        "top_nodes": top(by_node, 30),
    }
    out_path = args.out or os.path.join(PROJ, "results", f"profile_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report["buckets"], ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
