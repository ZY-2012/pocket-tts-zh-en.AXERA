#!/usr/bin/env python3
"""M2: verify mimi split (transformer + conv decoder) against the original graph.

Usage:
    python python/validate_mimi_split.py --samples 8
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TMPDIR", os.path.join(PROJ, ".work_tmp", "tmp"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402


def load_samples(tar_path: str, count: int, skip: int = 2) -> list[np.ndarray]:
    out = []
    with tarfile.open(tar_path) as tar:
        members = sorted((m for m in tar.getmembers() if m.name.endswith(".npy")),
                         key=lambda m: m.name)
        for member in members[skip:skip + count]:
            out.append(np.load(io.BytesIO(tar.extractfile(member).read())))
    return out


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb + 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser(description="validate mimi split")
    ap.add_argument("--original", default=os.path.join(PROJ, "models/subgraphs/onnxsim/mimi_decoder_step.onnx"))
    ap.add_argument("--split-dir", default=os.path.join(PROJ, "models/subgraphs/mimi_split"))
    ap.add_argument("--calib-dir", default=os.path.join(PROJ, "model_convert/calib_data/subgraphs/mimi_decoder_step"))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(PROJ, "results/validate_mimi_split.json"))
    args = ap.parse_args()

    orig = ort.InferenceSession(args.original, providers=["CPUExecutionProvider"])
    tf = ort.InferenceSession(os.path.join(args.split_dir, "mimi_transformer_step.onnx"),
                              providers=["CPUExecutionProvider"])
    conv = ort.InferenceSession(os.path.join(args.split_dir, "mimi_conv_step.onnx"),
                                providers=["CPUExecutionProvider"])

    data = {name: load_samples(os.path.join(args.calib_dir, f"{name}.tar.gz"), args.samples)
            for name in ("next_latent", "mimi_kv", "mimi_conv", "mimi_offset")}
    stats = {}

    def record(key, a, b):
        entry = stats.setdefault(key, {"max_abs_diff": 0.0, "min_cosine": 1.0})
        entry["max_abs_diff"] = max(entry["max_abs_diff"], float(np.max(np.abs(a - b))))
        entry["min_cosine"] = min(entry["min_cosine"], cosine(a, b))

    for i in range(args.samples):
        feed = {name: values[i] for name, values in data.items()}
        oo = dict(zip([o.name for o in orig.get_outputs()], orig.run(None, feed)))
        tf_out = dict(zip([o.name for o in tf.get_outputs()], tf.run(None, {
            "next_latent": feed["next_latent"],
            "mimi_kv": feed["mimi_kv"],
            "mimi_conv": feed["mimi_conv"],
            "mimi_offset": feed["mimi_offset"],
        })))
        conv_out = dict(zip([o.name for o in conv.get_outputs()], conv.run(None, {
            "decoder_embedding": tf_out["decoder_embedding"],
            "mimi_conv": feed["mimi_conv"],
            "upsample_state": tf_out["upsample_state"],
        })))
        record("audio", oo["audio"], conv_out["audio"])
        record("mimi_kv_new", oo["mimi_kv_new"], tf_out["mimi_kv_new"])
        record("mimi_conv_out", oo["mimi_conv_out"], conv_out["mimi_conv_out"])
        record("mimi_offset_out", oo["mimi_offset_out"].astype(np.int64),
               feed["mimi_offset"].astype(np.int64) + 16)

    print(json.dumps(stats, indent=2))
    with open(args.out, "w") as f:
        json.dump({"samples": args.samples, "stats": stats}, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
