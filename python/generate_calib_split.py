#!/usr/bin/env python3
"""M2: calibration archives for the split mimi conv decoder.

Replays the existing mimi_decoder_step calibration samples through the
mimi_transformer_step graph and archives decoder_embedding / upsample_state
together with the raw mimi_conv state.

Usage:
    python python/generate_calib_split.py
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tarfile

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TMPDIR", os.path.join(PROJ, ".work_tmp", "tmp"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402


def load_archive(path: str) -> list[np.ndarray]:
    out = []
    with tarfile.open(path) as tar:
        members = sorted((m for m in tar.getmembers() if m.name.endswith(".npy")),
                         key=lambda m: m.name)
        for member in members:
            out.append(np.load(io.BytesIO(tar.extractfile(member).read())))
    return out


def write_archive(out_dir: str, model_key: str, input_name: str,
                  samples: list[np.ndarray]) -> dict:
    safe = input_name.replace("/", "_")
    model_dir = os.path.join(out_dir, model_key)
    os.makedirs(model_dir, exist_ok=True)
    staging = os.path.join(model_dir, f".staging_{safe}")
    os.makedirs(staging, exist_ok=True)
    tar_path = os.path.join(model_dir, f"{safe}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for idx, value in enumerate(samples):
            npy = os.path.join(staging, f"{idx:05d}.npy")
            np.save(npy, np.asarray(value))
            tar.add(npy, arcname=f"{safe}/{idx:05d}.npy")
    shutil.rmtree(staging)
    return {"tar": os.path.relpath(tar_path, PROJ), "count": len(samples)}


def main() -> None:
    src_dir = os.path.join(PROJ, "model_convert", "calib_data", "subgraphs", "mimi_decoder_step")
    out_dir = os.path.join(PROJ, "model_convert", "calib_data", "subgraphs")
    tf = ort.InferenceSession(
        os.path.join(PROJ, "models", "subgraphs", "mimi_split", "mimi_transformer_step.onnx"),
        providers=["CPUExecutionProvider"])

    data = {name: load_archive(os.path.join(src_dir, f"{name}.tar.gz"))
            for name in ("next_latent", "mimi_kv", "mimi_conv", "mimi_offset")}
    n = min(len(v) for v in data.values())
    print(f"replaying {n} samples through mimi_transformer_step")

    embeddings, ups = [], []
    for i in range(n):
        out = dict(zip([o.name for o in tf.get_outputs()], tf.run(None, {
            "next_latent": data["next_latent"][i],
            "mimi_kv": data["mimi_kv"][i],
            "mimi_conv": data["mimi_conv"][i],
            "mimi_offset": data["mimi_offset"][i],
        })))
        embeddings.append(out["decoder_embedding"])
        ups.append(out["upsample_state"])

    manifest = {}
    for name, samples in (("decoder_embedding", embeddings), ("mimi_conv", data["mimi_conv"]),
                          ("upsample_state", ups)):
        manifest[name] = write_archive(out_dir, "mimi_conv_step", name, samples)
        print(f"  {name}: {manifest[name]['count']} samples -> {manifest[name]['tar']}")
    with open(os.path.join(out_dir, "mimi_conv_step", "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
