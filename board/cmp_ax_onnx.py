#!/usr/bin/env python3
"""Board single-call A/B: axmodel vs ONNX subgraph on identical calibration samples."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pocket_tts_axera import Session, cast_feed  # noqa: E402


def compare(ax: Session, on: Session, feed: dict, tag: str) -> None:
    ax_out = ax.run(cast_feed(ax, feed))
    on_out = on.run(cast_feed(on, feed))
    print(f"== {tag}")
    for key in ax_out:
        a = np.asarray(ax_out[key], dtype=np.float64).reshape(-1)
        b = np.asarray(on_out[key], dtype=np.float64).reshape(-1)
        if a.size == 0:
            continue
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        print(f"   {key:16s} shape={ax_out[key].shape} max|a|={np.abs(a).max():.4f} "
              f"max|on|={np.abs(b).max():.4f} maxdiff={np.abs(a - b).max():.4e} cos={cos:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="ax vs onnx single-sample compare")
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--axmodel-dir", required=True)
    ap.add_argument("--samples-dir", required=True)
    args = ap.parse_args()

    flow_ax = Session(os.path.join(args.axmodel_dir, "flow_net_step.axmodel"), "ax")
    flow_on = Session(os.path.join(args.onnx_dir, "flow_net_step.onnx"), "onnx")
    mimi_ax = Session(os.path.join(args.axmodel_dir, "mimi_decoder_step.axmodel"), "ax")
    mimi_on = Session(os.path.join(args.onnx_dir, "mimi_decoder_step.onnx"), "onnx")

    for idx in (20, 60):
        d = os.path.join(args.samples_dir, "flow_net_step")
        compare(flow_ax, flow_on, {
            "conditioning": np.load(os.path.join(d, f"conditioning_{idx:05d}.npy")),
            "noise": np.load(os.path.join(d, f"noise_{idx:05d}.npy")),
        }, f"flow_net sample {idx}")

        d = os.path.join(args.samples_dir, "mimi_decoder_step")
        compare(mimi_ax, mimi_on, {
            "next_latent": np.load(os.path.join(d, f"next_latent_{idx:05d}.npy")),
            "mimi_kv": np.load(os.path.join(d, f"mimi_kv_{idx:05d}.npy")),
            "mimi_conv": np.load(os.path.join(d, f"mimi_conv_{idx:05d}.npy")),
            "mimi_offset": np.load(os.path.join(d, f"mimi_offset_{idx:05d}.npy")),
        }, f"mimi sample {idx}")


if __name__ == "__main__":
    main()
