#!/usr/bin/env python3
"""M6.3: static-shape encoder variants for NPU (input frames 40 / 60 / 80).

Output: models/subgraphs/encoder/step_encoder_<F>f.onnx (+ equivalence check)

Usage:
    python python/make_encoder_static.py
"""
from __future__ import annotations

import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TMPDIR", os.path.join(PROJ, ".work_tmp", "tmp"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import onnxsim  # noqa: E402

MODEL_ROOT = os.environ.get("POCKET_TTS_ROOT", "/data/shared/huyuan/TTS/pocket-tts-zh-en")
SRC = os.path.join(MODEL_ROOT, "step_onnx", "step_encoder.onnx")
OUT_DIR = os.path.join(PROJ, "models", "subgraphs", "encoder")
TIERS = [40, 41, 44, 48]


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    dynamic = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    info = {"source": SRC, "tiers": {}}

    for frames in TIERS:
        samples = frames * 1920
        model = onnx.load(SRC, load_external_data=False)
        sim, ok = onnxsim.simplify(
            model, overwrite_input_shapes={"audio": [1, 1, samples]})
        if not ok:
            raise RuntimeError(f"onnxsim failed for tier {frames}")
        onnx.checker.check_model(sim)
        out_path = os.path.join(OUT_DIR, f"step_encoder_{frames}f.onnx")
        onnx.save(sim, out_path)

        x = rng.standard_normal((1, 1, samples)).astype(np.float32)
        a = dynamic.run(None, {"audio": x})[0]
        static_sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        b = static_sess.run(None, {"audio": x})[0]
        entry = {
            "frames": frames,
            "samples": samples,
            "seconds": round(samples / 24000, 3),
            "cond_frames": int(b.shape[1]),
            "onnx": os.path.relpath(out_path, PROJ),
            "size_mb": round(os.path.getsize(out_path) / 1e6, 1),
            "max_abs_diff": float(np.max(np.abs(a - b))),
            "cosine": cosine(a, b),
        }
        info["tiers"][f"{frames}f"] = entry
        print(f"tier {frames}f ({entry['seconds']}s) -> cond {b.shape} "
              f"max_abs={entry['max_abs_diff']:.2e} cos={entry['cosine']:.8f} "
              f"({entry['size_mb']} MB)")

    with open(os.path.join(OUT_DIR, "encoder_tiers.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"wrote {os.path.join(OUT_DIR, 'encoder_tiers.json')}")


if __name__ == "__main__":
    main()
