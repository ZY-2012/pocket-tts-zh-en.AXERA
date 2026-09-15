#!/usr/bin/env python3
"""M6.3: calibration archives for the static encoder tiers.

Inputs: the three reference voices (Vivian + moss prompts), padded/truncated to
each tier length, plus a silence sample.

Usage:
    python python/generate_calib_encoder.py
"""
from __future__ import annotations

import json
import os
import shutil
import tarfile

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TMPDIR", os.path.join(PROJ, ".work_tmp", "tmp"))

ZIPVOICE_ASSETS = os.environ.get("ZIPVOICE_ASSETS",
    "/data/shared/huyuan/TTS_quant/ZipVoice.AXERA/assets/moss_prompts")

import numpy as np  # noqa: E402

MODEL_ROOT = os.environ.get("POCKET_TTS_ROOT", "/data/shared/huyuan/TTS/pocket-tts-zh-en")
REFS = {"vivian": os.path.join(MODEL_ROOT, "Vivian.wav")}
for _name, _fname in (("en", "en_4_4p5s.wav"), ("zh", "zh_1_4p5s.wav")):
    _path = os.path.join(ZIPVOICE_ASSETS, _fname)
    if os.path.exists(_path):
        REFS[_name] = _path
TIERS = [40, 41, 44, 48]


def load_ref(path: str) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 24000:
        import math

        g = math.gcd(int(sr), 24000)
        audio = resample_poly(audio, 24000 // g, int(sr) // g)
    return audio.astype(np.float32)


def write_archive(out_dir: str, model_key: str, samples: list[np.ndarray]) -> dict:
    safe = "audio"
    model_dir = os.path.join(out_dir, model_key)
    os.makedirs(model_dir, exist_ok=True)
    staging = os.path.join(model_dir, f".staging_{safe}")
    os.makedirs(staging, exist_ok=True)
    tar_path = os.path.join(model_dir, f"{safe}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for idx, value in enumerate(samples):
            npy = os.path.join(staging, f"{idx:05d}.npy")
            np.save(npy, value)
            tar.add(npy, arcname=f"{safe}/{idx:05d}.npy")
    shutil.rmtree(staging)
    return {"tar": os.path.relpath(tar_path, PROJ), "count": len(samples),
            "shape": list(samples[0].shape)}


def main() -> None:
    out_dir = os.path.join(PROJ, "model_convert", "calib_data", "subgraphs")
    refs = {name: load_ref(path) for name, path in REFS.items()}
    manifest = {}
    for frames in TIERS:
        n = frames * 1920
        samples = []
        for name, audio in refs.items():
            if len(audio) >= n:
                samples.append(audio[:n][None, None])
            else:
                pad = np.zeros(n, dtype=np.float32)
                pad[: len(audio)] = audio
                samples.append(pad[None, None])
        samples.append(np.zeros((1, 1, n), dtype=np.float32))
        key = f"step_encoder_{frames}f"
        manifest[key] = write_archive(out_dir, key, samples)
        print(f"{key}: {manifest[key]['count']} samples shape {manifest[key]['shape']}")
    with open(os.path.join(out_dir, "encoder_calib_manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
