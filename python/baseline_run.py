#!/usr/bin/env python3
"""M0 baseline runner for pocket-tts-zh-en (vendor ONNX, host CPU).

Runs the vendor StepRuntime with fixed seed on a fixed text set for the given
models (step_onnx / step_onnx_int8), stores golden audio, token ids and timing
metrics under results/.

Host policy: never touch /tmp; all temp/caches stay inside the project.

Usage:
    python python/baseline_run.py --models step_onnx_int8
    python python/baseline_run.py --models step_onnx
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time

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
import soundfile as sf  # noqa: E402

import demo  # noqa: E402  (vendor text frontend: normalize_punctuation + spm)
from step_runtime import StepRuntime, resample_24k  # noqa: E402

SPM_MODEL = os.path.join(MODEL_ROOT, demo.SPM_MODEL)


def blake2b_hex(data: bytes, size: int = 16) -> str:
    return hashlib.blake2b(data, digest_size=size).hexdigest()


def pcm_digest(audio: np.ndarray) -> str:
    return blake2b_hex(np.ascontiguousarray(audio, dtype=np.float32).tobytes())


def load_reference(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample_24k(audio, sr).astype(np.float32)


def run_one(rt: StepRuntime, ids: list[int], ref: np.ndarray, cfg: dict) -> dict:
    gen = rt.stream(
        ids,
        ref,
        temp=cfg["temperature"],
        max_frames=cfg["max_frames"],
        seed=cfg["seed"],
    )
    t0 = time.perf_counter()
    frames = [next(gen)]
    t_first = time.perf_counter() - t0
    for frame in gen:
        frames.append(frame)
    t_total = time.perf_counter() - t0
    audio = np.concatenate(frames)
    seconds = len(audio) / rt.sample_rate
    return {
        "tokens": len(ids),
        "frames": len(frames),
        "audio_seconds": round(seconds, 4),
        "first_frame_ms": round(t_first * 1000.0, 1),
        "total_s": round(t_total, 4),
        "rtf": round(t_total / seconds, 4),
        "pcm_blake2b": pcm_digest(audio),
        "audio": audio,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="pocket-tts-zh-en M0 baseline")
    ap.add_argument("--models", nargs="+", default=["step_onnx", "step_onnx_int8"],
                    choices=["step_onnx", "step_onnx_int8"])
    ap.add_argument("--config", default=os.path.join(PROJ, "configs", "baseline_texts.json"))
    ap.add_argument("--out", default=os.path.join(PROJ, "results", "baseline.json"))
    ap.add_argument("--audio-dir", default=os.path.join(PROJ, "results", "audio"))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    ref_path = os.path.join(MODEL_ROOT, cfg["reference"])
    ref = load_reference(ref_path)
    os.makedirs(args.audio_dir, exist_ok=True)

    report = {
        "meta": {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "model_root": MODEL_ROOT,
            "reference": os.path.basename(ref_path),
            "reference_seconds": round(len(ref) / 24000, 4),
            "reference_pcm_blake2b": pcm_digest(ref),
            "seed": cfg["seed"],
            "temperature": cfg["temperature"],
            "max_frames": cfg["max_frames"],
            "threads": cfg["threads"],
            "note": "host CPU baseline; RTF is not the AX650 board RTF",
        },
        "texts": {t["tag"]: t["text"] for t in cfg["texts"]},
        "runs": [],
    }

    for model_name in args.models:
        model_dir = os.path.join(MODEL_ROOT, model_name)
        print(f"== {model_name} ==")
        t_load0 = time.perf_counter()
        rt = StepRuntime(model_dir, intra_op_num_threads=cfg["threads"])
        load_s = time.perf_counter() - t_load0
        print(f"load: {load_s:.2f}s")

        # warmup on the shortest text (excluded from metrics)
        warm_ids = demo.text2ids("你好。", SPM_MODEL)
        for _ in rt.stream(warm_ids, ref, temp=cfg["temperature"], max_frames=8, seed=cfg["seed"]):
            pass

        for item in cfg["texts"]:
            ids = demo.text2ids(item["text"], SPM_MODEL)
            res = run_one(rt, ids, ref, cfg)
            audio = res.pop("audio")
            wav_path = os.path.join(args.audio_dir, f"{model_name}_{item['tag']}.wav")
            sf.write(wav_path, audio, rt.sample_rate)
            res.update({
                "model": model_name,
                "tag": item["tag"],
                "text": item["text"],
                "token_ids": ids,
                "load_s": round(load_s, 3),
                "wav": os.path.relpath(wav_path, PROJ),
            })
            report["runs"].append(res)
            print(
                f"  {item['tag']:10s} tokens={res['tokens']:3d} frames={res['frames']:3d} "
                f"audio={res['audio_seconds']:6.2f}s first={res['first_frame_ms']:7.1f}ms "
                f"total={res['total_s']:7.2f}s RTF={res['rtf']:.3f}"
            )

    if os.path.exists(args.out):
        with open(args.out) as f:
            old = json.load(f)
        new_keys = {(r["model"], r["tag"]) for r in report["runs"]}
        kept = [r for r in old.get("runs", []) if (r["model"], r["tag"]) not in new_keys]
        report["runs"] = kept + report["runs"]
        if "meta" in old:
            report["meta"]["previous_models"] = sorted(
                {r["model"] for r in kept} | set(old["meta"].get("previous_models", []))
            )

    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
