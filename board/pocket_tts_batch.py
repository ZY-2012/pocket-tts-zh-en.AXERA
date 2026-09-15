#!/usr/bin/env python3
"""Board batch synthesis driver: load models once, synthesize N texts.

Used by the M5 listening pack and by the Voice_Test.AXERA benchmark (RTF main
convention: load once + warmup 1 item + time the rest).

Usage (board):
    python3 pocket_tts_batch.py --texts-json configs/listen_texts.json \
        --out-dir /path/out --onnx-dir models --axmodel-dir models \
        --mimi-split-dir models/mimi_split --spm models/xxx.bpe.model \
        --reference models/Vivian.wav --threads 4 --prefill-threads 8 \
        --flow-ar-model flow/flow_ar_step_int8.onnx \
        --flow-net-model flow_net_step_fp32.axmodel
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pocket_tts_axera import (  # noqa: E402
    SR, PocketTTS, TextFrontend, load_reference, write_wav,
)


def load_items(args) -> list[tuple[str, str]]:
    items = []
    if args.texts_json:
        with open(args.texts_json) as f:
            data = json.load(f)
        items = [(t["tag"], t["text"]) for t in data["texts"]]
    if args.texts_tsv:
        with open(args.texts_tsv) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                tag, text = line.split("\t", 1)
                items.append((tag, text))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="pocket-tts-zh-en board batch synthesis")
    ap.add_argument("--texts-json", default=None)
    ap.add_argument("--texts-tsv", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--axmodel-dir", required=True)
    ap.add_argument("--cpu-model-dir", default=None)
    ap.add_argument("--mimi-split-dir", default=None)
    ap.add_argument("--spm", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prefill-threads", type=int, default=None)
    ap.add_argument("--flow-ar-model", default=None)
    ap.add_argument("--flow-prefill-model", default="flow_step_windowed.onnx")
    ap.add_argument("--mimi-tf-model", default="mimi_transformer_step.onnx")
    ap.add_argument("--encoder-dir", default=None)
    ap.add_argument("--pause-ms", type=int, default=120)
    ap.add_argument("--flow-net-model", default="flow_net_step.axmodel")
    ap.add_argument("--mimi-conv-model", default="mimi_conv_step.axmodel")
    ap.add_argument("--flow-window", type=int, default=512)
    ap.add_argument("--max-frames", type=int, default=375)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1, help="warmup items excluded from metrics")
    ap.add_argument("--with-stages", type=int, default=1)
    args = ap.parse_args()

    items = load_items(args)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no texts")

    os.makedirs(args.out_dir, exist_ok=True)
    frontend = TextFrontend(args.spm)
    tts = PocketTTS(args.onnx_dir, args.axmodel_dir, cpu_model_dir=args.cpu_model_dir,
                    mimi_split_dir=args.mimi_split_dir,
                    threads=args.threads, flow_window=args.flow_window,
                    npu_flow_net=bool(args.flow_net_model.endswith(".axmodel")),
                    npu_mimi_conv=bool(args.mimi_conv_model.endswith(".axmodel")),
                    flow_net_model=args.flow_net_model,
                    mimi_conv_model=args.mimi_conv_model,
                    flow_ar_model=args.flow_ar_model,
                    flow_prefill_model=args.flow_prefill_model,
                    mimi_tf_model=args.mimi_tf_model,
                    encoder_dir=args.encoder_dir,
                    prefill_threads=args.prefill_threads)
    ref = load_reference(args.reference)
    print(f"load: {tts.load_s:.2f}s  ref={os.path.basename(args.reference)} {len(ref)/tts.cfg['sample_rate']:.2f}s")

    results = []
    measured = 0
    for idx, (tag, text) in enumerate(items):
        chunks = frontend.chunk_text(text) if args.chunk else [text]
        t_item = time.perf_counter()
        pieces = []
        first_frame_ms = None
        timing = {}
        pause = (np.zeros(int(SR * args.pause_ms / 1000), dtype=np.float32)
                 if args.pause_ms > 0 else None)
        for ci, chunk in enumerate(chunks):
            ids = frontend.text2ids(chunk)
            t0 = time.perf_counter()
            frames = []
            for frame in tts.stream(ids, ref, temp=args.temp, max_frames=args.max_frames,
                                    seed=args.seed,
                                    timing=timing if (args.with_stages and ci == len(chunks) - 1) else None):
                if first_frame_ms is None:
                    first_frame_ms = (time.perf_counter() - t0) * 1000
                frames.append(frame)
            pieces.append(np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32))
            if pause is not None and ci < len(chunks) - 1:
                pieces.append(pause)
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        total_s = time.perf_counter() - t_item
        seconds = len(audio) / tts.cfg["sample_rate"]
        wav_path = os.path.join(args.out_dir, f"{tag}.wav")
        write_wav(wav_path, audio)
        warm = idx < args.warmup
        entry = {
            "tag": tag,
            "text": text,
            "chunks": chunks,
            "tokens": len(frontend.text2ids(text)),
            "audio_seconds": round(seconds, 3),
            "total_s": round(total_s, 3),
            "rtf": round(total_s / seconds, 4) if seconds > 0 else None,
            "first_frame_ms": round(first_frame_ms or 0, 1),
            "warmup": warm,
            "wav": wav_path,
        }
        if args.with_stages and timing.get("stage_ms_mean"):
            entry["stage_ms_mean"] = timing["stage_ms_mean"]
        results.append(entry)
        if not warm:
            measured += 1
        print(f"{'[warm]' if warm else '      '} {tag:14s} chunks={len(chunks)} "
              f"audio={seconds:6.2f}s total={total_s:6.2f}s "
              f"RTF={(entry['rtf'] if entry['rtf'] is not None else -1):.3f}")

    measured_entries = [e for e in results if not e["warmup"] and e["rtf"]]
    summary = {
        "load_s": round(tts.load_s, 3),
        "n_items": len(results),
        "n_measured": len(measured_entries),
        "mean_rtf": round(float(np.mean([e["rtf"] for e in measured_entries])), 4)
        if measured_entries else None,
        "mean_first_frame_ms": round(float(np.mean([e["first_frame_ms"] for e in measured_entries])), 1)
        if measured_entries else None,
        "reference": os.path.basename(args.reference),
        "flow_ar_model": args.flow_ar_model,
        "flow_net_model": args.flow_net_model,
        "mimi_conv_model": args.mimi_conv_model,
        "threads": args.threads,
        "items": results,
    }
    out_json = os.path.join(args.out_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, ensure_ascii=False))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
