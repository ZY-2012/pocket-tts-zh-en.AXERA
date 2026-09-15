#!/usr/bin/env python3
"""M1: numerical equivalence check, fused step_model vs extracted subgraphs.

Runs the vendor StepRuntime (reference) and SubgraphRuntime on identical
inputs and compares every per-step output array.

Usage:
    python python/validate_subgraphs.py --variant raw
    python python/validate_subgraphs.py --variant onnxsim --refs en zh vivian
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.environ.get("POCKET_TTS_ROOT", "/data/shared/huyuan/TTS/pocket-tts-zh-en")
ZIPVOICE_ASSETS = "/data/shared/huyuan/TTS_quant/ZipVoice.AXERA/assets/moss_prompts"

WORK_TMP = os.path.join(PROJ, ".work_tmp")
os.makedirs(os.path.join(WORK_TMP, "tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.join(WORK_TMP, "tmp")

sys.path.insert(0, MODEL_ROOT)
sys.path.insert(0, os.path.join(PROJ, "python"))

import numpy as np  # noqa: E402

import demo  # noqa: E402
from step_runtime import StepRuntime, resample_24k  # noqa: E402
from subgraph_runtime import SubgraphRuntime  # noqa: E402

SPM_MODEL = os.path.join(MODEL_ROOT, demo.SPM_MODEL)

REFS = {
    "vivian": os.path.join(MODEL_ROOT, "Vivian.wav"),
    "en": os.path.join(ZIPVOICE_ASSETS, "en_4_4p5s.wav"),
    "zh": os.path.join(ZIPVOICE_ASSETS, "zh_1_4p5s.wav"),
}

FLOAT_KEYS = ["conditioning", "eos_logit", "flow_kv_new", "next_latent", "audio",
              "mimi_kv_new", "mimi_conv_out", "mimi_offset_out"]


def load_ref(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample_24k(audio, sr).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(a @ b / (na * nb))


class RecordingRuntime(StepRuntime):
    """Reference runtime, records every _run output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: list[dict[str, np.ndarray]] = []

    def _run(self, **kwargs):
        out = super()._run(**kwargs)
        self.records.append({k: np.array(v) for k, v in out.items()})
        return out


def compare(rec_a: list[dict], rec_b: list[dict]) -> dict:
    if len(rec_a) != len(rec_b):
        return {"step_count_match": False, "n_a": len(rec_a), "n_b": len(rec_b)}
    stats = {}
    for key in FLOAT_KEYS:
        max_abs = 0.0
        min_cos = 1.0
        shape_ok = True
        for ra, rb in zip(rec_a, rec_b):
            if key not in ra or key not in rb:
                continue
            a, b = ra[key], rb[key]
            if a.shape != b.shape:
                shape_ok = False
                break
            if np.issubdtype(a.dtype, np.floating):
                max_abs = max(max_abs, float(np.max(np.abs(a - b))))
                min_cos = min(min_cos, cosine(a, b))
        stats[key] = {"shape_match": shape_ok, "max_abs_diff": max_abs, "min_cosine": min_cos}
    return {"step_count_match": True, "n_steps": len(rec_a), "keys": stats}


def main() -> None:
    ap = argparse.ArgumentParser(description="validate extracted subgraphs")
    ap.add_argument("--model-dir", default=os.path.join(MODEL_ROOT, "step_onnx"))
    ap.add_argument("--manifest", default=os.path.join(PROJ, "models", "subgraphs", "manifest.json"))
    ap.add_argument("--variant", default="raw", choices=["raw", "onnxsim"])
    ap.add_argument("--flow-step", default=None, help="override flow_step onnx path")
    ap.add_argument("--flow-window", type=int, default=None, help="feed only last N flow KV")
    ap.add_argument("--refs", nargs="+", default=["vivian", "en", "zh"], choices=list(REFS))
    ap.add_argument("--texts", nargs="+", default=["zh_short", "en_short"])
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--force-frames", action="store_true",
                    help="disable EOS early-stop so both runtimes always emit max_frames")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(PROJ, "results", "validate_subgraphs.json"))
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(os.path.join(PROJ, "configs", "baseline_texts.json")) as f:
        texts = {t["tag"]: t["text"] for t in json.load(f)["texts"]}

    flow_path = os.path.join(PROJ, args.flow_step) if args.flow_step else None
    report = {"meta": {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "variant": args.variant,
        "flow_step": args.flow_step or f"<manifest:{args.variant}>",
        "flow_window": args.flow_window,
        "max_frames": args.max_frames,
        "threads": args.threads,
        "model_dir": args.model_dir,
    }, "cases": []}

    for ref_name in args.refs:
        ref = load_ref(REFS[ref_name])
        ref_rt = RecordingRuntime(args.model_dir, intra_op_num_threads=args.threads)
        sub_rt = SubgraphRuntime(args.model_dir, manifest, variant=args.variant,
                                 flow_step_path=flow_path, flow_window=args.flow_window,
                                 intra_op_num_threads=args.threads)
        sub_rt.records = []
        for tag in args.texts:
            ids = demo.text2ids(texts[tag], SPM_MODEL)
            ref_rt.records.clear()
            sub_rt.records.clear()
            for rt in (ref_rt, sub_rt):
                for _ in rt.stream(ids, ref, temp=0.0, max_frames=args.max_frames, seed=0,
                                   eos_threshold=1e9 if args.force_frames else None):
                    pass
            res = compare(ref_rt.records, sub_rt.records)
            res.update({"ref": ref_name, "text_tag": tag, "tokens": len(ids)})
            report["cases"].append(res)
            if not res["step_count_match"]:
                print(f"[FAIL] {ref_name}/{tag}: step count {res['n_a']} vs {res['n_b']}")
                continue
            worst = max((k["max_abs_diff"], name) for name, k in res["keys"].items()
                        if k["shape_match"])
            min_cos = min(k["min_cosine"] for k in res["keys"].values())
            print(f"[{'OK' if min_cos > 0.99999 and worst[0] < 1e-3 else 'DIFF'}] "
                  f"{ref_name}/{tag}: steps={res['n_steps']} min_cos={min_cos:.8f} "
                  f"max_abs={worst[0]:.3e} ({worst[1]})")

    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
