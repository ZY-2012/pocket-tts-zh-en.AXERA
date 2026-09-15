#!/usr/bin/env python3
"""M9: static QDQ quantization of the flow AR step graph.

The dynamic-int8 AR graph spends ~26% of its time in DynamicQuantizeLinear /
DynamicQuantizeMatMul (per-call activation range scans). Static QDQ precomputes
activation scales from real runtime trajectories.

Output: models/subgraphs/flow/flow_ar_step_static_int8.onnx

Usage:
    python python/quantize_ar_static.py --steps 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.environ.get("POCKET_TTS_ROOT", "/data/shared/huyuan/TTS/pocket-tts-zh-en")
os.environ.setdefault("TMPDIR", os.path.join(PROJ, ".work_tmp", "tmp"))
sys.path.insert(0, os.path.join(PROJ, "python"))
sys.path.insert(0, os.path.join(PROJ, "board"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnxruntime.quantization import (  # noqa: E402
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)

from pocket_tts_axera import TextFrontend, resample_24k  # noqa: E402
from subgraph_runtime import SubgraphRuntime  # noqa: E402

SPM = os.path.join(MODEL_ROOT, "chn_jpn_yue_eng_ko_spectok.bpe.model")
SRC = os.path.join(PROJ, "models", "subgraphs", "flow", "flow_ar_step.onnx")
DST = os.path.join(PROJ, "models", "subgraphs", "flow", "flow_ar_step_static_int8.onnx")

TEXTS = [
    "今天天气不错，我们一起去公园散步吧。",
    "订单号是2026年3月15日，单价128元，共3件。",
    "The train arrived before sunset, and the station became quiet.",
    "人工智能正在改变我们的生活方式。从智能手机到自动驾驶汽车，从语音助手到机器翻译，这些技术都在不断提升我们的工作效率。未来，随着算力的增长和算法的进步，我们将会看到更多令人惊叹的应用场景。",
]
REF = os.path.join(MODEL_ROOT, "Vivian.wav")


class SessionRecorder:
    """Records the AR-step feeds (seq==1) fed to the flow session."""

    def __init__(self, session, bucket: list):
        self.session = session
        self.bucket = bucket

    def run(self, output_names, feeds):
        if feeds["latent"].shape[1] == 1:
            self.bucket.append({k: np.array(v) for k, v in feeds.items()
                                if k in ("latent", "is_bos", "flow_kv", "flow_offset")})
        return self.session.run(output_names, feeds)

    def get_outputs(self):
        return self.session.get_outputs()


class FeedReader(CalibrationDataReader):
    def __init__(self, feeds: list[dict]):
        self.it = iter(feeds)

    def get_next(self):
        try:
            return next(self.it)
        except StopIteration:
            return None


def main() -> None:
    ap = argparse.ArgumentParser(description="static QDQ for flow AR")
    ap.add_argument("--steps", type=int, default=3, help="subsample every N AR steps")
    ap.add_argument("--max-samples", type=int, default=140)
    ap.add_argument("--per-channel", type=int, default=1)
    args = ap.parse_args()

    import soundfile as sf

    audio, sr = sf.read(REF, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    ref = resample_24k(audio, sr).astype(np.float32)
    frontend = TextFrontend(SPM)

    with open(os.path.join(PROJ, "models", "subgraphs", "manifest.json")) as f:
        manifest = json.load(f)
    rt = SubgraphRuntime(
        os.path.join(MODEL_ROOT, "step_onnx"), manifest, variant="raw",
        flow_step_path=os.path.join(PROJ, "models", "subgraphs", "windowed", "flow_step.onnx"),
        flow_window=512, intra_op_num_threads=8,
    )
    ar_feeds: list[dict] = []
    rt.flow_session = SessionRecorder(rt.flow_session, ar_feeds)

    for text in TEXTS:
        ids = frontend.text2ids(text)
        ar_feeds.clear()
        for _ in rt.stream(ids, ref, temp=0.0, max_frames=375, seed=0):
            pass
        picked = ar_feeds[:: args.steps][: args.max_samples]
        all_feeds.extend(picked)
        print(f"  collected {len(picked)} samples from {len(ar_feeds)} steps: {text[:24]}")

    print(f"total calibration samples: {len(all_feeds)}")
    t0 = time.perf_counter()
    quantize_static(
        SRC,
        DST,
        FeedReader(all_feeds),
        quant_format=QuantFormat.QOperator,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=bool(args.per_channel),
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={"ActivationSymmetric": False},
    )
    onnx.checker.check_model(DST)
    mb = os.path.getsize(DST) / 1e6
    print(f"wrote {DST} ({mb:.1f} MB, {time.perf_counter() - t0:.0f}s)")

    with open(os.path.join(PROJ, "models", "subgraphs", "flow", "ar_static_info.json"), "w") as f:
        json.dump({"source": os.path.relpath(SRC, PROJ), "static_int8": os.path.relpath(DST, PROJ),
                   "samples": len(all_feeds), "size_mb": round(mb, 1),
                   "per_channel": bool(args.per_channel)}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    all_feeds: list[dict] = []
    main()
