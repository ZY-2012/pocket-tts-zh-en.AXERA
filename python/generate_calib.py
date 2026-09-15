#!/usr/bin/env python3
"""M2.1: record calibration feeds for the static subgraphs (flow_net, mimi_decoder).

Runs the validated windowed SubgraphRuntime (fp32, exact) over several texts and
voices with temperature 0.3, records per-step inputs of the two quantization
candidates, then writes one tar.gz per tensor in Pulsar2 "Numpy" format
(inside: <name>/NNNNN.npy). Only the tar.gz files are kept (no npy duplicates).

Usage:
    python python/generate_calib.py --samples-per-run 32
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.environ.get("POCKET_TTS_ROOT", "/data/shared/huyuan/TTS/pocket-tts-zh-en")
ZIPVOICE_ASSETS = os.environ.get("ZIPVOICE_ASSETS",
    "/data/shared/huyuan/TTS_quant/ZipVoice.AXERA/assets/moss_prompts")

WORK_TMP = os.path.join(PROJ, ".work_tmp")
os.makedirs(os.path.join(WORK_TMP, "tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.join(WORK_TMP, "tmp")

sys.path.insert(0, MODEL_ROOT)
sys.path.insert(0, os.path.join(PROJ, "python"))

import numpy as np  # noqa: E402

import demo  # noqa: E402
from step_runtime import resample_24k  # noqa: E402
from subgraph_runtime import SubgraphRuntime  # noqa: E402

SPM_MODEL = os.path.join(MODEL_ROOT, demo.SPM_MODEL)

REFS = {"vivian": os.path.join(MODEL_ROOT, "Vivian.wav")}
for _name, _fname in (("en", "en_4_4p5s.wav"), ("zh", "zh_1_4p5s.wav")):
    _path = os.path.join(ZIPVOICE_ASSETS, _fname)
    if os.path.exists(_path):
        REFS[_name] = _path

EXTRA_TEXTS = {
    "zh_tech": "这次发布的新版本带来了更快的推理速度和更低的显存占用，开发者可以在本地设备上运行大模型。",
    "zh_poem": "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "en_long": "Autumn arrived early that year, and the narrow streets were covered with leaves "
               "as the old clock tower struck six in the evening.",
    "mix_long": "欢迎体验我们的语音合成 Demo，它支持 voice cloning 和 bilingual synthesis，效果怎么样？",
}


class SessionRecorder:
    """Wraps an ORT session and records the feed dict of every run()."""

    def __init__(self, session, bucket: list[dict[str, np.ndarray]]):
        self.session = session
        self.bucket = bucket

    def run(self, output_names, feeds):
        self.bucket.append({k: np.array(v) for k, v in feeds.items()})
        return self.session.run(output_names, feeds)

    def get_outputs(self):
        return self.session.get_outputs()


def safe_input_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def write_tensor_archive(out_dir: str, model_key: str, input_name: str,
                         samples: list[np.ndarray]) -> dict:
    safe = safe_input_name(input_name)
    model_dir = os.path.join(out_dir, model_key)
    os.makedirs(model_dir, exist_ok=True)
    staging = os.path.join(model_dir, f".staging_{safe}")
    os.makedirs(staging, exist_ok=True)
    tar_path = os.path.join(model_dir, f"{safe}.tar.gz")
    entries = []
    with tarfile.open(tar_path, "w:gz") as tar:
        for idx, value in enumerate(samples):
            value = np.asarray(value)
            if value.shape == ():
                value = value.reshape(1)
            npy_path = os.path.join(staging, f"{idx:05d}.npy")
            np.save(npy_path, value)
            tar.add(npy_path, arcname=f"{safe}/{idx:05d}.npy")
            entries.append({"file": f"{model_key}/{safe}/{idx:05d}.npy",
                            "shape": list(value.shape), "dtype": str(value.dtype)})
    shutil.rmtree(staging)
    return {"tar": os.path.relpath(tar_path, PROJ), "count": len(samples),
            "entries": entries[:3] + (["..."] if len(entries) > 3 else [])}


def main() -> None:
    ap = argparse.ArgumentParser(description="generate subgraph calibration data")
    ap.add_argument("--out-dir", default=os.path.join(PROJ, "model_convert", "calib_data"))
    ap.add_argument("--samples-per-run", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-frames", type=int, default=375)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--refs", nargs="+", default=list(REFS), choices=list(REFS))
    args = ap.parse_args()

    with open(os.path.join(PROJ, "configs", "baseline_texts.json")) as f:
        texts = {t["tag"]: t["text"] for t in json.load(f)["texts"]}
    texts.update(EXTRA_TEXTS)

    import soundfile as sf

    with open(os.path.join(PROJ, "models", "subgraphs", "manifest.json")) as f:
        manifest = json.load(f)
    rt = SubgraphRuntime(
        os.path.join(MODEL_ROOT, "step_onnx"), manifest, variant="raw",
        flow_step_path=os.path.join(PROJ, "models", "subgraphs", "windowed", "flow_step.onnx"),
        flow_window=512, intra_op_num_threads=args.threads,
    )
    flow_feeds: list[dict[str, np.ndarray]] = []
    mimi_feeds: list[dict[str, np.ndarray]] = []
    rt.flow_net_session = SessionRecorder(rt.flow_net_session, flow_feeds)
    rt.mimi_session = SessionRecorder(rt.mimi_session, mimi_feeds)

    out_dir = os.path.join(args.out_dir, "subgraphs")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    picks: list[list[int]] = []
    runs = []
    flow_by_key: dict[str, list[np.ndarray]] = {}
    mimi_by_key: dict[str, list[np.ndarray]] = {}
    for ref_name in args.refs:
        audio, sr = sf.read(REFS[ref_name], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        ref = resample_24k(audio, sr).astype(np.float32)
        for tag, text in texts.items():
            ids = demo.text2ids(text, SPM_MODEL)
            flow_feeds.clear()
            mimi_feeds.clear()
            frames = 0
            for _ in rt.stream(ids, ref, temp=args.temperature, max_frames=args.max_frames,
                               seed=args.seed):
                frames += 1
            n = len(mimi_feeds)
            k = min(args.samples_per_run, n)
            idx = np.unique(np.linspace(0, n - 1, k).round().astype(int)).tolist()
            flow_pick = [flow_feeds[i] for i in idx if i < len(flow_feeds)]
            mimi_pick = [mimi_feeds[i] for i in idx]
            picks.append(idx)
            runs.append({"ref": ref_name, "text_tag": tag, "tokens": len(ids),
                         "frames": frames, "recorded": n, "picked": len(idx)})
            print(f"  {ref_name:6s}/{tag:10s} tokens={len(ids):3d} frames={frames:3d} "
                  f"recorded={n:3d} picked={len(idx):3d}")
            for d in flow_pick:
                for key, value in d.items():
                    flow_by_key.setdefault(key, []).append(value)
            for d in mimi_pick:
                for key, value in d.items():
                    mimi_by_key.setdefault(key, []).append(value)

    manifest_out = {"runs": runs, "temperature": args.temperature, "seed": args.seed,
                    "flow_net_step": {}, "mimi_decoder_step": {}}
    for key, samples in flow_by_key.items():
        manifest_out["flow_net_step"][key] = write_tensor_archive(out_dir, "flow_net_step", key, samples)
    for key, samples in mimi_by_key.items():
        manifest_out["mimi_decoder_step"][key] = write_tensor_archive(out_dir, "mimi_decoder_step", key, samples)

    total_mb = 0.0
    for model in ("flow_net_step", "mimi_decoder_step"):
        for key, entry in manifest_out[model].items():
            p = os.path.join(PROJ, entry["tar"])
            total_mb += os.path.getsize(p) / 1e6
            print(f"  {model}/{key}: {entry['count']} samples, {os.path.getsize(p) / 1e6:.1f} MB")
    manifest_out["total_mb"] = round(total_mb, 1)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest_out, f, ensure_ascii=False, indent=2)
    print(f"wrote {os.path.join(out_dir, 'manifest.json')} (total {total_mb:.1f} MB)")


if __name__ == "__main__":
    main()
