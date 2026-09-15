#!/usr/bin/env python3
"""Runtime assembled from the extracted pocket-tts subgraphs.

Mirrors the vendor StepRuntime streaming algorithm, but calls the three
subgraph sessions (flow / flow_net / mimi_decoder) instead of the fused graph.

flow_window: if set, only the last N flow KV entries are fed each step; the
windowed flow_step graph must be used (see patch_flow_window.py).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import onnxruntime as ort

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.environ.get("POCKET_TTS_ROOT", "/data/shared/huyuan/TTS/pocket-tts-zh-en")

sys.path.insert(0, MODEL_ROOT)

from step_runtime import StepRuntime, _Config  # noqa: E402


class SubgraphRuntime(StepRuntime):
    def __init__(
        self,
        model_dir: str,
        manifest: dict,
        variant: str = "raw",
        flow_step_path: str | None = None,
        flow_window: int | None = None,
        intra_op_num_threads: int = 4,
        providers=None,
    ):
        self.dir = model_dir
        with open(os.path.join(model_dir, "step_config.json")) as f:
            self.config = _Config(**json.load(f))

        self.flow_window = flow_window
        if providers is None:
            providers = ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        if intra_op_num_threads > 0:
            options.intra_op_num_threads = intra_op_num_threads
        options.inter_op_num_threads = 1

        sg = manifest["subgraphs"]

        def path_of(name: str) -> str:
            entry = sg[name]
            p = entry["raw"] if variant == "raw" else entry["onnxsim"]
            return os.path.join(PROJ, p)

        self.flow_path = flow_step_path or path_of("flow_step")
        self.flow_session = ort.InferenceSession(self.flow_path, options, providers=providers)
        self.flow_net_session = ort.InferenceSession(path_of("flow_net_step"), options, providers=providers)
        self.mimi_session = ort.InferenceSession(path_of("mimi_decoder_step"), options, providers=providers)
        self.encoder = ort.InferenceSession(
            os.path.join(model_dir, "step_encoder.onnx"), options, providers=providers
        )
        self._flow_out_names = [o.name for o in self.flow_session.get_outputs()]
        self._mimi_out_names = [o.name for o in self.mimi_session.get_outputs()]
        self._voice_cache: dict[str, np.ndarray] = {}
        self.records: list[dict[str, np.ndarray]] | None = None

    def _record(self, out: dict) -> None:
        if self.records is not None:
            self.records.append({k: np.array(v) for k, v in out.items()})

    def flow_cache_view(self, flow_kv):
        if self.flow_window is not None and flow_kv.length > self.flow_window:
            return flow_kv.window(self.flow_window)
        return flow_kv.window()

    def _run(self, *, gates, seq, noise, flow_kv, mimi_kv, mimi_offset, mimi_conv,
             tokens=None, latent=None, is_bos=None, cond=None):
        cfg = self.config
        zeros = lambda *shape: np.zeros(shape, dtype=np.float32)  # noqa: E731
        flow_feeds = {
            "tokens": tokens if tokens is not None else np.zeros((1, seq), dtype=np.int64),
            "latent": latent if latent is not None else zeros(1, seq, cfg.latent_dim),
            "is_bos": is_bos if is_bos is not None else zeros(1, seq, 1),
            "cond": cond if cond is not None else zeros(1, seq, cfg.model_dim),
            "gates": gates,
            "flow_kv": self.flow_cache_view(flow_kv),
            "flow_offset": np.asarray(flow_kv.length, dtype=np.int64),
        }
        flow_out = dict(zip(self._flow_out_names, self.flow_session.run(None, flow_feeds)))
        c = flow_out["conditioning"]

        next_latent = self.flow_net_session.run(None, {
            "conditioning": c,
            "noise": noise,
            "decode_steps": np.asarray(float(cfg.decode_steps), dtype=np.float32),
        })[0]

        mimi_out = dict(zip(self._mimi_out_names, self.mimi_session.run(None, {
            "next_latent": next_latent,
            "mimi_kv": mimi_kv.window(cfg.mimi_kv_len),
            "mimi_conv": mimi_conv,
            "mimi_offset": mimi_offset,
        })))

        out = {
            "conditioning": c,
            "eos_logit": flow_out["eos_logit"],
            "flow_kv_new": flow_out["flow_kv_new"],
            "next_latent": next_latent,
            "audio": mimi_out["audio"],
            "mimi_kv_new": mimi_out["mimi_kv_new"],
            "mimi_conv_out": mimi_out["mimi_conv_out"],
            "mimi_offset_out": mimi_out["mimi_offset_out"],
        }
        self._record(out)
        return out
