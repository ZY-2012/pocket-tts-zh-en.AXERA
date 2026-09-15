#!/usr/bin/env python3
"""Board diagnostic: run axmodel and ONNX pipelines in lockstep.

Two independent universes (identical prefix state, identical zero noise):
ax = flow_net/mimi from axmodel; on = flow_net/mimi from ONNX.
The shared flow LM (CPU, fp32) is stepped twice per frame with each
universe's own KV cache / latent, so divergence is purely from the
quantized subgraphs.

Usage (board):
    python3 diag_stepwise.py --onnx-dir models --axmodel-dir models \
        --spm models/xxx.bpe.model --reference models/Vivian.wav --steps 20
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pocket_tts_axera import (  # noqa: E402
    Cache, PocketTTS, Session, TextFrontend, cast_feed, load_reference,
)


def clone_cache(cache: Cache) -> Cache:
    new = Cache(cache.shape, len(cache.buffer))
    new.buffer[: cache.length] = cache.buffer[: cache.length]
    new.length = cache.length
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description="lockstep ax vs onnx diagnostic")
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--axmodel-dir", required=True)
    ap.add_argument("--spm", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--text", default="今天天气不错，我们一起去公园散步吧。")
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()

    frontend = TextFrontend(args.spm)
    ids = frontend.text2ids(args.text)

    rt = PocketTTS(args.onnx_dir, args.axmodel_dir, cpu_model_dir=args.onnx_dir,
                   npu_flow_net=True, npu_mimi=True)
    ref = load_reference(args.reference)
    cfg = rt.cfg

    ax_flow_net = Session(os.path.join(args.axmodel_dir, "flow_net_step.axmodel"), "ax")
    on_flow_net = Session(os.path.join(args.onnx_dir, "flow_net_step.onnx"), "onnx")
    ax_mimi = Session(os.path.join(args.axmodel_dir, "mimi_decoder_step.axmodel"), "ax")
    on_mimi = Session(os.path.join(args.onnx_dir, "mimi_decoder_step.onnx"), "onnx")

    # shared prefix: voice + text prefill into a template flow KV; clone per universe
    template_kv, _ = rt.prefilled_voice(ref)
    tokens = np.asarray([ids], dtype=np.int64)
    out = rt._flow_step(tokens=tokens, gates=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                        seq=tokens.shape[1], flow_kv=template_kv)
    template_kv.append(out["flow_kv_new"])

    def init_state():
        kv = Cache((cfg["mimi_layers"], 2, 1, cfg["mimi_heads"], cfg["mimi_head_dim"]),
                   cfg["mimi_kv_len"] + 512 * cfg["steps_per_latent"])
        return {
            "flow_kv": clone_cache(template_kv),
            "latent": np.zeros((1, 1, cfg["latent_dim"]), dtype=np.float32),
            "is_bos": np.ones((1, 1, 1), dtype=np.float32),
            "mimi_kv": kv,
            "mimi_conv": np.zeros(cfg["conv_state_size"], dtype=np.float32),
            "mimi_offset": np.asarray(0, dtype=np.int64),
        }

    noise = np.zeros((1, cfg["latent_dim"]), dtype=np.float32)
    universes = {
        "ax": (init_state(), ax_flow_net, ax_mimi),
        "on": (init_state(), on_flow_net, on_mimi),
    }

    def step_universe(state, flow_net, mimi):
        fout = rt._flow_step(latent=state["latent"], is_bos=state["is_bos"],
                             gates=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                             seq=1, flow_kv=state["flow_kv"])
        state["flow_kv"].append(fout["flow_kv_new"])
        fo = flow_net.run(cast_feed(flow_net, {
            "conditioning": fout["conditioning"], "noise": noise}))
        latent = np.asarray(fo["next_latent"], dtype=np.float32).reshape(1, 1, cfg["latent_dim"])
        mo = mimi.run(cast_feed(mimi, {
            "next_latent": latent,
            "mimi_kv": state["mimi_kv"].window(cfg["mimi_kv_len"]),
            "mimi_conv": state["mimi_conv"],
            "mimi_offset": state["mimi_offset"],
        }))
        state["next_latent"] = latent
        state["mimi_out"] = mo

    print(f"step | latent_cos | audio_diff | kv_diff | conv_diff | eos(ax/on)")
    for step in range(args.steps):
        for tag, (state, flow_net, mimi) in universes.items():
            step_universe(state, flow_net, mimi)
        sax, sox = universes["ax"][0], universes["on"][0]
        a = sax["next_latent"].reshape(-1)
        b = sox["next_latent"].reshape(-1)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        ad = float(np.max(np.abs(np.asarray(sax["mimi_out"]["audio"]) -
                                 np.asarray(sox["mimi_out"]["audio"]))))
        kd = float(np.max(np.abs(np.asarray(sax["mimi_out"]["mimi_kv_new"]) -
                                 np.asarray(sox["mimi_out"]["mimi_kv_new"]))))
        cd = float(np.max(np.abs(np.asarray(sax["mimi_out"]["mimi_conv_out"]) -
                                 np.asarray(sox["mimi_out"]["mimi_conv_out"]))))
        print(f"{step:4d} | {cos:.6f} | {ad:.3e} | {kd:.3e} | {cd:.3e}")
        for state in (sax, sox):
            state["mimi_kv"].append(np.asarray(state["mimi_out"]["mimi_kv_new"], dtype=np.float32))
            state["mimi_conv"] = np.asarray(state["mimi_out"]["mimi_conv_out"],
                                            dtype=np.float32).reshape(-1)
            state["mimi_offset"] = np.asarray(
                state["mimi_out"]["mimi_offset_out"]).reshape(-1).astype(np.int64)
            state["latent"] = state["next_latent"]
            state["is_bos"] = np.zeros((1, 1, 1), dtype=np.float32)

    audio_ax = np.concatenate([np.asarray(universes["ax"][0]["mimi_out"]["audio"])[0, 0]])
    audio_on = np.concatenate([np.asarray(universes["on"][0]["mimi_out"]["audio"])[0, 0]])
    cos = float(audio_ax @ audio_on / (np.linalg.norm(audio_ax) * np.linalg.norm(audio_on) + 1e-30))
    print(f"final audio cos (last frame only, {len(audio_ax)} samples): {cos:.6f}")


if __name__ == "__main__":
    main()
