#!/usr/bin/env python3
"""pocket-tts-zh-en AX650 hybrid runtime.

NPU (axengine): flow_net_step.axmodel, mimi_decoder_step.axmodel
CPU (onnxruntime): windowed flow_step.onnx (W=512 KV window), step_encoder.onnx

Text frontend mirrors the vendor demo.py (punctuation normalization + the
shipped SentencePiece model), with punctuation-based chunking for long text.

Usage (board):
    python3 pocket_tts_axera.py --text "你好，世界。" --reference Vivian.wav --output out.wav
    python3 pocket_tts_axera.py --text-file long.txt --output long.wav
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import wave

import numpy as np

SR = 24000


# ---------------------------------------------------------------- vendor pieces
def resample_24k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return audio.astype(np.float32)
    from scipy.signal import resample_poly

    g = math.gcd(int(sr), SR)
    return resample_poly(audio, SR // g, int(sr) // g).astype(np.float32)


def normalize_punctuation(text: str) -> str:
    special = {
        "“": '"', "”": '"', "「": '"', "」": '"', "『": '"', "』": '"',
        "‘": "'", "’": "'", "【": "[", "】": "]", "…": "...", "｡": ".",
    }
    out = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E and ch not in "，。？、":
            out.append(chr(code - 0xFEE0))
        elif ch in special:
            out.append(special[ch])
        else:
            out.append(ch)
    return "".join(out)


_DIGIT = "零一二三四五六七八九"
_SMALL_UNIT = ["", "十", "百", "千"]
_BIG_UNIT = ["", "万", "亿", "兆"]


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _int_to_zh(digits: str) -> str:
    digits = digits.lstrip("0")
    if digits == "":
        return _DIGIT[0]
    if len(digits) > 16:
        return "".join(_DIGIT[int(c)] for c in digits)
    n = len(digits)
    out: list[str] = []
    zero_pending = False
    for i, ch in enumerate(digits):
        d = int(ch)
        pos = n - 1 - i
        small, big = pos % 4, pos // 4
        if d == 0:
            zero_pending = True
        else:
            if zero_pending:
                out.append(_DIGIT[0])
                zero_pending = False
            out.append(_DIGIT[d] + _SMALL_UNIT[small])
        if small == 0 and big > 0 and any(c != "0" for c in digits[max(0, i - 3): i + 1]):
            out.append(_BIG_UNIT[big])
            zero_pending = False
    reading = "".join(out)
    if reading.startswith("一十"):
        reading = reading[1:]
    return reading


import re  # noqa: E402

_RE_YEAR = re.compile(r"(?<![A-Za-z0-9])(\d{4})(?=年)")
_RE_NUM = re.compile(r"(?<![A-Za-z0-9])(-)?(\d[\d,，]*)(?:\.(\d+))?(?![A-Za-z])")


def numbers_to_zh(text: str) -> str:
    """Convert Arabic numbers to Chinese readings (only when the text has CJK)."""
    if not has_cjk(text):
        return text
    text = _RE_YEAR.sub(lambda m: "".join(_DIGIT[int(c)] for c in m.group(1)), text)

    def sub(match: re.Match) -> str:
        sign, int_part, dec_part = match.group(1), match.group(2), match.group(3)
        int_part = int_part.replace(",", "").replace("，", "")
        out = "负" if sign else ""
        out += _int_to_zh(int_part) if int_part else ""
        if dec_part:
            out += "点" + "".join(_DIGIT[int(c)] for c in dec_part)
        return out

    return _RE_NUM.sub(sub, text)


class TextFrontend:
    UNK_ID = 0
    SPACE_MARKER_ID = 124

    def __init__(self, spm_path: str, max_tokens: int = 48):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)
        self.max_tokens = max_tokens

    def text2ids(self, text: str) -> list[int]:
        text = numbers_to_zh(normalize_punctuation(text)).lower()
        ids = self.sp.EncodeAsIds("A " + text.strip())[1:]
        return [x for x in ids if x not in (self.UNK_ID, self.SPACE_MARKER_ID)]

    @staticmethod
    def _split_keep(text: str, separators: str) -> list[str]:
        pieces, buf = [], []
        for ch in text:
            buf.append(ch)
            if ch in separators:
                pieces.append("".join(buf).strip())
                buf = []
        if buf:
            pieces.append("".join(buf).strip())
        return [p for p in pieces if p]

    def chunk_text(self, text: str) -> list[str]:
        """Sentence-first, then clause, then hard split; each chunk <= max_tokens."""
        text = text.strip()
        if len(self.text2ids(text)) <= self.max_tokens:
            return [text]
        chunks: list[str] = []
        cur, cur_len = "", 0

        def flush() -> None:
            nonlocal cur, cur_len
            if cur:
                chunks.append(cur)
                cur, cur_len = "", 0

        def append_piece(piece: str) -> None:
            nonlocal cur, cur_len
            n = len(self.text2ids(piece))
            if cur and cur_len + n > self.max_tokens:
                flush()
            cur += piece
            cur_len += n

        def append_hard(piece: str) -> None:
            i = 0
            while i < len(piece):
                step = 16
                while step > 1 and len(self.text2ids(piece[i:i + step])) > self.max_tokens:
                    step //= 2
                append_piece(piece[i:i + step])
                i += step

        for sentence in self._split_keep(text, "。！？!?.;；\n"):
            if len(self.text2ids(sentence)) <= self.max_tokens:
                append_piece(sentence)
                continue
            flush()
            for clause in self._split_keep(sentence, "，、,:："):
                if len(self.text2ids(clause)) <= self.max_tokens:
                    append_piece(clause)
                else:
                    append_hard(clause)
        flush()
        return chunks or [text]


# ------------------------------------------------------------------- KV cache
class Cache:
    def __init__(self, shape, capacity: int):
        self.buffer = np.zeros((capacity, *shape), dtype=np.float32)
        self.shape = shape
        self.length = 0

    def reserve(self, extra: int) -> None:
        needed = self.length + extra
        if needed > len(self.buffer):
            grown = np.zeros((max(needed, 2 * len(self.buffer)), *self.shape), dtype=np.float32)
            grown[: self.length] = self.buffer[: self.length]
            self.buffer = grown

    def append(self, values: np.ndarray) -> None:
        self.reserve(len(values))
        self.buffer[self.length: self.length + len(values)] = values
        self.length += len(values)

    def window(self, size: int | None = None) -> np.ndarray:
        if size is None:
            return self.buffer[: self.length]
        start = self.length - size
        if start >= 0:
            return self.buffer[start: self.length]
        padded = np.zeros((size, *self.shape), dtype=np.float32)
        if self.length:
            padded[size - self.length:] = self.buffer[: self.length]
        return padded


def clone_cache(cache: "Cache") -> "Cache":
    new = Cache(cache.shape, len(cache.buffer))
    new.buffer[: cache.length] = cache.buffer[: cache.length]
    new.length = cache.length
    return new


class Session:
    """axengine (NPU) or onnxruntime (CPU) session with a common run()."""

    def __init__(self, path: str, backend: str, threads: int = 4):
        if backend == "ax":
            import axengine as axe

            self._sess = axe.InferenceSession(path)
        else:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            self._sess = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        self.inputs = [(i.name, getattr(i, "type", None) or getattr(i, "dtype", None),
                        getattr(i, "shape", None) or getattr(i, "dims", None))
                       for i in self._sess.get_inputs()]
        self.outputs = [(o.name, getattr(o, "type", None) or getattr(o, "dtype", None),
                         getattr(o, "shape", None) or getattr(o, "dims", None))
                        for o in self._sess.get_outputs()]

    def run(self, feed: dict) -> dict:
        values = self._sess.run(None, feed)
        return {name: np.asarray(value)
                for (name, _, _), value in zip(self.outputs, values)}


def cast_feed(session: Session, feed: dict) -> dict:
    out = {}
    types = {name: t for name, t, _ in session.inputs}
    shapes = {name: s for name, _, s in session.inputs}
    for name, value in feed.items():
        t = str(types.get(name, "")).lower()
        if "int32" in t:
            arr = np.asarray(value, dtype=np.int32)
        elif "int64" in t:
            arr = np.asarray(value, dtype=np.int64)
        else:
            arr = np.asarray(value, dtype=np.float32)
        shape = shapes.get(name)
        if shape:
            try:
                dims = [int(d) for d in shape]
                if all(d > 0 for d in dims) and tuple(dims) != arr.shape and int(np.prod(dims)) == arr.size:
                    arr = arr.reshape(dims)
            except (TypeError, ValueError):
                pass
        out[name] = arr
    return out


# ------------------------------------------------------------------- runtime
class PocketTTS:
    def __init__(self, onnx_dir: str, axmodel_dir: str,
                 cpu_model_dir: str | None = None,
                 mimi_split_dir: str | None = None,
                 threads: int = 4, flow_window: int = 512,
                 npu_flow_net: bool = True, npu_mimi_conv: bool = True,
                 flow_net_model: str = "flow_net_step.axmodel",
                 mimi_conv_model: str = "mimi_conv_step.axmodel",
                 flow_ar_model: str | None = None,
                 flow_prefill_model: str = "flow_step_windowed.onnx",
                 mimi_tf_model: str = "mimi_transformer_step.onnx",
                 encoder_dir: str | None = None,
                 prefill_threads: int | None = None):
        t0 = time.perf_counter()
        with open(os.path.join(onnx_dir, "step_config.json")) as f:
            self.cfg = json.load(f)
        self.flow_window = flow_window
        cpu_model_dir = cpu_model_dir or onnx_dir
        mimi_split_dir = mimi_split_dir or cpu_model_dir
        self.flow = Session(os.path.join(onnx_dir, flow_prefill_model), "onnx",
                            prefill_threads or threads)
        self.flow_ar = (Session(os.path.join(onnx_dir, flow_ar_model), "onnx", threads)
                        if flow_ar_model else None)
        self.encoder = Session(os.path.join(onnx_dir, "step_encoder.onnx"), "onnx", threads)
        self.encoder_tiers: dict[int, Session] = {}
        if encoder_dir:
            for name in sorted(os.listdir(encoder_dir)):
                match = re.fullmatch(r"step_encoder_(\d+)f\.axmodel", name)
                if match:
                    frames = int(match.group(1))
                    self.encoder_tiers[frames] = Session(
                        os.path.join(encoder_dir, name), "ax")
            if not self.encoder_tiers:
                print(f"[warn] no step_encoder_<F>f.axmodel found in {encoder_dir}")
            else:
                print(f"encoder tiers: {sorted(self.encoder_tiers)} frames")
        self.flow_net = (Session(os.path.join(axmodel_dir, flow_net_model), "ax")
                         if npu_flow_net else
                         Session(os.path.join(cpu_model_dir, "flow_net_step.onnx"), "onnx", threads))
        self.mimi_transformer = Session(
            os.path.join(mimi_split_dir, mimi_tf_model), "onnx", threads)
        self.mimi_conv = (Session(os.path.join(axmodel_dir, mimi_conv_model), "ax")
                          if npu_mimi_conv else
                          Session(os.path.join(mimi_split_dir, "mimi_conv_step.onnx"), "onnx",
                                  threads))
        self.load_s = time.perf_counter() - t0
        self._voice_cache = {}
        self._voice_prefix_cache = {}

    # voice conditioning (mimi encoder: NPU static tiers when available)
    def encode_voice(self, audio: np.ndarray) -> np.ndarray:
        frame_size = self.cfg["frame_size"]
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        key = audio.tobytes()[:64] + str(len(audio)).encode()
        cached = self._voice_cache.get(key)
        if cached is not None:
            return cached
        if self.encoder_tiers:
            tiers = sorted(self.encoder_tiers)
            padded_len = int(math.ceil(len(audio) / frame_size)) * frame_size
            tier = next((f for f in tiers if f * frame_size >= padded_len), tiers[-1])
            target = tier * frame_size
            if len(audio) >= target:
                clip = audio[-target:]
            else:
                clip = np.pad(audio, (target - len(audio), 0))
            out = self.encoder_tiers[tier].run({"audio": clip[None, None]})
        else:
            remainder = len(audio) % frame_size
            if remainder:
                audio = np.pad(audio, (0, frame_size - remainder))
            out = self.encoder.run({"audio": audio[None, None]})
        cond = out["cond"]
        self._voice_cache[key] = cond
        return cond

    def _flow_step(self, *, gates, seq, flow_kv, tokens=None, latent=None,
                   is_bos=None, cond=None):
        cfg = self.cfg
        zeros = lambda *shape: np.zeros(shape, dtype=np.float32)  # noqa: E731
        flow_cache = flow_kv.window()
        if self.flow_window is not None and flow_kv.length > self.flow_window:
            flow_cache = flow_kv.window(self.flow_window)
        feed = {
            "tokens": tokens if tokens is not None else np.zeros((1, seq), dtype=np.int64),
            "latent": latent if latent is not None else zeros(1, seq, cfg["latent_dim"]),
            "is_bos": is_bos if is_bos is not None else zeros(1, seq, 1),
            "cond": cond if cond is not None else zeros(1, seq, cfg["model_dim"]),
            "gates": gates,
            "flow_kv": flow_cache,
            "flow_offset": np.asarray(flow_kv.length, dtype=np.int64),
        }
        return self.flow.run(feed)

    def prefilled_voice(self, ref_audio: np.ndarray, timing: dict | None = None):
        cfg = self.cfg
        key = ref_audio.tobytes()[:64] + str(len(ref_audio)).encode()
        cached = self._voice_prefix_cache.get(key)
        if cached is not None:
            if timing is not None:
                timing["voice_cache_hit"] = True
            return clone_cache(cached[0]), cached[1]
        t0 = time.perf_counter()
        cond = self.encode_voice(ref_audio)
        t_enc = time.perf_counter()
        cache = Cache((cfg["flow_layers"], 2, 1, cfg["flow_heads"], cfg["flow_head_dim"]),
                      cond.shape[1] + 1)
        out = self._flow_step(
            cond=cond, gates=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            seq=cond.shape[1], flow_kv=cache)
        cache.append(out["flow_kv_new"])
        if timing is not None:
            timing["voice_cache_hit"] = False
            timing["encode_voice_ms"] = round((t_enc - t0) * 1000, 1)
            timing["voice_prefill_ms"] = round((time.perf_counter() - t_enc) * 1000, 1)
        self._voice_prefix_cache[key] = (clone_cache(cache), cond.shape[1])
        return cache, cond.shape[1]

    def stream(self, token_ids: list[int], ref_audio: np.ndarray, temp: float = 0.0,
               max_frames: int = 375, seed: int = 0, eos_threshold: float | None = None,
               timing: dict | None = None):
        cfg = self.cfg
        threshold = cfg["eos_threshold"] if eos_threshold is None else eos_threshold
        rng = np.random.default_rng(seed)
        latent_dim = cfg["latent_dim"]

        t0 = time.perf_counter()
        flow_kv, _ = self.prefilled_voice(ref_audio, timing=timing)
        t_voice = time.perf_counter()
        tokens = np.asarray([token_ids], dtype=np.int64)
        flow_kv.reserve(tokens.shape[1] + max_frames)

        mimi_kv = Cache((cfg["mimi_layers"], 2, 1, cfg["mimi_heads"], cfg["mimi_head_dim"]),
                        cfg["mimi_kv_len"] + max_frames * cfg["steps_per_latent"])
        mimi_conv = np.zeros(cfg["conv_state_size"], dtype=np.float32)
        mimi_offset = np.asarray(0, dtype=np.int64)

        out = self._flow_step(tokens=tokens, gates=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                              seq=tokens.shape[1], flow_kv=flow_kv)
        flow_kv.append(out["flow_kv_new"])
        if timing is not None:
            timing["text_prefill_ms"] = round((time.perf_counter() - t_voice) * 1000, 1)

        latent = np.zeros((1, 1, latent_dim), dtype=np.float32)
        is_bos = np.ones((1, 1, 1), dtype=np.float32)
        eos_frame = None
        frame_times = []
        stage_ms = {"flow": 0.0, "flow_net": 0.0, "mimi_transformer": 0.0, "mimi_conv": 0.0}
        for frame in range(max_frames):
            t_frame = time.perf_counter()
            noise = (rng.standard_normal((1, latent_dim)) * math.sqrt(temp)).astype(np.float32)
            t_stage = time.perf_counter()
            if self.flow_ar is not None:
                out = self.flow_ar.run({
                    "latent": latent,
                    "is_bos": is_bos,
                    "flow_kv": flow_kv.window(self.flow_window)
                    if (self.flow_window and flow_kv.length > self.flow_window)
                    else flow_kv.window(),
                    "flow_offset": np.asarray(flow_kv.length, dtype=np.int64),
                })
            else:
                out = self._flow_step(latent=latent, is_bos=is_bos,
                                      gates=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                                      seq=1, flow_kv=flow_kv)
            flow_kv.append(out["flow_kv_new"])
            conditioning = out["conditioning"]
            eos_logit = float(out["eos_logit"][0, 0])
            stage_ms["flow"] += (time.perf_counter() - t_stage) * 1000

            t_stage = time.perf_counter()
            flow_out = self.flow_net.run(cast_feed(self.flow_net, {
                "conditioning": conditioning, "noise": noise}))
            next_latent = np.asarray(flow_out["next_latent"], dtype=np.float32).reshape(1, 1, latent_dim)
            stage_ms["flow_net"] += (time.perf_counter() - t_stage) * 1000

            t_stage = time.perf_counter()
            tf_out = self.mimi_transformer.run({
                "next_latent": next_latent,
                "mimi_kv": mimi_kv.window(cfg["mimi_kv_len"]),
                "mimi_conv": mimi_conv,
                "mimi_offset": np.asarray(mimi_offset, dtype=np.int64),
            })
            mimi_kv.append(np.asarray(tf_out["mimi_kv_new"], dtype=np.float32))
            stage_ms["mimi_transformer"] += (time.perf_counter() - t_stage) * 1000

            t_stage = time.perf_counter()
            conv_out = self.mimi_conv.run(cast_feed(self.mimi_conv, {
                "decoder_embedding": tf_out["decoder_embedding"],
                "mimi_conv": mimi_conv,
                "upsample_state": tf_out["upsample_state"],
            }))
            mimi_conv = np.asarray(conv_out["mimi_conv_out"], dtype=np.float32).reshape(-1)
            stage_ms["mimi_conv"] += (time.perf_counter() - t_stage) * 1000
            mimi_offset = int(mimi_offset) + cfg["steps_per_latent"]
            latent = next_latent
            is_bos = np.zeros((1, 1, 1), dtype=np.float32)
            frame_times.append(time.perf_counter() - t_frame)

            if eos_frame is None and eos_logit > threshold:
                eos_frame = frame
            if eos_frame is not None and frame >= eos_frame:
                break
            yield np.asarray(conv_out["audio"], dtype=np.float32)[0, 0]

        if timing is not None:
            timing["per_frame_ms"] = [round(t * 1000, 2) for t in frame_times]
            n = max(len(frame_times), 1)
            timing["stage_ms_mean"] = {k: round(v / n, 2) for k, v in stage_ms.items()}

    def synthesize(self, token_ids: list[int], ref_audio: np.ndarray, **kwargs):
        return np.concatenate(list(self.stream(token_ids, ref_audio, **kwargs)))


def write_wav(path: str, audio: np.ndarray) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


class StreamingWavWriter:
    """WAV writer with a placeholder header so the file is readable while growing."""

    def __init__(self, path: str, sample_rate: int = SR):
        self.sample_rate = sample_rate
        self.samples = 0
        self._f = open(path, "wb")
        self._f.write(b"RIFF" + (0xFFFFFFFF).to_bytes(4, "little") + b"WAVE" +
                      b"fmt " + (16).to_bytes(4, "little") + (1).to_bytes(2, "little") +
                      (1).to_bytes(2, "little") + sample_rate.to_bytes(4, "little") +
                      (sample_rate * 2).to_bytes(4, "little") + (2).to_bytes(2, "little") +
                      (16).to_bytes(2, "little") + b"data" +
                      (0xFFFFFFFF).to_bytes(4, "little"))
        self._f.flush()

    def write(self, audio: np.ndarray) -> None:
        pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        self._f.write(pcm.tobytes())
        self._f.flush()
        self.samples += pcm.size

    def close(self) -> None:
        data_bytes = self.samples * 2
        self._f.seek(4)
        self._f.write((36 + data_bytes).to_bytes(4, "little"))
        self._f.seek(40)
        self._f.write(data_bytes.to_bytes(4, "little"))
        self._f.close()


def load_reference(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample_24k(audio, sr)


def main() -> None:
    ap = argparse.ArgumentParser(description="pocket-tts-zh-en AX650 runtime")
    ap.add_argument("--text", default="欢迎使用图灵云语音合成服务!我们提供高质量的中文语音合成技术。")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--onnx-dir", required=True,
                    help="dir with flow_step_windowed.onnx + step_encoder.onnx + step_config.json")
    ap.add_argument("--axmodel-dir", required=True)
    ap.add_argument("--cpu-model-dir", default=None,
                    help="dir with flow_net_step.onnx for --npu-flow-net 0")
    ap.add_argument("--mimi-split-dir", default=None,
                    help="dir with mimi_transformer_step.onnx / mimi_conv_step.onnx")
    ap.add_argument("--encoder-dir", default=None,
                    help="dir with step_encoder_<F>f.axmodel NPU encoder tiers")
    ap.add_argument("--npu-flow-net", type=int, default=1)
    ap.add_argument("--npu-mimi-conv", type=int, default=1)
    ap.add_argument("--flow-net-model", default="flow_net_step.axmodel")
    ap.add_argument("--flow-ar-model", default=None, help="optional lean AR graph (onnx-dir relative)")
    ap.add_argument("--flow-prefill-model", default="flow_step_windowed.onnx")
    ap.add_argument("--mimi-tf-model", default="mimi_transformer_step.onnx")
    ap.add_argument("--mimi-conv-model", default="mimi_conv_step.axmodel")
    ap.add_argument("--spm", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prefill-threads", type=int, default=None)
    ap.add_argument("--flow-window", type=int, default=512)
    ap.add_argument("--max-frames", type=int, default=375)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1, help="chunk long text")
    ap.add_argument("--pause-ms", type=int, default=120, help="silence between chunks")
    ap.add_argument("--stream", type=int, default=0, help="write frames while generating")
    ap.add_argument("--timing-json", default=None)
    args = ap.parse_args()

    text = args.text
    if args.text_file:
        with open(args.text_file) as f:
            text = f.read().strip()

    frontend = TextFrontend(args.spm)
    chunks = frontend.chunk_text(text) if args.chunk else [text]

    tts = PocketTTS(args.onnx_dir, args.axmodel_dir, cpu_model_dir=args.cpu_model_dir,
                    mimi_split_dir=args.mimi_split_dir,
                    threads=args.threads, flow_window=args.flow_window,
                    npu_flow_net=bool(args.npu_flow_net),
                    npu_mimi_conv=bool(args.npu_mimi_conv),
                    flow_net_model=args.flow_net_model,
                    mimi_conv_model=args.mimi_conv_model,
                    flow_ar_model=args.flow_ar_model,
                    flow_prefill_model=args.flow_prefill_model,
                    mimi_tf_model=args.mimi_tf_model,
                    encoder_dir=args.encoder_dir,
                    prefill_threads=args.prefill_threads)
    print(f"load: {tts.load_s:.2f}s")
    print(f"flow inputs : {tts.flow.inputs}")
    print(f"flow_net io : {tts.flow_net.inputs} -> {tts.flow_net.outputs}")
    print(f"mimi tf io  : {tts.mimi_transformer.inputs} -> {tts.mimi_transformer.outputs}")
    print(f"mimi conv io: {tts.mimi_conv.inputs} -> {tts.mimi_conv.outputs}")

    ref = load_reference(args.reference)
    print(f"reference: {len(ref) / SR:.2f}s, chunks={len(chunks)}")

    t_all = time.perf_counter()
    pieces = []
    first_frame_ms = None
    n_frames = 0
    timing = {}
    writer = StreamingWavWriter(args.output) if args.stream else None
    pause = (np.zeros(int(SR * args.pause_ms / 1000), dtype=np.float32)
             if args.pause_ms > 0 else None)
    for i, chunk in enumerate(chunks):
        ids = frontend.text2ids(chunk)
        print(f"  chunk {i}: {len(ids)} tokens: {chunk[:40]}")
        t0 = time.perf_counter()
        gen = tts.stream(ids, ref, temp=args.temp, max_frames=args.max_frames,
                         seed=args.seed, timing=timing if i == len(chunks) - 1 else None)
        chunk_frames = 0
        chunk_samples = 0
        for frame in gen:
            if first_frame_ms is None:
                first_frame_ms = (time.perf_counter() - t0) * 1000
            if writer is not None:
                writer.write(frame)
            else:
                pieces.append(np.asarray(frame, dtype=np.float32))
            chunk_frames += 1
            chunk_samples += len(frame)
        n_frames += chunk_frames
        if writer is not None and pause is not None and i < len(chunks) - 1:
            writer.write(pause)
        if writer is None and pause is not None and i < len(chunks) - 1:
            pieces.append(pause)
        print(f"    -> {chunk_frames} frames, {chunk_samples / SR:.2f}s, "
              f"{(time.perf_counter() - t0):.2f}s")
    if writer is not None:
        writer.close()
        audio_seconds = writer.samples / SR
    else:
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        write_wav(args.output, audio)
        audio_seconds = len(audio) / SR
    total_s = time.perf_counter() - t_all
    seconds = audio_seconds
    rtf = total_s / seconds if seconds > 0 else float("inf")
    per_frame = timing.get("per_frame_ms", [])
    stats = {
        "text": text,
        "chunks": chunks,
        "load_s": round(tts.load_s, 3),
        "first_frame_ms": round(first_frame_ms or 0, 1),
        "audio_seconds": round(seconds, 4),
        "total_s": round(total_s, 4),
        "rtf": round(rtf, 4),
        "frames": n_frames,
        "per_frame_ms_mean": round(float(np.mean(per_frame)), 2) if per_frame else None,
        "per_frame_ms_p50": round(float(np.median(per_frame)), 2) if per_frame else None,
        "per_frame_ms_max": round(float(np.max(per_frame)), 2) if per_frame else None,
        "stage_ms_mean": timing.get("stage_ms_mean"),
        "encode_voice_ms": timing.get("encode_voice_ms"),
        "voice_prefill_ms": timing.get("voice_prefill_ms"),
        "voice_cache_hit": timing.get("voice_cache_hit"),
        "text_prefill_ms": timing.get("text_prefill_ms"),
        "npu_flow_net": bool(args.npu_flow_net),
        "npu_mimi_conv": bool(args.npu_mimi_conv),
        "flow_prefill_model": args.flow_prefill_model,
        "mimi_tf_model": args.mimi_tf_model,
        "encoder_dir": args.encoder_dir,
        "stream": bool(args.stream),
        "pause_ms": args.pause_ms,
        "output": args.output,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.timing_json:
        with open(args.timing_json, "w") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
