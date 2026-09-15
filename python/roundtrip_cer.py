#!/usr/bin/env python3
"""Round-trip CER: transcribe generated wavs with local SenseVoiceSmall and
compare against the input text (char-level, punctuation stripped).

Host-side quick quality probe (board ASR comes later from the benchmark).

Usage:
    python python/roundtrip_cer.py --wavs results/audio/xxx.wav --text "..."
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENSEVOICE_DIR = os.path.expanduser("~/.cache/modelscope/hub/models/iic/SenseVoiceSmall")

_PUNCT = re.compile(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()\[\]【】\-—…·~～《》<>]")
_TAGS = re.compile(r"<\|[^|]*\|>")


def normalize(text: str) -> str:
    return _PUNCT.sub("", _TAGS.sub("", text)).lower()


def spoken_reference(text: str) -> str:
    """Apply the TTS frontend normalisation (numbers -> hanzi) to the reference."""
    try:
        sys.path.insert(0, os.path.join(PROJ, "board"))
        from pocket_tts_axera import normalize_punctuation, numbers_to_zh

        return numbers_to_zh(normalize_punctuation(text))
    except Exception:
        return text


def cer(ref: str, hyp: str) -> tuple[float, int, int, int]:
    ref, hyp = normalize(ref), normalize(hyp)
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    dist = dp[m]
    return (dist / max(n, 1), dist, n, m)


def main() -> None:
    ap = argparse.ArgumentParser(description="round-trip CER with SenseVoiceSmall")
    ap.add_argument("--wavs", nargs="+", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--language", default="zh", choices=["zh", "en"])
    ap.add_argument("--model-dir", default=SENSEVOICE_DIR)
    ap.add_argument("--out", default=os.path.join(PROJ, "results", "roundtrip_cer.json"))
    args = ap.parse_args()

    from funasr import AutoModel

    model = AutoModel(model=args.model_dir, disable_update=True, device="cpu", disable_pbar=True)
    results = []
    for wav in args.wavs:
        res = model.generate(input=wav, language=args.language, use_itn=False, batch_size_s=300)
        hyp = res[0]["text"] if res else ""
        ref = spoken_reference(args.text)
        rate, dist, n, m = cer(ref, hyp)
        results.append({"wav": os.path.relpath(wav, PROJ), "hyp": hyp,
                        "cer": round(rate, 4), "edit": dist, "ref_len": n, "hyp_len": m,
                        "spoken_ref": normalize(ref)})
        print(f"{os.path.basename(wav):42s} CER={rate * 100:6.2f}%  hyp={hyp}")

    with open(args.out, "w") as f:
        json.dump({"text": args.text, "reference": normalize(spoken_reference(args.text)),
                   "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
