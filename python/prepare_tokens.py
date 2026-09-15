#!/usr/bin/env python3
"""Host-side tokenizer for the C++ runtime: text -> request file.

Writes one line per chunk (comma-separated token ids) reusing the exact board
text frontend (punctuation normalization, numbers_to_zh, chunking <=48 tokens).

Usage:
    python python/prepare_tokens.py --spm models/xxx.bpe.model \
        --text "你好，世界。" --out request.tokens
    python python/prepare_tokens.py --spm ... --text-file long.txt --out request.tokens
"""
from __future__ import annotations

import argparse
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "board"))

from pocket_tts_axera import TextFrontend  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="text -> C++ request tokens")
    ap.add_argument("--spm", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()

    text = args.text
    if args.text_file:
        with open(args.text_file) as f:
            text = f.read().strip()
    if not text:
        raise SystemExit("no text")

    frontend = TextFrontend(args.spm, max_tokens=args.max_tokens)
    chunks = frontend.chunk_text(text)
    with open(args.out, "w") as f:
        for chunk in chunks:
            ids = frontend.text2ids(chunk)
            f.write(",".join(str(i) for i in ids) + "\n")
            print(f"chunk: {len(ids)} tokens | {chunk[:40]}")
    print(f"wrote {args.out} ({len(chunks)} chunk(s))")


if __name__ == "__main__":
    main()
