#!/usr/bin/env python3
"""M5: listening pack report - CER + duration for board vs host reference.

Usage:
    python python/listen_pack_report.py --pack-dir results/audio/listen_pack
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "python"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from roundtrip_cer import cer, normalize, spoken_reference  # noqa: E402

SENSEVOICE_DIR = os.path.expanduser("~/.cache/modelscope/hub/models/iic/SenseVoiceSmall")


def main() -> None:
    ap = argparse.ArgumentParser(description="listen pack CER report")
    ap.add_argument("--pack-dir", default=os.path.join(PROJ, "results", "audio", "listen_pack"))
    ap.add_argument("--texts-json", default=os.path.join(PROJ, "configs", "listen_texts.json"))
    ap.add_argument("--model-dir", default=SENSEVOICE_DIR)
    args = ap.parse_args()

    with open(args.texts_json) as f:
        texts = json.load(f)["texts"]

    from funasr import AutoModel

    asr = AutoModel(model=args.model_dir, disable_update=True, device="cpu", disable_pbar=True)

    rows = []
    for item in texts:
        tag, text = item["tag"], item["text"]
        lang = "en" if tag.startswith("en_") else "zh"
        row = {"tag": tag, "text": text, "lang": lang, "spoken_ref": normalize(spoken_reference(text))}
        for side in ("board", "host_ref"):
            wav = os.path.join(args.pack_dir, side, f"{tag}.wav")
            if not os.path.exists(wav):
                row[f"{side}_cer"] = None
                continue
            hyp = asr.generate(input=wav, language=lang, use_itn=False, batch_size_s=300)[0]["text"]
            rate, dist, n, m = cer(spoken_reference(text), hyp)
            row[f"{side}_seconds"] = round(sf.info(wav).duration, 2)
            row[f"{side}_cer"] = round(rate * 100, 2)
            row[f"{side}_hyp"] = normalize(hyp)
        rows.append(row)
        print(f"{tag:14s} board CER={row.get('board_cer')}% ({row.get('board_seconds')}s)  "
              f"host CER={row.get('host_ref_cer')}% ({row.get('host_ref_seconds')}s)")

    tsv = os.path.join(args.pack_dir, "CER.tsv")
    with open(tsv, "w") as f:
        f.write("tag\tboard_cer%\thost_ref_cer%\tboard_s\thost_ref_s\ttext\tboard_hyp\thost_hyp\n")
        for r in rows:
            f.write(f"{r['tag']}\t{r.get('board_cer')}\t{r.get('host_ref_cer')}\t"
                    f"{r.get('board_seconds')}\t{r.get('host_ref_seconds')}\t{r['text']}\t"
                    f"{r.get('board_hyp','')}\t{r.get('host_ref_hyp','')}\n")
    with open(os.path.join(args.pack_dir, "CER.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"wrote {tsv}")


if __name__ == "__main__":
    main()
