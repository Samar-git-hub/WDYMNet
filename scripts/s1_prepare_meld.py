"""
S1 — Prepare MELD (emotion).

Annotations: official CSVs from github.com/declare-lab/MELD.
Audio: the official raw tarball is 10.9 GB, so we use the audio-only HF mirror
ajyy/MELD_audio (1.5 GB; 16 kHz mono FLAC extracted from the official MP4s,
named {split}/dia{D}_utt{U}.flac). Clips missing from the mirror or unreadable
are logged to data/MELD/meld_skipped.txt.

Usage: python scripts/s1_prepare_meld.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import soundfile as sf

from wdymnet.data.download import download, extract_tar
from wdymnet.data.manifest import clean_text, split_words, write_manifest
from wdymnet.utils.common import load_config, set_seed

CSV_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{}"
AUDIO_URL = "https://huggingface.co/datasets/ajyy/MELD_audio/resolve/main/archive/{}.tar.gz"
EMOTION_MAP = {"neutral": 0, "surprise": 1, "fear": 2, "sadness": 3, "joy": 4, "disgust": 5, "anger": 6}
# (our split, official csv, mirror archive/folder name)
SPLITS = [("train", "train_sent_emo.csv", "train"), ("val", "dev_sent_emo.csv", "dev"),
          ("test", "test_sent_emo.csv", "test")]


def main():
    set_seed(1337)
    cfg = load_config("configs/data.yaml")
    root = Path(cfg["paths"]["data_root"])
    meld_dir, dl_dir = root / "MELD", root / "downloads"
    audio_dir = meld_dir / "audio"

    print("=" * 60 + "\nS1 — Prepare MELD\n" + "=" * 60)
    entries, skipped = [], []
    for split, csv_name, arch in SPLITS:
        csv_path = download(CSV_URL.format(csv_name), meld_dir / csv_name)
        tar_path = download(AUDIO_URL.format(arch), dl_dir / f"meld_{arch}.tar.gz")
        extract_tar(tar_path, audio_dir, marker=arch)

        df = pd.read_csv(csv_path, encoding="utf-8")
        for _, row in df.iterrows():
            cid = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            wav = audio_dir / arch / f"{cid}.flac"
            text = clean_text(row["Utterance"])
            words = split_words(text)
            reason = None
            if not wav.exists():
                reason = "missing audio"
            elif not words:
                reason = "empty transcript"
            else:
                try:
                    info = sf.info(str(wav))
                    if info.frames < 0.1 * info.samplerate:
                        reason = "audio < 0.1 s"
                except Exception as e:
                    reason = f"unreadable ({e})"
            if reason:
                skipped.append(f"{split}/{cid}: {reason}")
                continue
            entries.append({
                "id": f"meld_{split}_{cid}",
                "wav": wav.as_posix(),
                "text": text,
                "words": [{"w": w} for w in words],
                "labels": {"emotion": EMOTION_MAP[row["Emotion"].strip().lower()]},
                "speaker": str(row["Speaker"]),
                "split": split,
                "dataset": "meld",
            })

    (meld_dir / "meld_skipped.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")
    out = Path(cfg["paths"]["manifests"]) / "meld.jsonl"
    write_manifest(entries, out)
    print(f"\n[OK] {len(entries)} entries → {out} | skipped {len(skipped)} (see meld_skipped.txt)")
    print("  splits:", dict(Counter(e["split"] for e in entries)))
    inv = {v: k for k, v in EMOTION_MAP.items()}
    print("  emotions:", {inv[k]: v for k, v in sorted(Counter(e["labels"]["emotion"] for e in entries).items())})


if __name__ == "__main__":
    main()
