"""
S1 — Prepare MUStARD++ (sarcasm).

CSV (mustard++_text.csv) columns: SCENE, KEY, SENTENCE, END_TIME, SPEAKER, SHOW,
Sarcasm, Sarcasm_Type, Implicit_Emotion, Explicit_Emotion, Valence, Arousal.
Rows with KEY ending in "_u" are the main utterances (1,202); "_c_NN" rows are context.
We only use the main-utterance videos (Drive subfolder "final_utterance_videos",
files named {KEY}.mp4). gdown --folder caps at 50 files, so we list the folder and
download each file by id, convert to 16 kHz mono wav, and delete the mp4.

Splits: speaker-grouped ~70/15/15 → splits/mustardpp_{train,val,test}.json.

Usage: python scripts/s1_prepare_mustardpp.py
"""
import json
import subprocess
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from wdymnet.data.download import download, gdrive_folder_listing
from wdymnet.data.manifest import clean_text, split_words, write_manifest
from wdymnet.utils.common import check_ffmpeg, load_config, set_seed

CSV_URL = "https://raw.githubusercontent.com/cfiltnlp/MUStARD_Plus_Plus/main/mustard++_text.csv"
UTTERANCE_FOLDER_ID = "1xcvGIPGzrcNKkNBwBSoik9sh_SDSH2ov"  # final_utterance_videos
DRIVE_FILE_URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"


def fetch_clip(file_id: str, key: str, video_dir: Path, wav_dir: Path) -> str | None:
    """Download one mp4 and convert it to wav. Returns an error string or None."""
    wav = wav_dir / f"{key}.wav"
    if wav.exists():
        return None
    mp4 = video_dir / f"{key}.mp4"
    try:
        if not mp4.exists():
            tmp = mp4.with_suffix(".part")
            urllib.request.urlretrieve(DRIVE_FILE_URL.format(file_id), tmp)
            tmp.rename(mp4)
        r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(mp4),
                            "-vn", "-ac", "1", "-ar", "16000", "-y", str(wav)], capture_output=True)
        if r.returncode != 0:
            wav.unlink(missing_ok=True)
            return f"ffmpeg: {r.stderr.decode(errors='ignore')[:200]}"
        mp4.unlink()
        return None
    except Exception as e:
        return str(e)


def speaker_grouped_split(speakers: list[str], ratios=(0.70, 0.15, 0.15), seed=1337) -> list[str]:
    """Greedy group split by utterance count: no speaker appears in two splits."""
    counts = Counter(speakers)
    rng = np.random.default_rng(seed)
    order = sorted(counts, key=lambda s: (-counts[s], rng.random()))
    names, target = ["train", "val", "test"], np.array(ratios) * len(speakers)
    have, assign = np.zeros(3), {}
    for spk in order:
        k = int(np.argmax(target - have))  # split with the largest remaining deficit
        assign[spk] = names[k]
        have[k] += counts[spk]
    return [assign[s] for s in speakers]


def main():
    set_seed(1337)
    cfg = load_config("configs/data.yaml")
    if not check_ffmpeg():
        sys.exit(1)
    root = Path(cfg["paths"]["data_root"]) / "MUStARD++"
    video_dir, wav_dir = root / "videos", root / "wavs"
    video_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60 + "\nS1 — Prepare MUStARD++\n" + "=" * 60)
    csv_path = download(CSV_URL, root / "mustard++_text.csv")
    df = pd.read_csv(csv_path)
    print(f"  CSV columns: {list(df.columns)}")
    utt = df[df["KEY"].str.endswith("_u")].reset_index(drop=True)
    print(f"  main utterances: {len(utt)} | sarcastic: {int((utt['Sarcasm'] == 1).sum())}")

    # ---- videos → wavs ----
    listing = {name.rsplit(".", 1)[0]: fid for fid, name in gdrive_folder_listing(UTTERANCE_FOLDER_ID)}
    print(f"  Drive folder lists {len(listing)} utterance videos")
    todo = [(listing[k], k) for k in utt["KEY"] if k in listing and not (wav_dir / f"{k}.wav").exists()]
    errors = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_clip, fid, k, video_dir, wav_dir): k for fid, k in todo}
        for fut in tqdm(futs, desc="  download+ffmpeg"):
            if err := fut.result():
                errors[futs[fut]] = err
    if errors:
        print(f"  [WARN] {len(errors)} clips failed; first: {next(iter(errors.items()))}")
        print(f"  If Drive rate-limits, re-run later (resume-safe) or download manually from\n"
              f"  https://drive.google.com/drive/folders/{UTTERANCE_FOLDER_ID} into {video_dir}")

    # ---- speaker-grouped splits ----
    utt["split"] = speaker_grouped_split(utt["SPEAKER"].astype(str).tolist(), tuple(cfg["mustardpp"]["split_ratios"]))
    splits_dir = Path(cfg["paths"]["splits"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    for s in ["train", "val", "test"]:
        keys = utt.loc[utt["split"] == s, "KEY"].tolist()
        (splits_dir / f"mustardpp_{s}.json").write_text(json.dumps(keys, indent=0))

    # ---- manifest ----
    entries, missing = [], []
    for _, r in utt.iterrows():
        wav = wav_dir / f"{r['KEY']}.wav"
        words = split_words(r["SENTENCE"])
        if not wav.exists() or not words:
            missing.append(r["KEY"])
            continue
        entries.append({
            "id": f"mustardpp_{r['KEY']}",
            "wav": wav.as_posix(),
            "text": clean_text(r["SENTENCE"]),
            "words": [{"w": w} for w in words],
            "labels": {"sarcasm": int(r["Sarcasm"]),
                       "sarcasm_type": str(r["Sarcasm_Type"]),
                       "implicit_emotion": str(r["Implicit_Emotion"]),
                       "explicit_emotion": str(r["Explicit_Emotion"])},
            "speaker": f"{r['SHOW']}:{r['SPEAKER']}",
            "split": r["split"],
            "dataset": "mustardpp",
        })
    out = Path(cfg["paths"]["manifests"]) / "mustardpp.jsonl"
    write_manifest(entries, out)
    print(f"\n[OK] {len(entries)} entries → {out} | missing audio/text: {len(missing)}")
    for s in ["train", "val", "test"]:
        sub = [e for e in entries if e["split"] == s]
        print(f"  {s}: {len(sub)} utts, sarcastic={sum(e['labels']['sarcasm'] for e in sub)}, "
              f"speakers={len({e['speaker'] for e in sub})}")


if __name__ == "__main__":
    main()
