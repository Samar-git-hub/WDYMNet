"""
S2 — Alignment + feature caching (resume-safe).

Per manifest entry:
  1. word timestamps: HPC/MELD/MUStARD++ gold transcript → MMS_FA forced alignment
     (fallback: faster-whisper + difflib, ≥80% words must match). Utterances whose mean
     MMS word score is below alignment.min_mms_score are dropped (bad clip/transcript match).
  2. emotion2vec+ frames → per-word concat(mean, max)  [N, 1536]; 9-class scores (baseline B0)
     and the hidden state entering the last 3 blocks (Stage B) → data/cache_mid/{id}.pt
  3. RoBERTa word vectors [N, 768] (frozen)
  4. prosody [N, 8] (pyin; computed in a CPU process pool)
→ data/cache/{id}.pt.  Drops/failures are logged to data/cache_log.jsonl.

Usage: python scripts/s2_align_and_cache.py [--datasets meld mustardpp hpc] [--limit N]
"""
import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from wdymnet.align.aligner import MMSAligner, align_words_to_gold, whisper_words
from wdymnet.data.manifest import read_manifest
from wdymnet.features.prosody import extract_word_prosody
from wdymnet.features.speech import Emotion2vecEncoder, pool_frames_to_words
from wdymnet.features.text import TextEncoder
from wdymnet.utils.common import load_config, print_vram, reset_vram_stats, set_seed

SR = 16000


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(audio, dtype=np.float32)


def prosody_job(wav_path: str, word_times: list[dict], fmin: float, fmax: float) -> np.ndarray:
    return extract_word_prosody(load_audio(wav_path), SR, word_times, fmin, fmax)


def labels_for(entry: dict) -> dict:
    out, lab = {}, entry["labels"]
    if entry["dataset"] == "hpc":
        out["emph_labels"] = torch.tensor([w["label"] for w in entry["words"]], dtype=torch.long)
    if lab.get("emotion", -1) >= 0:
        out["emo_label"] = int(lab["emotion"])
    if lab.get("sarcasm", -1) >= 0:
        out["sarc_label"] = int(lab["sarcasm"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hpc", "mustardpp", "meld"])
    ap.add_argument("--limit", type=int, default=0, help="process at most N entries per dataset (debug)")
    ap.add_argument("--workers", type=int, default=8, help="CPU processes for pyin prosody")
    args = ap.parse_args()

    set_seed(1337)
    cfg = load_config("configs/data.yaml")
    acfg, pcfg = cfg["alignment"], cfg["prosody"]
    cache_dir, mid_dir = Path(cfg["paths"]["cache_root"]), Path(cfg["paths"]["cache_mid"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    mid_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg["paths"]["data_root"]) / "cache_log.jsonl"
    reset_vram_stats()

    entries = []
    for ds in args.datasets:
        rows = read_manifest(Path(cfg["paths"]["manifests"]) / f"{ds}.jsonl")
        entries += rows[: args.limit] if args.limit else rows
    done = {p.stem for p in cache_dir.glob("*.pt")}
    logged = set()
    if log_path.exists():
        logged = {json.loads(l)["id"] for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [e for e in entries if e["id"] not in done and e["id"] not in logged]
    print(f"S2: {len(entries)} entries | cached {len(entries) - len(todo)} | to do {len(todo)}")
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    speech, text, mms = Emotion2vecEncoder(device), TextEncoder(device=device), MMSAligner(device)
    whisper = None  # loaded lazily, only if MMS fails somewhere
    print(f"models loaded in {time.time() - t0:.0f}s")
    print_vram("models loaded")

    stats = Counter()
    log = open(log_path, "a", encoding="utf-8")
    pending = {}  # future → (entry, partial cache dict)

    def flush(block: bool):
        for fut in [f for f in pending if block or f.done()]:
            entry, data = pending.pop(fut)
            try:
                data["prosody_feats"] = torch.from_numpy(fut.result())
                torch.save(data, cache_dir / f"{entry['id']}.pt")
                stats["ok"] += 1
            except Exception as ex:
                drop(entry, f"prosody: {ex}")

    def drop(entry, reason):
        stats[reason.split(":")[0]] += 1
        log.write(json.dumps({"id": entry["id"], "reason": reason}) + "\n")
        log.flush()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n_done, entry in enumerate(tqdm(todo, desc="caching", smoothing=0.05)):
            if n_done % 200 == 0 and device == "cuda":
                torch.cuda.empty_cache()  # long clips fragment the allocator; >6 GB spills to shared memory on Windows
            try:
                audio = load_audio(entry["wav"])
            except Exception as ex:
                drop(entry, f"unreadable: {ex}")
                continue
            if len(audio) < 0.2 * SR:
                drop(entry, "too short")
                continue
            if len(audio) > acfg["max_clip_s"] * SR:  # a few MELD clips are minutes long (bad segmentation)
                drop(entry, f"too long: {len(audio) / SR:.0f}s")
                continue
            gold = [w["w"] for w in entry["words"]]

            # ---- 1. alignment ----
            res = mms.align(audio, gold)
            if res is not None:
                word_times, score = res
                method = "mms"
                if score < acfg["min_mms_score"]:
                    drop(entry, f"low alignment score: {score:.3f}")
                    continue
            else:
                if whisper is None:
                    from faster_whisper import WhisperModel
                    whisper = WhisperModel("small", device=device, compute_type="int8_float16")
                word_times = align_words_to_gold(whisper_words(whisper, audio), gold, acfg["min_word_match_ratio"])
                method, score = "whisper", None
                if word_times is None:
                    drop(entry, "whisper alignment < 80% match")
                    continue
            stats[method] += 1

            # ---- 2. speech ----
            enc = speech.encode(audio)
            audio_feats = pool_frames_to_words(enc["frames"], word_times)
            torch.save({"mid": enc["mid"]}, mid_dir / f"{entry['id']}.pt")

            # ---- 3. text ----
            text_feats = torch.from_numpy(text.extract(gold))

            data = {
                "id": entry["id"], "dataset": entry["dataset"], "split": entry["split"],
                "words": [{"w": w["w"], "start": w["start"], "end": w["end"]} for w in word_times],
                "align_method": method, "align_score": score,
                "n_frames": enc["frames"].shape[0], "duration": len(audio) / SR,
                "audio_feats": audio_feats.half(), "text_feats": text_feats.half(),
                "e2v_scores": enc["scores"], **labels_for(entry),
            }
            # ---- 4. prosody (CPU pool) ----
            fut = pool.submit(prosody_job, entry["wav"], data["words"], pcfg["f0_fmin"], pcfg["f0_fmax"])
            pending[fut] = (entry, data)
            if len(pending) > 4 * args.workers:
                flush(block=False)
                while len(pending) > 8 * args.workers:
                    time.sleep(0.05)
                    flush(block=False)
        flush(block=True)
    log.close()

    print(f"\n[OK] {dict(stats)} in {(time.time() - t0) / 60:.1f} min (drops logged to {log_path})")
    print_vram("S2 complete")


if __name__ == "__main__":
    main()
