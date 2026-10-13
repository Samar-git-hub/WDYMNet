"""
S1 — Prepare the Helsinki Prosody Corpus (word-level emphasis) + LibriTTS audio.

HPC files (github.com/Helsinki-NLP/prosody, data/{dev,test,train_100,train_360}.txt):
  "<file>\t{libritts_id}.txt" starts a sentence; word lines are
  word \t prominence(0/1/2/NA) \t boundary \t real_prominence \t real_boundary.
NA rows are punctuation and are dropped (no label, nothing to align).

Plan: HPC dev → train (≈10% of speakers held out as val); HPC test → test.
Audio: LibriTTS dev-clean / test-clean from OpenSLR SLR60 (~1.2 GB each).

Usage: python scripts/s1_prepare_hpc.py
"""
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import GroupShuffleSplit

from wdymnet.data.download import download, extract_tar
from wdymnet.data.manifest import write_manifest
from wdymnet.utils.common import load_config, set_seed

LIBRITTS_URL = "https://openslr.elda.org/resources/60/{}.tar.gz"  # EU mirror of openslr.org/60 (much faster)


def parse_hpc_file(path: Path) -> list[dict]:
    """→ [{file_id, words:[{w, label}]}], NA (punctuation) tokens dropped."""
    sents, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("<file>"):
            cur = {"file_id": line.split()[1].removesuffix(".txt"), "words": []}
            sents.append(cur)
            continue
        parts = line.split("\t")
        if cur is not None and len(parts) >= 2 and parts[1] in ("0", "1", "2"):
            cur["words"].append({"w": parts[0], "label": int(parts[1])})
    return [s for s in sents if s["words"]]


def find_wav(file_id: str, libritts_dir: Path, split: str) -> Path | None:
    spk, chap = file_id.split("_")[:2]
    p = libritts_dir / split / spk / chap / f"{file_id}.wav"
    return p if p.exists() else None


def main():
    set_seed(1337)
    cfg = load_config("configs/data.yaml")
    root = Path(cfg["paths"]["data_root"])
    hpc_dir, libritts_dir = root / "prosody", root / "LibriTTS"

    print("=" * 60 + "\nS1 — Prepare HPC + LibriTTS\n" + "=" * 60)
    if not (hpc_dir / "data" / "dev.txt").exists():
        subprocess.run(["git", "clone", "--depth", "1", cfg["hpc"]["repo_url"], str(hpc_dir)], check=True)

    for split in cfg["hpc"]["libritts_splits"]:
        tar = download(LIBRITTS_URL.format(split), root / "downloads" / f"{split}.tar.gz")
        extract_tar(tar, root, marker=split)  # archive root is LibriTTS/{split}/...

    entries = []
    for hpc_name, libri_split, our_split in [("dev", "dev-clean", "train"), ("test", "test-clean", "test")]:
        sents = parse_hpc_file(hpc_dir / "data" / f"{hpc_name}.txt")
        missing, n_checked, n_text_ok = 0, 0, 0
        for s in sents:
            wav = find_wav(s["file_id"], libritts_dir, libri_split)
            if wav is None:
                missing += 1
                continue
            # spot-check that HPC words match the LibriTTS normalized transcript
            if n_checked < 20:
                n_checked += 1
                norm = wav.with_suffix(".normalized.txt").read_text(encoding="utf-8").lower()
                n_text_ok += all(w["w"].strip("'\"").lower() in norm for w in s["words"][:5])
            entries.append({
                "id": f"hpc_{s['file_id']}",
                "wav": wav.as_posix(),
                "text": " ".join(w["w"] for w in s["words"]),
                "words": s["words"],
                "labels": {},
                "speaker": s["file_id"].split("_")[0],
                "split": our_split,
                "dataset": "hpc",
            })
        print(f"  HPC {hpc_name}: {len(sents)} sentences, {missing} without wav, "
              f"transcript spot-check {n_text_ok}/{n_checked} OK")

    # hold out ~10% of HPC-dev speakers for validation
    train = [e for e in entries if e["split"] == "train"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=1337)
    _, val_idx = next(gss.split(train, groups=[e["speaker"] for e in train]))
    for i in val_idx:
        train[i]["split"] = "val"

    out = Path(cfg["paths"]["manifests"]) / "hpc.jsonl"
    write_manifest(entries, out)
    print(f"\n[OK] {len(entries)} entries → {out}")
    for s in ["train", "val", "test"]:
        sub = [e for e in entries if e["split"] == s]
        labs = Counter(w["label"] for e in sub for w in e["words"])
        print(f"  {s}: {len(sub)} sents, {sum(labs.values())} words, speakers={len({e['speaker'] for e in sub})}, "
              f"labels={dict(sorted(labs.items()))}")


if __name__ == "__main__":
    main()
