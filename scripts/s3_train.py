"""
S3 — Stage A training on cached features (frozen encoders) + baselines.

  ours-A : WDYMNet fusion model (emphasis + MIL emotion/sarcasm + incongruity)
  B0     : emotion2vec+ 9-class head mapped to MELD (no training; from S2 cached scores)
  B1     : utterance-only fusion (mean-pooled audio+text → classifier)
  B2     : Emo-FiLM-style per-word audio classifier on copied utterance labels

Usage: python scripts/s3_train.py [--config configs/train_a.yaml] [--only ours b1 b2]
Writes results/checkpoints/{stage_a_best,b1,b2}.pt and results/s3_results.json.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from wdymnet.eval.metrics import emotion_metrics
from wdymnet.models.baselines import map_e2v_scores_to_meld
from wdymnet.features.speech import ENCODER_NAME
from wdymnet.train import load_cache, save_json, summary_line, train_stage_a
from wdymnet.utils.common import load_config, print_vram, reset_vram_stats, set_seed


def b0_metrics(items: list[dict]) -> dict | None:
    if not items:
        return None
    preds = [int(torch.tensor(map_e2v_scores_to_meld(x["e2v_scores"].tolist())).argmax()) for x in items]
    return emotion_metrics(preds, [x["emo_label"] for x in items])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_a.yaml")
    ap.add_argument("--only", nargs="+", default=["ours", "b1", "b2"])
    args = ap.parse_args()

    cfg, data_cfg = load_config(args.config), load_config("configs/data.yaml")
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_vram_stats()
    t0 = time.time()
    data = load_cache(data_cfg["paths"]["cache_root"])
    for s in data:
        print(f"  {s}: " + ", ".join(f"{d}={len(v)}" for d, v in data[s].items()))

    out_path = Path(cfg.get("results_path", "results/s3_results.json"))
    results = {}
    if out_path.exists():
        import json
        results = json.loads(out_path.read_text(encoding="utf-8"))
    results["speech_encoder"] = ENCODER_NAME
    results["B0_val"] = b0_metrics(data["val"]["meld"])
    if results["B0_val"]:
        print(f"B0 (emotion2vec+ zero-shot) val: wF1 {results['B0_val']['weighted_f1']:.3f}")

    names = {"ours": ("wdymnet", "stage_a_best"), "b1": ("b1", "b1"), "b2": ("b2", "b2")}
    for key in args.only:
        set_seed(cfg["seed"])
        kind, out_name = names[key]
        r = train_stage_a(kind, cfg, data_cfg, data, device, out_name)
        results[out_name] = r
        print(f"[{out_name}] best val: {summary_line(r['best_val'])}")

    results["time_min"] = round((time.time() - t0) / 60, 1)
    results["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0
    save_json(results, out_path)
    print(f"\nsaved {out_path} | {results['time_min']} min")
    print_vram("S3 complete")


if __name__ == "__main__":
    main()
