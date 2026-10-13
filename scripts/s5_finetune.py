"""
S5 — Stage B fine-tuning: top-3 emotion2vec blocks (lr 2e-5) + RoBERTa LoRA (lr 2e-4) + fusion/heads (lr 5e-4),
initialised from Stage A. Batch 8 × grad-accum 4, fp16, utterances ≤ 10 s.
Keeps the better checkpoint on validation: writes results/checkpoints/stage_b_best.pt only if it beats Stage A.

Usage: python scripts/s5_finetune.py [--config configs/train_b.yaml] [--max-steps N (debug)]
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from tqdm import tqdm

from wdymnet.finetune import StageBFeaturizer
from wdymnet.train import (collate, cosine_warmup, load_cache, load_checkpoint, make_loss, metrics_from_preds,
                           modality_drop, predict, round_robin, n_batches, save_json, summary_line, to_device,
                           compute_loss, oversample, DATASETS)
from wdymnet.utils.common import load_config, print_vram, reset_vram_stats, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_b.yaml")
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    cfg, data_cfg = load_config(args.config), load_config("configs/data.yaml")
    set_seed(cfg["seed"])
    device = torch.device("cuda")
    reset_vram_stats()
    tc, ft = cfg["training"], cfg["model"]["finetune"]
    rng = random.Random(cfg["seed"])
    t_start = time.time()

    data = load_cache(data_cfg["paths"]["cache_root"], splits=("train", "val"))
    cap = tc["max_utterance_len_s"]
    train = oversample({d: [x for x in data["train"][d] if x["duration"] <= cap] for d in DATASETS}, tc.get("oversample"))
    val_items = [x for d in DATASETS for x in data["val"][d]]
    print("train (≤%.0fs): %s | val %d" % (cap, {d: len(v) for d, v in train.items()}, len(val_items)))

    ck_dir = Path(cfg["checkpoint"]["save_dir"])
    model, _, ck_a = load_checkpoint(cfg["checkpoint"]["init_from"], device)
    stage_a_val = ck_a["val"]["val_macro_f1"]
    feat = StageBFeaturizer(data_cfg["paths"]["cache_mid"], ft["lora"], "cuda")
    n_lora = sum(p.numel() for p in feat.text.parameters() if p.requires_grad)
    n_top = sum(p.numel() for p in feat.top.parameters() if p.requires_grad)
    print(f"trainable: heads {sum(p.numel() for p in model.parameters()):,} | e2v top-3 {n_top:,} | LoRA {n_lora:,}")

    opt = torch.optim.AdamW([
        {"params": model.parameters(), "lr": tc["lr"]},
        {"params": [p for p in feat.top.parameters() if p.requires_grad], "lr": ft["speech_encoder_lr"]},
        {"params": [p for p in feat.text.parameters() if p.requires_grad], "lr": ft["lora"]["lr"]},
    ], weight_decay=tc["weight_decay"])
    accum = tc["grad_accumulation_steps"]
    steps = n_batches(train, tc["batch_size"]) * tc["max_epochs"] // accum
    sched = cosine_warmup(opt, steps, tc["warmup_ratio"])
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = make_loss(cfg, data_cfg, device)
    params = [p for g in opt.param_groups for p in g["params"]]

    def fwd(b):
        return feat(model, b)

    best, history = stage_a_val, []
    print(f"Stage A val macro-F1 to beat: {stage_a_val:.4f}")
    for ep in range(tc["max_epochs"]):
        model.train(); feat.train(); t0 = time.time(); tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        total = n_batches(train, tc["batch_size"])
        for step, (ds, items) in enumerate(tqdm(round_robin(train, tc["batch_size"], rng), total=total, desc=f"ep{ep + 1}")):
            b = to_device(collate(items), device)
            da, dt = modality_drop(cfg, rng)
            with torch.autocast("cuda", dtype=torch.float16):
                out = feat(model, b, da, dt)
                loss, _ = compute_loss("wdymnet", out, b, loss_fn)
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == total:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, tc["gradient_clip"])
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            tot += loss.item(); nb += 1
            if args.max_steps and step + 1 >= args.max_steps:
                break
        feat.eval()
        m = metrics_from_preds(val_items, predict(model, "wdymnet", val_items, device, bs=16, fwd=fwd))
        history.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), **m})
        print(f"  ep {ep + 1} loss {tot / max(nb, 1):.3f} | val {summary_line(m)} | {time.time() - t0:.0f}s")
        print_vram(f"epoch {ep + 1}")
        state = {"kind": "wdymnet", "model": model.state_dict(), "config": cfg, "epoch": ep + 1, "val": m,
                 **feat.state_for_checkpoint()}
        torch.save(state, ck_dir / "stage_b_last.pt")
        if m["val_macro_f1"] > best:
            best = m["val_macro_f1"]
            torch.save(state, ck_dir / "stage_b_best.pt")
            print(f"  ✓ new best (beats Stage A {stage_a_val:.4f}) → stage_b_best.pt")
        if args.max_steps:
            break

    res = {"stage_a_val_macro_f1": stage_a_val, "best_val_macro_f1": best, "beats_stage_a": best > stage_a_val,
           "history": history, "time_min": round((time.time() - t_start) / 60, 1),
           "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)}
    save_json(res, cfg.get("results_path", "results/s5_results.json"))
    print(f"\nStage B best val macro-F1 {best:.4f} vs Stage A {stage_a_val:.4f} | {res['time_min']} min")
    print_vram("S5 complete")


if __name__ == "__main__":
    main()
