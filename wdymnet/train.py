"""
Shared training / evaluation code for Stage A, baselines B1/B2 and Stage B.

Data: one cached .pt per utterance (from S2), all loaded into RAM.
Multi-task sampling: every batch comes from ONE dataset, round-robin over
MELD / MUStARD++ / HPC until each is exhausted; heads without labels are skipped.
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from wdymnet.eval.metrics import emotion_metrics, emphasis_metrics, sarcasm_metrics
from wdymnet.losses import WDYMNetLoss
from wdymnet.models.baselines import EmoFiLMBaseline, UtteranceFusionBaseline
from wdymnet.models.fusion import WDYMNet

DATASETS = ["meld", "mustardpp", "hpc"]


# ---------------------------------------------------------------- data
def load_cache(cache_dir: str | Path, splits=("train", "val", "test"), datasets=DATASETS) -> dict:
    """→ {split: {dataset: [item, ...]}} (items sorted by id for determinism)."""
    out = {s: {d: [] for d in datasets} for s in splits}
    paths = sorted(Path(cache_dir).glob("*.pt"))
    for p in tqdm(paths, desc="loading cache", leave=False):
        try:
            x = torch.load(p, weights_only=False)
        except Exception:  # file still being written by S2
            continue
        if x["split"] in out and x["dataset"] in datasets:
            out[x["split"]][x["dataset"]].append(x)
    return out


def collate(items: list[dict]) -> dict:
    B, N = len(items), max(len(x["words"]) for x in items)
    batch = {
        "audio": torch.zeros(B, N, items[0]["audio_feats"].shape[1]),
        "text": torch.zeros(B, N, items[0]["text_feats"].shape[1]),
        "pros": torch.zeros(B, N, items[0]["prosody_feats"].shape[1]),
        "mask": torch.zeros(B, N, dtype=torch.bool),
        "emph": torch.full((B, N), -1, dtype=torch.long),
        "emo": torch.full((B,), -1, dtype=torch.long),
        "sarc": torch.full((B,), -1, dtype=torch.long),
        "items": items,
    }
    for i, x in enumerate(items):
        n = len(x["words"])
        batch["audio"][i, :n] = x["audio_feats"].float()
        batch["text"][i, :n] = x["text_feats"].float()
        batch["pros"][i, :n] = x["prosody_feats"].float()
        batch["mask"][i, :n] = True
        if "emph_labels" in x:
            batch["emph"][i, :n] = x["emph_labels"]
        batch["emo"][i] = x.get("emo_label", -1)
        batch["sarc"][i] = x.get("sarc_label", -1)
    return batch


def batches(items: list[dict], bs: int, shuffle: bool, rng: random.Random):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i:i + bs]]


def round_robin(per_ds: dict[str, list], bs: int, rng: random.Random):
    """Yield (dataset, item list) — one dataset per batch, cycling until all are exhausted."""
    its = {d: batches(v, bs, True, rng) for d, v in per_ds.items() if v}
    while its:
        for d in list(its):
            b = next(its[d], None)
            if b is None:
                del its[d]
            else:
                yield d, b


def oversample(per_ds: dict[str, list], factors: dict[str, int] | None) -> dict[str, list]:
    """Repeat small datasets so their head gets enough batches per epoch (MUStARD++ is ~10x smaller than MELD)."""
    return {d: v * int((factors or {}).get(d, 1)) for d, v in per_ds.items()}


def n_batches(per_ds: dict[str, list], bs: int) -> int:
    return sum(math.ceil(len(v) / bs) for v in per_ds.values())


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ---------------------------------------------------------------- models
def build_model(kind: str, mcfg: dict) -> nn.Module:
    if kind == "wdymnet":
        return WDYMNet(proj_dim=mcfg["proj_dim"], n_heads=mcfg["n_heads"], n_layers=mcfg["n_layers"],
                       ff_dim=mcfg["ff_dim"], dropout=mcfg["dropout"], num_emotions=mcfg["num_emotions"],
                       num_emphasis=mcfg["num_emphasis"], use_prosody=mcfg.get("use_prosody", True),
                       use_incongruity=mcfg.get("use_incongruity", True))
    if kind == "b1":
        return UtteranceFusionBaseline()
    if kind == "b2":
        return EmoFiLMBaseline()
    raise ValueError(kind)


def forward(model, kind: str, b: dict, drop_audio=False, drop_text=False) -> dict:
    if kind == "wdymnet":
        return model(b["audio"], b["text"], b["pros"], b["mask"], drop_audio=drop_audio, drop_text=drop_text)
    if kind == "b1":
        return model(b["audio"], b["text"], b["mask"])
    out = model(b["audio"])  # b2: per-word audio classifier; utterance = mean of word probabilities
    p = F.softmax(out["emo_logits"].float(), -1) * b["mask"][..., None]
    out["emo_utt_logits"] = torch.log(p.sum(1) / b["mask"].sum(1, keepdim=True).clamp(min=1) + 1e-8)
    return out


def compute_loss(kind: str, out: dict, b: dict, loss_fn: WDYMNetLoss) -> tuple[torch.Tensor, dict]:
    if kind == "wdymnet":
        tgt = {}
        if (b["emph"] >= 0).any():
            tgt["emph_labels"] = b["emph"]
        if (b["emo"] >= 0).all():
            tgt["emo_label"] = b["emo"]
        if (b["sarc"] >= 0).all():
            tgt["sarc_label"] = b["sarc"]
        return loss_fn(out, tgt, b["mask"])
    zero = out["emo_utt_logits"].sum() * 0
    if kind == "b1":
        if (b["emo"] >= 0).all():
            l = loss_fn.emo_ce(out["emo_utt_logits"].float(), b["emo"])
        elif (b["sarc"] >= 0).all():
            l = loss_fn.sarc_ce(out["sarc_utt_logits"].float(), b["sarc"])
        else:
            l = zero
        return l, {"total": l.item()}
    # b2: copy the utterance label to every word (Emo-FiLM-style pseudo-labels)
    if not (b["emo"] >= 0).all():
        return zero, {"total": 0.0}
    word_lab = b["emo"][:, None].expand_as(b["mask"]).masked_fill(~b["mask"], -100)
    l = F.cross_entropy(out["emo_logits"].float().flatten(0, 1), word_lab.flatten(),
                        weight=loss_fn.emo_ce.weight, ignore_index=-100)
    return l, {"total": l.item()}


def train_datasets(kind: str) -> list[str]:
    return {"wdymnet": DATASETS, "b1": ["meld", "mustardpp"], "b2": ["meld"]}[kind]


# ---------------------------------------------------------------- eval
@torch.no_grad()
def predict(model, kind: str, items: list[dict], device, bs: int = 64, fwd=None) -> list[dict]:
    """Per-item outputs (numpy) for metrics, tables and faithfulness. `fwd` overrides forward (Stage B)."""
    model.eval()
    res = []
    for chunk in batches(items, bs, False, random.Random(0)):
        b = to_device(collate(chunk), device)
        # fp32 for the small cached-feature model (fp16 occasionally overflowed to NaN); fp16 only for Stage B encoders
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda" and fwd is not None):
            out = fwd(b) if fwd else forward(model, kind, b)
        for i, x in enumerate(chunk):
            n = len(x["words"])
            r = {"id": x["id"], "dataset": x["dataset"], "n": n}
            for k in ["emph_logits", "emo_logits", "sarc_logits", "audio_emo_logits", "text_emo_logits"]:
                if k in out:
                    r[k.replace("_logits", "_prob")] = F.softmax(out[k][i, :n].float(), -1).cpu().numpy()
            for k in ["emo_attn", "sarc_attn", "incongruity"]:
                if k in out:
                    r[k] = out[k][i, :n].float().cpu().numpy()
            for k in ["emo_utt_logits", "sarc_utt_logits", "audio_emo_utt", "text_emo_utt"]:
                if k in out:
                    r[k.replace("_logits", "")] = F.softmax(out[k][i].float(), -1).cpu().numpy()
            res.append(r)
    return res


def metrics_from_preds(items: list[dict], preds: list[dict]) -> dict:
    m = {}
    emph_p, emph_l, emo_p, emo_l, sarc_p, sarc_l, sarc_s = [], [], [], [], [], [], []
    for x, p in zip(items, preds):
        if "emph_labels" in x and "emph_prob" in p:
            lab = x["emph_labels"].numpy()
            keep = lab >= 0
            emph_p += list(p["emph_prob"].argmax(-1)[keep]); emph_l += list(lab[keep])
        if "emo_label" in x and "emo_utt" in p:
            emo_p.append(int(p["emo_utt"].argmax())); emo_l.append(x["emo_label"])
        if "sarc_label" in x and "sarc_utt" in p:
            sarc_p.append(int(p["sarc_utt"].argmax())); sarc_l.append(x["sarc_label"]); sarc_s.append(float(p["sarc_utt"][1]))
    if emph_p:
        m["emph"] = emphasis_metrics(emph_p, emph_l)
    if emo_p:
        m["emo"] = emotion_metrics(emo_p, emo_l)
    if sarc_p:
        m["sarc"] = sarcasm_metrics(sarc_p, sarc_l, sarc_s)
    # model selection: emphasis macro-F1, emotion macro-F1, sarcasm AUC (threshold-free; its threshold is calibrated later)
    f1s = [m["emph"]["f1_3way"]] if "emph" in m else []
    f1s += [m["emo"]["macro_f1"]] if "emo" in m else []
    f1s += [m["sarc"]["auc"] if m["sarc"]["auc"] is not None else m["sarc"]["macro_f1"]] if "sarc" in m else []
    m["val_macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return m


def summary_line(m: dict) -> str:
    parts = [f"macroF1 {m['val_macro_f1']:.3f}"]
    if "emph" in m:
        parts.append(f"emph acc3 {m['emph']['acc_3way']:.3f} acc2 {m['emph']['acc_2way']:.3f}")
    if "emo" in m:
        parts.append(f"emo wF1 {m['emo']['weighted_f1']:.3f} mF1 {m['emo']['macro_f1']:.3f}")
    if "sarc" in m:
        parts.append(f"sarc mF1 {m['sarc']['macro_f1']:.3f} AUC {m['sarc']['auc'] or 0:.3f}")
    return " | ".join(parts)


# ---------------------------------------------------------------- Stage A loop
def make_loss(cfg: dict, data_cfg: dict, device) -> WDYMNetLoss:
    lc = cfg["loss"]
    return WDYMNetLoss(emo_class_weights=data_cfg["meld"]["class_weights"], emph_weight=lc["emph_weight"],
                       emo_weight=lc["emo_weight"], sarc_weight=lc["sarc_weight"],
                       unimodal_weight=lc["unimodal_weight"], sparsity_weight=lc["sparsity_weight"],
                       device=str(device))


def cosine_warmup(optimizer, total: int, warmup_ratio: float):
    warm = max(1, int(total * warmup_ratio))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


def modality_drop(cfg: dict, rng: random.Random) -> tuple[bool, bool]:
    md = cfg["model"].get("modality_dropout", {})
    da, dt = rng.random() < md.get("audio_p", 0.0), rng.random() < md.get("text_p", 0.0)
    if da and dt:  # never drop both
        da, dt = (True, False) if rng.random() < 0.5 else (False, True)
    return da, dt


def train_stage_a(kind: str, cfg: dict, data_cfg: dict, data: dict, device, out_name: str, log=print) -> dict:
    """Train on cached features; early stopping on val macro-F1; saves results/checkpoints/{out_name}.pt."""
    tc = cfg["training"]
    rng = random.Random(cfg["seed"])
    ds_names = train_datasets(kind)
    train = oversample({d: data["train"][d] for d in ds_names}, tc.get("oversample"))
    val_items = [x for d in ds_names for x in data["val"][d]]

    model = build_model(kind, cfg["model"]).to(device)
    loss_fn = make_loss(cfg, data_cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps = n_batches(train, tc["batch_size"]) * tc["max_epochs"]
    sched = cosine_warmup(opt, steps, tc["warmup_ratio"])
    scaler = torch.amp.GradScaler("cuda", enabled=tc.get("fp16", True) and device.type == "cuda")
    log(f"[{out_name}] params {sum(p.numel() for p in model.parameters()):,} | "
        f"train {({d: len(v) for d, v in train.items()})} | val {len(val_items)}")

    ckpt_dir = Path(cfg["checkpoint"]["save_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best, best_m, patience, history = -1.0, None, 0, []
    for ep in range(tc["max_epochs"]):
        model.train(); t0 = time.time(); tot, nb = 0.0, 0
        for ds, items in tqdm(round_robin(train, tc["batch_size"], rng), total=n_batches(train, tc["batch_size"]),
                              desc=f"ep{ep + 1}", leave=False):
            b = to_device(collate(items), device)
            da, dt = modality_drop(cfg, rng) if kind == "wdymnet" else (False, False)
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                out = forward(model, kind, b, da, dt)
                loss, _ = compute_loss(kind, out, b, loss_fn)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); nb += 1
        m = metrics_from_preds(val_items, predict(model, kind, val_items, device))
        history.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), **{k: v for k, v in m.items()}})
        log(f"  ep {ep + 1:2d} loss {tot / max(nb, 1):.3f} | val {summary_line(m)} | {time.time() - t0:.0f}s")
        if m["val_macro_f1"] > best:
            best, best_m, patience = m["val_macro_f1"], m, 0
            torch.save({"kind": kind, "model": model.state_dict(), "config": cfg, "epoch": ep + 1, "val": m},
                       ckpt_dir / f"{out_name}.pt")
        else:
            patience += 1
            if patience >= tc["early_stopping_patience"]:
                log(f"  early stop at epoch {ep + 1}")
                break
    return {"best_val": best_m, "history": history}


def load_checkpoint(path: str | Path, device) -> tuple[nn.Module, str, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["kind"], ck["config"]["model"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck["kind"], ck


def save_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=2, default=conv), encoding="utf-8")
