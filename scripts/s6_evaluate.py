"""
S6 — Sanity evaluation (one table + data for plots).

  1. Test metrics: emphasis (HPC, 3-/2-way), emotion (MELD wF1: B0, B1, B2, ours-A, ours-B),
     sarcasm (MUStARD++ unseen speakers, macro-F1).
  2. Conflict: on sarcastic test clips, sarcasm detection rate with text-only / audio-only / fused input.
  3. Faithfulness: for correctly detected sarcastic / emotional test clips, mute the audio of the top-k
     tagged words vs k random words (k=1,2), re-extract features, report mean probability drop.
  4. Ablations on ours (Stage A retrained): −prosody, −incongruity, −modality dropout; −fine-tuning = A vs B.
  5. Word-tag display thresholds tuned on validation → results/thresholds.json.
  6. Live latency / VRAM on 5 s clips → results/latency.json.

Usage: python scripts/s6_evaluate.py [--skip-ablations] [--faith-n 80]
Writes results/s6_results.json, results/main_table.csv, results/preds_test.pt (for S8 figures).
"""
import argparse
import copy
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import f1_score

from wdymnet.eval.faithfulness import faithfulness_trial
from wdymnet.eval.metrics import emotion_metrics
from wdymnet.models.baselines import map_e2v_scores_to_meld
from wdymnet.train import (DATASETS, collate, forward, load_cache, load_checkpoint, metrics_from_preds, predict,
                           save_json, to_device, train_stage_a)
from wdymnet.utils.common import load_config, print_vram, reset_vram_stats, set_seed

CK = Path("results/checkpoints")
SR = 16000


_WAVS: dict = {}


def load_audio_for(item_id: str) -> np.ndarray:
    """16 kHz mono audio of a cached item (wav path looked up in the manifests)."""
    import soundfile as sf
    from wdymnet.data.manifest import read_manifest
    from wdymnet.pipeline import to_mono16k
    if not _WAVS:
        for d in DATASETS:
            _WAVS.update({e["id"]: e["wav"] for e in read_manifest(Path("data/manifests") / f"{d}.jsonl")})
    a, sr = sf.read(_WAVS[item_id], dtype="float32")
    return to_mono16k(a, sr)


def flat(data, split, names=DATASETS):
    return [x for d in names for x in data[split][d]]


@torch.no_grad()
def conflict_eval(model, items, device, fwd_kw=None) -> dict:
    """Sarcasm detection rate on sarcastic clips with one modality dropped."""
    sarc = [x for x in items if x.get("sarc_label") == 1]
    out = {}
    for name, (da, dt) in {"text_only": (True, False), "audio_only": (False, True), "fused": (False, False)}.items():
        hits = 0
        for i in range(0, len(sarc), 64):
            b = to_device(collate(sarc[i:i + 64]), device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=False):  # Stage A model runs in fp32
                o = forward(model, "wdymnet", b, da, dt)
            hits += int((o["sarc_utt_logits"].argmax(-1) == 1).sum())
        out[name] = hits / max(len(sarc), 1)
    out["n"] = len(sarc)
    return out


def tune_thresholds(val_items, val_preds, max_false_alarm: float = 0.10) -> dict:
    """Word-tag display thresholds for the live demo, from the deployed model's validation predictions.
    sarcasm: MUStARD++ is 50% sarcastic but ordinary speech is not, so pick the lowest threshold whose false-alarm
             rate on non-sarcastic validation speech (MUStARD++ non-sarcastic + MELD + HPC read speech) is <= 10%.
    emotion: threshold on the max non-neutral word prob (only words where it beats neutral) that best separates
             neutral vs emotional MELD validation utterances (macro-F1)."""
    neg = [p["sarc_utt"][1] for x, p in zip(val_items, val_preds) if x.get("sarc_label", 0) == 0]
    pos = [p["sarc_utt"][1] for x, p in zip(val_items, val_preds) if x.get("sarc_label") == 1]
    t_s = float(np.quantile(neg, 1 - max_false_alarm))
    def word_emo(p):
        nn = p["emo_prob"][:, 1:].max(1)
        nn = nn[nn > p["emo_prob"][:, 0]]
        return nn.max() if len(nn) else 0.0
    e = [(int(x["emo_label"] != 0), word_emo(p)) for x, p in zip(val_items, val_preds) if "emo_label" in x]
    grid = np.round(np.arange(0.2, 0.81, 0.05), 2)
    t_e = max(grid, key=lambda t: f1_score([a for a, _ in e], [int(b > t) for _, b in e], average="macro"))
    return {"sarcasm_threshold": round(t_s, 3), "emotion_threshold": float(t_e),
            "sarcasm_val_false_alarm": float(np.mean(np.array(neg) > t_s)),
            "sarcasm_val_recall": float(np.mean(np.array(pos) > t_s))}


def faithfulness_eval(tagger, test_items, preds, n_max: int, rng, sarc_th: float) -> dict:
    """Mute top-k tagged vs random words; re-extract features from the muted audio with the same word times."""
    res = {}
    for task in ["sarcasm", "emotion"]:
        if task == "sarcasm":
            cand = [(x, p, 1) for x, p in zip(test_items, preds) if x.get("sarc_label") == 1 and p["sarc_utt"][1] > sarc_th]
        else:
            cand = [(x, p, x["emo_label"]) for x, p in zip(test_items, preds)
                    if x.get("emo_label", 0) != 0 and "emo_utt" in p and p["emo_utt"].argmax() == x["emo_label"]]
        rng.shuffle(cand)
        cand = cand[:n_max]
        for k in (1, 2):
            trials = []
            for x, p, c in cand:
                audio = load_audio_for(x["id"])
                words = x["words"]

                def prob(a, w, c=c):
                    o = tagger.run_model(tagger.features(a, w))
                    return o["sarc_utt"] if task == "sarcasm" else float(o["emo_utt"][c])

                # rank words the way the demo marks them: sarcasm by MIL attention, emotion by word prob
                scores = p["sarc_attn"] if task == "sarcasm" else p["emo_prob"][:, c]
                t = faithfulness_trial(prob, audio, SR, words, scores, k, rng)
                if t:
                    trials.append(t)
            res[f"{task}_k{k}"] = {"n": len(trials),
                                   "tagged_drop": float(np.mean([t["tagged_drop"] for t in trials])) if trials else None,
                                   "random_drop": float(np.mean([t["random_drop"] for t in trials])) if trials else None,
                                   "tagged_gt_random": float(np.mean([t["tagged_drop"] > t["random_drop"] for t in trials])) if trials else None}
            print(f"  faithfulness {task} k={k}: {res[f'{task}_k{k}']}")
    return res


def latency_eval(tagger, items, n=20) -> dict:
    clips = [x for x in items if 4.0 <= x["duration"] <= 6.0][:n]
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for x in clips:
        r = tagger.tag(load_audio_for(x["id"]), SR)
        if r["status"] == "ok":
            rows.append(r["timing"])
    keys = ["vad", "asr", "features", "model", "total", "audio_s"]
    return {"n": len(rows), "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
            "p90": {k: float(np.percentile([r[k] for r in rows], 90)) for k in keys},
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--faith-n", type=int, default=80, help="clips per task for the muting test")
    args = ap.parse_args()
    set_seed(1337)
    device = torch.device("cuda")
    reset_vram_stats()
    data_cfg, cfg_a = load_config("configs/data.yaml"), load_config("configs/train_a.yaml")
    t0 = time.time()
    data = load_cache(data_cfg["paths"]["cache_root"])
    test, val = flat(data, "test"), flat(data, "val")
    R = {}

    # ---- models ----
    models = {"B1": CK / "b1.pt", "B2": CK / "b2.pt", "ours-A": CK / "stage_a_best.pt"}
    preds = {}
    for name, p in models.items():
        if p.exists():
            m, kind, _ = load_checkpoint(p, device)
            preds[name] = predict(m, kind, test, device)
            R[name] = metrics_from_preds(test, preds[name])
    meld_test = data["test"]["meld"]
    R["B0"] = {"emo": emotion_metrics(
        [int(np.argmax(map_e2v_scores_to_meld(x["e2v_scores"].tolist()))) for x in meld_test],
        [x["emo_label"] for x in meld_test])}

    # Stage B is reported even when it is not selected (S5 only writes stage_b_best.pt if it beats Stage A on val)
    b_selected = (CK / "stage_b_best.pt").exists()
    stage_b = CK / ("stage_b_best.pt" if b_selected else "stage_b_last.pt")
    val_preds_b = None
    if stage_b.exists():
        from wdymnet.finetune import StageBFeaturizer
        mb, _, ckb = load_checkpoint(stage_b, device)
        feat = StageBFeaturizer(data_cfg["paths"]["cache_mid"], ckb["config"]["model"]["finetune"]["lora"]).eval()
        feat.load_from_checkpoint(ckb)
        preds["ours-B"] = predict(mb, "wdymnet", test, device, bs=16, fwd=lambda b: feat(mb, b))
        R["ours-B"] = metrics_from_preds(test, preds["ours-B"])
        R["ours-B"]["selected_on_val"] = b_selected
        if b_selected:
            val_preds_b = predict(mb, "wdymnet", val, device, bs=16, fwd=lambda b: feat(mb, b))
        del feat
        torch.cuda.empty_cache()
    best_name = "ours-B" if b_selected else "ours-A"
    R["best_model"] = best_name

    # ---- conflict (Stage A model; Stage B uses the same heads) ----
    ma, _, _ = load_checkpoint(CK / "stage_a_best.pt", device)
    R["conflict_ours-A"] = conflict_eval(ma, data["test"]["mustardpp"], device)
    print("conflict:", R["conflict_ours-A"])

    # ---- thresholds on validation ----
    val_best = val_preds_b if best_name == "ours-B" else predict(ma, "wdymnet", val, device)
    R["thresholds"] = tune_thresholds(val, val_best)
    save_json(R["thresholds"], "results/thresholds.json")
    print("thresholds:", R["thresholds"])

    # ---- ablations (retrain Stage A variants; evaluate on test) ----
    if not args.skip_ablations:
        variants = {"-prosody": ("use_prosody", False), "-incongruity": ("use_incongruity", False),
                    "-modality dropout": ("modality_dropout", {"text_p": 0.0, "audio_p": 0.0})}
        R["ablations"] = {}
        for name, (key, val_) in variants.items():
            cfg = copy.deepcopy(cfg_a)
            cfg["model"][key] = val_
            set_seed(cfg["seed"])
            tag = "abl_" + name.strip("-").replace(" ", "_")
            train_stage_a("wdymnet", cfg, data_cfg, data, device, tag, log=lambda s: None)
            m, _, _ = load_checkpoint(CK / f"{tag}.pt", device)
            R["ablations"][name] = metrics_from_preds(test, predict(m, "wdymnet", test, device))
            print(f"  ablation {name}: macroF1 {R['ablations'][name]['val_macro_f1']:.3f}")

    # ---- faithfulness + latency (full pipeline with the best checkpoint) ----
    from wdymnet.pipeline import Tagger
    live = load_config("configs/live.yaml")
    tagger = Tagger(CK / ("stage_b_best.pt" if best_name == "ours-B" else "stage_a_best.pt"),
                    vad_cfg=live["vad"], beam_size=live["asr"].get("beam_size", 1))
    rng = np.random.default_rng(1337)
    R["faithfulness"] = faithfulness_eval(tagger, test, preds[best_name], args.faith_n, rng,
                                          R["thresholds"]["sarcasm_threshold"])
    # live-style sarcasm decisions on test at the calibrated threshold
    th_s = R["thresholds"]["sarcasm_threshold"]
    tp = [(x.get("sarc_label"), p["sarc_utt"][1]) for x, p in zip(test, preds[best_name]) if "sarc_label" in x]
    hpc = [p["sarc_utt"][1] for x, p in zip(test, preds[best_name]) if x["dataset"] == "hpc"]
    R["sarcasm_calibrated_test"] = {"threshold": th_s,
                                    "recall": float(np.mean([s > th_s for y, s in tp if y == 1])),
                                    "false_alarm_mustardpp": float(np.mean([s > th_s for y, s in tp if y == 0])),
                                    "false_alarm_read_speech": float(np.mean(np.array(hpc) > th_s))}
    print("calibrated sarcasm on test:", R["sarcasm_calibrated_test"])
    R["latency"] = latency_eval(tagger, data["test"]["mustardpp"] + data["test"]["meld"])
    save_json(R["latency"], "results/latency.json")
    print("latency:", R["latency"]["mean"], "peak VRAM", R["latency"]["peak_vram_mb"])

    # ---- outputs ----
    R["time_min"] = round((time.time() - t0) / 60, 1)
    R["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    save_json(R, "results/s6_results.json")
    torch.save({"ids": [x["id"] for x in test], "preds": preds, "words": [x["words"] for x in test]},
               "results/preds_test.pt")
    write_main_table(R)
    print(f"\nsaved results/s6_results.json + main_table.csv | {R['time_min']} min")
    print_vram("S6 complete")


def write_main_table(R: dict):
    def g(name, *path):
        v = R.get(name)
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
        return "" if v is None else f"{100 * v:.1f}"
    rows = [["Model", "Emph acc 3-way", "Emph acc 2-way", "Emph F1 3-way", "Emotion wF1 (MELD)",
             "Sarcasm macro-F1 (MUStARD++)", "Sarcasm AUC"]]
    for n in ["B0", "B1", "B2", "ours-A", "ours-B"]:
        if n in R:
            label = f"{n} (not selected on val)" if n == "ours-B" and not R[n].get("selected_on_val", True) else n
            rows.append([label, g(n, "emph", "acc_3way"), g(n, "emph", "acc_2way"), g(n, "emph", "f1_3way"),
                         g(n, "emo", "weighted_f1"), g(n, "sarc", "macro_f1"), g(n, "sarc", "auc")])
    for n, m in R.get("ablations", {}).items():
        R["_tmp"] = m
        rows.append([f"ours {n}", g("_tmp", "emph", "acc_3way"), g("_tmp", "emph", "acc_2way"),
                     g("_tmp", "emph", "f1_3way"), g("_tmp", "emo", "weighted_f1"), g("_tmp", "sarc", "macro_f1"), g("_tmp", "sarc", "auc")])
    R.pop("_tmp", None)
    with open("results/main_table.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    main()
