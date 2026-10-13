"""
S8 — Paper artifacts from results/ (no training, no GPU needed).

  paper_artifacts/main_table.{csv,tex}      sanity table (S6)
  paper_artifacts/confusion_meld.png        MELD test confusion matrix (best model)
  paper_artifacts/incongruity_examples.png  per-word incongruity / sarcasm / emphasis for 4 sarcastic test clips
  paper_artifacts/faithfulness.png          prob. drop: mute top-k tagged vs k random words
  paper_artifacts/latency_vram.png          live latency per stage + peak VRAM
  paper_artifacts/architecture.mmd          architecture diagram (mermaid)

Usage: python scripts/s8_make_figures.py
"""
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = Path("paper_artifacts")
EMOTIONS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
# reference categorical palette (light mode), fixed order
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})


def tables(R: dict):
    shutil.copy("results/main_table.csv", OUT / "main_table.csv")
    rows = list(csv.reader(open(OUT / "main_table.csv", encoding="utf-8")))
    tex = ["\\begin{tabular}{l" + "c" * (len(rows[0]) - 1) + "}", "\\toprule",
           " & ".join(rows[0]) + " \\\\", "\\midrule"]
    tex += [" & ".join(c if c else "--" for c in r).replace("-prosody", "$-$prosody") + " \\\\" for r in rows[1:]]
    if "conflict_ours-A" in R:
        c = R["conflict_ours-A"]
        tex += ["\\midrule", f"\\multicolumn{{{len(rows[0])}}}{{l}}{{Sarcastic test clips detected (n={c['n']}): "
                f"text-only {100 * c['text_only']:.1f}, audio-only {100 * c['audio_only']:.1f}, "
                f"fused {100 * c['fused']:.1f}}} \\\\"]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "main_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def confusion(R: dict):
    cm = np.array(R[R["best_model"]]["emo"]["confusion_matrix"], dtype=float)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(7), EMOTIONS, rotation=40, ha="right"); ax.set_yticks(range(7), EMOTIONS)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold")
    ax.set_title(f"MELD test emotion — {R['best_model']} (row-normalised)", color=INK, fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(OUT / "confusion_meld.png"); plt.close(fig)


def incongruity_examples(R: dict, n=4):
    P = torch.load("results/preds_test.pt", weights_only=False)
    preds, ids, words = P["preds"][R["best_model"]], P["ids"], P["words"]
    cands = [(float(p["incongruity"].max()), i) for i, p in enumerate(preds)
             if p["dataset"] == "mustardpp" and p["sarc_utt"][1] > 0.5 and 4 <= p["n"] <= 12]
    picks = [i for _, i in sorted(cands, reverse=True)[:n]]
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(8, 1.55 * len(picks)))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, picks):
        p = preds[i]
        M = np.stack([p["incongruity"], p["sarc_prob"][:, 1], 1 - p["emph_prob"][:, 0]])
        ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
        for r in range(3):
            for c in range(M.shape[1]):
                ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[r, c] > 0.6 else INK)
        ax.set_xticks(range(p["n"]), [w["w"] for w in words[i]], fontsize=8.5, color=INK)
        ax.set_yticks(range(3), ["incongruity", "sarcasm", "emphasis"], fontsize=8)
        ax.set_title(f"{ids[i]}  (utterance sarcasm p={p['sarc_utt'][1]:.2f})", fontsize=8.5, color=INK2, loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "incongruity_examples.png"); plt.close(fig)


def faithfulness(R: dict):
    F = R.get("faithfulness", {})
    keys = [k for k in ["sarcasm_k1", "sarcasm_k2", "emotion_k1", "emotion_k2"] if F.get(k, {}).get("n")]
    if not keys:
        return
    x = np.arange(len(keys)); w = 0.36
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for off, key, col, lab in [(-w / 2, "tagged_drop", C["blue"], "mute top-k tagged words"),
                               (w / 2, "random_drop", C["orange"], "mute k random words")]:
        vals = [F[k][key] for k in keys]
        bars = ax.bar(x + off, vals, w - 0.03, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004 * np.sign(v or 1), f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [f"{k.split('_')[0]}\nk={k[-1]} (n={F[k]['n']})" for k in keys])
    ax.set_ylabel("mean drop in predicted prob.")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Faithfulness: muting tagged vs random words", color=INK, fontsize=10, loc="left")
    fig.savefig(OUT / "faithfulness.png"); plt.close(fig)


def latency(R: dict):
    L = R.get("latency")
    if not L or not L.get("n"):
        return
    stages = [("vad", "VAD", C["blue"]), ("asr", "ASR (whisper-small)", C["orange"]),
              ("features", "features", C["aqua"]), ("model", "WDYMNet", C["yellow"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 2.4), gridspec_kw={"width_ratios": [3, 1]})
    left = 0.0
    for k, lab, col in stages:
        v = L["mean"][k]
        a1.barh([0], [v], left=left, color=col, height=0.5, label=f"{lab} {1000 * v:.0f} ms",
                edgecolor="#fcfcfb", linewidth=2)
        left += v
    a1.axvline(1.5, color=INK2, ls="--", lw=1)
    a1.text(1.5, 0.95, " 1.5 s target", color=INK2, fontsize=8, va="top", transform=a1.get_xaxis_transform())
    a1.set_yticks([]); a1.set_xlabel("seconds"); a1.set_xlim(0, max(1.6, left * 1.08))
    a1.set_title(f"Live latency, {L['mean']['audio_s']:.1f} s utterances (mean of {L['n']}): "
                 f"{L['mean']['total']:.2f} s total", color=INK, fontsize=9.5, loc="left")
    a1.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    a2.bar([0], [L["peak_vram_mb"] / 1024], color=C["blue"], width=0.5)
    a2.axhline(6.0, color=INK2, ls="--", lw=1); a2.text(0.3, 6.05, "6 GB", fontsize=8, color=INK2)
    a2.text(0, L["peak_vram_mb"] / 1024 + 0.1, f"{L['peak_vram_mb'] / 1024:.2f} GB", ha="center", fontsize=8.5, color=INK)
    a2.set_ylim(0, 6.6); a2.set_xticks([]); a2.set_ylabel("GB")
    a2.set_title("Peak VRAM (live)", color=INK, fontsize=9.5, loc="left")
    fig.savefig(OUT / "latency_vram.png"); plt.close(fig)


ARCH = """flowchart LR
  mic[16 kHz speech] --> vad[Silero VAD] --> asr[faster-whisper small<br/>words + timestamps]
  vad --> e2v[emotion2vec+ base<br/>50 Hz frames, 768-d]
  asr --> rob[RoBERTa-base + LoRA<br/>mean of subwords]
  asr --> pros[prosody: dur, F0, RMS, pauses]
  e2v -->|mean+max per word| fuse
  rob --> fuse
  pros --> fuse
  fuse[gated fusion d=256<br/>+ 2-layer Transformer] --> emph[Emphasis head<br/>3 classes, supervised: HPC]
  fuse --> emo[Emotion head<br/>per-word logits + MIL attention: MELD]
  fuse --> sarc[Sarcasm head<br/>per-word logits + MIL attention: MUStARD++]
  e2v -.audio-only.-> ae[audio emotion head]
  rob -.text-only.-> te[text emotion head]
  ae --> inc[incongruity = JS div / ln 2]
  te --> inc
  inc --> sarc
"""


def main():
    OUT.mkdir(exist_ok=True)
    R = json.loads(Path("results/s6_results.json").read_text(encoding="utf-8"))
    (OUT / "architecture.mmd").write_text(ARCH, encoding="utf-8")
    tables(R); confusion(R); incongruity_examples(R); faithfulness(R); latency(R)
    print("wrote:", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
