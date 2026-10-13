# WDYMNet — *What Do You Mean?*
### Word-level detection of emphasis, emotion and sarcasm in speech

WDYMNet listens to live microphone speech and returns a **tone-tagged transcript**: every word gets an
emphasis level, an emotion tag, a sarcasm tag and an **incongruity** score (how much the voice disagrees with the words).

```
I <sarcasm p=.81><emph lvl=2>love</emph></sarcasm> waiting <anger p=.66>in queues</anger>
```

**Full project write-up** (problem, design, results, decisions, limitations — for faculty and teammates):
[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)

---

## Quick start (Windows or Linux, Python 3.12, NVIDIA GPU ≥ 6 GB)

```bash
python -m venv .venv
.venv\Scripts\activate                     # Linux: source .venv/bin/activate
pip install -r requirements.txt            # CUDA 12.6 wheels; ffmpeg must be on PATH (winget install Gyan.FFmpeg)

python scripts/s0_env_check.py             # S0  GPU / ffmpeg / load all models, peak VRAM
python scripts/s1_prepare_meld.py          # S1  data (~4 GB download in total, all < 5 GB each)
python scripts/s1_prepare_mustardpp.py
python scripts/s1_prepare_hpc.py
python scripts/s2_align_and_cache.py       # S2  forced alignment + feature cache (~1.5 h, resume-safe)
python scripts/s3_train.py                 # S3  Stage A + baselines B1/B2 (+ B0 zero-shot)
python -m wdymnet.live.app                 # S4  live demo → http://127.0.0.1:7860
python scripts/s5_finetune.py              # S5  Stage B fine-tuning
python scripts/s6_evaluate.py              # S6  sanity table, conflict, faithfulness, ablations, latency
python scripts/s8_make_figures.py          # S8  tables + figures → paper_artifacts/
python -m pytest                           # tests
```

**Just the demo:** the trained model (`results/checkpoints/stage_a_best.pt`, 11 MB) and its display thresholds
are in the repo, so after `pip install -r requirements.txt` you can run `python -m wdymnet.live.app` directly —
no datasets needed. The speech/text encoders and the rewriter LLM (~2.7 GB) download on first launch.

Every number in `results/` is produced by exactly one of these scripts. Seed 1337 throughout; configs in `configs/`.

## The demo

* **Push-to-talk** — record (or upload) a sentence; it's tagged as soon as you stop.
* **Continuous** — listens on the default mic (sounddevice, 16 kHz), Silero VAD cuts phrases at pauses.
* Words are **coloured by emotion**, **bold by emphasis**, **dotted-underlined for sarcasm** (only the words the model relied on); hover a word for all
  probabilities, the incongruity bar and "words say X · voice says Y".
* **What you meant**: below the tagged transcript, a pretrained instruction-tuned LLM (Qwen3-4B-Instruct-2507,
  4-bit AWQ, used as-is, no fine-tuning) rewrites each utterance from the ASR words plus the voice cues the tagger
  found (overall tone, sarcasm and the words it sits on, emphasised words, confident word-level emotions, words
  whisper was unsure about). It first lists words that make no sense in context, then returns a **corrected**
  sentence (sound-alikes and contradictions fixed, e.g. "I went for a walk tomorrow" → "today"; punctuation that
  matches the tone) and one line on the **intended meaning** (for sarcasm, the meaning behind it). On a 10-sentence
  check of grammatical-but-wrong ASR errors (`scripts/check_rewriter.py`) it fixes 9/10; Qwen2.5-1.5B and
  Qwen3-1.7B fixed 4/10. It runs on
  CTranslate2 (~2.7 GB VRAM, ~4–5 s per sentence, shown after the tags); the AWQ checkpoint is converted once by
  `wdymnet/ct2_awq.py` (no gptqmodel needed). See `wdymnet/rewrite.py`; switch it off with `rewrite.enabled: false` in `configs/live.yaml`.

## Method (matches the code)

Per utterance with *N* words (`wdymnet/models/fusion.py`):

| Stream | How a word vector is built |
|---|---|
| audio | emotion2vec+ base frames (50 Hz, 768-d) inside `[start, end)` → concat(mean, max) = 1536-d. Frame index = ⌊t·50⌋; words shorter than one frame take the nearest frame (`features/speech.py`). |
| text | RoBERTa-base, mean of the word's sub-word states via `word_ids()` (`features/text.py`). |
| prosody | duration, F0 mean/max/slope (pyin 65–400 Hz, voiced frames only), RMS mean/max, pause before/after; z-normalised within the utterance (`features/prosody.py`). |

* Each stream is projected to d = 256; **gated fusion** `g = σ(W[a; t; p])`, `h = g_a·a + g_t·t + g_p·p`;
  sinusoidal positions; 2-layer Transformer encoder (4 heads, ff 512, dropout 0.1).
* **Emphasis**: per-word linear → 3 classes (CE, punctuation ignored). Supervised by HPC.
* **Emotion / sarcasm (multiple-instance learning)**: per-word logits; attention pooling
  `a = softmax(w·tanh(V h))`, utterance logits = Σ aᵢ·logitᵢ, trained on utterance labels only.
  At inference the per-word logits are the word tags (shown above a threshold tuned on validation).
  Sparsity loss = normalised entropy of the attention.
* **Unimodal auxiliary heads**: per-word emotion from the audio-only and text-only streams, pooled the same way.
* **Incongruity**ᵢ = JS(text-only emotion distᵢ ‖ audio-only emotion distᵢ) / ln 2 ∈ [0, 1]; it is also an input
  feature of the sarcasm head.
* **Loss** = CE_emph + CE_emo (class-weighted, MELD weights) + CE_sarc + 0.3·(CE_text-only + CE_audio-only) + 0.01·sparsity.
* **Training**: each batch comes from one dataset (round-robin MELD / MUStARD++ / HPC); heads without labels
  are skipped. MUStARD++ (825 utterances, ~10× smaller than MELD) is repeated 4× per epoch; without this the
  sarcasm head received ~6% of the updates and stayed at chance. Modality dropout: text p = 0.3, audio p = 0.3,
  never both. Model selection uses the mean of emphasis macro-F1, emotion macro-F1 and sarcasm **AUC**
  (threshold-free, since the sarcasm threshold is calibrated separately).
* **Display rules (demo)**: a word is coloured with its strongest non-neutral emotion only if that emotion beats
  the word's own neutral probability and the tuned threshold. Sarcasm is shown only when the utterance
  probability passes a threshold calibrated for ≤ 10% false alarms on non-sarcastic validation speech
  (MUStARD++ non-sarcastic, MELD, read speech), and then only on the ≤ 2 words with clearly above-uniform
  sarcasm attention.
  * **Stage A** — frozen encoders, cached features, fp32; AdamW 1e-3, batch 32, ≤ 15 epochs, cosine + 5% warm-up,
    early stopping on the mean validation score (patience 3).
  * **Stage B** — the top 3 of emotion2vec's 8 transformer blocks are trained (lr 2e-5) from cached hidden
    states entering block 6; RoBERTa gets LoRA r=8, α=16, dropout 0.1 on query/value (lr 2e-4); heads lr 5e-4;
    batch 8 × accumulation 4, fp16, 3 epochs, utterances ≤ 10 s. Kept only if it beats Stage A on validation.
* **Word timestamps**: gold transcripts are force-aligned with torchaudio MMS_FA (fallback: faster-whisper words
  + difflib, ≥ 80% match); utterances with mean alignment score < 0.1 are dropped. Live mode uses faster-whisper
  small (int8_float16, greedy) word timestamps after Silero VAD gating.

## Datasets

| Dataset | Used for | Our split | Notes |
|---|---|---|---|
| [MELD](https://github.com/declare-lab/MELD) | utterance emotion (7 classes) | official train / dev / test | audio from the 16 kHz audio-only mirror [`ajyy/MELD_audio`](https://huggingface.co/datasets/ajyy/MELD_audio) (1.5 GB, extracted from the official MP4s; the official 10.9 GB video tarball is not needed). 17 clips missing/unreadable → `data/MELD/meld_skipped.txt`. |
| [MUStARD++](https://github.com/cfiltnlp/MUStARD_Plus_Plus) | utterance sarcasm | **speaker-grouped** 70/15/15 (840 / 180 / 181), no speaker in two splits | main-utterance video only; 1,201 utterances (601 sarcastic). Greedy grouping by utterance count leaves the test split less sarcastic (56/181). |
| [Helsinki Prosody Corpus](https://github.com/Helsinki-NLP/prosody) + [LibriTTS](https://www.openslr.org/60/) | word prominence 0/1/2 | HPC dev → train (36 speakers) / val (4 held-out speakers); HPC test → test | all 10,549 sentences matched to LibriTTS dev-clean / test-clean wavs; punctuation (NA) rows dropped. |

Licences and terms: [DATA_LICENSES.md](DATA_LICENSES.md). No audio, video or features are committed (`data/` is gitignored).

## Results

Sanity checks, not a comparative study. Produced by `scripts/s6_evaluate.py` → `results/s6_results.json`,
`results/main_table.csv`; figures and LaTeX by `scripts/s8_make_figures.py` → `paper_artifacts/`.
Speech encoder: emotion2vec+ base. Test sets are the clips that passed S2 alignment
(HPC 4,817/4,822 sentences, MELD 2,264/2,610, MUStARD++ 169/181).

| Model | Emph acc 3-way | Emph acc 2-way | Emph F1 3-way | Emotion wF1 (MELD) | Sarcasm macro-F1 | Sarcasm AUC |
|---|---|---|---|---|---|---|
| B0 emotion2vec+ zero-shot | – | – | – | 46.2 | – | – |
| B1 utterance fusion | – | – | – | **49.3** | 57.0 | 63.3 |
| B2 Emo-FiLM-style word tagger | – | – | – | 43.7 | – | – |
| **ours, Stage A** (deployed) | **72.1** | **84.3** | **69.0** | 46.7 | 56.6 | 65.3 |
| ours, Stage B (not selected on val) | 71.2 | 83.1 | 67.5 | 46.7 | **60.8** | **68.5** |
| ours − prosody | 59.9 | 77.2 | 50.3 | 43.8 | 57.0 | 64.8 |
| ours − incongruity | 64.1 | 78.7 | 62.5 | 45.0 | 52.8 | 54.3 |
| ours − modality dropout | 64.8 | 81.0 | 62.3 | 46.1 | 52.6 | 64.7 |

(Ablations are Stage A retrains. HPC reference: a text-only BERT trained on train-360 reaches 68.6 / 83.2; we train
on the much smaller HPC dev set but also use audio. MUStARD++ is speaker-independent: 169 test clips, 54 sarcastic.)

* **What helps**: prosody is the most important input for emphasis (−12 points 3-way accuracy without it);
  the incongruity feature is the most important input for sarcasm (AUC 65.3 → 54.3 without it); modality
  dropout helps emphasis and sarcasm.
* **Fine-tuning vs pretrained-only**: our Stage A beats the pretrained-only emotion2vec head (B0) on emotion
  (46.7 vs 46.2, marginal). Stage B (partial fine-tuning + LoRA) improves sarcasm on test (AUC 68.5) but not
  emphasis or emotion, and it did not beat Stage A on validation, so Stage A is the deployed model. The
  utterance-only baseline B1 is best on emotion: word-level structure costs a little utterance accuracy.
* **Conflict** (Stage A, 54 sarcastic test clips; share predicted sarcastic at 0.5): text-only 63.0%,
  audio-only 70.4%, fused 63.0%. Fusion does not yet beat the better single modality.
* **Sarcasm in the demo** is calibrated, not taken at 0.5: MUStARD++ is 50% sarcastic but ordinary speech is not.
  The threshold (0.981) gives ≤ 10% false alarms on non-sarcastic validation speech. On test this means
  15% recall, with 7% false alarms on non-sarcastic MUStARD++ and 14% on read speech. When the demo marks
  sarcasm it is a hint, not a verdict.
* **Faithfulness** (mean drop in the predicted-class probability after zeroing the audio of k words):

  | | tagged k=1 | random k=1 | tagged k=2 | random k=2 |
  |---|---|---|---|---|
  | emotion (n=67 / 61) | 0.121 | 0.110 | 0.176 | 0.171 |
  | sarcasm (n=8) | 0.000 | 0.070 | **0.196** | 0.047 |

  Emotion word tags are **not** more faithful than random words (tagged > random in only 49% of clips): the
  utterance emotion decision is spread across the sentence. Sarcasm (k=2) looks faithful, but only 8 test clips
  pass the calibrated threshold, too few to conclude anything.
* **Live**: 0.75 s mean end-to-end on 3.8 s utterances (VAD 118 ms, ASR 390 ms, features 227 ms, model 15 ms; the rewrite follows ~4–5 s later),
  peak 1.1 GB VRAM. Training: Stage A 8 min / 0.19 GB, Stage B 37 min / 4.7 GB.

**Training fixes that changed these numbers** (found while debugging "everything looks sarcastic"):
(1) MUStARD++ was ~6% of batches, so the sarcasm head never trained, now oversampled 4×; (2) the attention-entropy
loss divided by log(1) = 0 on one-word utterances, giving huge gradients, now fixed; (3) fp16 training of the small
fusion model overflowed to NaN after a few epochs, so Stage A now trains in fp32 (Stage B keeps fp16 + GradScaler).

## Limitations

* **Domain gap**: MELD and MUStARD++ are sitcom audio (laugh tracks, overlapping speakers, music); live speech
  into a laptop mic is very different. MELD clips often contain other speakers' speech, which also hurts forced alignment.
* **Weak word labels**: emotion and sarcasm word tags come from multiple-instance learning on utterance labels;
  there is no gold per-word emotion/sarcasm annotation, so word-level claims rest on the faithfulness test.
* **Automatic emphasis labels**: HPC prominence labels are produced automatically from acoustic features (Wavelet
  Prosody Toolkit), not by human listeners; we also train on HPC *dev* only (≈ 87k words), not train-360.
* **Small sarcasm data**: 840 training utterances from a handful of speakers, no dialogue context; speaker-disjoint
  test results are noisy and sarcasm detection is weak (AUC ≈ 0.65). A linear probe on the text features alone
  reaches AUC ≈ 0.71, so a dedicated sarcasm head could do better.
* **Word-level emotion tags are not yet faithful**: muting the tagged words hurts the prediction no more than
  muting random words.
* **The rewriter is only spot-checked** (10 hand-written sentences): the "what you meant" output comes from an
  off-the-shelf 4B LLM prompted with the cues; it inherits the tagger's errors (a missed or false sarcasm cue changes the stated meaning) and is a
  demo feature, not a measured result.

## Related work (prior work, not ours)

* **ConAttSD** (Zhang et al., 2021) — utterance-level text-vs-audio incongruity for sarcasm detection.
* **Emo-FiLM** (2025) — word-level emotion from emotion2vec frames trained with utterance pseudo-labels (our baseline B2 follows this recipe).
* **Text + speech fusion for speech emotion diarization** (Interspeech 2025).
* **ProSarc** (2026) — audio-only sarcasm onset localisation.
* **Helsinki Prosody Corpus** (Talman et al., 2019) — prominence prediction from text; reference text-only BERT on
  train-360: 83.2% (2-way) / 68.6% (3-way) test accuracy.
* **Speech emotion diarization + ZED** (Wang et al., 2023).

**What this project adds**: (1) one word-level model that jointly tags emphasis (supervised) and emotion + sarcasm
(weakly supervised via MIL) from text + audio; (2) an interpretable per-word incongruity score; (3) a live demo
that turns the tagged words into a corrected sentence and its intended meaning; (4) a comparison of fine-tuning against the pretrained-only model.

## Repository layout

```
configs/      data.yaml, train_a.yaml, train_b.yaml, live.yaml
scripts/      s0_env_check · s1_prepare_{meld,mustardpp,hpc} · s2_align_and_cache · s3_train · s5_finetune
              s6_evaluate · s8_make_figures
wdymnet/      data/ align/ features/{speech,text,prosody} models/{fusion,baselines} losses train finetune pipeline
              eval/{metrics,faithfulness} live/{app,stream,render} rewrite ct2_awq utils/
tests/        pytest suite
results/      metrics JSON/CSV + checkpoints/stage_a_best.pt (demo model; others gitignored)
paper_artifacts/  tables + figures
```

Code: MIT.
