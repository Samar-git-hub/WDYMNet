"""
Word alignment — gold transcript → word start/end times.

Primary: torchaudio MMS_FA forced alignment (model loaded once).
Fallback: faster-whisper word timestamps + difflib alignment to the gold words
(drop the utterance if < 80% of gold words match).
Live mode uses faster-whisper words directly (see whisper_words).
"""
import difflib
import re

import numpy as np
import torch

SR = 16000


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


class MMSAligner:
    """CTC forced alignment with torchaudio's MMS_FA bundle."""

    def __init__(self, device: str = "cuda"):
        import torchaudio
        self.bundle = torchaudio.pipelines.MMS_FA
        self.model = self.bundle.get_model(with_star=True).to(device).eval()
        self.tokenizer = self.bundle.get_tokenizer()
        self.aligner = self.bundle.get_aligner()
        self.device = device

    @staticmethod
    def _mms_word(w: str) -> str:
        """MMS dictionary is a-z + apostrophe; anything else (digits, symbols) → star token."""
        w = re.sub(r"[^a-z']", "", w.lower()).strip("'")
        return w or "*"

    @torch.inference_mode()
    def align(self, audio: np.ndarray, words: list[str]) -> tuple[list[dict], float] | None:
        """→ ([{w, start, end, score}], mean word score) or None if alignment is impossible."""
        if not words:
            return None
        wav = torch.from_numpy(audio).float().to(self.device)[None]
        emission, _ = self.model(wav)
        tokens = self.tokenizer([self._mms_word(w) for w in words])
        try:
            spans = self.aligner(emission[0], tokens)
        except Exception:  # e.g. audio too short for the token sequence
            return None
        sec_per_frame = wav.shape[1] / emission.shape[1] / SR
        out = []
        for w, sp in zip(words, spans):
            score = float(sum(s.score * (s.end - s.start) for s in sp) / max(sum(s.end - s.start for s in sp), 1))
            out.append({"w": w, "start": round(sp[0].start * sec_per_frame, 3),
                        "end": round(sp[-1].end * sec_per_frame, 3), "score": round(score, 3)})
        return out, float(np.mean([o["score"] for o in out]))


def align_words_to_gold(asr_words: list[dict], gold_words: list[str], min_match_ratio: float = 0.80) -> list[dict] | None:
    """Map ASR word timestamps onto gold words via difflib; unmatched gold words are
    interpolated between their matched neighbours. None if < min_match_ratio of gold words match."""
    if not asr_words or not gold_words:
        return None
    a = [normalize_word(w["word"]) for w in asr_words]
    g = [normalize_word(w) for w in gold_words]
    g2a = {}
    for blk in difflib.SequenceMatcher(None, a, g, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            g2a[blk.b + k] = blk.a + k
    if len(g2a) / len(gold_words) < min_match_ratio:
        return None

    out = [None] * len(gold_words)
    for gi, ai in g2a.items():
        out[gi] = {"w": gold_words[gi], "start": asr_words[ai]["start"], "end": asr_words[ai]["end"]}
    i = 0
    while i < len(out):  # fill runs of unmatched words by splitting the gap evenly
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] is None:
            j += 1
        lo = out[i - 1]["end"] if i > 0 else asr_words[0]["start"]
        hi = out[j]["start"] if j < len(out) else asr_words[-1]["end"]
        hi = max(hi, lo)
        step = (hi - lo) / (j - i)
        for k in range(i, j):
            out[k] = {"w": gold_words[k], "start": lo + (k - i) * step, "end": lo + (k - i + 1) * step}
        i = j
    return out


def whisper_words(model, audio: np.ndarray, beam_size: int = 5) -> list[dict]:
    """faster-whisper → [{word, start, end, probability}] (empty list if nothing recognised)."""
    segments, _ = model.transcribe(audio, word_timestamps=True, vad_filter=False, language="en",
                                   beam_size=beam_size, condition_on_previous_text=False)
    words = []
    for seg in segments:
        for w in seg.words or []:
            if w.word.strip():
                words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end),
                              "probability": float(w.probability)})
    return fix_overlaps(words)


def fix_overlaps(words: list[dict]) -> list[dict]:
    """Make timestamps monotonic and non-overlapping (whisper occasionally overlaps words)."""
    prev_end = 0.0
    for w in words:
        w["start"] = max(w["start"], prev_end)
        w["end"] = max(w["end"], w["start"])
        prev_end = w["end"]
    return words
