"""
End-to-end inference: waveform → VAD gate → faster-whisper words → features → WDYMNet → per-word tags.
Used by the live demo and the faithfulness test.
"""
import time
from pathlib import Path

import numpy as np
import torch

from wdymnet.align.aligner import fix_overlaps, whisper_words
from wdymnet.features.prosody import extract_word_prosody
from wdymnet.features.speech import Emotion2vecEncoder, pool_frames_to_words
from wdymnet.features.text import TextEncoder
from wdymnet.train import load_checkpoint

SR = 16000
EMOTIONS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]


def to_mono16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Any dtype / channels / rate → float32 mono 16 kHz in [-1, 1] (handles clipped/int input)."""
    a = np.asarray(audio)
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / np.iinfo(a.dtype).max
    a = a.astype(np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1 if a.shape[1] < a.shape[0] else 0)
    if sr != SR:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    peak = np.abs(a).max() if a.size else 0.0
    if peak > 1.0:
        a = a / peak
    return np.clip(a, -1.0, 1.0)


def speech_segments(vad_model, audio: np.ndarray, threshold=0.5, min_speech_ms=250, min_silence_ms=300) -> list[dict]:
    """Silero VAD → [{start, end}] in samples (empty on silence)."""
    from silero_vad import get_speech_timestamps
    if len(audio) < 512:
        return []
    return get_speech_timestamps(torch.from_numpy(audio), vad_model, sampling_rate=SR, threshold=threshold,
                                 min_speech_duration_ms=min_speech_ms, min_silence_duration_ms=min_silence_ms)


class Tagger:
    def __init__(self, checkpoint: str | Path, device: str = "cuda", whisper_size: str = "small",
                 compute_type: str = "int8_float16", vad_cfg: dict | None = None, beam_size: int = 1):
        from faster_whisper import WhisperModel
        from silero_vad import load_silero_vad
        self.device = torch.device(device)
        self.vad_cfg = vad_cfg or {}
        self.beam_size = beam_size  # greedy decoding keeps live ASR latency low
        self.vad = load_silero_vad()
        self.whisper = WhisperModel(whisper_size, device=device, compute_type=compute_type)
        self.speech = Emotion2vecEncoder(device)
        self.text = TextEncoder(device=device)
        self.model, self.kind, ck = load_checkpoint(checkpoint, self.device)
        self.stage = "B" if "speech_top" in ck else "A"
        if "speech_top" in ck:  # Stage B: fine-tuned top-3 emotion2vec blocks + RoBERTa LoRA
            self.speech.net.blocks[self.speech.split:].load_state_dict(ck["speech_top"])
            lc = ck["config"]["model"]["finetune"]["lora"]
            self.text.add_lora(lc["r"], lc["alpha"], lc["dropout"], lc["target_modules"])
            self.text.model.load_state_dict(ck["lora"], strict=False)
            self.text.model.eval()
        self.checkpoint = str(checkpoint)

    def gate(self, audio: np.ndarray) -> np.ndarray | None:
        """Crop to the voiced region (±0.2 s); None if there is no speech at all."""
        segs = speech_segments(self.vad, audio, self.vad_cfg.get("threshold", 0.5),
                               self.vad_cfg.get("min_speech_duration_ms", 250),
                               self.vad_cfg.get("min_silence_duration_ms", 300))
        if not segs:
            return None
        pad = int(0.2 * SR)
        return audio[max(segs[0]["start"] - pad, 0): segs[-1]["end"] + pad]

    @torch.no_grad()
    def features(self, audio: np.ndarray, words: list[dict]) -> dict:
        enc = self.speech.encode(audio)
        a = pool_frames_to_words(enc["frames"], words)
        t = torch.from_numpy(self.text.extract([w["w"] for w in words]))
        p = torch.from_numpy(extract_word_prosody(audio, SR, words))
        return {"audio": a[None], "text": t[None], "pros": p[None], "e2v_scores": enc["scores"]}

    @torch.no_grad()
    def run_model(self, feats: dict) -> dict:
        n = feats["audio"].shape[1]
        b = {"audio": feats["audio"].float().to(self.device), "text": feats["text"].float().to(self.device),
             "pros": feats["pros"].float().to(self.device),
             "mask": torch.ones(1, n, dtype=torch.bool, device=self.device)}
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            out = self.model(b["audio"], b["text"], b["pros"], b["mask"])
        sm = lambda k: torch.softmax(out[k][0].float(), -1).cpu().numpy()
        return {"emph": sm("emph_logits"), "emo": sm("emo_logits"), "sarc": sm("sarc_logits")[:, 1],
                "audio_emo": sm("audio_emo_logits"), "text_emo": sm("text_emo_logits"),
                "incongruity": out["incongruity"][0].float().cpu().numpy(),
                "emo_utt": sm("emo_utt_logits"), "sarc_utt": float(sm("sarc_utt_logits")[1]),
                "emo_attn": out["emo_attn"][0].float().cpu().numpy(),
                "sarc_attn": out["sarc_attn"][0].float().cpu().numpy()}

    def tag(self, audio: np.ndarray, sr: int = SR, words: list[dict] | None = None) -> dict:
        """→ {words:[{w,start,end,emph,emph_p,emo,emo_p,emo_probs,sarc_p,incongruity,...}], utterance:{...}, timing:{...}}
        Pass `words` (with start/end) to skip ASR (e.g. gold-aligned clips)."""
        t0 = time.perf_counter()
        timing = {}
        audio = to_mono16k(audio, sr)
        offset = 0.0
        if words is None:
            gated = self.gate(audio)
            timing["vad"] = time.perf_counter() - t0
            if gated is None:
                return {"words": [], "utterance": None, "timing": timing, "status": "no speech detected"}
            audio = gated
            t1 = time.perf_counter()
            asr = whisper_words(self.whisper, audio, self.beam_size)
            timing["asr"] = time.perf_counter() - t1
            words = [{"w": w["word"], "start": w["start"], "end": w["end"], "asr_p": w["probability"]} for w in asr]
            words = [w for w in words if any(c.isalnum() for c in w["w"])]
            if not words:
                return {"words": [], "utterance": None, "timing": timing, "status": "no words recognised"}
        words = fix_overlaps([dict(w) for w in words])
        t2 = time.perf_counter()
        feats = self.features(audio, words)
        timing["features"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        o = self.run_model(feats)
        timing["model"] = time.perf_counter() - t3
        timing["total"] = time.perf_counter() - t0
        timing["audio_s"] = len(audio) / SR

        out_words = []
        for i, w in enumerate(words):
            emo_probs = o["emo"][i]
            nn_idx = int(np.argmax(emo_probs[1:]) + 1)  # strongest non-neutral emotion
            out_words.append({
                **w, "start": round(w["start"] + offset, 3), "end": round(w["end"] + offset, 3),
                "emph": int(o["emph"][i].argmax()), "emph_probs": o["emph"][i].round(3).tolist(),
                "emo": EMOTIONS[nn_idx], "emo_p": float(emo_probs[nn_idx]),
                "emo_probs": dict(zip(EMOTIONS, emo_probs.round(3).tolist())),
                "audio_emo_probs": dict(zip(EMOTIONS, o["audio_emo"][i].round(3).tolist())),
                "text_emo_probs": dict(zip(EMOTIONS, o["text_emo"][i].round(3).tolist())),
                "sarc_p": float(o["sarc"][i]), "incongruity": float(o["incongruity"][i]),
                "emo_attn": float(o["emo_attn"][i]), "sarc_attn": float(o["sarc_attn"][i]),
            })
        utt = {"emotion": EMOTIONS[int(o["emo_utt"].argmax())],
               "emo_probs": dict(zip(EMOTIONS, o["emo_utt"].round(3).tolist())),
               "sarcasm_p": o["sarc_utt"]}
        return {"words": out_words, "utterance": utt, "timing": timing, "status": "ok"}
