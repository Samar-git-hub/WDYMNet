"""
Hand-crafted per-word prosody (librosa).
Columns: duration, F0 mean, F0 max, F0 slope (pyin, voiced frames only), RMS mean, RMS max,
pause before, pause after. NaN-safe; z-normalized within the utterance.
"""
import warnings

import numpy as np

PROSODY_DIM = 8
HOP = 320  # 20 ms at 16 kHz
FRAME = 1024


def extract_word_prosody(audio: np.ndarray, sr: int, word_times: list[dict],
                         f0_fmin: float = 65.0, f0_fmax: float = 400.0) -> np.ndarray:
    import librosa

    n = len(word_times)
    if n == 0:
        return np.zeros((0, PROSODY_DIM), dtype=np.float32)
    audio = audio.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, _, _ = librosa.pyin(audio, fmin=f0_fmin, fmax=f0_fmax, sr=sr, frame_length=FRAME, hop_length=HOP)
    rms = librosa.feature.rms(y=audio, frame_length=FRAME, hop_length=HOP)[0]
    dur_total = len(audio) / sr

    feats = np.zeros((n, PROSODY_DIM), dtype=np.float32)
    for i, wt in enumerate(word_times):
        s, e = wt["start"], wt["end"]
        fs = min(int(s * sr / HOP), len(rms) - 1)
        fe = max(min(int(np.ceil(e * sr / HOP)), len(rms)), fs + 1)
        seg = f0[fs:fe]
        voiced = seg[np.isfinite(seg)]
        f0_mean = f0_max = slope = 0.0
        if len(voiced):
            f0_mean, f0_max = float(voiced.mean()), float(voiced.max())
            if len(voiced) > 1:
                slope = float(np.polyfit(np.arange(len(voiced)) * HOP / sr, voiced, 1)[0])
        r = rms[fs:fe]
        pause_before = s - word_times[i - 1]["end"] if i > 0 else s
        pause_after = word_times[i + 1]["start"] - e if i < n - 1 else dur_total - e
        feats[i] = [e - s, f0_mean, f0_max, slope, r.mean(), r.max(), max(pause_before, 0.0), max(pause_after, 0.0)]

    feats = np.nan_to_num(feats)
    if n > 1:
        std = feats.std(0, keepdims=True)
        std[std < 1e-8] = 1.0
        feats = (feats - feats.mean(0, keepdims=True)) / std
    else:
        feats = np.zeros_like(feats)  # a single word has no within-utterance contrast
    return feats.astype(np.float32)
