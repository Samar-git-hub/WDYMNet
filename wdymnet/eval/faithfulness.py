"""
Faithfulness: zero the audio of the top-k tagged words vs k random words, re-run the model
with the same word timestamps, and compare the drop in the predicted class probability.
If the word tags are faithful, muting tagged words should hurt more than muting random ones.
"""
import numpy as np


def mute_words(audio: np.ndarray, sr: int, words: list[dict], idx: list[int]) -> np.ndarray:
    """Copy of `audio` with the [start, end) span of each selected word set to zero."""
    out = audio.copy()
    for i in idx:
        s = max(0, min(int(words[i]["start"] * sr), len(out)))
        e = max(s, min(int(np.ceil(words[i]["end"] * sr)), len(out)))
        out[s:e] = 0.0
    return out


def faithfulness_trial(predict_fn, audio: np.ndarray, sr: int, words: list[dict], word_scores: np.ndarray,
                       k: int, rng: np.random.Generator, n_random: int = 3) -> dict | None:
    """predict_fn(audio, words) → probability of the originally-predicted class.
    Returns drops for top-k tagged vs random-k words (None if the clip has ≤ k words)."""
    n = len(words)
    if n <= k:
        return None
    p0 = predict_fn(audio, words)
    top = list(np.argsort(-word_scores)[:k])
    tagged_drop = p0 - predict_fn(mute_words(audio, sr, words, top), words)
    rand_drops = [p0 - predict_fn(mute_words(audio, sr, words, list(rng.choice(n, k, replace=False))), words)
                  for _ in range(n_random)]
    return {"p0": float(p0), "tagged_drop": float(tagged_drop), "random_drop": float(np.mean(rand_drops))}
