"""
Speech encoder: emotion2vec+ base (FunASR), frame features at 50 Hz, dim 768.

We wrap the underlying torch module (8 transformer blocks) so we can split it:
  bottom = CNN feature extractor + prenet + blocks[0:5]   (frozen, output cached in S2)
  top    = blocks[5:8]                                     (fine-tuned in Stage B)
`encode()` reproduces funasr's `generate(granularity="frame")` exactly (verified: max diff 0.0)
while also returning the hidden state entering the top-3 blocks and the 9-class scores.

Gotchas (from the emotion2vec docs): pass true-length 16 kHz mono float32 audio with no
padding, and never pure silence (it is classified "surprised") — callers VAD-gate first.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SPEECH_FRAME_DIM = 768
SPEECH_WORD_DIM = 768 * 2  # mean + max pooling
FRAME_RATE = 50.0
N_TOP_BLOCKS = 3
# emotion2vec 9-class order: 0 angry, 1 disgusted, 2 fearful, 3 happy, 4 neutral, 5 other, 6 sad, 7 surprised, 8 unknown
E2V_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]
ENCODER_NAME = "emotion2vec_plus_base"


class Emotion2vecEncoder(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        from funasr import AutoModel
        try:
            am = AutoModel(model="iic/emotion2vec_plus_base", hub="hf", disable_update=True, device=device)
        except Exception as e:
            print(f"[SpeechEnc] hf hub failed ({e}); trying modelscope")
            am = AutoModel(model="iic/emotion2vec_plus_base", hub="ms", disable_update=True, device=device)
        self.net = am.model.eval()
        self.fe = self.net.modality_encoders["AUDIO"]
        self.n_extra = self.fe.modality_cfg.num_extra_tokens
        self.split = len(self.net.blocks) - N_TOP_BLOCKS
        self.device = device
        self.requires_grad_(False)

    # ---- pieces ----
    def _alibi(self, T: int, B: int, dtype, device) -> torch.Tensor:
        """ALiBi bias for T frames (+ extra tokens), identical to the feature extractor's."""
        ab = self.fe.get_alibi_bias(batch_size=B, time_steps=T, heads=self.fe.modality_cfg.num_alibi_heads,
                                    dtype=torch.float32, device=device)
        if self.fe.alibi_scale is not None:
            ab = ab * self.fe.alibi_scale.clamp_min(0).squeeze(0).type_as(ab)
        return F.pad(ab, (self.n_extra, 0, self.n_extra, 0)).to(dtype)

    @torch.no_grad()
    def bottom(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [S] → hidden entering the top blocks [T + n_extra, 768]."""
        src = F.layer_norm(wav, wav.shape).view(1, -1)
        out = self.fe(src, None, False, remove_masked=False, clone_batch=1, mask_seeds=None, precomputed_mask=None)
        x, ab = out["x"], out["alibi_bias"]
        for blk in self.net.blocks[: self.split]:
            x, _ = blk(x, padding_mask=None, alibi_bias=ab)
        return x[0]

    def top(self, mid: torch.Tensor, n_frames: torch.Tensor | None = None) -> torch.Tensor:
        """mid [B, T + n_extra, 768] (zero-padded) → frames [B, T, 768]. Differentiable."""
        B, L, _ = mid.shape
        T = L - self.n_extra
        pad = None
        if n_frames is not None:
            idx = torch.arange(T, device=mid.device)[None]
            pad = F.pad(idx >= n_frames[:, None], (self.n_extra, 0), value=False)  # True = padding
        x = mid
        ab = self._alibi(T, B, mid.dtype, mid.device)
        for blk in self.net.blocks[self.split:]:
            x, _ = blk(x, padding_mask=pad, alibi_bias=ab)
        if self.net.norm is not None:
            x = self.net.norm(x)
        return x[:, self.n_extra:]

    def scores(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [T, 768] → 9-class softmax (emotion2vec's own head; used for baseline B0)."""
        return torch.softmax(self.net.proj(frames.mean(0, keepdim=True)).float(), dim=-1)[0]

    @torch.no_grad()
    def encode(self, audio: np.ndarray) -> dict:
        """audio: 1-D float32 16 kHz, true length. → {frames [T,768], mid [T+extra,768], scores [9]}"""
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(self.device)
        mid = self.bottom(wav)
        frames = self.top(mid[None])[0]
        return {"frames": frames.float().cpu(), "mid": mid.half().cpu(), "scores": self.scores(frames).cpu()}


def time_to_frame(t: float, frame_rate: float = FRAME_RATE) -> int:
    """frame index = floor(t * frame_rate)"""
    return int(np.floor(t * frame_rate + 1e-9))


def word_frame_spans(word_times: list[dict], n_frames: int, frame_rate: float = FRAME_RATE) -> list[tuple[int, int]]:
    """[start, end) frame span per word, clipped to the clip; sub-frame words get their nearest frame."""
    spans = []
    for wt in word_times:
        s = min(time_to_frame(wt["start"], frame_rate), n_frames - 1)
        e = min(time_to_frame(wt["end"], frame_rate), n_frames)
        if e <= s:  # shorter than one frame (or past the end) → nearest frame
            s = min(max(int(round((wt["start"] + wt["end"]) / 2 * frame_rate - 0.5)), 0), n_frames - 1)
            e = s + 1
        spans.append((s, e))
    return spans


def pool_frames_to_words(frames, word_times: list[dict], frame_rate: float = FRAME_RATE):
    """frames [T, D] (np or torch) → [N, 2D] = concat(mean, max) over each word's frames."""
    is_np = isinstance(frames, np.ndarray)
    fr = torch.from_numpy(frames) if is_np else frames
    D = fr.shape[-1]
    if len(word_times) == 0 or fr.shape[0] == 0:
        out = fr.new_zeros((len(word_times), 2 * D))
    else:
        out = torch.stack([torch.cat([fr[s:e].mean(0), fr[s:e].amax(0)])
                           for s, e in word_frame_spans(word_times, fr.shape[0], frame_rate)])
    return out.numpy().astype(np.float32) if is_np else out
