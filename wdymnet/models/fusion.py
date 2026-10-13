"""
WDYMNet Fusion Model — gated multimodal fusion with per-word task heads.

Architecture:
  Per word: project audio/text/prosody → gated fusion → Transformer encoder
  Heads: Emphasis (per-word CE), Emotion + Sarcasm (MIL attention pooling),
         Unimodal auxiliary heads, Incongruity (JS divergence).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GatedFusion(nn.Module):
    """Gated combination of audio, text, and prosody features."""

    def __init__(self, audio_dim: int, text_dim: int, prosody_dim: int, out_dim: int):
        super().__init__()
        self.proj_audio = nn.Linear(audio_dim, out_dim)
        self.proj_text = nn.Linear(text_dim, out_dim)
        self.proj_pros = nn.Linear(prosody_dim, out_dim)

        # Gate takes concatenated projected features
        self.gate = nn.Linear(out_dim * 3, 3)

    def forward(self, audio: torch.Tensor, text: torch.Tensor, prosody: torch.Tensor,
                drop_audio: bool = False, drop_text: bool = False) -> torch.Tensor:
        """
        Args:
            audio: [B, N, audio_dim]
            text: [B, N, text_dim]
            prosody: [B, N, prosody_dim]
            drop_audio: Zero out audio stream (modality dropout).
            drop_text: Zero out text stream (modality dropout).

        Returns:
            fused: [B, N, out_dim]
        """
        a = self.proj_audio(audio)
        t = self.proj_text(text)
        p = self.proj_pros(prosody)

        if drop_audio:
            a = torch.zeros_like(a)
        if drop_text:
            t = torch.zeros_like(t)

        concat = torch.cat([a, t, p], dim=-1)  # [B, N, 3*out_dim]
        g = torch.sigmoid(self.gate(concat))    # [B, N, 3]

        fused = g[..., 0:1] * a + g[..., 1:2] * t + g[..., 2:3] * p
        return fused


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MILAttentionPool(nn.Module):
    """
    Multiple Instance Learning attention pooling.
    Per-word logits → attention-weighted utterance logits.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, logits: torch.Tensor,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [B, N, D] hidden states
            logits: [B, N, C] per-word logits
            mask: [B, N] boolean mask (True = valid)

        Returns:
            utt_logits: [B, C] utterance-level logits
            attn_weights: [B, N] attention weights
        """
        # Compute attention scores
        scores = self.attention(h).squeeze(-1)  # [B, N]

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # [B, N]

        # Weighted sum of per-word logits
        utt_logits = torch.bmm(attn_weights.unsqueeze(1), logits).squeeze(1)  # [B, C]

        return utt_logits, attn_weights


class WDYMNet(nn.Module):
    """
    Word-level Detection of Emphasis, Emotion and Sarcasm in Speech.

    Jointly tags:
    - Emphasis (per-word, 3 classes, supervised)
    - Emotion (per-word via MIL, 7 classes, weakly supervised)
    - Sarcasm (per-word via MIL, binary, weakly supervised)
    - Incongruity (per-word, JS divergence of unimodal emotion dists)
    """

    def __init__(
        self,
        audio_dim: int = 1536,   # 768 * 2 (mean + max)
        text_dim: int = 768,
        prosody_dim: int = 8,
        proj_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
        num_emotions: int = 7,
        num_emphasis: int = 3,
        use_prosody: bool = True,
        use_incongruity: bool = True,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_emotions = num_emotions
        self.use_prosody = use_prosody          # ablation: −prosody
        self.use_incongruity = use_incongruity  # ablation: −incongruity (sarcasm head sees zeros)

        # Gated fusion
        self.fusion = GatedFusion(audio_dim, text_dim, prosody_dim, proj_dim)

        # Positional encoding + Transformer
        self.pos_enc = PositionalEncoding(proj_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ---- Task heads ----
        # Emphasis (per-word, supervised)
        self.emph_head = nn.Linear(proj_dim, num_emphasis)

        # Emotion (per-word logits + MIL pooling)
        self.emo_head = nn.Linear(proj_dim, num_emotions)
        self.emo_pool = MILAttentionPool(proj_dim)

        # Sarcasm (per-word logits + MIL pooling)
        # Also takes incongruity as input feature
        self.sarc_proj = nn.Linear(proj_dim + 1, proj_dim)  # +1 for incongruity
        self.sarc_head = nn.Linear(proj_dim, 2)
        self.sarc_pool = MILAttentionPool(proj_dim)

        # ---- Unimodal auxiliary heads (for incongruity computation) ----
        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.audio_emo_head = nn.Linear(proj_dim, num_emotions)
        self.audio_emo_pool = MILAttentionPool(proj_dim)

        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.text_emo_head = nn.Linear(proj_dim, num_emotions)
        self.text_emo_pool = MILAttentionPool(proj_dim)

    def compute_incongruity(
        self, audio_emo_logits: torch.Tensor, text_emo_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-word incongruity = JS divergence of unimodal emotion distributions,
        normalized to [0, 1] by dividing by ln(2).

        Args:
            audio_emo_logits: [B, N, C] audio-only emotion logits
            text_emo_logits: [B, N, C] text-only emotion logits

        Returns:
            incongruity: [B, N] in [0, 1]
        """
        # fp32: under fp16 autocast 1e-8 underflows and log() gives -inf
        p = F.softmax(audio_emo_logits.float(), dim=-1).clamp(min=1e-8)
        q = F.softmax(text_emo_logits.float(), dim=-1).clamp(min=1e-8)
        m = 0.5 * (p + q)

        kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
        kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
        js = 0.5 * (kl_pm + kl_qm)

        # Normalize by ln(2) to get [0, 1]
        return (js / math.log(2)).clamp(0.0, 1.0)

    def forward(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor,
        prosody_feats: torch.Tensor,
        mask: torch.Tensor | None = None,
        drop_audio: bool = False,
        drop_text: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            audio_feats: [B, N, audio_dim]
            text_feats: [B, N, text_dim]
            prosody_feats: [B, N, prosody_dim]
            mask: [B, N] boolean mask (True = valid word)
            drop_audio: Modality dropout for audio.
            drop_text: Modality dropout for text.

        Returns:
            dict with keys:
                emph_logits: [B, N, 3] emphasis per-word
                emo_logits: [B, N, 7] emotion per-word
                emo_utt_logits: [B, 7] emotion utterance-level (MIL)
                emo_attn: [B, N] emotion attention weights
                sarc_logits: [B, N, 2] sarcasm per-word
                sarc_utt_logits: [B, 2] sarcasm utterance-level (MIL)
                sarc_attn: [B, N] sarcasm attention weights
                incongruity: [B, N] incongruity scores [0,1]
                audio_emo_utt: [B, 7] unimodal audio emotion
                text_emo_utt: [B, 7] unimodal text emotion
        """
        if not self.use_prosody:
            prosody_feats = torch.zeros_like(prosody_feats)

        # Gated fusion
        h = self.fusion(audio_feats, text_feats, prosody_feats,
                        drop_audio=drop_audio, drop_text=drop_text)

        # Positional encoding + Transformer
        h = self.pos_enc(h)
        if mask is not None:
            # TransformerEncoder uses src_key_padding_mask (True = ignore)
            h = self.transformer(h, src_key_padding_mask=~mask)
        else:
            h = self.transformer(h)

        # ---- Emphasis head ----
        emph_logits = self.emph_head(h)  # [B, N, 3]

        # ---- Emotion head (MIL) ----
        emo_logits = self.emo_head(h)  # [B, N, 7]
        emo_utt_logits, emo_attn = self.emo_pool(h, emo_logits, mask)

        # ---- Unimodal auxiliary heads ----
        audio_h = F.relu(self.audio_proj(audio_feats))  # [B, N, proj_dim]
        audio_emo_logits = self.audio_emo_head(audio_h)
        audio_emo_utt, _ = self.audio_emo_pool(audio_h, audio_emo_logits, mask)

        text_h = F.relu(self.text_proj(text_feats))
        text_emo_logits = self.text_emo_head(text_h)
        text_emo_utt, _ = self.text_emo_pool(text_h, text_emo_logits, mask)

        # ---- Incongruity ----
        incongruity = self.compute_incongruity(audio_emo_logits, text_emo_logits)
        inc_feat = incongruity if self.use_incongruity else torch.zeros_like(incongruity)

        # ---- Sarcasm head (uses incongruity as extra feature) ----
        sarc_input = torch.cat([h, inc_feat.unsqueeze(-1).to(h.dtype)], dim=-1)
        sarc_h = F.relu(self.sarc_proj(sarc_input))
        sarc_logits = self.sarc_head(sarc_h)
        sarc_utt_logits, sarc_attn = self.sarc_pool(sarc_h, sarc_logits, mask)

        return {
            "emph_logits": emph_logits,
            "emo_logits": emo_logits,
            "emo_utt_logits": emo_utt_logits,
            "emo_attn": emo_attn,
            "sarc_logits": sarc_logits,
            "sarc_utt_logits": sarc_utt_logits,
            "sarc_attn": sarc_attn,
            "incongruity": incongruity,
            "audio_emo_utt": audio_emo_utt,
            "text_emo_utt": text_emo_utt,
            "audio_emo_logits": audio_emo_logits,
            "text_emo_logits": text_emo_logits,
        }
