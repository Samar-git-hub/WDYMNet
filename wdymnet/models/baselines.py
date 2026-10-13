"""
Baseline models for comparison with WDYMNet.

B0: Pretrained-only (emotion2vec+ 9-class → mapped to MELD 7-class)
B1: Utterance-only fusion (mean-pooled audio+text → classifier)
B2: Emo-FiLM-style (copy utterance label to every word, per-word audio classifier)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# emotion2vec 9-class → MELD 7-class mapping
# emotion2vec: 0=angry, 1=disgusted, 2=fearful, 3=happy, 4=neutral, 5=other, 6=sad, 7=surprised, 8=unknown
# MELD:        0=neutral, 1=surprise, 2=fear, 3=sadness, 4=joy, 5=disgust, 6=anger
E2V_TO_MELD = {
    0: 6,  # angry → anger
    1: 5,  # disgusted → disgust
    2: 2,  # fearful → fear
    3: 4,  # happy → joy
    4: 0,  # neutral → neutral
    6: 3,  # sad → sadness
    7: 1,  # surprised → surprise
}
# Indices 5 (other) and 8 (unknown) are ignored


def map_e2v_scores_to_meld(e2v_scores: list[float]) -> list[float]:
    """Map emotion2vec 9-class scores to MELD 7-class."""
    meld_scores = [0.0] * 7
    for e2v_idx, meld_idx in E2V_TO_MELD.items():
        if e2v_idx < len(e2v_scores):
            meld_scores[meld_idx] = e2v_scores[e2v_idx]
    # Re-normalize
    total = sum(meld_scores)
    if total > 0:
        meld_scores = [s / total for s in meld_scores]
    return meld_scores


class UtteranceFusionBaseline(nn.Module):
    """
    B1: Mean-pool audio + text features → classifier.
    No word-level tags.
    """

    def __init__(self, audio_dim: int = 1536, text_dim: int = 768,
                 hidden_dim: int = 256, num_emotions: int = 7):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(audio_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.emo_head = nn.Linear(hidden_dim, num_emotions)
        self.sarc_head = nn.Linear(hidden_dim, 2)

    def forward(self, audio_feats: torch.Tensor, text_feats: torch.Tensor,
                mask: torch.Tensor | None = None) -> dict:
        """
        Args:
            audio_feats: [B, N, audio_dim]
            text_feats: [B, N, text_dim]
            mask: [B, N]
        """
        concat = torch.cat([audio_feats, text_feats], dim=-1)  # [B, N, A+T]

        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)
            pooled = (concat * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            pooled = concat.mean(dim=1)

        h = self.proj(pooled)
        return {
            "emo_utt_logits": self.emo_head(h),
            "sarc_utt_logits": self.sarc_head(h),
        }


class EmoFiLMBaseline(nn.Module):
    """
    B2: Emo-FiLM-style — copy utterance emotion label to every word,
    train a per-word audio-only classifier with those pseudo-labels.
    """

    def __init__(self, audio_dim: int = 1536, hidden_dim: int = 256,
                 num_emotions: int = 7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_emotions),
        )

    def forward(self, audio_feats: torch.Tensor) -> dict:
        """
        Args:
            audio_feats: [B, N, audio_dim]

        Returns:
            emo_logits: [B, N, num_emotions] per-word logits
        """
        return {"emo_logits": self.classifier(audio_feats)}
