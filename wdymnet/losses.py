"""
WDYMNet loss functions.

Combined loss:
  CE_emph + CE_emo (class-weighted) + CE_sarc
  + 0.3 * (CE_text_only + CE_audio_only)
  + 0.01 * sparsity (entropy of MIL attention)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WDYMNetLoss(nn.Module):
    """Multi-task loss for WDYMNet."""

    def __init__(
        self,
        emo_class_weights: list[float] | None = None,
        emph_weight: float = 1.0,
        emo_weight: float = 1.0,
        sarc_weight: float = 1.0,
        unimodal_weight: float = 0.3,
        sparsity_weight: float = 0.01,
        device: str = "cuda",
    ):
        super().__init__()
        self.emph_weight = emph_weight
        self.emo_weight = emo_weight
        self.sarc_weight = sarc_weight
        self.unimodal_weight = unimodal_weight
        self.sparsity_weight = sparsity_weight

        # Emphasis CE (ignore index -1 for NA/punctuation)
        self.emph_ce = nn.CrossEntropyLoss(ignore_index=-1)

        # Emotion CE (class-weighted for imbalanced MELD)
        if emo_class_weights is not None:
            w = torch.tensor(emo_class_weights, dtype=torch.float32, device=device)
            self.emo_ce = nn.CrossEntropyLoss(weight=w)
        else:
            self.emo_ce = nn.CrossEntropyLoss()

        # Sarcasm CE
        self.sarc_ce = nn.CrossEntropyLoss()

    def sparsity_loss(self, attn: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Entropy of attention weights (encourage sparse attention → few key words).
        Lower entropy = sparser attention.
        We minimize entropy to encourage the model to focus on fewer words.
        """
        attn = attn.float()
        if mask is None:
            mask = torch.ones_like(attn, dtype=torch.bool)
        # entropy over real words only (padding must not contribute), normalised to [0, 1] by log(N);
        # 1-word utterances have nothing to sparsify → 0 (dividing by log(1) blew up gradients and fp16)
        entropy = -(attn * attn.clamp(min=1e-8).log()).masked_fill(~mask, 0.0).sum(dim=-1)
        n_words = mask.float().sum(dim=-1)
        entropy = torch.where(n_words > 1, entropy / n_words.clamp(min=2).log(), torch.zeros_like(entropy))
        return entropy.mean()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor | None],
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined loss. Only active heads contribute (skip if target is None).

        Args:
            outputs: Model output dict.
            targets: Dict with optional keys:
                emph_labels: [B, N] long (-1 for ignore)
                emo_label: [B] long (utterance emotion)
                sarc_label: [B] long (utterance sarcasm)
            mask: [B, N] boolean mask.

        Returns:
            total_loss: scalar
            loss_dict: individual losses for logging
        """
        total = torch.tensor(0.0, device=outputs["emph_logits"].device)
        loss_dict = {}

        # ---- Emphasis (per-word, supervised) ----
        if targets.get("emph_labels") is not None:
            emph_logits = outputs["emph_logits"]  # [B, N, 3]
            emph_labels = targets["emph_labels"]   # [B, N]
            B, N, C = emph_logits.shape
            loss_emph = self.emph_ce(emph_logits.reshape(-1, C), emph_labels.reshape(-1))
            total = total + self.emph_weight * loss_emph
            loss_dict["emph"] = loss_emph.item()

        # ---- Emotion (utterance-level MIL) ----
        if targets.get("emo_label") is not None:
            loss_emo = self.emo_ce(outputs["emo_utt_logits"], targets["emo_label"])
            total = total + self.emo_weight * loss_emo
            loss_dict["emo"] = loss_emo.item()

            # Unimodal auxiliary losses
            loss_audio_emo = self.emo_ce(outputs["audio_emo_utt"], targets["emo_label"])
            loss_text_emo = self.emo_ce(outputs["text_emo_utt"], targets["emo_label"])
            total = total + self.unimodal_weight * (loss_audio_emo + loss_text_emo)
            loss_dict["audio_emo"] = loss_audio_emo.item()
            loss_dict["text_emo"] = loss_text_emo.item()

            # Sparsity on emotion attention
            loss_sparse_emo = self.sparsity_loss(outputs["emo_attn"], mask)
            total = total + self.sparsity_weight * loss_sparse_emo
            loss_dict["sparse_emo"] = loss_sparse_emo.item()

        # ---- Sarcasm (utterance-level MIL) ----
        if targets.get("sarc_label") is not None:
            loss_sarc = self.sarc_ce(outputs["sarc_utt_logits"], targets["sarc_label"])
            total = total + self.sarc_weight * loss_sarc
            loss_dict["sarc"] = loss_sarc.item()

            # Sparsity on sarcasm attention
            loss_sparse_sarc = self.sparsity_loss(outputs["sarc_attn"], mask)
            total = total + self.sparsity_weight * loss_sparse_sarc
            loss_dict["sparse_sarc"] = loss_sparse_sarc.item()

        loss_dict["total"] = total.item()
        return total, loss_dict
