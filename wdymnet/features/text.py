"""
Text encoder: roberta-base, word vector = mean of its subword states (via tokenizer word_ids()).
Frozen for feature caching (Stage A); `add_lora()` wraps it with PEFT LoRA for Stage B.
"""
import numpy as np
import torch
import torch.nn as nn

TEXT_DIM = 768


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "roberta-base", device: str = "cuda"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)  # fast tokenizer
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device

    def add_lora(self, r=8, alpha=16, dropout=0.1, target_modules=("query", "value")):
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(target_modules))
        self.model = get_peft_model(self.model, cfg)
        return self

    def forward(self, batch_words: list[list[str]]) -> torch.Tensor:
        """list of word lists → [B, N_max, 768] word vectors (zeros where padded)."""
        enc = self.tokenizer(batch_words, is_split_into_words=True, return_tensors="pt",
                             padding=True, truncation=True, max_length=512).to(self.device)
        hidden = self.model(**enc).last_hidden_state  # [B, L, 768]
        n_max = max(len(w) for w in batch_words)
        out = hidden.new_zeros(len(batch_words), n_max, hidden.shape[-1])
        for b in range(len(batch_words)):
            wid = torch.tensor([-1 if i is None else i for i in enc.word_ids(b)], device=hidden.device)
            valid = wid >= 0
            # scatter-mean subword states into their word slots
            out[b].index_add_(0, wid[valid], hidden[b, valid])
            cnt = torch.bincount(wid[valid], minlength=n_max).clamp(min=1)[:n_max]
            out[b] = out[b] / cnt[:, None].to(out.dtype)
        return out

    @torch.no_grad()
    def extract(self, words: list[str]) -> np.ndarray:
        if not words:
            return np.zeros((0, TEXT_DIM), dtype=np.float32)
        return self.forward([words])[0].float().cpu().numpy()
