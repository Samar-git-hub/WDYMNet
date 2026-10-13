"""
Stage B featurizer: recomputes word features with trainable encoders instead of the S2 cache.
  audio: cached hidden state entering emotion2vec's last 3 blocks → (trainable) top blocks → word pooling
  text : RoBERTa + LoRA on the word list
  prosody: from the S2 cache
"""
from pathlib import Path

import torch
import torch.nn.functional as F

from wdymnet.features.speech import Emotion2vecEncoder, pool_frames_to_words
from wdymnet.features.text import TextEncoder


class StageBFeaturizer(torch.nn.Module):
    def __init__(self, mid_dir: str | Path, lora_cfg: dict, device: str = "cuda"):
        super().__init__()
        self.mid_dir = Path(mid_dir)
        self.speech = Emotion2vecEncoder(device)
        self.top = self.speech.net.blocks[self.speech.split:]
        self.top.requires_grad_(True)
        self.text = TextEncoder(device=device).add_lora(lora_cfg["r"], lora_cfg["alpha"], lora_cfg["dropout"],
                                                        lora_cfg["target_modules"])
        self.device = device

    def train(self, mode: bool = True):
        super().train(mode)
        self.speech.net.eval()  # keep the frozen bottom (and its dropout/layerdrop) in eval mode
        self.top.train(mode)
        return self

    # attention + ALiBi tensors grow with B·L², so long (eval) clips run in smaller sub-batches
    MAX_B_L2 = 8 * 520 ** 2

    def audio_feats(self, items: list[dict]) -> torch.Tensor:
        mids = [torch.load(self.mid_dir / f"{x['id']}.pt", weights_only=True)["mid"] for x in items]
        N = max(len(x["words"]) for x in items)
        out = torch.zeros(len(items), N, mids[0].shape[-1] * 2, device=self.device)
        i = 0
        while i < len(items):
            j = i + 1
            while j < len(items) and (j + 1 - i) * max(m.shape[0] for m in mids[i:j + 1]) ** 2 <= self.MAX_B_L2:
                j += 1
            L = max(m.shape[0] for m in mids[i:j])
            mid = torch.stack([F.pad(m, (0, 0, 0, L - m.shape[0])) for m in mids[i:j]]).to(self.device).float()
            n_frames = torch.tensor([m.shape[0] - self.speech.n_extra for m in mids[i:j]], device=self.device)
            frames = self.speech.top(mid, n_frames)
            for k, x in enumerate(items[i:j]):
                out[i + k, :len(x["words"])] = pool_frames_to_words(frames[k, :n_frames[k]], x["words"]).to(out.dtype)
            i = j
        return out

    def forward(self, model, b: dict, drop_audio=False, drop_text=False) -> dict:
        items = b["items"]
        audio = self.audio_feats(items)
        text = self.text([[w["w"] for w in x["words"]] for x in items])
        return model(audio, text, b["pros"], b["mask"], drop_audio=drop_audio, drop_text=drop_text)

    def state_for_checkpoint(self) -> dict:
        lora = {k: v.detach().cpu() for k, v in self.text.model.state_dict().items() if "lora_" in k}
        return {"speech_top": {k: v.detach().cpu() for k, v in self.top.state_dict().items()}, "lora": lora}

    def load_from_checkpoint(self, ck: dict):
        self.top.load_state_dict(ck["speech_top"])
        self.text.model.load_state_dict(ck["lora"], strict=False)
