"""
Convert an AutoAWQ (GEMM, 4-bit) Hugging Face checkpoint to CTranslate2 without `gptqmodel`.

CTranslate2's converter normally instantiates the model with transformers, which for AWQ checkpoints requires
the gptqmodel package (no Windows wheels). Its model loaders only read plain tensors (qweight / qzeros / scales,
norm weights, embeddings), so we hand them a lightweight attribute tree built straight from the safetensors files.
"""
from pathlib import Path
from types import SimpleNamespace

import torch


def _tree(tensors: dict[str, torch.Tensor]) -> SimpleNamespace:
    root: dict = {}
    for name, t in tensors.items():
        node = root
        *parents, leaf = name.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = t.half() if t.is_floating_point() else t

    def build(d):
        if d and all(k.isdigit() for k in d):  # numbered children → list (e.g. layers.0, layers.1, ...)
            return [build(d[k]) for k in sorted(d, key=int)]
        return SimpleNamespace(**{k: build(v) if isinstance(v, dict) else v for k, v in d.items()})
    return build(root)


def convert_awq(model_id: str, out_dir: str | Path) -> Path:
    from ctranslate2.converters import TransformersConverter
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    local = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges*",
                                                        "vocab*", "tokenizer*"])

    class AWQConverter(TransformersConverter):
        def load_model(self, model_class, model_name_or_path, **kwargs):
            config = kwargs["config"]
            qc = config.quantization_config
            config.quantization_config = SimpleNamespace(**qc) if isinstance(qc, dict) else qc
            tensors = {}
            for f in sorted(Path(local).glob("*.safetensors")):
                tensors.update(load_file(str(f)))
            model = _tree(tensors)
            model.config = config
            if not hasattr(model, "lm_head"):  # tied embeddings
                model.lm_head = SimpleNamespace(weight=model.model.embed_tokens.weight)
            return model

    AWQConverter(local).convert(str(out_dir), force=True)
    return Path(out_dir)
