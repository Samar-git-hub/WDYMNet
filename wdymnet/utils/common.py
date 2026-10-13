"""
WDYMNet utility functions — seeds, config loading, VRAM reporting.
"""
import random
import yaml
import torch
import numpy as np
from pathlib import Path


def set_seed(seed: int = 1337):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_vram(tag: str = ""):
    """Print current and peak GPU VRAM usage."""
    if not torch.cuda.is_available():
        print(f"[VRAM {tag}] CUDA not available")
        return
    allocated = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"[VRAM {tag}] Allocated: {allocated:.0f} MB | Peak: {peak:.0f} MB | Total: {total:.0f} MB")


def reset_vram_stats():
    """Reset peak VRAM tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on PATH."""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        print(f"[OK] ffmpeg found: {path}")
        return True
    else:
        print("[WARN] ffmpeg not found on PATH!")
        print("  Install: https://ffmpeg.org/download.html")
        print("  Windows: winget install Gyan.FFmpeg")
        print("  Or: choco install ffmpeg")
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
