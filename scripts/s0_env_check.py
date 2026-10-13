"""
S0 — Environment check for WDYMNet.
Verifies: GPU, CUDA, ffmpeg, model loading (whisper-small + emotion2vec+ base + roberta-base).
Prints peak VRAM after loading all models.
"""
import sys
import time

import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wdymnet.utils.common import set_seed, check_ffmpeg, print_vram, reset_vram_stats


def main():
    set_seed(1337)
    print("=" * 60)
    print("WDYMNet S0 — Environment Check")
    print("=" * 60)

    # ---- Python version ----
    print(f"\n[INFO] Python: {sys.version}")

    # ---- PyTorch + CUDA ----
    import torch
    import torchaudio
    print(f"[INFO] PyTorch: {torch.__version__}")
    print(f"[INFO] torchaudio: {torchaudio.__version__}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"[INFO] CUDA version: {torch.version.cuda}")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"[INFO] GPU memory: {total_mem:.0f} MB")
        bf16_ok = torch.cuda.is_bf16_supported()
        print(f"[INFO] bf16 supported: {bf16_ok}")
    else:
        print("[WARN] No CUDA GPU detected! Training will be very slow on CPU.")
        print("  Make sure you installed the CUDA build of PyTorch:")
        print("  pip install -r requirements.txt  (pulls the cu126 wheels)")

    # ---- ffmpeg ----
    print()
    check_ffmpeg()

    # ---- Load models ----
    if not torch.cuda.is_available():
        print("\n[SKIP] Skipping model loading test — no CUDA GPU.")
        print("[INFO] Install CUDA PyTorch first, then re-run this script.")
        return

    device = "cuda"
    reset_vram_stats()

    # 1. faster-whisper (small)
    print("\n--- Loading faster-whisper (small) ---")
    t0 = time.time()
    try:
        from faster_whisper import WhisperModel
        whisper = WhisperModel("small", device=device, compute_type="int8_float16")
        # Dummy transcribe
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second silence
        segments, info = whisper.transcribe(dummy_audio, word_timestamps=True,
                                            vad_filter=False, language="en")
        # Consume the generator
        for seg in segments:
            pass
        print(f"[OK] faster-whisper loaded in {time.time()-t0:.1f}s")
        print_vram("after whisper")
    except Exception as e:
        print(f"[ERROR] faster-whisper failed: {e}")

    # 2. emotion2vec+ base
    print("\n--- Loading emotion2vec+ base ---")
    t0 = time.time()
    e2v_model = None
    e2v_source = None
    try:
        from funasr import AutoModel
        try:
            e2v_model = AutoModel(model="iic/emotion2vec_plus_base", hub="hf", disable_update=True)
            e2v_source = "hf"
        except Exception:
            print("[INFO] HuggingFace hub failed, trying ModelScope...")
            e2v_model = AutoModel(model="iic/emotion2vec_plus_base", hub="ms", disable_update=True)
            e2v_source = "ms"

        # Dummy forward
        dummy = np.random.randn(16000).astype(np.float32)
        res = e2v_model.generate(dummy, granularity="frame", extract_embedding=True)
        feats = res[0]["feats"]
        print(f"[OK] emotion2vec+ base loaded from {e2v_source} in {time.time()-t0:.1f}s")
        print(f"[INFO] Frame features shape: {feats.shape} (expect [T, 768])")
        print_vram("after emotion2vec")
    except Exception as e:
        print(f"[ERROR] emotion2vec+ base failed: {e}")
        print("[INFO] Will fall back to WavLM if needed")

    # 3. roberta-base
    print("\n--- Loading roberta-base ---")
    t0 = time.time()
    try:
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        roberta = RobertaModel.from_pretrained("roberta-base").to(device)
        # Dummy forward
        inputs = tokenizer("Hello world, this is a test.", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = roberta(**inputs)
        print(f"[OK] roberta-base loaded in {time.time()-t0:.1f}s")
        print(f"[INFO] Hidden size: {outputs.last_hidden_state.shape[-1]}")
        print_vram("after roberta")
    except Exception as e:
        print(f"[ERROR] roberta-base failed: {e}")

    # 4. Silero VAD
    print("\n--- Loading Silero VAD ---")
    t0 = time.time()
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        vad_model = load_silero_vad()
        assert get_speech_timestamps(torch.zeros(32000), vad_model, sampling_rate=16000) == []
        print(f"[OK] Silero VAD loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"[ERROR] Silero VAD failed: {e}")

    # ---- Final VRAM report ----
    print("\n" + "=" * 60)
    print_vram("FINAL — all models loaded")
    print("=" * 60)

    # Cleanup
    del whisper, roberta, tokenizer
    if e2v_model is not None:
        del e2v_model
    torch.cuda.empty_cache()
    print("\n[OK] Environment check complete!")


if __name__ == "__main__":
    main()
