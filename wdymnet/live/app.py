"""
WDYMNet live demo (Gradio).

  python -m wdymnet.live.app [--config configs/live.yaml] [--checkpoint PATH]

Speak (push-to-talk, upload, or continuous mic) → tone-tagged transcript → a corrected sentence and its intended
meaning, rewritten by a small LLM from the transcript plus the voice cues.
"""
import argparse
import json
from pathlib import Path

import gradio as gr
import numpy as np

from wdymnet.live.render import CSS, corrected_html, legend_html, transcript_html
from wdymnet.live.stream import MicStream
from wdymnet.pipeline import Tagger
from wdymnet.utils.common import load_config

WAVEFORM = gr.WaveformOptions(waveform_color="#71717a", waveform_progress_color="#a1a1aa")
THEME = gr.themes.Base(primary_hue=gr.themes.colors.zinc, secondary_hue=gr.themes.colors.zinc,
                       neutral_hue=gr.themes.colors.zinc, radius_size=gr.themes.sizes.radius_sm,
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]).set(
    block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
    block_label_text_color="*neutral_500", block_label_text_color_dark="*neutral_400",
    block_title_background_fill="transparent", block_title_background_fill_dark="transparent",
    block_title_text_color="*neutral_500", block_title_text_color_dark="*neutral_400",
    body_background_fill_dark="*neutral_950", block_background_fill_dark="*neutral_900")


def pick_checkpoint(cfg: dict, override: str | None) -> Path:
    for c in ([override] if override else []) + cfg["model"]["checkpoints"]:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(f"No checkpoint found (tried {cfg['model']['checkpoints']}). Run S3 first.")


def build_ui(analyze, add_rewrite, cfg: dict, model_label: str) -> gr.Blocks:
    """analyze(audio, sr) → tagged result (shown at once); add_rewrite(result) fills result["rewrite"] after."""
    th = cfg["inference"]
    mic = MicStream(analyze, cfg["vad"], post=add_rewrite)

    def views(results):
        newest_first = results[::-1]
        return transcript_html(newest_first, th), corrected_html(newest_first)

    def on_audio(audio, results):
        if audio is None:
            return (results, *views(results))
        sr, data = audio
        res = analyze(data, sr)
        if res["words"]:
            results = results + [res]
        elif not results:
            return results, transcript_html([res], th), corrected_html([])
        return (results, *views(results))

    def rewrite_pending(results):
        for r in results:
            if r["words"] and "rewrite" not in r:
                add_rewrite(r)
        return (results, *views(results))

    def toggle(running):
        if running:
            mic.stop()
            return False, gr.update(value="Start listening"), gr.Timer(active=False)
        mic.results.clear()
        mic.start()
        return True, gr.update(value="Stop listening"), gr.Timer(active=True)

    def poll():
        return (f"<span class='wd-muted'>{mic.status}</span>", *views(mic.results))

    with gr.Blocks(title="WDYMNet") as demo:
        gr.Markdown("## WDYMNet\n<span class='wd-muted'>Word-level emphasis, emotion and sarcasm from speech, "
                    f"then what you meant. {model_label}</span>")
        gr.HTML(legend_html())
        with gr.Row():
            with gr.Column(scale=2):
                audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Push to talk",
                                    waveform_options=WAVEFORM)
            with gr.Column(scale=1):
                gr.Markdown("<span class='wd-muted'>Continuous mode listens on the default microphone and "
                            "tags each phrase after a pause.</span>")
                running = gr.State(False)
                cont_btn = gr.Button("Start listening")
                status = gr.Markdown("<span class='wd-muted'>idle</span>")
                clear = gr.Button("Clear", size="sm")
        transcript = gr.HTML(transcript_html([], th))
        corrected = gr.HTML(corrected_html([]))
        results = gr.State([])
        timer = gr.Timer(0.5, active=False)

        outs = [results, transcript, corrected]
        audio_in.stop_recording(on_audio, [audio_in, results], outs).then(rewrite_pending, results, outs)
        audio_in.upload(on_audio, [audio_in, results], outs).then(rewrite_pending, results, outs)
        clear.click(lambda: ([], transcript_html([], th), corrected_html([])), None, outs)
        cont_btn.click(toggle, running, [running, cont_btn, timer])
        timer.tick(poll, None, [status, transcript, corrected])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live.yaml")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    th_file = Path("results/thresholds.json")
    if th_file.exists():  # display thresholds calibrated on validation in S6
        cfg["inference"].update(json.loads(th_file.read_text()))
    th = cfg["inference"]

    ckpt = pick_checkpoint(cfg, args.checkpoint)
    print(f"loading models (checkpoint {ckpt}) …")
    a = cfg["asr"]
    tagger = Tagger(ckpt, device=cfg["model"]["device"], whisper_size=a["model_size"],
                    compute_type=a["compute_type"], vad_cfg=cfg["vad"], beam_size=a.get("beam_size", 1))
    rewriter = None
    if cfg["rewrite"]["enabled"]:
        from wdymnet.rewrite import Rewriter
        rewriter = Rewriter(cfg["rewrite"]["model"], device=cfg["model"]["device"],
                            max_new_tokens=cfg["rewrite"]["max_new_tokens"], ct2_dir=cfg["rewrite"]["ct2_dir"])

    def analyze(audio, sr):
        return tagger.tag(audio, sr)

    def add_rewrite(res):
        res["rewrite"] = rewriter.rewrite(res, th) if rewriter is not None else {}

    # warm-up: CUDA kernels, numba (pyin), whisper and the LLM, so the first real utterance is fast
    warm = (np.random.randn(16000) * 0.01).astype(np.float32)
    list(tagger.whisper.transcribe(warm, language="en")[0])
    w = tagger.tag(warm, words=[{"w": "hello", "start": 0.1, "end": 0.5}, {"w": "there", "start": 0.5, "end": 0.9}])
    if rewriter is not None:
        rewriter.rewrite(w, th)

    label = f"Tagger: {ckpt.stem} (stage {tagger.stage})" + (f" · rewriter: {rewriter.name}" if rewriter else "")
    demo = build_ui(analyze, add_rewrite, cfg, label)
    demo.launch(server_port=cfg["ui"]["server_port"], share=cfg["ui"]["share"], css=CSS, theme=THEME)


if __name__ == "__main__":
    main()
