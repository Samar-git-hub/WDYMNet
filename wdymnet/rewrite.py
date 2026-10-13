"""
Tone-aware rewriting: a pretrained instruction-tuned LLM (Qwen3-4B-Instruct-2507, 4-bit AWQ, used as-is, no
fine-tuning) turns the raw ASR transcript + WDYMNet's voice cues into
  corrected — the sentence the speaker most likely meant: misheard sound-alikes and words that contradict the rest
              of the sentence fixed, grammatical, punctuation matching the tone
  meaning   — one short line on what they actually mean (sarcasm → the intended meaning)
Only the cues the demo displays are passed (thresholded tags, strongest few words), so the LLM is not fed noise.
Runs on CTranslate2 (~2.7 GB VRAM); the checkpoint is converted once into data/models/ on first use.
Smaller models (Qwen2.5-1.5B, Qwen3-1.7B) were tried and scored 4/10 on nonsense-but-grammatical ASR errors
("I went for a walk tomorrow"); this one scores 9/10.
"""
import re
import time
from pathlib import Path

from wdymnet.live.render import utterance_tags

MAX_LISTED = 4  # strongest words per cue; long lists (noisy TV audio) only confuse a small LLM

SYSTEM = (
    "You fix speech-recognition transcripts. They often contain real, grammatical words that are still wrong: "
    "sound-alikes (whether/weather), or words that contradict the rest of the sentence (a past-tense verb with "
    "'tomorrow'). You also get cues about HOW it was said, from the speaker's voice. Reply with exactly three lines:\n"
    "Problems: each word that makes no sense in context -> what it should be; or 'none'. Be brief.\n"
    "Corrected: the sentence the speaker meant, grammatical, with those fixes and otherwise their own words; "
    "punctuation matching the tone ('!' for anger or excitement). No new facts.\n"
    "Meaning: one short sentence on what they actually mean, using the tone cues; if sarcasm is detected, the "
    "intended (usually opposite) meaning."
)

# worked examples: one tone/sarcasm case, one sense-error case
EXAMPLES = [
    ('Transcript: "i love waiting in queues for like hours"\n'
     "Voice cues:\n- overall tone: anger (moderate)\n- sarcasm: detected (strong), on: \"love\"\n"
     '- emphasised words: "love", "hours"\n- possibly misheard words: none',
     "Problems: none\n"
     "Corrected: I love waiting in queues for hours!\n"
     "Meaning: They hate waiting in long queues."),
    ('Transcript: "I cooked dinner tomorrow and it were delicious, the meet was perfect."\n'
     "Voice cues:\n- overall tone: joy (moderate)\n- sarcasm: not detected\n"
     '- emphasised words: "delicious"\n- possibly misheard words: none',
     "Problems: tomorrow -> yesterday (past tense); were -> was; meet -> meat\n"
     "Corrected: I cooked dinner yesterday and it was delicious, the meat was perfect!\n"
     "Meaning: They are happy with the dinner they cooked."),
]


def _strength(p: float) -> str:
    return "strong" if p >= 0.6 else "moderate" if p >= 0.4 else "weak"


def _quoted(ws: list[str]) -> str:
    return ", ".join(f'"{w}"' for w in ws) if ws else "none"


def build_cues(res: dict, th: dict) -> str:
    """Voice cues in plain language, from the same thresholded tags the transcript display uses."""
    words, utt = res["words"], res["utterance"]
    tags = utterance_tags(res, th)
    emo, p_emo = utt["emotion"], utt["emo_probs"][utt["emotion"]]
    lines = [f"- overall tone: {emo} ({_strength(p_emo)})" if emo != "neutral" or p_emo >= 0.4
             else "- overall tone: unclear"]
    sarc_words = [w["w"] for w, t in zip(words, tags) if t["sarc"]]
    lines.append(f"- sarcasm: detected ({_strength(utt['sarcasm_p'])}), on: {_quoted(sarc_words)}" if sarc_words
                 else "- sarcasm: not detected")
    emph = sorted([w for w, t in zip(words, tags) if t["emph"] == 2], key=lambda w: -w["emph_probs"][2])
    lines.append("- emphasised words: " + _quoted([w["w"] for w in emph[:MAX_LISTED]]))
    by_emo: dict[str, list[dict]] = {}
    for w, t in zip(words, tags):
        if t["emo"] and w["emo_p"] >= 0.5:  # only confident word-level emotions
            by_emo.setdefault(t["emo"], []).append(w)
    for e, ws in by_emo.items():
        ws = sorted(ws, key=lambda w: -w["emo_p"])
        where = "most of the sentence" if len(ws) > MAX_LISTED else _quoted([w["w"] for w in ws])
        lines.append(f"- {e} in the voice on: {where}")
    misheard = [w["w"] for w in words if w.get("asr_p", 1.0) < 0.5]
    lines.append("- possibly misheard words: " + _quoted(misheard[:MAX_LISTED]))
    return "\n".join(lines)


def _ensure_ct2(model_name: str, out_dir: Path) -> Path:
    if not (out_dir / "model.bin").exists():
        print(f"[rewrite] converting {model_name} to CTranslate2 (one-off) → {out_dir}")
        if "awq" in model_name.lower():  # 4-bit AWQ checkpoint: keep its weights as-is
            from wdymnet.ct2_awq import convert_awq
            convert_awq(model_name, out_dir)
        else:
            from ctranslate2.converters import TransformersConverter
            TransformersConverter(model_name).convert(str(out_dir), quantization="int8_float16", force=True)
    return out_dir


class Rewriter:
    def __init__(self, model_name: str = "Eslzzyl/Qwen3-4B-Instruct-2507-AWQ", device: str = "cuda",
                 max_new_tokens: int = 128, ct2_dir: str = "data/models/qwen3-4b-2507-awq-ct2"):
        import ctranslate2
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        path = _ensure_ct2(model_name, Path(ct2_dir))
        self.gen = ctranslate2.Generator(str(path), device=device)  # dtype from the checkpoint (AWQ int4 / int8)
        self.max_new_tokens, self.name = max_new_tokens, model_name.split("/")[-1]

    def prompt(self, res: dict, th: dict) -> list[dict]:
        transcript = " ".join(w["w"] for w in res["words"])
        msgs = [{"role": "system", "content": SYSTEM}]
        for u, a in EXAMPLES:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        return msgs + [{"role": "user", "content": f'Transcript: "{transcript}"\nVoice cues:\n{build_cues(res, th)}'}]

    def rewrite(self, res: dict, th: dict) -> dict:
        if not res or not res.get("words"):
            return {}
        t0 = time.perf_counter()
        # enable_thinking=False: Qwen3 would otherwise reason at length first (slow); ignored by other templates
        ids = self.tok.apply_chat_template(self.prompt(res, th), add_generation_prompt=True, enable_thinking=False)
        ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
        out = self.gen.generate_batch([self.tok.convert_ids_to_tokens(ids)], max_length=self.max_new_tokens,
                                      sampling_topk=1, include_prompt_in_result=False,
                                      end_token=[self.tok.eos_token, "<|im_end|>"])
        text = self.tok.decode(out[0].sequences_ids[0], skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        field = lambda k: (m.group(1).strip() if (m := re.search(rf"{k}:\s*(.+)", text)) else "")
        problems = field("Problems")
        return {"corrected": field("Corrected") or " ".join(w["w"] for w in res["words"]),  # fall back to ASR text
                "meaning": field("Meaning"),
                "problems": "" if problems.lower().rstrip(".") == "none" else problems,
                "seconds": time.perf_counter() - t0, "raw": text}
