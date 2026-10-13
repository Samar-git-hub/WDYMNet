"""
Spot-check of the "what you meant" rewriter on transcripts with grammatical-but-wrong ASR errors (no audio needed):
sound-alikes, tense/time contradictions, plus two already-correct sentences that must stay unchanged.

Usage: python scripts/check_rewriter.py [HF_MODEL_ID CT2_DIR]     (defaults: the model in wdymnet/rewrite.py)
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wdymnet.pipeline import EMOTIONS
from wdymnet.rewrite import Rewriter

CASES = [  # (ASR transcript, what a good correction should contain / must not contain)
    ("I went for a walk tomorrow and I did 6 kilometers. It was a very tiring walk I did tomorrow.", "not:tomorrow"),
    ("I did this tomorrow and I love doing it tomorrow.", "not:did this tomorrow"),
    ("Can you pass me the salt? The whether is really nice today.", "weather"),
    ("I want to by a new phone because mine is broken.", "buy"),
    ("She ate her breakfast next week and then went to work.", "any:yesterday|this morning|today|earlier|last week"),
    ("I will see you yesterday at the station.", "any:tomorrow|later|today|soon"),
    ("My throat hurts so I am going to the dentist.", "any:doctor"),
    ("Their going to the cinema later, do you want to come?", "They're"),
    ("I love waiting in queues for hours.", "love waiting"),            # already fine: keep as is
    ("Could you please send me the report by Friday?", "Friday"),      # already fine: keep as is
]

def fake_res(text, tone="neutral", p=0.9):
    words = [{"w": w, "start": i * .3, "end": i * .3 + .25, "asr_p": 0.9, "emph": 0, "emph_probs": [.8, .1, .1],
              "emo": "joy", "emo_p": .05, "emo_probs": {**{e: .0 for e in EMOTIONS}, "neutral": .9},
              "sarc_p": .1, "sarc_attn": .1} for i, w in enumerate(text.split())]
    return {"words": words, "utterance": {"emotion": tone, "emo_probs": {**{e: .0 for e in EMOTIONS}, tone: p},
                                          "sarcasm_p": .1}}

th = json.load(open("results/thresholds.json"))
kw = dict(model_name=sys.argv[1], ct2_dir=sys.argv[2]) if len(sys.argv) > 2 else {}
rw = Rewriter(**kw)
ok, times = 0, []
for text, check in CASES:
    o = rw.rewrite(fake_res(text), th)
    c = o["corrected"]
    if check.startswith("not:"):
        good = check[4:].lower() not in c.lower()
    elif check.startswith("any:"):
        good = any(k in c.lower() for k in check[4:].split("|"))
    else:
        good = check.lower() in c.lower()
    ok += good; times.append(o["seconds"])
    print(f"[{'OK ' if good else 'BAD'}] {text}\n      -> {c}\n      meaning: {o['meaning']}")
print(f"SCORE {ok}/{len(CASES)}  mean {sum(times)/len(times):.2f}s")
