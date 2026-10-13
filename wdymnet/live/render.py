"""
Render tagged words as (a) HTML for the demo and (b) inline tagged text, e.g.
  I <sarcasm p=.81><emph lvl=2>love</emph></sarcasm> waiting <anger p=.66>in queues</anger>

Display rules (thresholds from configs/live.yaml, re-calibrated by S6 → results/thresholds.json):
  emphasis — argmax level (1 = semibold, 2 = bold)
  emotion  — the word's strongest non-neutral emotion, only if it beats the word's own neutral prob
             and exceeds emotion_threshold
  sarcasm  — only if the utterance sarcasm prob exceeds sarcasm_threshold (calibrated for a low false-alarm
             rate on ordinary speech); then only words with clearly above-uniform sarcasm attention (max 2)
"""
import html

EMO_COLORS = {"joy": "#c9a227", "sadness": "#6b93c7", "anger": "#cf6a5f", "fear": "#9a86c4",
              "disgust": "#6fa27a", "surprise": "#c47b9f", "neutral": "inherit"}
SARC_COLOR = "#a58fd6"

CSS = """
.wd-card {font-size: 1.22rem; line-height: 2.1rem; padding: 14px 18px; border-radius: 8px;
          border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary);}
.wd-utt {padding: 6px 0 10px; border-bottom: 1px solid var(--border-color-primary);}
.wd-utt:last-child {border-bottom: none;}
.wd-meta {font-size: .74rem; line-height: 1.1rem; opacity: .6; margin-top: 4px; font-family: var(--font-mono);}
.wd-meta span + span::before {content: "·"; margin: 0 8px; opacity: .6;}
.wd-w {position: relative; display: inline-block; padding: 0 2px; border-radius: 3px; cursor: default;}
.wd-w:hover {background: var(--background-fill-primary);}
.wd-e1 {font-weight: 600;}
.wd-e2 {font-weight: 750;}
.wd-sarc {text-decoration: underline dotted; text-decoration-color: """ + SARC_COLOR + """;
          text-decoration-thickness: 2px; text-underline-offset: 6px;}
.wd-tip {visibility: hidden; opacity: 0; transition: opacity .1s; position: absolute; z-index: 50;
         bottom: 120%; left: 50%; transform: translateX(-50%); width: 220px; padding: 10px 12px;
         font-size: .72rem; line-height: 1.05rem; font-weight: 400; color: var(--body-text-color);
         background: var(--background-fill-primary); border: 1px solid var(--border-color-primary);
         border-radius: 6px; box-shadow: 0 6px 18px rgba(0,0,0,.25); text-align: left;}
.wd-w:hover .wd-tip {visibility: visible; opacity: 1;}
.wd-row {display: flex; align-items: center; gap: 6px; margin: 1px 0;}
.wd-row .lab {width: 62px; opacity: .7;}
.wd-row .bar {flex: 1; height: 4px; border-radius: 2px; background: var(--border-color-primary); overflow: hidden;}
.wd-row .bar > i {display: block; height: 100%;}
.wd-row .val {width: 30px; text-align: right; font-family: var(--font-mono); opacity: .8;}
.wd-tip .hd {font-weight: 600; margin-bottom: 4px;}
.wd-tip hr {border: none; border-top: 1px solid var(--border-color-primary); margin: 6px 0;}
.wd-legend {display: flex; flex-wrap: wrap; gap: 14px; font-size: .78rem; opacity: .75; margin: 0 0 4px;}
.wd-dot {display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px;}
.wd-empty {opacity: .5; font-size: .95rem;}
.wd-muted {opacity: .6; font-size: .9rem;}
.wd-label {font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; opacity: .5; margin-bottom: 4px;}
.wd-fix {font-size: 1.15rem; line-height: 1.7rem;}
.wd-mean {font-size: .92rem; opacity: .75; margin-top: 2px;}
"""


def utterance_tags(res: dict, th: dict) -> list[dict]:
    """Per-word display tags for one tagged utterance (see module docstring)."""
    words, utt = res["words"], res["utterance"] or {}
    sarcastic = utt.get("sarcasm_p", 0.0) > th["sarcasm_threshold"]
    top_sarc = set()
    if sarcastic:  # words the sarcasm attention clearly relies on (> 1.5x uniform), at most 2, else the single top word
        order = sorted(range(len(words)), key=lambda i: -words[i]["sarc_attn"])
        top_sarc = {i for i in order[:2] if words[i]["sarc_attn"] > 1.5 / len(words)} or {order[0]}
    tags = []
    for i, w in enumerate(words):
        emo_ok = w["emo_p"] > th["emotion_threshold"] and w["emo_p"] > w["emo_probs"]["neutral"]
        tags.append({"emph": w["emph"], "emo": w["emo"] if emo_ok else None, "sarc": i in top_sarc})
    return tags


def _bar(label: str, p: float, color: str) -> str:
    return (f'<div class="wd-row"><span class="lab">{label}</span><span class="bar"><i style="width:{p * 100:.0f}%;'
            f'background:{color}"></i></span><span class="val">{p:.2f}</span></div>')


def _tooltip(w: dict) -> str:
    rows = [f'<div class="hd">{html.escape(w["w"])} <span style="opacity:.5;font-weight:400">'
            f'{w["start"]:.2f}–{w["end"]:.2f}s</span></div>',
            _bar("emphasis", 1 - w["emph_probs"][0], "#9a9a9a"),
            _bar("sarcasm", w["sarc_p"], SARC_COLOR),
            _bar("mismatch", w["incongruity"], "#c08552"), "<hr>"]
    for e, p in sorted(w["emo_probs"].items(), key=lambda kv: -kv[1])[:4]:
        rows.append(_bar(e, p, EMO_COLORS[e] if e != "neutral" else "#9a9a9a"))
    ta = max(w["text_emo_probs"], key=w["text_emo_probs"].get)
    aa = max(w["audio_emo_probs"], key=w["audio_emo_probs"].get)
    rows.append(f'<hr><div style="opacity:.7">words: {ta} · voice: {aa}</div>')
    return f'<span class="wd-tip">{"".join(rows)}</span>'


def utterance_html(res: dict | None, th: dict, show_meta: bool = True) -> str:
    if not res or not res.get("words"):
        msg = (res or {}).get("status", "Nothing yet. Record a sentence to see it tagged.")
        return f'<div class="wd-utt wd-empty">{html.escape(msg)}</div>'
    utt, parts = res["utterance"], []
    for w, t in zip(res["words"], utterance_tags(res, th)):
        cls = ["wd-w"] + ([f"wd-e{t['emph']}"] if t["emph"] else []) + (["wd-sarc"] if t["sarc"] else [])
        color = EMO_COLORS[t["emo"]] if t["emo"] else "inherit"
        parts.append(f'<span class="{" ".join(cls)}" style="color:{color}">{html.escape(w["w"])}{_tooltip(w)}</span>')
    meta = ""
    if show_meta:
        tm = res["timing"]
        sarc = "sarcastic" if utt["sarcasm_p"] > th["sarcasm_threshold"] else "not sarcastic"
        meta = (f'<div class="wd-meta"><span>{utt["emotion"]} {utt["emo_probs"][utt["emotion"]]:.2f}</span>'
                f'<span>{sarc} ({utt["sarcasm_p"]:.2f})</span>'
                f'<span>{tm["audio_s"]:.1f}s audio, {tm["total"] * 1000:.0f} ms</span></div>')
    return f'<div class="wd-utt">{" ".join(parts)}{meta}</div>'


def corrected_html(results: list[dict]) -> str:
    """Rewriter output per utterance (newest first): corrected sentence + intended meaning."""
    rows = []
    for r in results:
        rw = r.get("rewrite") or {}
        if not rw:
            continue
        meaning = f'<div class="wd-mean">{html.escape(rw["meaning"])}</div>' if rw.get("meaning") else ""
        rows.append(f'<div class="wd-utt"><div class="wd-fix">{html.escape(rw["corrected"])}</div>{meaning}'
                    f'<div class="wd-meta"><span>rewritten in {rw["seconds"] * 1000:.0f} ms</span></div></div>')
    body = "".join(rows) or '<div class="wd-utt wd-empty">The corrected sentence and its meaning appear here.</div>'
    return f'<div class="wd-card"><div class="wd-label">What you meant</div>{body}</div>'


def legend_html() -> str:
    items = [f'<span><span class="wd-dot" style="background:{EMO_COLORS[e]}"></span>{e}</span>'
             for e in ["joy", "sadness", "anger", "fear", "disgust", "surprise"]]
    items += ['<span><b>bold</b> emphasis</span>', '<span class="wd-sarc">dotted</span>&nbsp;sarcasm',
              '<span>hover a word for details</span>']
    return f'<div class="wd-legend">{"".join(items)}</div>'


def transcript_html(results: list[dict], th: dict) -> str:
    body = "".join(utterance_html(r, th) for r in results) if results else utterance_html(None, th)
    return f'<div class="wd-card">{body}</div>'


def tagged_text(res: dict, th: dict) -> str:
    """Inline tags; consecutive words sharing an emotion are grouped into one span."""
    if not res or not res.get("words"):
        return ""
    ws, tags, out, i = res["words"], utterance_tags(res, th), [], 0
    while i < len(ws):
        emo, j = tags[i]["emo"], i
        while emo and j + 1 < len(ws) and tags[j + 1]["emo"] == emo:
            j += 1
        toks = []
        for k in range(i, j + 1):
            s = ws[k]["w"]
            if tags[k]["emph"]:
                s = f'<emph lvl={tags[k]["emph"]}>{s}</emph>'
            if tags[k]["sarc"]:
                s = f'<sarcasm p={res["utterance"]["sarcasm_p"]:.2f}>{s}</sarcasm>'.replace("p=0.", "p=.")
            toks.append(s)
        span = " ".join(toks)
        if emo:
            p = max(ws[k]["emo_p"] for k in range(i, j + 1))
            span = f"<{emo} p={p:.2f}>{span}</{emo}>".replace("p=0.", "p=.")
        out.append(span)
        i = j + 1
    return " ".join(out)
