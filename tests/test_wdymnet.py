"""
WDYMNet tests: word-span pooling, 50 Hz time→frame mapping, VAD on silence, JS-divergence bounds,
word muting, emotion2vec called with unpadded audio, LoRA trainable params, manifest schema,
plus model/loss edge cases (1-word utterances, missing labels).
Run: python -m pytest tests -q
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from wdymnet.features.speech import pool_frames_to_words, time_to_frame, word_frame_spans


# ---------------------------------------------------------------- pooling / frames
class TestFramePooling:
    def test_shape_and_values(self):
        frames = np.arange(100, dtype=np.float32)[:, None].repeat(4, 1)  # frame i has value i
        out = pool_frames_to_words(frames, [{"start": 0.0, "end": 0.1}, {"start": 0.5, "end": 1.0}])
        assert out.shape == (2, 8)
        assert out[0, 0] == pytest.approx(2.0)   # mean of frames 0..4
        assert out[0, 4] == pytest.approx(4.0)   # max of frames 0..4
        assert out[1, 0] == pytest.approx(37.0)  # mean of frames 25..49

    def test_subframe_word_takes_nearest_frame(self):
        frames = np.arange(10, dtype=np.float32)[:, None]
        assert word_frame_spans([{"start": 0.101, "end": 0.105}], 10) == [(5, 6)]
        out = pool_frames_to_words(frames, [{"start": 0.101, "end": 0.105}])
        assert out[0, 0] == 5.0 and not np.isnan(out).any()

    def test_word_at_clip_end(self):
        frames = np.ones((50, 3), dtype=np.float32)
        spans = word_frame_spans([{"start": 0.9, "end": 1.5}, {"start": 1.2, "end": 1.3}], 50)
        assert spans[0] == (45, 50)          # clipped to the last frame
        assert spans[1] == (49, 50)          # entirely past the end → last frame
        assert not np.isnan(pool_frames_to_words(frames, [{"start": 1.2, "end": 1.3}])).any()

    def test_empty(self):
        assert pool_frames_to_words(np.zeros((5, 3), np.float32), []).shape == (0, 6)

    def test_torch_input_is_differentiable(self):
        fr = torch.randn(20, 3, requires_grad=True)
        pool_frames_to_words(fr, [{"start": 0.0, "end": 0.2}]).sum().backward()
        assert fr.grad is not None


def test_time_to_frame_50hz():
    assert [time_to_frame(t) for t in (0.0, 0.019, 0.02, 0.5, 1.0, 0.06)] == [0, 0, 1, 25, 50, 3]


# ---------------------------------------------------------------- VAD
def test_vad_no_segments_on_silence():
    from silero_vad import load_silero_vad
    from wdymnet.pipeline import speech_segments
    vad = load_silero_vad()
    assert speech_segments(vad, np.zeros(16000 * 2, np.float32)) == []
    assert speech_segments(vad, np.zeros(100, np.float32)) == []


# ---------------------------------------------------------------- incongruity
class TestJSDivergence:
    def test_identical_is_zero(self):
        from wdymnet.models.fusion import WDYMNet
        x = torch.randn(2, 5, 7)
        assert torch.allclose(WDYMNet().compute_incongruity(x, x), torch.zeros(2, 5), atol=1e-5)

    def test_bounds(self):
        from wdymnet.models.fusion import WDYMNet
        m = WDYMNet()
        a, b = torch.zeros(1, 1, 7), torch.zeros(1, 1, 7)
        a[..., 0], b[..., 6] = 100.0, 100.0
        assert m.compute_incongruity(a, b).item() == pytest.approx(1.0, abs=1e-3)  # disjoint → 1
        r = m.compute_incongruity(torch.randn(8, 10, 7) * 5, torch.randn(8, 10, 7) * 5)
        assert (r >= 0).all() and (r <= 1).all()

    def test_fp16_autocast_no_nan(self):
        from wdymnet.models.fusion import WDYMNet
        a = torch.randn(1, 3, 7, dtype=torch.float16) * 30
        assert torch.isfinite(WDYMNet().compute_incongruity(a, -a)).all()


# ---------------------------------------------------------------- muting
def test_mute_edits_only_target_span():
    from wdymnet.eval.faithfulness import mute_words
    audio = np.ones(32000, dtype=np.float32)
    words = [{"start": 0.1, "end": 0.3}, {"start": 0.5, "end": 1.0}]
    out = mute_words(audio, 16000, words, [1])
    assert np.all(out[8000:16000] == 0)
    assert np.all(out[:8000] == 1) and np.all(out[16000:] == 1)
    assert np.all(audio == 1)  # input untouched


# ---------------------------------------------------------------- emotion2vec input
def test_emotion2vec_gets_unpadded_audio():
    from wdymnet.features.speech import Emotion2vecEncoder
    seen = {}
    enc = Emotion2vecEncoder.__new__(Emotion2vecEncoder)  # skip model loading
    torch.nn.Module.__init__(enc)
    enc.device = "cpu"
    enc.bottom = lambda wav: seen.setdefault("len", wav.shape[0]) and torch.zeros(12, 4)
    enc.top = lambda mid: torch.zeros(1, 2, 4)
    enc.scores = lambda fr: torch.zeros(9)
    audio = np.random.randn(12345).astype(np.float32)
    enc.encode(audio)
    assert seen["len"] == 12345  # exact length, no padding


# ---------------------------------------------------------------- LoRA
@pytest.mark.slow
def test_lora_trainable_and_base_frozen():
    from wdymnet.features.text import TextEncoder
    te = TextEncoder(device="cpu").add_lora()
    trainable = [n for n, p in te.named_parameters() if p.requires_grad]
    assert len(trainable) > 0 and all("lora_" in n for n in trainable)
    assert sum(p.numel() for n, p in te.named_parameters() if p.requires_grad) > 0
    out = te([["hello", "world's", "end"], ["hi"]])
    assert out.shape == (2, 3, 768) and torch.all(out[1, 1:] == 0)


# ---------------------------------------------------------------- manifests
class TestManifest:
    def test_schema_ok_and_bad(self):
        from wdymnet.data.manifest import validate_entry
        e = {"id": "x", "wav": "a.wav", "text": "hi there", "words": [{"w": "hi", "start": 0.0, "end": 0.2}],
             "labels": {"emotion": 0}, "speaker": "s", "split": "train", "dataset": "meld"}
        validate_entry(e)
        with pytest.raises(ValueError):
            validate_entry({**e, "split": "dev"})
        with pytest.raises(ValueError):
            validate_entry({k: v for k, v in e.items() if k != "words"})
        with pytest.raises(ValueError):
            validate_entry({**e, "words": [{"w": "hi", "start": 0.5, "end": 0.2}]})

    @pytest.mark.parametrize("name", ["meld", "mustardpp", "hpc"])
    def test_real_manifests(self, name):
        from wdymnet.data.manifest import read_manifest, validate_entry
        p = Path("data/manifests") / f"{name}.jsonl"
        if not p.exists():
            pytest.skip("run S1 first")
        rows = read_manifest(p)
        for e in rows[:500]:
            validate_entry(e)
        assert {e["split"] for e in rows} == {"train", "val", "test"}

    def test_split_words(self):
        from wdymnet.data.manifest import split_words
        assert split_words("Oh my God, he’s lost it - ...") == ["Oh", "my", "God,", "he's", "lost", "it"]


def test_mustardpp_split_is_speaker_disjoint():
    import sys
    sys.path.insert(0, "scripts")
    from s1_prepare_mustardpp import speaker_grouped_split
    spk = ["a"] * 50 + ["b"] * 30 + ["c"] * 10 + ["d"] * 5 + ["e"] * 5
    sp = speaker_grouped_split(spk)
    for s in set(spk):
        assert len({x for x, y in zip(sp, spk) if y == s}) == 1


# ---------------------------------------------------------------- model / loss
class TestModel:
    def test_output_shapes_and_one_word(self):
        from wdymnet.models.fusion import WDYMNet
        m = WDYMNet().eval()
        for B, N in [(2, 5), (1, 1)]:
            out = m(torch.randn(B, N, 1536), torch.randn(B, N, 768), torch.randn(B, N, 8), torch.ones(B, N, dtype=torch.bool))
            assert out["emph_logits"].shape == (B, N, 3) and out["emo_utt_logits"].shape == (B, 7)
            assert out["sarc_utt_logits"].shape == (B, 2) and out["incongruity"].shape == (B, N)
            assert torch.isfinite(out["emo_utt_logits"]).all()

    def test_padding_does_not_change_outputs(self):
        from wdymnet.models.fusion import WDYMNet
        m = WDYMNet().eval()
        a, t, p = torch.randn(1, 3, 1536), torch.randn(1, 3, 768), torch.randn(1, 3, 8)
        o1 = m(a, t, p, torch.ones(1, 3, dtype=torch.bool))
        pad = lambda x: torch.cat([x, torch.randn(1, 2, x.shape[-1])], 1)
        o2 = m(pad(a), pad(t), pad(p), torch.tensor([[True] * 3 + [False] * 2]))
        assert torch.allclose(o1["emo_utt_logits"], o2["emo_utt_logits"], atol=1e-4)

    def test_loss_skips_missing_heads(self):
        from wdymnet.losses import WDYMNetLoss
        from wdymnet.models.fusion import WDYMNet
        out = WDYMNet()(torch.randn(2, 4, 1536), torch.randn(2, 4, 768), torch.randn(2, 4, 8))
        loss, d = WDYMNetLoss(device="cpu")(out, {"emph_labels": torch.tensor([[0, 1, -1, 2], [2, -1, -1, -1]])})
        assert loss.requires_grad and "emph" in d and "emo" not in d and "sarc" not in d


def test_sparsity_loss_bounded_with_padding_and_one_word():
    from wdymnet.losses import WDYMNetLoss
    lf = WDYMNetLoss(device="cpu")
    mask = torch.tensor([[True, False, False, False], [True, True, True, True]])
    attn = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]], requires_grad=True)
    loss = lf.sparsity_loss(attn, mask)
    assert loss.item() == pytest.approx(0.5)  # 1-word → 0, uniform over 4 → 1
    loss.backward()
    assert torch.isfinite(attn.grad).all() and attn.grad.abs().max() < 10


def test_e2v_to_meld_mapping():
    from wdymnet.models.baselines import E2V_TO_MELD, map_e2v_scores_to_meld
    assert sorted(E2V_TO_MELD.values()) == list(range(7))
    m = map_e2v_scores_to_meld([0.1, 0, 0, 0.3, 0.4, 0.05, 0.1, 0.05, 0])
    assert len(m) == 7 and sum(m) == pytest.approx(1.0) and int(np.argmax(m)) == 0  # neutral


def test_prosody_edge_cases():
    from wdymnet.features.prosody import PROSODY_DIM, extract_word_prosody
    audio = (np.sin(np.linspace(0, 400 * np.pi, 32000)) * 0.3).astype(np.float32)
    f = extract_word_prosody(audio, 16000, [{"start": 0.0, "end": 0.5}, {"start": 0.6, "end": 1.0}, {"start": 1.9, "end": 2.5}])
    assert f.shape == (3, PROSODY_DIM) and np.isfinite(f).all()
    assert extract_word_prosody(audio, 16000, []).shape == (0, PROSODY_DIM)
    assert np.all(extract_word_prosody(audio, 16000, [{"start": 0.1, "end": 0.4}]) == 0)  # single word


def test_render_tagged_text():
    from wdymnet.live.render import tagged_text, utterance_tags
    from wdymnet.pipeline import EMOTIONS
    def w(word, emo=None, emph=0, attn=0.1):
        probs = {e: 0.0 for e in EMOTIONS}
        probs["neutral"] = 0.2 if emo else 0.8
        return {"w": word, "emph": emph, "emo": emo or "joy", "emo_p": 0.66 if emo else 0.1, "sarc_p": 0.6,
                "sarc_attn": attn, "emo_probs": probs}
    res = {"utterance": {"sarcasm_p": 0.81},
           "words": [w("I"), w("love", emph=2, attn=0.6), w("waiting"), w("in", "anger"), w("queues", "anger")]}
    th = {"emotion_threshold": 0.3, "sarcasm_threshold": 0.5}
    assert tagged_text(res, th) == "I <sarcasm p=.81><emph lvl=2>love</emph></sarcasm> waiting <anger p=.66>in queues</anger>"
    # non-sarcastic utterance → no sarcasm marks at all, however high the word scores
    res["utterance"]["sarcasm_p"] = 0.3
    assert not any(t["sarc"] for t in utterance_tags(res, th))
    # an emotion that does not beat the word's own neutral prob is not shown
    res["words"][3]["emo_probs"]["neutral"] = 0.9
    assert utterance_tags(res, th)[3]["emo"] is None


def test_rewrite_cues_follow_display_tags():
    from wdymnet.pipeline import EMOTIONS
    from wdymnet.rewrite import build_cues
    def w(word, emph=0, attn=0.1, asr=0.9):
        probs = {e: 0.0 for e in EMOTIONS}; probs["neutral"] = 0.9
        return {"w": word, "emph": emph, "emph_probs": [0.1, 0.2, 0.7] if emph == 2 else [0.8, 0.1, 0.1],
                "emo": "joy", "emo_p": 0.05, "sarc_p": 0.5, "sarc_attn": attn, "emo_probs": probs, "asr_p": asr}
    res = {"utterance": {"emotion": "anger", "emo_probs": {**{e: 0.0 for e in EMOTIONS}, "anger": 0.7}, "sarcasm_p": 0.99},
           "words": [w("I"), w("love", emph=2, attn=0.7), w("queues", asr=0.2)]}
    cues = build_cues(res, {"emotion_threshold": 0.3, "sarcasm_threshold": 0.9})
    assert '- overall tone: anger (strong)' in cues
    assert 'sarcasm: detected (strong), on: "love"' in cues
    assert '- emphasised words: "love"' in cues and '- possibly misheard words: "queues"' in cues
    res["utterance"]["sarcasm_p"] = 0.2
    assert "sarcasm: not detected" in build_cues(res, {"emotion_threshold": 0.3, "sarcasm_threshold": 0.9})
