"""
Continuous mode: sounddevice mic (16 kHz mono) → streaming Silero VAD → finished speech
segments are analysed (tagged + rewritten) in a worker thread; the UI polls `results`.
"""
import queue
import threading

import numpy as np

SR = 16000
CHUNK = 512  # silero streaming window at 16 kHz


class MicStream:
    def __init__(self, analyze, vad_cfg: dict, max_segment_s: float = 10.0, post=None):
        # analyze(audio, sr) → result (shown at once); post(result) adds slower extras in place (the rewrite)
        self.analyze, self.post, self.vad_cfg, self.max_len = analyze, post, vad_cfg, int(max_segment_s * SR)
        self.results: list[dict] = []
        self.status = "idle"
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._threads = [threading.Thread(target=self._capture, daemon=True),
                         threading.Thread(target=self._worker, daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        self.status = "stopped"

    def _capture(self):
        import sounddevice as sd
        import torch
        from silero_vad import VADIterator, load_silero_vad
        vad = VADIterator(load_silero_vad(), threshold=self.vad_cfg.get("threshold", 0.5), sampling_rate=SR,
                          min_silence_duration_ms=self.vad_cfg.get("min_silence_duration_ms", 300), speech_pad_ms=100)
        pre_roll = np.zeros(0, dtype=np.float32)
        seg, in_speech = [], False
        try:
            with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=CHUNK) as st:
                self.status = "listening"
                while not self._stop.is_set():
                    chunk, _ = st.read(CHUNK)
                    chunk = np.clip(chunk[:, 0], -1, 1)
                    ev = vad(torch.from_numpy(chunk))
                    if ev and "start" in ev:
                        in_speech, seg = True, [pre_roll]
                        self.status = "hearing speech…"
                    if in_speech:
                        seg.append(chunk)
                    else:
                        pre_roll = np.concatenate([pre_roll, chunk])[-int(0.3 * SR):]
                    seg_len = sum(len(s) for s in seg)
                    if in_speech and ((ev and "end" in ev) or seg_len >= self.max_len):
                        self._q.put(np.concatenate(seg))
                        in_speech, seg = False, []
                        vad.reset_states()
                        self.status = "listening"
        except Exception as e:  # no mic / device busy
            self.status = f"mic error: {e}"
            self._stop.set()

    def _worker(self):
        while not self._stop.is_set() or not self._q.empty():
            try:
                audio = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            res = self.analyze(audio, SR)
            if res["words"]:
                self.results.append(res)
                if self.post:
                    self.post(res)
