#!/usr/bin/env python3
"""Real-time Russian ASR for a client-operator dialogue: Silero VAD + GigaAM-v3.

Pipeline: mic (or --test wav) -> VAD utterances -> GigaAM transcribe -> stdout
"[HH:MM:SS] text" lines (+ optional transcript file via --out). No speaker
diarization: role attribution is expected to be recovered downstream by the
LLM fact-extractor from the turn structure of the dialogue.

Segments are fed to the model as in-memory tensors (no temp files, no ffmpeg
subprocess per utterance like model.transcribe(path) would do).
"""
import argparse
import math
import os
import queue
import sys
import tempfile
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
from silero_vad import VADIterator, load_silero_vad

SAMPLE_RATE = 16000
BLOCK = 512                        # 32 ms per VAD chunk
MIN_LEN = int(0.15 * SAMPLE_RATE)  # utterances shorter than this are dropped
PREROLL = int(0.12 * SAMPLE_RATE)  # audio kept before a VAD "start"
PREROLL_BLOCKS = math.ceil(PREROLL / BLOCK)
TEST_TAIL_SEC = 0.6                # silence appended in --test mode so VAD closes the last utterance
WATCHDOG_SEC = 6
SILENCE_RMS_WARN = 3e-4
PREROLL_QUIET_RMS = 8e-3
LAG_WARN = 3                       # warn when this many segments await transcription
DEFAULT_REPO = "ai-sage/GigaAM-v3"

IDLE_MSG = "🎤  слушаю…  (Ctrl+C — выход)"
SPEECH_MSG = "🗣   речь…"
LONG_MSG = "🗣   речь… (длинный сегмент, отправлен)"
STOP_MSG = "Остановлено. Пока!"
MIC_WARN_MSG = (
    "⚠  Микрофон молчит (сигнал ≈ 0). Проверь доступ: Системные\n"
    "   настройки → Конфиденциальность и защита → Микрофон → Terminal,\n"
    "   и/или выбери другое устройство: ./run.sh --list-devices, --input N")

_STOP = object()  # worker shutdown sentinel


class Console:
    """Thread-safe single-line status + line output (ANSI clear-to-EOL)."""

    CLEAR = "\r\033[K"

    def __init__(self):
        self._lock = threading.Lock()

    def status(self, msg):
        with self._lock:
            sys.stdout.write(self.CLEAR + msg)
            sys.stdout.flush()

    def line(self, msg):
        with self._lock:
            sys.stdout.write(self.CLEAR + msg + "\n")
            sys.stdout.flush()


class Transcriber:
    """GigaAM-v3 transcription from an in-memory float32 waveform.

    Mirrors GigaAMASR.transcribe() internals but skips its file loading path
    (which spawns an ffmpeg subprocess and quantizes audio to int16). Falls
    back to a temp-file transcribe if the model exposes no decoding/head.
    """

    def __init__(self, model):
        self.model = model
        param = next(model.parameters())
        self.device = param.device
        self.dtype = param.dtype
        self._direct = hasattr(model, "decoding") and hasattr(model, "head")

    @torch.inference_mode()
    def __call__(self, wav):
        if self._direct:
            x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
            x = x.to(self.device, self.dtype).unsqueeze(0)
            length = torch.tensor([x.shape[-1]], device=self.device)
            encoded, encoded_len = self.model.forward(x, length)
            text = self.model.decoding.decode(
                self.model.head, encoded, encoded_len)[0]
            return text.strip()
        return self._via_file(wav)

    def _via_file(self, wav):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, wav, SAMPLE_RATE, subtype="FLOAT")
            return self.model.transcribe(path).strip()
        finally:
            os.remove(path)


class Feed:
    """VAD state machine: raw BLOCK-sized samples -> (timestamp, wav) jobs.

    process() takes ownership of the passed array slice (it must not be
    mutated afterwards); copies are made only when an utterance is emitted.
    """

    def __init__(self, vad, job_q, console, max_seg):
        self.vad = vad
        self.job_q = job_q
        self.console = console
        self.max_seg = max_seg
        self.buf = None       # list of blocks of the current utterance
        self.buf_len = 0      # total samples in self.buf (O(1) length check)
        self.preroll = deque((np.zeros(BLOCK, np.float32)
                              for _ in range(PREROLL_BLOCKS)),
                             maxlen=PREROLL_BLOCKS)
        self.peak = 0.0
        self.warned = False

    def _emit(self, seg):
        self.job_q.put((time.strftime("%H:%M:%S"), seg))

    def process(self, x):
        self.peak = max(self.peak, float(x.max()), -float(x.min()))
        ev = self.vad(torch.from_numpy(x)) or {}
        if "start" in ev:
            pre = np.concatenate(self.preroll)[-PREROLL:]
            quiet = float(np.sqrt((pre ** 2).mean())) < PREROLL_QUIET_RMS
            self.buf = [pre] if quiet else []
            self.buf_len = len(pre) if quiet else 0
            self.console.status(SPEECH_MSG)
        if self.buf is not None:
            self.buf.append(x)
            self.buf_len += len(x)
            if self.buf_len >= self.max_seg:
                self._emit(np.concatenate(self.buf))
                self.buf, self.buf_len = [], 0
                self.console.status(LONG_MSG)
        else:
            self.preroll.append(x)
        if "end" in ev:
            buf, self.buf = self.buf or [], None
            self.buf_len = 0
            if buf:
                self._emit(np.concatenate(buf))
            self.console.status(IDLE_MSG)

    def start_watchdog(self):
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self):
        while not self.warned:
            time.sleep(WATCHDOG_SEC)
            if self.peak < SILENCE_RMS_WARN and self.buf is None:
                self.warned = True
                self.console.line(MIC_WARN_MSG)


def asr_worker(job_q, transcribe, console, log=None, warn_lag=False):
    try:
        while True:
            job = job_q.get()
            if job is _STOP:
                job_q.task_done()
                return
            ts, seg = job
            try:
                if warn_lag and job_q.qsize() >= LAG_WARN:
                    console.line(f"⚠  ASR не успевает: {job_q.qsize()} сегментов в очереди")
                if len(seg) >= MIN_LEN:
                    text = transcribe(seg)
                    if text:
                        console.line(f"[{ts}] {text}")
                        if log:
                            log.write(f"{ts}\t{text}\n")
                            log.flush()
            except Exception as e:  # noqa: BLE001
                console.line(f"⚠  ошибка обработки сегмента: {e}")
            finally:
                job_q.task_done()
            console.status(IDLE_MSG)
    finally:
        if log:
            log.close()


def load_model(name, device, repo):
    print(f"Loading GigaAM-v3 '{name}' on {device} ...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(repo, revision=name, trust_remote_code=True)
    return model.to(device).eval()


def load_vad():
    print("Loading Silero VAD ...")
    try:
        return load_silero_vad(onnx=True)  # avoids deprecated torch.jit.load
    except Exception:  # noqa: BLE001
        return load_silero_vad()


def run_test(args, feed):
    wav, sr = sf.read(args.test, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
    wav = np.concatenate(
        [wav, np.zeros(int(TEST_TAIL_SEC * SAMPLE_RATE), np.float32)])
    for i in range(0, len(wav), BLOCK):
        x = wav[i:i + BLOCK]
        if len(x) < BLOCK:
            x = np.pad(x, (0, BLOCK - len(x)))
        feed.process(x)


def run_mic(args, feed, console):
    dev = args.input if args.input is not None else sd.default.device[0]
    if dev is None or int(dev) < 0:
        sys.exit("⚠  Не найдено устройство ввода: ./run.sh --list-devices, --input N")
    info = sd.query_devices(dev, "input")
    in_sr = int(round(info["default_samplerate"])) or SAMPLE_RATE
    print(f"🎙  Вход: [{dev}] {info['name']} @ {in_sr} Гц → 16 кГц")
    chunk = max(1, round(BLOCK * in_sr / SAMPLE_RATE))
    resampler = (torchaudio.transforms.Resample(in_sr, SAMPLE_RATE)
                 if in_sr != SAMPLE_RATE else None)
    pending = np.zeros(0, np.float32)
    feed.start_watchdog()
    try:
        with sd.InputStream(device=dev, samplerate=in_sr, channels=1,
                            dtype="float32", blocksize=chunk) as stream:
            while True:
                data, err = stream.read(chunk)
                if err:
                    raise RuntimeError(err)
                x = torch.from_numpy(data[:, 0].copy())
                if resampler is not None:
                    x = resampler(x)
                pending = np.concatenate([pending, x.numpy()])
                while len(pending) >= BLOCK:
                    feed.process(pending[:BLOCK])
                    pending = pending[BLOCK:]
    except KeyboardInterrupt:
        console.line(STOP_MSG)
    except sd.PortAudioError as e:
        console.line(f"Ошибка микрофона: {e}\n"
                     "Дайте доступ: Системные настройки → Конфиденциальность "
                     "→ Микрофон → Terminal")
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Silero VAD + GigaAM-v3 real-time ASR (client-operator dialogue)")
    p.add_argument("--model", default="e2e_rnnt",
                   choices=["e2e_rnnt", "e2e_ctc", "rnnt", "ctc"],
                   help="GigaAM-v3 variant (default e2e_rnnt)")
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"HuggingFace repo of the model (default {DEFAULT_REPO})")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--max-seg", type=float, default=12.0,
                   help="force-flush utterance after N seconds")
    p.add_argument("--silence-ms", type=int, default=200,
                   help="end-of-speech silence")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="VAD speech threshold")
    p.add_argument("--pad-ms", type=int, default=100)
    p.add_argument("--test", metavar="WAV",
                   help="simulate mic from a wav file (any sample rate)")
    p.add_argument("--input", type=int, default=None,
                   help="input device index (default: system default)")
    p.add_argument("--out", metavar="LOG",
                   help="append 'ts<TAB>text' transcript lines to LOG")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    console = Console()
    transcriber = Transcriber(load_model(args.model, device, args.repo))
    t0 = time.time()
    transcriber(np.zeros(SAMPLE_RATE, dtype=np.float32))
    print(f"Ready ({time.time() - t0:.1f}s warm-up).")

    vad = VADIterator(load_vad(), threshold=args.threshold,
                      sampling_rate=SAMPLE_RATE,
                      min_silence_duration_ms=args.silence_ms,
                      speech_pad_ms=args.pad_ms)

    log = open(args.out, "a", encoding="utf-8") if args.out else None
    job_q: queue.Queue = queue.Queue()
    worker = threading.Thread(target=asr_worker,
                              args=(job_q, transcriber, console, log),
                              kwargs=dict(warn_lag=not args.test),
                              daemon=True)
    worker.start()

    console.status(IDLE_MSG)
    feed = Feed(vad, job_q, console, int(args.max_seg * SAMPLE_RATE))
    try:
        if args.test:
            run_test(args, feed)
        else:
            run_mic(args, feed, console)
    finally:
        job_q.join()  # drain pending segments before exit
        job_q.put(_STOP)
        worker.join(timeout=15)


if __name__ == "__main__":
    main()
