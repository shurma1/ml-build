# -*- coding: utf-8 -*-
"""Сквозная проверка ASR БЕЗ видеокарты: загрузка + VAD + распознавание.

Запускать ДО пересборки образа, а не после деплоя. Здесь ловятся почти все
отказы загрузки: они происходят при сборке модели на CPU, до переноса
на устройство. Один прогон на ноутбуке вскрыл три ошибки подряд, каждая
из которых иначе стоила бы полного круга «пересборка — деплой — падение».

    python3 -m venv /tmp/tv && /tmp/tv/bin/pip install \
        torch torchaudio transformers hydra-core omegaconf sentencepiece \
        pyannote.core silero-vad soundfile soxr onnxruntime numpy
    /tmp/tv/bin/python gpu/tools/e2e_asr.py

Проверяет не только то, что модель собралась, но и то, что весь конвейер
выдаёт осмысленный текст. Загрузиться без ошибки мало: частично собранная
модель выдаёт правдоподобный мусор.
"""
import os, sys, time
sys.path.insert(0, "/Users/dmitriy/Desktop/ML")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "hf"))

import numpy as np, soundfile as sf
from gpu.gateway.asr import AsrWorker
from gpu.gateway import config as C

WAV = "/Users/dmitriy/Desktop/ML/sber ctc/Test audio/output.wav"

print("загружаю GigaAM...", flush=True)
t0 = time.time()
w = AsrWorker()
w.load("cpu")
print(f"  загрузилась за {time.time()-t0:.1f} с, способ: {w.load_mode}", flush=True)
p = next(w.model.model.parameters())
print(f"  параметров {sum(x.numel() for x in w.model.model.parameters())/1e6:.0f}М, "
      f"устройство {p.device}, тип {p.dtype}", flush=True)

wav, sr = sf.read(WAV, dtype="float32", always_2d=False)
if wav.ndim > 1:
    wav = wav.mean(axis=1)
if sr != C.SAMPLE_RATE:
    import soxr
    wav = soxr.resample(wav, sr, C.SAMPLE_RATE)
wav = np.concatenate([wav, np.zeros(int(0.6 * C.SAMPLE_RATE), np.float32)]).astype(np.float32)
print(f"\nаудио: {len(wav)/C.SAMPLE_RATE:.1f} с, {sr} Гц -> 16000", flush=True)

st = w.stream()
segs = st.feed(wav)
print(f"VAD нарезал реплик: {len(segs)}\n", flush=True)

total = 0.0
for i, (a, b, seg) in enumerate(segs, 1):
    text, ms = w.transcribe(seg)
    total += ms
    print(f"  [{a:6.2f}–{b:6.2f}]  {ms:5d} мс  {text}", flush=True)

print(f"\nвсего распознавания {total/1000:.1f} с на {len(wav)/C.SAMPLE_RATE:.1f} с аудио")
