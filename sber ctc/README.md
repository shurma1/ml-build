# Realtime ASR (русский)

Скрипт `realtime_asr.py`: речь с микрофона → текст в реальном времени.

## Стек

| Компонент | Технология | Зачем |
|---|---|---|
| VAD | Silero VAD (ONNX) | детект начала/конца реплики |
| ASR | GigaAM-v3 (ai-sage, RNNT/CTC) | распознавание речи |
| Аудио | sounddevice + soundfile | захват с микрофона, чтение wav |
| Inference | PyTorch (MPS/CPU) | запуск моделей |

## Как работает

```
микрофон (512 сэмплов / 32 мс, ресемпл в 16 кГц)
   → VAD-машина (preroll 120 мс, конец по тишине 200 мс, force-flush 12 с)
   → очередь задач (поток-воркер)
   → GigaAM напрямую тензором (без temp-файлов и ffmpeg)
   → "[HH:MM:SS] текст" в stdout (+ --out лог)
```

Таймстемп ставится в момент окончания реплики, воркер дренирует очередь при Ctrl+C.

## Запуск

- **macOS**, Python 3.11 в `.venv`, доступ к микрофону для Terminal.
- Apple Silicon (MPS) по умолчанию; `--device cpu` — Intel/CPU.

```bash
./run.sh                                  # микрофон
./run.sh --test "Test audio/output.wav"   # симуляция по файлу
./run.sh --list-devices                   # устройства → --input N
./run.sh --model e2e_ctc --out log.tsv    # CTC-вариант + запись транскрипта
```

Зависимости: `pip install -r requirements.txt`. Модель скачивается с HuggingFace при первом запуске.
