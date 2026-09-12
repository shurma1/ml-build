# -*- coding: utf-8 -*-
"""ASR-воркер: Silero VAD на соединение + одна общая GigaAM-v3 на GPU.

Взято из sber ctc/realtime_asr.py и переложено на много соединений сразу.
Что сохранено дословно: конечный автомат VAD (preroll 120 мс, конец по 200 мс
тишины, принудительный сброс на 12 с) и подача сегмента в модель тензором,
без временных файлов и ffmpeg.

Что добавлено: сериализация GPU на одном потоке и отбрасывание сегментов,
устаревших в очереди. Пока сегмент ждал, посетитель уже сказал следующую фразу —
такой ответ никому не нужен, и лучше честно сказать «dropped», чем выдать
оператору реплику, которой он уже не ждёт.

Частичных гипотез нет и не планируется: GigaAM декодирует завершённый сегмент
целиком, а не по кадрам. Единица выдачи — реплика, закрытая VAD.
"""
import math
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from . import config as C

PREROLL_BLOCKS = math.ceil(C.PREROLL / C.BLOCK)


class Transcriber:
    """GigaAM-v3 из float32-волны в памяти.

    Повторяет внутренности GigaAMASR.transcribe(), минуя путь через файл:
    тот поднимает ffmpeg отдельным процессом на каждый сегмент и режет
    аудио до int16.
    """

    def __init__(self, model):
        self.model = model
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self._direct = hasattr(model, "decoding") and hasattr(model, "head")

    @torch.inference_mode()
    def __call__(self, wav):
        if not self._direct:                      # незнакомая сборка модели
            raise RuntimeError("модель без decoding/head — путь через тензор недоступен")
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
        x = x.to(self.device, self.dtype).unsqueeze(0)
        length = torch.tensor([x.shape[-1]], device=self.device)
        encoded, encoded_len = self.model.forward(x, length)
        return self.model.decoding.decode(self.model.head, encoded, encoded_len)[0].strip()


class VadStream:
    """Конечный автомат VAD одного соединения: блоки по 512 сэмплов -> реплики.

    Состояние VADIterator живёт внутри соединения: у каждого посетителя своя
    история тишины, общая на всех она была бы бессмысленной.
    """

    def __init__(self, vad_model, offset=0.0):
        from silero_vad import VADIterator
        self.it = VADIterator(vad_model, threshold=C.VAD_THRESHOLD,
                              sampling_rate=C.SAMPLE_RATE,
                              min_silence_duration_ms=C.VAD_SILENCE_MS,
                              speech_pad_ms=C.VAD_PAD_MS)
        self.max_seg = int(C.MAX_SEG_SEC * C.SAMPLE_RATE)
        self.pre = deque(maxlen=PREROLL_BLOCKS)
        self.buf: list = []
        self.active = False
        self.t = offset            # секунды от начала соединения
        self.seg_start = 0.0
        self._tail = np.zeros(0, np.float32)

    def feed(self, pcm: np.ndarray):
        """Принять произвольный кусок float32 и вернуть готовые реплики.

        -> [(t0, t1, wav), ...]. Кадры с провода приходят по 20-40 мс, шаг VAD —
        ровно 512 сэмплов, поэтому остаток переносится в следующий вызов.
        """
        out = []
        data = np.concatenate([self._tail, pcm]) if len(self._tail) else pcm
        n = (len(data) // C.BLOCK) * C.BLOCK
        self._tail = data[n:].copy()
        for i in range(0, n, C.BLOCK):
            seg = self._block(data[i:i + C.BLOCK])
            if seg:
                out.append(seg)
        return out

    def _block(self, x):
        self.t += C.BLOCK / C.SAMPLE_RATE
        ev = self.it(torch.from_numpy(x))
        if not self.active:
            self.pre.append(x.copy())
            if ev and "start" in ev:
                self.active = True
                self.buf = list(self.pre)
                self.seg_start = self.t - len(self.buf) * C.BLOCK / C.SAMPLE_RATE
                self.pre.clear()
            return None

        self.buf.append(x.copy())
        if ev and "end" in ev:
            return self._flush()
        if sum(len(b) for b in self.buf) >= self.max_seg:
            # Длинная реплика без паузы: отдаём как есть и продолжаем слушать.
            # Иначе монолог посетителя на минуту не доедет до оператора вовсе.
            seg = self._flush()
            self.it.reset_states()
            self.active = True
            self.buf = []
            self.seg_start = self.t
            return seg
        return None

    def _flush(self):
        wav = np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)
        t0, t1 = self.seg_start, self.t
        self.active, self.buf = False, []
        self.pre.clear()
        if len(wav) < C.MIN_LEN:
            return None
        return (round(t0, 2), round(t1, 2), wav)

    def close(self):
        try:
            self.it.reset_states()
        except Exception:
            pass


class AsrWorker:
    """Одна модель на процесс, один поток на GPU.

    GigaAM на карте всё равно исполняется по одному сегменту за раз, поэтому
    пул из одного потока — не ограничение, а честное признание факта: он
    убирает работу с event loop и даёт естественную сериализацию.
    """

    def __init__(self):
        self.model = None
        self.vad = None
        self.ready = False
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self._lock = threading.Lock()

    def load(self, device="cuda"):
        from transformers import AutoModel
        from silero_vad import load_silero_vad
        # revision, а НЕ subfolder: варианты GigaAM (e2e_rnnt, e2e_ctc, rnnt, ctc)
        # лежат в git-ревизиях репозитория, подкаталогов с такими именами нет.
        # Точно такой же вызов в sber ctc/realtime_asr.py — он рабочий, и менять
        # в нём что-либо по памяти не стоило.
        # Тип данных не навязываем: модель весит 0.4 ГБ, экономить нечего,
        # а fp16 в чужом remote-code — лишний риск на распознавании речи.
        model = AutoModel.from_pretrained(
            C.ASR_REPO, revision=C.ASR_VARIANT, trust_remote_code=True,
        ).to(device).eval()
        self.model = Transcriber(model)
        try:
            self.vad = load_silero_vad(onnx=True)
        except Exception:                      # ONNX-бэкенд необязателен
            self.vad = load_silero_vad()
        # Прогрев обязателен: первый вызов компилирует ядра и стоит десятки
        # секунд. Без него они достанутся первому живому посетителю.
        self.model(np.zeros(C.SAMPLE_RATE, dtype=np.float32))
        self.ready = True

    def stream(self, offset=0.0) -> VadStream:
        return VadStream(self.vad, offset)

    def transcribe(self, wav):
        t0 = time.perf_counter()
        with self._lock:
            text = self.model(wav)
        return text, round((time.perf_counter() - t0) * 1000)

    def info(self):
        return {"repo": C.ASR_REPO, "variant": C.ASR_VARIANT, "ready": self.ready}


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    """Кадр с провода -> волна. Формат зафиксирован контрактом: s16le, 16 кГц, моно.

    Ресемплинг сознательно оставлен клиенту: в браузере он бесплатен через
    AudioContext, а на GPU-машине это лишний CPU на каждое соединение.
    """
    if len(raw) % 2:
        raw = raw[:-1]
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
