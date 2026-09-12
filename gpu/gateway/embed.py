# -*- coding: utf-8 -*-
"""Эмбеддер: одна модель, префикс на сервере, отпечаток конфигурации в ответе.

Два решения, которые здесь важнее самой модели.

1. Префикс `Instruct: ...\\nQuery: ` добавляется ЗДЕСЬ, по полю kind, а не
   вызывающим кодом. Если оставить это вызывающему, индекс и запрос однажды
   разъедутся, и поиск продолжит возвращать правдоподобные результаты —
   то есть никто не заметит. Цена потери префикса на этом корпусе измерена:
   -19 п.п. R@1.

2. Ответ несёт fingerprint — модель, длина, нормализация, хэш префиксов.
   Он пишется в meta при построении индекса и сверяется core-api при старте.
   Эмбеддер в системе один-единственный и живёт на арендованной машине:
   чужая модель после перезапуска инстанса обесценивает весь индекс молча.
"""
import hashlib
import threading
import time
from collections import OrderedDict

from . import config as C


def fingerprint() -> str:
    h = hashlib.sha256(
        f"{C.EMB_MODEL}|{C.EMB_MAX_SEQ}|norm|{C.QUERY_PREFIX}|{C.DOC_PREFIX}".encode()
    ).hexdigest()[:8]
    return f"{C.EMB_MODEL}/{C.EMB_MAX_SEQ}/norm/{h}"


class EmbedWorker:
    def __init__(self, cache_size=4096):
        self.model = None
        self.dim = None
        self.ready = False
        self._lock = threading.Lock()
        self._cache = OrderedDict()
        self._cap = cache_size
        self.hits = self.misses = 0

    def load(self, device="cuda"):
        import torch
        from sentence_transformers import SentenceTransformer
        # trust_remote_code нужен части моделей (Giga-Embeddings) и мешает другим
        trust = "bge" not in C.EMB_MODEL.lower()
        self.model = SentenceTransformer(
            C.EMB_MODEL, trust_remote_code=trust, device=device,
            model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {},
        )
        self.model.max_seq_length = C.EMB_MAX_SEQ
        v = self.model.encode(["прогрев"], normalize_embeddings=True, convert_to_numpy=True)
        self.dim = int(v.shape[1])
        self.ready = True

    def _prefix(self, kind):
        return C.QUERY_PREFIX if kind == "query" else C.DOC_PREFIX

    def encode(self, texts, kind):
        """-> (vectors, ms, cached). Батч почти бесплатен — отсюда правило
        «все намерения одного хода диалога уходят одним запросом»: пять строк
        стоят x1.23 от одной, а не x5."""
        t0 = time.perf_counter()
        pref = self._prefix(kind)
        keys = [f"{kind}\x00{t}" for t in texts]

        miss_idx = [i for i, k in enumerate(keys) if k not in self._cache]
        if miss_idx:
            batch = [pref + texts[i] for i in miss_idx]
            with self._lock:
                vecs = self.model.encode(batch, normalize_embeddings=True,
                                         convert_to_numpy=True, batch_size=len(batch))
            for i, v in zip(miss_idx, vecs):
                self._put(keys[i], v)
        self.misses += len(miss_idx)
        self.hits += len(keys) - len(miss_idx)

        out = []
        for k in keys:
            v = self._cache[k]
            self._cache.move_to_end(k)
            out.append([round(float(x), 6) for x in v])
        return out, round((time.perf_counter() - t0) * 1000, 1), len(keys) - len(miss_idx)

    def _put(self, key, vec):
        self._cache[key] = vec
        if len(self._cache) > self._cap:
            self._cache.popitem(last=False)

    def info(self):
        return {"model": C.EMB_MODEL, "dim": self.dim, "fingerprint": fingerprint(),
                "ready": self.ready, "cache": {"hits": self.hits, "misses": self.misses}}
