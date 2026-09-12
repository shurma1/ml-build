# -*- coding: utf-8 -*-
"""Сетевой эмбеддер: то единственное, ради чего правится embending/v2/dialog.py.

Раньше `DialogSearch` держал `SentenceTransformer` в своём процессе. Теперь модель
живёт на арендованной видеокарте, и её заменяет HTTP-вызов `POST /v1/embed`.

Три вещи, которые здесь важнее самого запроса.

1. **Префикс не добавляется.** Его ставит шлюз по полю `kind`. Если добавить его
   ещё и тут, запрос закодируется с двойным префиксом, а индекс — с одинарным;
   выдача останется правдоподобной, и никто ничего не заметит.
   Цена рассогласования перемерена на нынешней конфигурации и невелика: двойной префикс -0.7 п.п. R@1, отсутствие префикса -0.9, чужой формат -1.0 — все три в пределах доверительного интервала. Стоявшая здесь цифра -19 п.п. относится к другому замеру (Giga-480M на плоском индексе из 732 карточек, embending/RESULTS.md), а не к mE5-large-instruct на 321 типе. Инвариант это не отменяет: он стоит дёшево, а рассогласование по выдаче не видно.

2. **Кэш переживает шлюз.** Эмбеддер один и живёт на машине, которую могут
   вытеснить. Намерения в диалогах повторяются, поэтому вектор кладётся в таблицу
   `query_cache`: тёплый запрос это 7-9 мс против 120 мс холодного, и после
   пересоздания пода кэш остаётся.

3. **Кэш привязан к отпечатку.** Строки с чужим `fingerprint` не читаются: вектор
   от другой модели в кэше — это та же тихая порча выдачи, что и потерянный префикс.

Синхронный по необходимости: вызывается из потока, где работает синхронный
`DialogSearch`. Обращения к базе — короткие, через отдельный однопоточный пул.
"""
import hashlib
import logging
import threading
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

log = logging.getLogger("core.embed")


class EmbedUnavailable(RuntimeError):
    """Шлюз не ответил. Вызывающий обязан уйти в лексический режим, а не в 500."""


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).digest()


def as_vector(v):
    """pgvector отдаёт свой тип Vector, numpy его сам не разбирает."""
    if hasattr(v, "to_numpy"):
        return v.to_numpy().astype(np.float32)
    if hasattr(v, "to_list"):
        return np.asarray(v.to_list(), dtype=np.float32)
    return np.asarray(v, dtype=np.float32)


class _CacheConn:
    """Ленивое соединение к базе на поток. Кэш — это оптимизация: любая его
    ошибка гасится и логируется, поиск от этого падать не должен."""

    def __init__(self, dsn):
        self.dsn = dsn
        self._local = threading.local()

    def get(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None and not conn.closed:
            return conn
        conn = psycopg.connect(self.dsn, autocommit=True)
        register_vector(conn)
        self._local.conn = conn
        return conn

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None and not conn.closed:
            conn.close()
        self._local.conn = None


class GatewayEmbedder:
    """Замена SentenceTransformer в DialogSearch. Интерфейс — один метод encode()."""

    def __init__(self, gateway=None, dsn=None, kind="query", use_cache=True):
        if gateway is None:
            from .gateway import gateway as shared
            gateway = shared
        if dsn is None:
            from . import config as C
            dsn = C.DB
        self.gateway = gateway
        if not gateway.configured:
            # Отдельный запуск v2 (eval/acceptance.py) — конфигурации из meta нет,
            # поднимаемся из окружения. В сервере эта ветка не срабатывает.
            gateway.ensure_from_env()
        elif gateway.fingerprint is None:
            gateway.probe_fingerprint_sync()
        self.kind = kind
        self.use_cache = use_cache
        self._db = _CacheConn(dsn) if use_cache else None
        self.stats = {"gateway": 0, "cache_hit": 0, "texts": 0, "ms": 0.0}

    # --- кэш ---------------------------------------------------------------

    def _cache_read(self, texts, fingerprint):
        if not self.use_cache or not fingerprint:
            return {}
        try:
            conn = self._db.get()
            shas = [text_sha(t) for t in texts]
            rows = conn.execute(
                "SELECT text_sha, embedding FROM query_cache "
                "WHERE text_sha = ANY(%s) AND fingerprint = %s", (shas, fingerprint)).fetchall()
            by_sha = {bytes(r[0]): as_vector(r[1]) for r in rows}
            hit = {t: by_sha[s] for t, s in zip(texts, shas) if s in by_sha}
            if hit:
                conn.execute(
                    "UPDATE query_cache SET hits = hits + 1, last_seen = now() "
                    "WHERE text_sha = ANY(%s)", ([text_sha(t) for t in hit],))
            return hit
        except Exception as e:                                  # noqa: BLE001
            log.warning("кэш векторов недоступен на чтении: %s", e)
            return {}

    def _cache_write(self, pairs, fingerprint):
        if not self.use_cache or not fingerprint or not pairs:
            return
        try:
            conn = self._db.get()
            conn.cursor().executemany(
                "INSERT INTO query_cache (text_sha, text, fingerprint, embedding) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (text_sha) DO UPDATE SET "
                "embedding = EXCLUDED.embedding, fingerprint = EXCLUDED.fingerprint, "
                "last_seen = now()",
                [(text_sha(t), t[:4000], fingerprint, np.asarray(v, dtype=np.float32))
                 for t, v in pairs])
        except Exception as e:                                  # noqa: BLE001
            log.warning("кэш векторов недоступен на записи: %s", e)

    def invalidate(self, fingerprint=None):
        """Смена отпечатка обесценивает кэш целиком — иначе в нём останутся
        векторы от другой модели, а это молчаливая порча выдачи."""
        if not self.use_cache:
            return 0
        try:
            conn = self._db.get()
            if fingerprint:
                cur = conn.execute("DELETE FROM query_cache WHERE fingerprint <> %s",
                                   (fingerprint,))
            else:
                cur = conn.execute("DELETE FROM query_cache")
            return cur.rowcount
        except Exception as e:                                  # noqa: BLE001
            log.warning("кэш векторов не очищен: %s", e)
            return 0

    # --- собственно кодирование --------------------------------------------

    def encode(self, texts):
        """-> список np.ndarray в порядке texts. Бросает EmbedUnavailable, если шлюз молчит."""
        texts = list(texts)
        if not texts:
            return []
        t0 = time.perf_counter()
        fp = self.gateway.fingerprint
        cached = self._cache_read(texts, fp)
        miss = [t for t in texts if t not in cached]

        fresh = {}
        if miss:
            # Один батч на все промахи: пять строк стоят x1.23 от одной, а не x5.
            data = self._post(miss)
            vectors = data.get("vectors") or []
            got_fp = data.get("fingerprint")
            if len(vectors) != len(miss):
                raise EmbedUnavailable(
                    f"шлюз вернул {len(vectors)} векторов на {len(miss)} строк")
            if got_fp and got_fp != fp:
                # Шлюз сменился под нами — кэш от прежней модели больше не годится.
                log.warning("отпечаток эмбеддера изменился: %s -> %s", fp, got_fp)
                self.gateway.fingerprint = got_fp
                self.invalidate(got_fp)
                fp = got_fp
            fresh = {t: np.asarray(v, dtype=np.float32) for t, v in zip(miss, vectors)}
            self._cache_write(list(fresh.items()), fp)

        self.stats["cache_hit"] += len(cached)
        self.stats["texts"] += len(texts)
        self.stats["ms"] += (time.perf_counter() - t0) * 1000
        return [cached.get(t) if t in cached else fresh[t] for t in texts]

    def _post(self, texts):
        try:
            client = self.gateway.sclient()
        except Exception as e:                                  # noqa: BLE001
            raise EmbedUnavailable(f"адрес gpu-шлюза не настроен: {e}") from e
        try:
            r = client.post("/v1/embed", json={"texts": texts, "kind": self.kind})
        except Exception as e:                                  # noqa: BLE001
            self.stats["gateway"] += 1
            raise EmbedUnavailable(f"{type(e).__name__}: {e}") from e
        if r.status_code >= 400:
            try:
                err = (r.json() or {}).get("error") or {}
            except Exception:                                   # noqa: BLE001
                err = {}
            raise EmbedUnavailable(
                f"шлюз ответил {r.status_code}: {err.get('code') or ''} {err.get('message') or ''}".strip())
        body = r.json()
        self.stats["gateway"] += 1
        return body.get("data") or {}
