# -*- coding: utf-8 -*-
"""Клиент gpu-шлюза с адресом, который меняется на ходу.

Зачем вообще: арендованный под получает новый адрес при каждом пересоздании.
Перенастраивать сервер переменными окружения с перезапуском — значит ронять все
живые сессии ради строки конфигурации. Поэтому адрес живёт в `meta['gpu_gateway']`
в базе, а здесь — держатель, который умеет пересобрать пулы соединений под новый
адрес, не останавливая процесс.

Пулов два, потому что потребителя два:
  * async (`httpx.AsyncClient`) — вся работа FastAPI: extract, ask, explain, readyz;
  * sync (`httpx.Client`) — эмбеддинг, который вызывается из потока, где крутится
    синхронный DialogSearch из embending/v2.
Оба пересобираются одной операцией и видят одну и ту же конфигурацию.

Токен наружу не отдаётся никогда: `public()` показывает `mask()`, а в логах его
вырезает фильтр из logging_conf.
"""
import asyncio
import logging
import threading
import time
from urllib.parse import urlsplit

import httpx

from . import config as C
from .errors import ApiError
from .logging_conf import mask, remember_secret

log = logging.getLogger("core.gateway")


class GatewayConfigError(ValueError):
    """base_url не прошёл проверку — сохранять нечего."""


def validate_base_url(raw):
    """Схема только https (http допустим для localhost), без пути и параметров.

    Путь и query запрещены не из педантизма: адрес склеивается с `/v1/embed`
    и подобным, и база вида `https://host/api?x=1` превратит запрос в мусор,
    который отвалится не здесь, а на первом поиске у живого посетителя.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise GatewayConfigError("base_url пуст")
    url = raw.strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise GatewayConfigError("схема должна быть https (или http для localhost)")
    if not parts.hostname:
        raise GatewayConfigError("в base_url нет хоста")
    local = parts.hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
    if parts.scheme == "http" and not local:
        raise GatewayConfigError("http разрешён только для localhost, иначе только https")
    if parts.path not in ("", "/"):
        raise GatewayConfigError("base_url не должен содержать путь")
    if parts.query or parts.fragment:
        raise GatewayConfigError("base_url не должен содержать параметров запроса")
    if parts.username or parts.password:
        raise GatewayConfigError("учётные данные в base_url недопустимы")
    return url


class Gateway:
    """Единственный держатель адреса, токена и пулов соединений к шлюзу."""

    def __init__(self):
        self.base_url = ""
        self.token = ""
        self.fingerprint = None          # отпечаток эмбеддера, известный по /v1/models
        self.updated_at = None
        self._async = None
        self._sync = None
        self._lock = threading.RLock()
        self._gen = 0                    # поколение конфигурации, для наблюдаемости

    # --- конфигурация -------------------------------------------------------

    @property
    def configured(self):
        return bool(self.base_url)

    def configure(self, base_url, token, fingerprint=None, updated_at=None):
        """Пересобрать пулы под новый адрес. Процесс не перезапускается.

        Старые клиенты закрываются после подмены, а не до: запрос, который уже
        летит по старому адресу, доживает свой таймаут и возвращает ответ,
        вместо того чтобы упасть с ошибкой закрытого пула.
        """
        url = validate_base_url(base_url)
        with self._lock:
            old_a, old_s = self._async, self._sync
            self.base_url = url
            self.token = token or ""
            self.fingerprint = fingerprint
            self.updated_at = updated_at
            self._gen += 1
            limits = httpx.Limits(max_connections=C.GW_MAX_CONNECTIONS,
                                  max_keepalive_connections=C.GW_MAX_CONNECTIONS)
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            self._async = httpx.AsyncClient(base_url=url, headers=headers, limits=limits,
                                            timeout=C.GW_TIMEOUT)
            self._sync = httpx.Client(base_url=url, headers=headers, limits=limits,
                                      timeout=C.GW_EMBED_TIMEOUT)
        remember_secret(self.token)
        _close_later(old_a, old_s)
        log.info("шлюз переключён на %s (токен %s, поколение %s)",
                 url, mask(self.token), self._gen)
        return self.public()

    def public(self, extra=None):
        """Наружу — без токена. Только первые 4 символа и звёздочки."""
        d = {"base_url": self.base_url or None, "token": mask(self.token),
             "fingerprint": self.fingerprint, "updated_at": self.updated_at,
             "generation": self._gen, "configured": self.configured}
        if extra:
            d.update(extra)
        return d

    # --- сырые клиенты ------------------------------------------------------

    def aclient(self):
        with self._lock:
            if self._async is None:
                raise ApiError("gpu_unavailable", "адрес gpu-шлюза не настроен")
            return self._async

    def sclient(self):
        with self._lock:
            if self._sync is None:
                raise ApiError("gpu_unavailable", "адрес gpu-шлюза не настроен")
            return self._sync

    async def aclose(self):
        with self._lock:
            a, s = self._async, self._sync
            self._async = self._sync = None
        if a is not None:
            await a.aclose()
        if s is not None:
            s.close()

    # --- вызовы -------------------------------------------------------------

    @staticmethod
    def _unwrap(resp):
        """Конверт шлюза: {"data": …, "meta": …} либо {"error": {code,message,retryable}}."""
        try:
            body = resp.json()
        except Exception:                                      # noqa: BLE001
            body = {}
        if resp.status_code >= 400:
            err = (body or {}).get("error") or {}
            code = err.get("code") or "internal"
            msg = err.get("message") or f"шлюз ответил {resp.status_code}"
            if code in ("not_ready", "oom", "internal", "rate_limited"):
                raise ApiError("gpu_unavailable", f"gpu-шлюз: {msg}", status=503)
            if code == "stale_segment":
                raise StaleSegment(msg)
            if code == "unauthorized":
                raise ApiError("gpu_unavailable", "gpu-шлюз отверг токен", status=503)
            raise ApiError("gpu_unavailable", f"gpu-шлюз: {msg}", status=503)
        return body.get("data", body), body.get("meta") or {}

    async def readyz(self, timeout=None):
        """-> (ready: bool, payload: dict). Исключений не бросает: это опрос."""
        t0 = time.perf_counter()
        try:
            r = await self.aclient().get("/readyz", timeout=timeout or C.GW_READY_TIMEOUT)
            payload = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            return r.status_code == 200, {**payload,
                                          "status": r.status_code,
                                          "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except ApiError:
            raise
        except Exception as e:                                 # noqa: BLE001
            return False, {"error": f"{type(e).__name__}: {e}",
                           "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    async def models(self):
        r = await self.aclient().get("/v1/models")
        if r.status_code >= 400:
            raise ApiError("gpu_unavailable", f"/v1/models ответил {r.status_code}")
        return r.json()

    async def embed(self, texts, kind="query"):
        """Префикс здесь НЕ добавляется — его ставит шлюз по полю kind.

        Это инвариант, а не стиль. Цена рассогласования перемерена на нынешней конфигурации и невелика: двойной префикс -0.7 п.п. R@1, отсутствие префикса -0.9, чужой формат -1.0 — все три в пределах доверительного интервала. Стоявшая здесь цифра -19 п.п. относится к другому замеру (Giga-480M на плоском индексе из 732 карточек, embending/RESULTS.md), а не к mE5-large-instruct на 321 типе. Инвариант это не отменяет: он стоит дёшево, а рассогласование по выдаче не видно.
        """
        r = await self.aclient().post("/v1/embed", json={"texts": list(texts), "kind": kind},
                                      timeout=C.GW_EMBED_TIMEOUT)
        data, meta = self._unwrap(r)
        return data, meta

    async def extract(self, session_id, turns):
        r = await self.aclient().post(
            "/v1/llm/extract", json={"session_id": str(session_id), "turns": turns})
        data, meta = self._unwrap(r)
        return data, meta

    async def explain(self, title, status, reasons, facts=None):
        """Проговорить ГОТОВЫЙ вердикт человеческим языком.

        Вердикт уходит на вход, модель его не выносит: рантайм-проверка права
        силами LLM измерена и отвергнута (14% самопротиворечий, 17% ложных
        отказов, 2.7 с на услугу). Привязка есть, потому что она часть контракта
        шлюза; публичного роута под неё в core-api.yaml пока не описано, и
        выдумывать его сверх контракта здесь незачем.
        """
        r = await self.aclient().post("/v1/llm/explain", json={
            "title": title, "status": status, "reasons": reasons, "facts": facts or {}})
        data, _ = self._unwrap(r)
        return data

    def ask_stream(self, document, question, history=None):
        """Контекстный менеджер стрима SSE от шлюза — проксируется наружу как есть."""
        return self.aclient().stream("POST", "/v1/llm/ask", json={
            "document": document, "question": question,
            "history": history or [], "stream": True}, timeout=None)

    def ensure_from_env(self):
        """Поднять конфигурацию из окружения, если её ещё нет.

        Нужно ровно для одного случая: `embending/v2` запускают отдельным
        скриптом (`eval/acceptance.py`, `run_eval.py`), без жизненного цикла
        FastAPI, который обычно читает адрес из `meta`. В работающем сервере
        этот путь не срабатывает — там конфигурация уже стоит из базы.
        """
        with self._lock:
            if self.configured:
                return self
        if not C.GW_BASE_URL:
            raise ApiError("gpu_unavailable",
                           "адрес gpu-шлюза не настроен: ни в meta, ни в GW_BASE_URL")
        self.configure(C.GW_BASE_URL, C.GW_TOKEN)
        self.probe_fingerprint_sync()
        return self

    def probe_fingerprint_sync(self):
        """Синхронно узнать отпечаток эмбеддера. Молча уходит ни с чем, если
        шлюз не ответил: отпечаток нужен кэшу, а не поиску."""
        try:
            r = self.sclient().get("/v1/models", timeout=C.GW_READY_TIMEOUT)
            if r.status_code == 200:
                fp = ((r.json() or {}).get("embed") or {}).get("fingerprint")
                if fp:
                    self.fingerprint = fp
                return fp
        except Exception as e:                                 # noqa: BLE001
            log.warning("отпечаток эмбеддера не получен: %s", e)
        return None



class StaleSegment(Exception):
    """Диалог ушёл вперёд, пока запрос стоял в очереди. Не ошибка — норма."""


def _close_later(aclient, sclient):
    """Старые пулы закрываются в фоне: запрос в полёте должен доиграть."""
    if sclient is not None:
        threading.Timer(C.GW_TIMEOUT, sclient.close).start()
    if aclient is None:
        return

    async def _shut():
        await asyncio.sleep(C.GW_TIMEOUT)
        try:
            await aclient.aclose()
        except Exception:                                      # noqa: BLE001
            pass

    try:
        asyncio.get_running_loop().create_task(_shut())
    except RuntimeError:
        pass


gateway = Gateway()
