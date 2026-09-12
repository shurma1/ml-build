# -*- coding: utf-8 -*-
"""Смена адреса gpu-шлюза без перезапуска процесса.

Задача практическая: арендованный под получает новый адрес при каждом
пересоздании. Перенастраивать сервер переменными окружения — значит ронять все
живые сессии ради одной строки.

Порядок действий строгий, и каждый шаг отвечает за свой класс ошибки:

  1. секрет сверяется `hmac.compare_digest` — сравнение за константное время.
     Обычное `==` на строках выходит из цикла на первом несовпавшем байте, и по
     времени ответа секрет подбирается посимвольно;
  2. `base_url` проверяется до любых сетевых обращений: только https (http лишь
     для localhost), без пути и параметров;
  3. `GET /readyz` — не ответил, значит 502 и НИЧЕГО не сохраняем. Записать
     мёртвый адрес хуже, чем отказать: следующий рестарт поднимется в никуда;
  4. `GET /v1/models` — оттуда берётся отпечаток эмбеддера;
  5. отпечаток сверяется с `meta['embed_fingerprint']`. Не совпал — 409 и
     не сохраняем: индекс собран одной моделью, запросы кодировались бы другой,
     и выдача осталась бы правдоподобной. Исключение — пустая meta: индекс ещё
     не строился, отпечаток записывается как эталонный;
  6. конфигурация ложится в `meta['gpu_gateway']` — в БАЗУ, а не в память:
     настройка обязана пережить перезапуск процесса;
  7. пулы HTTP-клиентов пересобираются на новый адрес на ходу;
  8. ответ: {ok, ready, models, latency_ms, fingerprint_match}.

Токен наружу не отдаётся никогда — ни в ответе, ни в логе: только первые
четыре символа и звёздочки.
"""
import hmac
import logging
import time
from collections import deque

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse

from .. import config as C
from .. import db
from ..errors import error_body
from ..gateway import gateway, GatewayConfigError, validate_base_url
from ..logging_conf import mask, remember_secret
from ..watchdog import watchdog

log = logging.getLogger("core.admin")

# include_in_schema=False: роут не должен светиться в публичной спецификации.
router = APIRouter(prefix="/v1/admin", tags=["admin"], include_in_schema=False)

META_KEY = "gpu_gateway"
FP_KEY = "embed_fingerprint"

_attempts = {}


def _client_ip(request: Request):
    """Адрес, по которому считается лимит попыток.

    X-Forwarded-For берётся только при явно включённом ADMIN_TRUST_PROXY:
    заголовок ставит кто угодно, и доверять ему по умолчанию значит отдать
    обход лимита любому, кто умеет менять строку в запросе.
    """
    if C.ADMIN_TRUST_PROXY:
        fwd = request.headers.get("x-forwarded-for") or ""
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _rate_limited(ip):
    """Не более 10 попыток в минуту с одного адреса."""
    now = time.monotonic()
    q = _attempts.setdefault(ip, deque())
    while q and now - q[0] > C.ADMIN_RATE_WINDOW:
        q.popleft()
    if len(q) >= C.ADMIN_RATE_LIMIT:
        return True
    q.append(now)
    return False


def _authorized(supplied):
    """hmac.compare_digest, а не ==: сравнение за константное время."""
    expected = C.admin_secret()
    a = (supplied or "").encode("utf-8")
    b = expected.encode("utf-8")
    return hmac.compare_digest(a, b)


def _guard(request, secret):
    ip = _client_ip(request)
    if _rate_limited(ip):
        log.warning("админ-роут: превышен лимит попыток с %s", ip)
        return JSONResponse(status_code=429,
                            content=error_body("rate_limited",
                                               f"не более {C.ADMIN_RATE_LIMIT} попыток в минуту",
                                               True))
    if not _authorized(secret):
        log.warning("админ-роут: неверный секрет с %s", ip)
        return JSONResponse(status_code=401,
                            content=error_body("unauthorized", "неверный X-Admin-Secret", False))
    return None


def _fingerprint_of(models):
    return ((models or {}).get("embed") or {}).get("fingerprint")


async def _stored_fingerprint():
    v = await db.meta_get(FP_KEY)
    if isinstance(v, dict):
        return v.get("fingerprint") or v.get("value")
    return v if isinstance(v, str) else None


@router.get("/gateway")
async def get_gateway(request: Request, x_admin_secret: str = Header(default="")):
    """Текущая конфигурация. Токен замаскирован."""
    bad = _guard(request, x_admin_secret)
    if bad:
        return bad
    stored = await db.meta_get(META_KEY) or {}
    return {
        "current": gateway.public({"status": watchdog.status,
                                   "mode": _mode(),
                                   "failures": watchdog.failures}),
        "stored": {"base_url": stored.get("base_url"),
                   "token": mask(stored.get("token")),
                   "fingerprint": stored.get("fingerprint"),
                   "updated_at": stored.get("updated_at")},
        "index_fingerprint": await _stored_fingerprint(),
    }


@router.post("/gateway")
async def set_gateway(body: dict, request: Request, x_admin_secret: str = Header(default="")):
    bad = _guard(request, x_admin_secret)
    if bad:
        return bad

    # 2. base_url — до любых сетевых обращений
    try:
        base_url = validate_base_url((body or {}).get("base_url"))
    except GatewayConfigError as e:
        return JSONResponse(status_code=400,
                            content=error_body("schema_validation_failed", str(e), False))
    token = (body or {}).get("token") or ""
    if not isinstance(token, str):
        return JSONResponse(status_code=400,
                            content=error_body("schema_validation_failed",
                                               "token должен быть строкой", False))

    # 3-4. живость и модели проверяются ПРОБНЫМ клиентом: пока проверка не прошла,
    #      рабочие пулы остаются на прежнем адресе.
    import httpx
    t0 = time.perf_counter()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with httpx.AsyncClient(base_url=base_url, headers=headers,
                                     timeout=C.GW_READY_TIMEOUT) as probe:
            r = await probe.get("/readyz")
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            if r.status_code != 200:
                return JSONResponse(status_code=502, content=error_body(
                    "gpu_unavailable",
                    f"{base_url}/readyz ответил {r.status_code} — адрес не сохранён", True))
            ready_body = _json_or_empty(r)
            rm = await probe.get("/v1/models")
            models = _json_or_empty(rm) if rm.status_code == 200 else {}
    except Exception as e:                                      # noqa: BLE001
        log.warning("админ-роут: %s не отвечает: %s", base_url, type(e).__name__)
        return JSONResponse(status_code=502, content=error_body(
            "gpu_unavailable",
            f"{base_url} не ответил на /readyz за {C.GW_READY_TIMEOUT} с — адрес не сохранён",
            True))

    # 5. сверка отпечатка
    fp = _fingerprint_of(models)
    stored_fp = await _stored_fingerprint()
    if stored_fp and not fp:
        # Отпечаток не прочитался — чаще всего шлюз отверг токен на /v1/models.
        # Сохранять в этом месте нельзя: неизвестный отпечаток это НЕ совпавший.
        # Разница видна только по выдаче, а она останется правдоподобной —
        # ровно тот случай, ради которого отпечаток и заведён.
        log.error("админ-роут: отпечаток эмбеддера не прочитан (/v1/models молчит "
                  "или отверг токен) при заданном эталоне %s — не сохраняем", stored_fp)
        return JSONResponse(status_code=409, content=error_body(
            "index_fingerprint_mismatch",
            "не удалось прочитать отпечаток эмбеддера у шлюза: /v1/models не ответил "
            "или отверг токен. Сверить с индексом нечего, адрес не сохранён", False))
    if stored_fp and fp != stored_fp:
        log.error("админ-роут: отпечаток эмбеддера чужой (%s вместо %s) — не сохраняем",
                  fp, stored_fp)
        return JSONResponse(status_code=409, content=error_body(
            "index_fingerprint_mismatch",
            f"эмбеддер шлюза ({fp}) не тот, которым собран индекс ({stored_fp}); "
            "адрес не сохранён", False))
    fingerprint_match = bool(stored_fp and fp == stored_fp)
    if not stored_fp:
        # Индекс ещё не строился — эталона нет и сверять не с чем.
        if fp:
            await db.meta_set(FP_KEY, {"fingerprint": fp, "source": "admin_gateway"})
            stored_fp, fingerprint_match = fp, True
            log.info("отпечаток эмбеддера записан как эталонный: %s", fp)
        else:
            log.warning("админ-роут: отпечаток не прочитан и эталона нет — "
                        "сохраняем адрес, но сверять при старте будет нечего")

    # 6. в базу, а не в память: настройка обязана пережить перезапуск
    record = {"base_url": base_url, "token": token, "fingerprint": fp,
              "updated_at": _now()}
    await db.meta_set(META_KEY, record)
    remember_secret(token)

    # 7. пересобрать пулы на ходу
    gateway.configure(base_url, token, fingerprint=fp, updated_at=record["updated_at"])
    watchdog.reset()
    await watchdog.check_once()
    _invalidate_cache(fp)

    log.info("админ-роут: шлюз переключён на %s (токен %s), отпечаток %s",
             base_url, mask(token), "совпал" if fingerprint_match else "принят как эталонный")

    # 8. ответ
    return {"ok": True, "ready": bool(ready_body.get("ready", True)),
            "models": _safe_models(models), "latency_ms": latency_ms,
            "fingerprint_match": fingerprint_match,
            "fingerprint": fp, "mode": _mode(),
            "gateway": gateway.public()}


@router.post("/gateway/check")
async def check_gateway(request: Request, x_admin_secret: str = Header(default="")):
    """Принудительная проверка живости — не дожидаясь очередного опроса сторожа."""
    bad = _guard(request, x_admin_secret)
    if bad:
        return bad
    if not gateway.configured:
        return JSONResponse(status_code=502, content=error_body(
            "gpu_unavailable", "адрес gpu-шлюза не настроен", True))
    t0 = time.perf_counter()
    ready = await watchdog.check_once()
    models = {}
    if ready:
        try:
            models = await gateway.models()
        except Exception as e:                                  # noqa: BLE001
            log.warning("проверка: /v1/models недоступен: %s", e)
    fp = _fingerprint_of(models)
    stored_fp = await _stored_fingerprint()
    return {"ok": bool(ready), "ready": bool(ready),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "models": _safe_models(models),
            "fingerprint": fp, "index_fingerprint": stored_fp,
            "fingerprint_match": bool(fp and stored_fp and fp == stored_fp),
            "status": watchdog.status, "mode": _mode(),
            "gateway": gateway.public()}


def _mode():
    from ..search_service import search_service
    return search_service.mode


def _invalidate_cache(fp):
    """Кэш векторов от прежней модели обесценивается вместе со сменой отпечатка."""
    if not fp:
        return
    try:
        from ..embed_client import GatewayEmbedder
        n = GatewayEmbedder(kind="query").invalidate(fp)
        if n:
            log.info("кэш векторов очищен от чужого отпечатка: %s строк", n)
    except Exception as e:                                      # noqa: BLE001
        log.warning("кэш векторов не очищен: %s", e)


def _json_or_empty(r):
    try:
        return r.json() or {}
    except Exception:                                           # noqa: BLE001
        return {}


def _safe_models(models):
    """Наружу — только версии. Ничего, что может оказаться секретом."""
    m = models or {}
    return {"asr": (m.get("asr") or {}), "embed": (m.get("embed") or {}),
            "llm": {k: v for k, v in (m.get("llm") or {}).items() if k != "api_key"},
            "prompt_version": m.get("prompt_version")}


def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


async def load_from_meta():
    """Поднять сохранённую конфигурацию при старте. Здесь же — bootstrap из
    окружения, если в базе ещё пусто: первый запуск должен с чего-то начаться."""
    stored = await db.meta_get(META_KEY) or {}
    base_url, token = stored.get("base_url"), stored.get("token") or ""
    source = "meta"
    if not base_url and C.GW_BASE_URL:
        base_url, token, source = C.GW_BASE_URL, C.GW_TOKEN, "окружение"
    if not base_url:
        log.warning("адрес gpu-шлюза не настроен: ни meta['gpu_gateway'], ни GW_BASE_URL. "
                    "Поиск поднимется в режиме degraded_lexical, настройте POST /v1/admin/gateway")
        return None
    try:
        gateway.configure(base_url, token, fingerprint=stored.get("fingerprint"),
                          updated_at=stored.get("updated_at"))
    except GatewayConfigError as e:
        log.error("сохранённый адрес шлюза негоден (%s): %s", base_url, e)
        return None
    log.info("адрес gpu-шлюза взят из источника «%s»: %s", source, base_url)
    return gateway
