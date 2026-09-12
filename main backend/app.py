# -*- coding: utf-8 -*-
"""core-api: рабочее место оператора МФЦ.

Основной сервер без GPU. Владеет Postgres+pgvector, состоянием диалога и всей
бизнес-логикой; модели (ASR, эмбеддер, LLM) вызывает через gpu-шлюз по HTTP.

Два инварианта, которые задают всю форму этого сервиса:

  * **право проверяется ПОСЛЕ поиска, никогда вместо него.** Услуга, на которую
    клиент не имеет права, обязана быть найдена и показана со статусом `blocked`
    и дословной цитатой из регламента;
  * **LLM не выносит вердикт о праве**, а только проговаривает готовый. Вердикт
    считает детерминированная таблица предикатов.

Порядок подъёма важен: окружение проверяется ДО того, как открыт порт. Процесс
без `ADMIN_SECRET` не стартует вовсе — молчаливый дефолт у секрета, которым
закрыта смена адреса шлюза, недопустим.
"""
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from core import config as C                                     # noqa: E402
from core.logging_conf import setup as setup_logging, remember_secret  # noqa: E402

# --- проверка окружения до открытия порта ---
try:
    C.check_env()
except C.ConfigError as e:
    C.fail_fast(e)

setup_logging(C.LOG_LEVEL)
remember_secret(C.GW_TOKEN)
remember_secret(os.getenv("ADMIN_SECRET", ""))

from core import db                                              # noqa: E402
from core.app_state import state                                 # noqa: E402
from core.catalog import catalog                                 # noqa: E402
from core.errors import ApiError, error_body                     # noqa: E402
from core.events import hub, DEGRADED, degraded_payload         # noqa: E402
from core.gateway import gateway                                 # noqa: E402
from core.purge import purger                                    # noqa: E402
from core.routers import admin as admin_router                   # noqa: E402
from core.routers import reference, search, services, sessions    # noqa: E402
from core.search_service import search_service, LEXICAL           # noqa: E402
from core.watchdog import watchdog                                # noqa: E402

log = logging.getLogger("core.app")

# Пути, открытые без публичного токена: проверка живости и версии.
OPEN_PATHS = {"/v1/health", "/v1/version", "/health", "/docs", "/openapi.json", "/redoc"}
# Пути, которые обязаны работать даже при рассинхроне отпечатка — иначе чинить нечем.
MISMATCH_ALLOWED = OPEN_PATHS | {"/v1/admin/gateway", "/v1/admin/gateway/check", "/v1/branches"}


async def verify_fingerprint():
    """Сверка отпечатка при старте.

    Три исхода:
      * совпал — работаем;
      * не совпал — отказ обслуживать, 503 index_fingerprint_mismatch;
      * шлюз не ответил — это НЕ рассинхрон, а недоступность: уходим
        в degraded_lexical и продолжаем работать на лексике.
    """
    stored = await db.meta_get("embed_fingerprint")
    index_fp = stored.get("fingerprint") if isinstance(stored, dict) else stored
    state.index_fingerprint = index_fp
    if not gateway.configured:
        log.warning("шлюз не настроен — отпечаток не сверяется, режим %s", LEXICAL)
        search_service.set_mode(LEXICAL)
        return None
    try:
        models = await gateway.models()
    except Exception as e:                                       # noqa: BLE001
        log.warning("отпечаток не сверить, шлюз не ответил (%s) — режим %s", e, LEXICAL)
        search_service.set_mode(LEXICAL)
        return None
    gw_fp = ((models or {}).get("embed") or {}).get("fingerprint")
    state.gateway_fingerprint = gw_fp
    gateway.fingerprint = gw_fp
    if not index_fp:
        # Индекс строился до того, как завели отпечаток — фиксируем эталон.
        if gw_fp:
            await db.meta_set("embed_fingerprint", {"fingerprint": gw_fp, "source": "startup"})
            state.index_fingerprint = gw_fp
            log.info("отпечаток эмбеддера записан как эталонный: %s", gw_fp)
        return gw_fp
    if gw_fp and gw_fp != index_fp:
        state.set_mismatch(index_fp, gw_fp)
        return gw_fp
    state.clear_mismatch()
    log.info("отпечаток эмбеддера сошёлся: %s", index_fp)
    return gw_fp


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.started_at = time.time()
    await db.open_pool()
    await db.apply_runtime_schema()
    catalog.load()
    await admin_router.load_from_meta()
    await search_service.start()
    search_service.on_degraded(
        lambda detail: hub.broadcast(DEGRADED, degraded_payload("gateway", LEXICAL, detail)))
    await verify_fingerprint()
    if gateway.configured and not state.fingerprint_mismatch:
        await watchdog.check_once()
    watchdog.start()
    purger.start()
    log.info("core-api поднят: режим %s, корпус %s карточек / %s типов",
             search_service.mode, catalog.version.get("cards"), catalog.version.get("types"))
    try:
        yield
    finally:
        await watchdog.stop()
        await purger.stop()
        await search_service.stop()
        await gateway.aclose()
        await db.close_pool()


app = FastAPI(title="МФЦ Поиск — core-api", version=C.API_VERSION, lifespan=lifespan)


if C.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=C.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        # X-Admin-Secret нужен странице настройки шлюза: без него браузер
        # не пропустит даже предварительный запрос. Сам секрет в сборку фронта
        # не попадает — его вводит руками тот, кто меняет адрес.
        allow_headers=["Authorization", "Content-Type", "X-Request-Id", "X-Admin-Secret"],
        expose_headers=["X-Request-Id"],
    )


@app.middleware("http")
async def envelope(request: Request, call_next):
    rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
    path = request.url.path

    if C.API_TOKEN and path not in OPEN_PATHS and not path.startswith("/v1/admin"):
        if request.headers.get("authorization", "") != f"Bearer {C.API_TOKEN}":
            return JSONResponse(status_code=401,
                                content=error_body("unauthorized", "нужен Bearer-токен",
                                                   False, rid))
    # Рассинхрон отпечатка: обслуживание остановлено, чинить — через админ-роут.
    if state.fingerprint_mismatch and path not in MISMATCH_ALLOWED:
        return JSONResponse(status_code=503, content=error_body(
            "index_fingerprint_mismatch",
            state.fingerprint_mismatch, False, rid))

    try:
        resp = await call_next(request)
    except ApiError as e:
        return e.response(rid)
    except Exception as e:                                       # noqa: BLE001
        log.exception("необработанная ошибка на %s", path)
        return JSONResponse(status_code=500, content=error_body(
            "internal", f"{type(e).__name__}: {e}", True, rid))
    resp.headers["X-Request-Id"] = rid
    return resp


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return exc.response(request.headers.get("X-Request-Id"))


app.include_router(sessions.router)
app.include_router(search.router)
app.include_router(services.router)
app.include_router(reference.router)
app.include_router(admin_router.router)


@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "alive", "mode": search_service.mode,
            "uptime_s": round(time.time() - (state.started_at or time.time()))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("PORT", "8080")), log_level=C.LOG_LEVEL.lower())
