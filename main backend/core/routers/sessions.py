# -*- coding: utf-8 -*-
"""Сессии: HTTP-ручки и WebSocket рабочего места оператора.

WebSocket делает две вещи одновременно: гонит кадры PCM вверх, в ASR шлюза,
и отдаёт события вниз. Поэтому внутри две независимые задачи — «наверх» и
«вниз» — и падение шлюза гасит только первую: оператор продолжает получать
`state.updated` и `results.updated` по репликам, введённым руками.
"""
import asyncio
import json
import logging

from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect

from .. import config as C
from .. import db
from ..catalog import catalog
from ..errors import ApiError, error_body
from ..events import hub, DEGRADED, ERROR, READY, degraded_payload
from ..gateway import gateway
from ..search_service import search_service
from ..sessions import STATE_FIELDS, manager

log = logging.getLogger("core.routers.sessions")
router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.post("", status_code=201)
async def create_session(body: dict, request: Request):
    branch_id = (body or {}).get("branch_id")
    operator_id = (body or {}).get("operator_id")
    if not branch_id or not operator_id:
        raise ApiError("schema_validation_failed", "нужны branch_id и operator_id")
    vad = (body or {}).get("vad_silence_ms")
    if vad is not None:
        lo, hi = C.ASR_SILENCE_LIMITS
        if not isinstance(vad, (int, float)) or isinstance(vad, bool) or not lo <= vad <= hi:
            raise ApiError("schema_validation_failed",
                           f"vad_silence_ms: целое от {lo} до {hi}")
        vad = int(vad)
    s = await manager.create(branch_id, operator_id, (body or {}).get("window"), vad)
    return {"session_id": s.id, "ws_url": _ws_url(request, s.id),
            "municipality": s.municipality,
            "corpus_version": await _corpus_version()}


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Полный снимок — для восстановления UI после обрыва WebSocket."""
    s = await manager.get(session_id)
    return {"session_id": s.id, "branch_id": s.branch_id, "operator_id": s.operator_id,
            "municipality": s.municipality,
            "started_at": s.started_at.isoformat() if hasattr(s.started_at, "isoformat")
            else s.started_at,
            "turns": [_turn(t) for t in s.turns], "state": s.render(),
            "search": s.last_search, "answers": [{"key": k, "value": v} for k, v in s.answers],
            "mode": search_service.mode}


@router.post("/{session_id}/turns")
async def add_turn(session_id: str, body: dict):
    """Ручной ввод: тот же конвейер, что у реплики от ASR, только source другой."""
    text = ((body or {}).get("text") or "").strip()
    if not text:
        raise ApiError("schema_validation_failed", "text обязателен")
    if len(text) > 2000:
        raise ApiError("schema_validation_failed", "text длиннее 2000 символов")
    speaker = (body or {}).get("speaker")
    if speaker not in (None, "client", "operator"):
        raise ApiError("schema_validation_failed", "speaker: client | operator | null")
    s = await manager.get(session_id)
    if s.closed:
        raise ApiError("session_not_found", "сессия уже закрыта")
    return await manager.add_turn(s, text, speaker=speaker, source="operator_typed")


@router.post("/{session_id}/answers")
async def answer_questions(session_id: str, body: dict):
    answers = (body or {}).get("answers")
    if not isinstance(answers, list) or not answers:
        raise ApiError("schema_validation_failed", "answers: непустой список {key, value}")
    for a in answers:
        if not isinstance(a, dict) or "key" not in a or "value" not in a:
            raise ApiError("schema_validation_failed", "каждый ответ — {key, value}")
        # `fact` необязателен и приходит из того же вопроса, что показали оператору.
        # Проверяем по белому списку полей состояния: клиент не должен уметь
        # записать ответом произвольный ключ.
        fact = a.get("fact")
        if fact is not None and fact not in STATE_FIELDS:
            raise ApiError("schema_validation_failed", f"fact: неизвестное поле {fact}")
    s = await manager.get(session_id)
    r = await manager.answer(s, answers)
    return r or {"results": [], "questions": [], "mode": search_service.mode}


@router.post("/{session_id}/facts")
async def set_facts(session_id: str, body: dict):
    setter = (body or {}).get("set") or {}
    unpin = (body or {}).get("unpin") or []
    if not isinstance(setter, dict) or not isinstance(unpin, list):
        raise ApiError("schema_validation_failed", "set — объект, unpin — список полей")
    dismiss = _fact_refs((body or {}).get("dismiss"), "dismiss")
    restore = _fact_refs((body or {}).get("restore"), "restore")
    clean = _validate_facts(setter)
    s = await manager.get(session_id)
    return await manager.set_facts(s, clean, unpin, dismiss, restore)


@router.post("/{session_id}/close", status_code=204)
async def close_session(session_id: str, body: dict = None):
    body = body or {}
    outcome = body.get("outcome")
    allowed = ("served", "not_found", "wrong_branch", "refused", "abandoned")
    if outcome is not None and outcome not in allowed:
        raise ApiError("schema_validation_failed", f"outcome: {' | '.join(allowed)}")
    chosen = body.get("chosen_service_id")
    if chosen and chosen not in catalog.card_by_id:
        raise ApiError("service_not_found", f"услуга {chosen} неизвестна")
    await manager.close(session_id, chosen, outcome, body.get("operator_note"))
    # Именно Response, а не JSONResponse(content=None): последний отдал бы тело
    # «null» при статусе 204, и uvicorn обрывает такое соединение.
    return Response(status_code=204)


# --- WebSocket --------------------------------------------------------------

def _ws_authorized(ws: WebSocket):
    """Проверка токена на WebSocket.

    Отдельная, потому что middleware на неё не распространяется: оно объявлено
    для scope «http», а рукопожатие приходит со scope «websocket». Без этой
    проверки HTTP оказывался закрыт, а поток событий — открыт всем, кто знает
    идентификатор сессии.

    Источника два. Заголовок ставит реверс-прокси — это рабочий путь, при нём
    токен вообще не попадает в браузер. Query-параметр остаётся для прямого
    подключения: заголовки браузерному WebSocket задать нечем. В логах он
    вырезается фильтром, но в истории и в прокси-логах чужого сервера
    он всё же оседает — поэтому путь через прокси предпочтительнее.
    """
    if not C.API_TOKEN:
        return True
    auth = ws.headers.get("authorization", "")
    if auth == f"Bearer {C.API_TOKEN}":
        return True
    return ws.query_params.get("token") == C.API_TOKEN


@router.websocket("/{session_id}/stream")
async def stream(ws: WebSocket, session_id: str):
    if not _ws_authorized(ws):
        log.warning("сессия %s: WebSocket без токена отклонён", session_id)
        await ws.close(code=4401)
        return
    try:
        s = await manager.get(session_id)
    except ApiError:
        await ws.close(code=4404)
        return
    await ws.accept()
    q = hub.subscribe(s.id)
    hub.emit(s.id, READY, {"corpus_version": await _corpus_version(),
                           "models": {"embed_fingerprint": gateway.fingerprint},
                           "mode": search_service.mode})
    if search_service.mode != "dense":
        hub.emit(s.id, DEGRADED, degraded_payload("gateway", search_service.mode))

    up = asyncio.create_task(_pump_up(ws, s))
    down = asyncio.create_task(_pump_down(ws, q))
    try:
        done, pending = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
    finally:
        hub.unsubscribe(s.id, q)
        for t in (up, down):
            t.cancel()
        try:
            await ws.close()
        except Exception:                                       # noqa: BLE001
            pass


async def _pump_down(ws, q):
    """События вниз. Одно событие — одно сообщение, порядок по `seq`."""
    while True:
        ev = await q.get()
        await ws.send_text(json.dumps(ev, ensure_ascii=False, default=str))


async def _pump_up(ws, s):
    """Кадры PCM наверх, в ASR шлюза. Соединение к шлюзу поднимается лениво:
    пока оператор не заговорил, незачем занимать канал на арендованной машине.

    Потеря шлюза здесь не роняет сессию — она гасит только голос. Оператор
    получает `degraded` и продолжает вводить реплики руками.
    """
    asr = None
    paused = False
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("text") is not None:
                cmd = _cmd(msg["text"])
                if cmd == "pause":
                    paused = True
                elif cmd == "resume":
                    paused = False
                elif cmd == "close":
                    return
                continue
            raw = msg.get("bytes")
            if not raw or paused:
                continue
            if asr is None:
                asr = await _open_asr(s)
                if asr is None:
                    paused = True        # голос выключен, ручной ввод работает
                    continue
            try:
                await asr.send(raw)
            except Exception as e:                              # noqa: BLE001
                log.warning("сессия %s: канал ASR оборвался: %s", s.id, e)
                hub.emit(s.id, DEGRADED, degraded_payload("asr", search_service.mode, e))
                asr = None
                paused = True
    except WebSocketDisconnect:
        return
    finally:
        if asr is not None:
            await asr.close()


class _AsrLink:
    """Мост к WS /v1/asr/stream шлюза. Реплика вниз -> тот же конвейер, что
    у ручного ввода: turn -> extract -> состояние -> поиск."""

    def __init__(self, ws, session, task):
        self.ws, self.session, self.task = ws, session, task

    async def send(self, data):
        await self.ws.send(data)

    async def close(self):
        self.task.cancel()
        try:
            await self.ws.close()
        except Exception:                                       # noqa: BLE001
            pass


async def _open_asr(s):
    import websockets
    if not gateway.configured:
        hub.emit(s.id, DEGRADED, degraded_payload("asr", search_service.mode,
                                                  "адрес gpu-шлюза не настроен"))
        return None
    url = gateway.base_url.replace("https://", "wss://").replace("http://", "ws://")
    url = f"{url}/v1/asr/stream?session_id={s.id}&sample_rate=16000&format=pcm_s16le"
    # Пауза VAD — на соединение, а не на весь шлюз: приёмы идут одновременно, и
    # общая настройка означала бы, что подкрутка под одного посетителя достаётся
    # всем окнам сразу.
    if s.vad_silence_ms:
        url += f"&silence_ms={s.vad_silence_ms}"
    if gateway.token:
        url += f"&token={gateway.token}"
    try:
        conn = await websockets.connect(url, max_size=None, open_timeout=10)
    except Exception as e:                                      # noqa: BLE001
        log.warning("сессия %s: ASR недоступен: %s", s.id, type(e).__name__)
        hub.emit(s.id, DEGRADED, degraded_payload("asr", search_service.mode, type(e).__name__))
        return None

    async def reader():
        try:
            async for raw in conn:
                try:
                    ev = json.loads(raw)
                except Exception:                               # noqa: BLE001
                    continue
                if ev.get("type") == "utterance" and (ev.get("text") or "").strip():
                    try:
                        await manager.add_turn(
                            s, ev["text"], speaker=ev.get("speaker"), source="asr",
                            t0=ev.get("t0"), t1=ev.get("t1"), asr_ms=ev.get("ms"))
                    except Exception as e:                      # noqa: BLE001
                        log.error("сессия %s: ход не обработан: %s", s.id, e)
                        hub.emit(s.id, ERROR, error_body("internal", str(e)))
                elif ev.get("type") == "ready":
                    # Шлюз сообщает, какие параметры VAD он в итоге применил.
                    log.info("сессия %s: ASR подключён, VAD=%s", s.id, ev.get("vad"))
                elif ev.get("type") == "error":
                    hub.emit(s.id, ERROR, error_body(
                        ev.get("code") or "internal", ev.get("message") or "ошибка ASR"))
        except asyncio.CancelledError:
            raise
        except Exception as e:                                  # noqa: BLE001
            log.info("сессия %s: чтение ASR завершено: %s", s.id, type(e).__name__)

    return _AsrLink(conn, s, asyncio.create_task(reader()))


# --- вспомогательное --------------------------------------------------------

def _cmd(text):
    try:
        return (json.loads(text) or {}).get("type")
    except Exception:                                           # noqa: BLE001
        return None


def _fact_refs(raw, name):
    """`[{key, value}]` — снять факт с работы или вернуть его обратно.

    Значение обязательно, и это не формальность: снимается конкретное значение,
    а не поле целиком. Иначе одно нажатие по «город Тула» глушило бы
    муниципалитет до конца приёма, и поправку клиента услышать было бы нечем.
    """
    if raw in (None, []):
        return []
    if not isinstance(raw, list):
        raise ApiError("schema_validation_failed", f"{name} — список объектов {{key, value}}")
    out = []
    for x in raw:
        key = x.get("key") if isinstance(x, dict) else None
        value = x.get("value") if isinstance(x, dict) else None
        if key not in STATE_FIELDS or value in (None, "", [], {}):
            raise ApiError("schema_validation_failed",
                           f"{name}: {{key, value}}, key из " + ", ".join(STATE_FIELDS))
        out.append({"key": key, "value": str(value)})
    return out


def _validate_facts(setter):
    """Ручная правка проверяется теми же enum'ами, что и извлечение из речи:
    оператор ошибается реже модели, но схема одна."""
    import sys
    if C.V2_PATH not in sys.path:
        sys.path.insert(0, C.V2_PATH)
    from extraction_schema import CATEGORIES, MUNICIPALITIES, RECIPIENTS, LIFE_SITUATIONS

    out = {}
    for k, v in (setter or {}).items():
        if k == "age":
            if v is None or (isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 120):
                out[k] = v
            else:
                raise ApiError("schema_validation_failed", "age: целое 0..120 или null")
        elif k == "children":
            if v is None or (isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 15):
                out[k] = v
            else:
                raise ApiError("schema_validation_failed", "children: целое 0..15 или null")
        elif k == "categories":
            if not isinstance(v, list) or any(x not in CATEGORIES for x in v):
                raise ApiError("schema_validation_failed",
                               "categories: из списка " + ", ".join(CATEGORIES))
            out[k] = sorted(set(v)) or None
        elif k == "recipient":
            if v is not None and v not in RECIPIENTS:
                raise ApiError("schema_validation_failed", "recipient: person | ip | organization")
            out[k] = v
        elif k == "municipality":
            if v is not None and v not in MUNICIPALITIES:
                from facets import find_municipality
                mo = find_municipality(v or "")
                if not mo:
                    raise ApiError("schema_validation_failed", f"муниципалитет «{v}» неизвестен")
                v = mo
            out[k] = v
        elif k == "life_situation":
            if v is not None and v not in LIFE_SITUATIONS:
                raise ApiError("schema_validation_failed",
                               "life_situation вне справочника жизненных ситуаций")
            out[k] = v
    return out


def _turn(t):
    return {"seq": t["seq"], "t0": t.get("t0"), "t1": t.get("t1"), "text": t.get("text"),
            "speaker": t.get("speaker"), "source": t.get("source"),
            "asr_ms": t.get("asr_ms"), "empty": bool(t.get("empty"))}


def _ws_url(request, session_id):
    """Адрес WebSocket для клиента.

    За реверс-прокси адрес сокета и адрес, по которому core-api себя видит, —
    разные вещи: сервер видит апстрим (127.0.0.1:8080), а браузеру нужен
    внешний адрес. Отдать первый значит послать клиента мимо прокси — то есть
    мимо подстановки токена, и соединение будет отклонено.

    Поэтому учитываются стандартные заголовки прокси. `PUBLIC_WS_BASE`
    остаётся ручным переопределением для схем, где заголовки не проставляются.
    """
    if C.PUBLIC_WS_BASE:
        return f"{C.PUBLIC_WS_BASE.rstrip('/')}/v1/sessions/{session_id}/stream"
    proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip()
    host = (request.headers.get("x-forwarded-host") or "").split(",")[0].strip()
    scheme = "wss" if (proto or request.url.scheme) == "https" else "ws"
    return f"{scheme}://{host or request.url.netloc}/v1/sessions/{session_id}/stream"


async def _corpus_version():
    v = dict(catalog.version or {})
    stored = await db.meta_get("corpus_version") or {}
    v.setdefault("built_at", stored.get("built_at"))
    return v
