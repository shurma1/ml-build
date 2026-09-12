# -*- coding: utf-8 -*-
"""Карточка услуги и вопрос по её регламенту.

Карточка отдаётся ИСХОДНЫМИ полями выгрузки, а не пересказом: оператор
показывает посетителю дословную строку регламента, и смещения в `highlights`
адресуют именно эти поля.
"""
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..catalog import catalog
from ..errors import ApiError, error_body
from ..gateway import gateway
from ..reasons import humanize_reasons
from ..sessions import manager
from .. import highlights as H

log = logging.getLogger("core.routers.services")
router = APIRouter(prefix="/v1/services", tags=["services"])


@router.get("/{service_id}")
async def get_service(service_id: str, session_id: str = None):
    card = catalog.card_by_id.get(service_id)
    if not card:
        raise ApiError("service_not_found", f"услуга {service_id} не найдена")
    t = catalog.types.get(card["type_id"])
    fields = catalog.raw_fields(service_id)

    facts, status, reasons = {}, None, []
    if session_id:
        try:
            s = await manager.get(session_id)
        except ApiError:
            s = None
        if s is not None:
            d = s.state.as_features()
            facts = dict(d.get("facts") or {})
            if d.get("recipient"):
                facts["recipient"] = d["recipient"]
            # Оператор открыл карточку — это и есть неявная разметка релевантности.
            await manager.mark_opened(session_id, service_id)
            from eligibility import check as check_eligibility
            status, reasons = check_eligibility(t, facts, catalog.src.get(card["type_id"]))
            reasons = humanize_reasons(reasons)

    branches = await _branches(service_id)
    raw = catalog.raw.get(service_id) or {}
    return {
        "service_id": service_id, "type_id": card["type_id"],
        "title": card.get("raw_title") or card.get("title"),
        "municipality": card.get("municipality"),
        "department": card.get("department_full") or card.get("department"),
        # `fields` — сырые поля выгрузки как есть. Смещения в `highlights` и в
        # пунктах `view` считаются именно по ним, поэтому трогать их нельзя.
        "fields": fields,
        # Структурные поля выгрузки, которые не строки и поэтому в `fields`
        # не помещаются, а интерфейсу нужны: число отделений он печатает
        # в строке выдачи, виды заявителей — в «кому положено».
        "mfcCount": raw.get("mfcCount"),
        "recipientIds": list(raw.get("recipientIds") or []),
        "lifeSituationNames": list(raw.get("lifeSituationNames") or []),
        "departmentName": raw.get("departmentName"),
        # Разобранный регламент. Раньше это считал фронт по своим регуляркам
        # и расходился с этим же сервером; теперь источник один.
        "view": catalog.view(service_id),
        **catalog.structural(service_id),
        "status": status, "reasons": reasons,
        "highlights": H.for_type(t, fields, facts) if t else [],
        "branches": branches,
    }


@router.post("/{service_id}/ask")
async def ask_service(service_id: str, body: dict, request: Request):
    """Проксируется в SSE шлюза как есть. Ответ заземлён на одну карточку:
    всегда с цитатой из документа, без догадок и без отписок."""
    card = catalog.card_by_id.get(service_id)
    if not card:
        raise ApiError("service_not_found", f"услуга {service_id} не найдена")
    question = ((body or {}).get("question") or "").strip()
    if not question:
        raise ApiError("schema_validation_failed", "question обязателен")
    if len(question) > 1000:
        raise ApiError("schema_validation_failed", "question длиннее 1000 символов")
    history = (body or {}).get("history") or []
    if not isinstance(history, list) or len(history) > 20:
        raise ApiError("schema_validation_failed", "history: до 20 сообщений")

    document = _ask_document(service_id)
    if not gateway.configured:
        raise ApiError("gpu_unavailable", "gpu-шлюз недоступен: вопросы по документу выключены")

    async def proxy():
        try:
            async with gateway.ask_stream(document, question, history) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    payload = error_body("gpu_unavailable",
                                         f"шлюз ответил {resp.status_code}", True)
                    yield _sse("error", payload)
                    return
                async for chunk in resp.aiter_text():
                    if chunk:
                        yield chunk
        except Exception as e:                                  # noqa: BLE001
            log.warning("вопрос по карточке %s оборвался: %s", service_id, e)
            yield _sse("error", error_body("gpu_unavailable", str(e), True))

    return StreamingResponse(proxy(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _ask_document(service_id):
    """Регламент для модели = сырые текстовые поля + структурные факты карточки.

    Модель отвечает только о том, что ей дали.
    А карточка на экране печатает ведомство, число отделений и правовое
    основание: они берутся из структурных полей выгрузки (departmentName,
    mfcCount) и из разбора cardview, которых среди сырых текстов нет.
    Оператор своими глазами видит «124 отделения», спрашивает модель и получает
    отказ — и перестаёт верить ответам. Документ для модели и карточка на экране
    обязаны собираться из одних данных.

    Секции-добавки названы по-русски и отделены от сырых полей: смещения цитат
    считаются по тексту секции, а интерфейс печатает только текст цитаты,
    поэтому на подсветку это не влияет.
    """
    raw = catalog.raw.get(service_id) or {}
    card = catalog.card_by_id.get(service_id) or {}
    view = catalog.view(service_id) or {}
    doc = dict(catalog.raw_fields(service_id))
    if raw.get("departmentName"):
        doc["Ведомство"] = raw["departmentName"]
    if card.get("municipality"):
        doc["Муниципальное образование"] = card["municipality"]
    if raw.get("mfcCount") is not None:
        # Фразой, а не голым числом: измерено на этой же модели — «## Число
        # отделений МФЦ\n124» она отвечала «в регламенте не указано», а полное
        # предложение из того же числа цитирует без вопросов.
        doc["Число отделений МФЦ"] = (
            f"Количество отделений МФЦ, в которых можно получить услугу: "
            f"{raw['mfcCount']}.")
    life = raw.get("lifeSituationNames") or []
    if life:
        # Те же метки, что интерфейс печатает в карточке, и тот же ответ на
        # косвенный вопрос («а если потерял?»): слово «утрата» живёт не в
        # тексте регламента, а в классификаторе услуги. Без этой секции
        # регламент честно молчал о сценариях, для которых услугу и берут.
        doc["Жизненные ситуации услуги"] = (
            "Эта услуга относится к жизненным ситуациям: " + ", ".join(life) + ".")
    if view.get("legalBasis"):
        doc["Правовое основание"] = view["legalBasis"]
    return doc


async def _branches(service_id):
    from .. import db
    rows = await db.fetch(
        "SELECT b.branch_id, b.name FROM service_branch sb "
        "JOIN branch b ON b.branch_id = sb.branch_id WHERE sb.id = %s ORDER BY b.name",
        (service_id,))
    return [r["name"] for r in rows]
