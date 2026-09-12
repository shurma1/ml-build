# -*- coding: utf-8 -*-
"""Ежесуточная чистка персональных данных.

`turn.text` и `fact_state.state` — расшифровка разговора с посетителем, то есть
персональные данные. Они живут до `session.purge_after` (30 дней по умолчанию)
и затем стираются.

`search_log` переживает чистку: прямой речи в нём нет — только признаки, ранги
и статусы. Так живой лог для метрики копится бессрочно, а сырая речь не хранится
дольше срока, на который получено согласие.

Строки `turn` не удаляются целиком: тогда развалилась бы нумерация реплик, по
которой `search_log` связан с ходом диалога. Затирается ровно текст.
"""
import asyncio
import logging

from . import config as C
from . import db

log = logging.getLogger("core.purge")


async def purge_once():
    expired = await db.fetch(
        "SELECT session_id FROM session WHERE purge_after < now() "
        "AND EXISTS (SELECT 1 FROM turn t WHERE t.session_id = session.session_id "
        "            AND t.text <> '')")
    ids = [r["session_id"] for r in expired]
    if not ids:
        return {"sessions": 0, "turns": 0, "states": 0}
    turns = await db.execute(
        "UPDATE turn SET text = '' WHERE session_id = ANY(%s) AND text <> ''", (ids,))
    states = await db.execute(
        "UPDATE fact_state SET state = '{}'::jsonb, changed = '{}'::jsonb "
        "WHERE session_id = ANY(%s) AND state <> '{}'::jsonb", (ids,))
    log.info("чистка ПДн: сессий %s, реплик %s, снимков состояния %s",
             len(ids), turns, states)
    return {"sessions": len(ids), "turns": turns, "states": states}


class Purger:
    def __init__(self):
        self.task = None
        self.last = None

    async def _loop(self):
        while True:
            try:
                self.last = await purge_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:                              # noqa: BLE001
                log.error("чистка ПДн не прошла: %s", e)
            await asyncio.sleep(C.PURGE_INTERVAL)

    def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self._loop())
        return self

    async def stop(self):
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None


purger = Purger()
