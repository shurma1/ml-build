# -*- coding: utf-8 -*-
"""Слежение за gpu-шлюзом: раз в 15 секунд GET /readyz.

Два отказа подряд — уходим в `degraded_lexical` и рассылаем `degraded` во все
живые сессии. Ответил снова — возвращаемся в `dense`.

Почему два, а не один: арендованный под живёт за туннелем, и одиночный таймаут
там обычное дело. Переключать режим по каждому чиху значит дёргать оператора
плашкой «распознавание недоступно» несколько раз за смену. Почему не пять:
между отказом и переключением оператор ждёт вектор, которого не будет.
"""
import asyncio
import logging

from . import config as C
from .events import hub, DEGRADED, degraded_payload
from .gateway import gateway
from .search_service import search_service, DENSE, LEXICAL

log = logging.getLogger("core.watchdog")


class Watchdog:
    def __init__(self):
        self.task = None
        self.failures = 0
        self.last_ok = None
        self.last_payload = {}
        self.checks = 0

    @property
    def status(self):
        if not gateway.configured:
            return "down"
        if self.failures == 0:
            return "up"
        return "degraded" if self.failures < C.WATCHDOG_FAILURES else "down"

    async def check_once(self):
        if not gateway.configured:
            self._fail("адрес gpu-шлюза не настроен")
            return False
        ready, payload = await gateway.readyz()
        self.checks += 1
        self.last_payload = payload
        if ready:
            self._ok()
        else:
            self._fail(payload.get("error") or f'/readyz -> {payload.get("status")}')
        return ready

    def _ok(self):
        was = self.failures
        self.failures = 0
        self.last_ok = asyncio.get_event_loop().time()
        if search_service.mode != DENSE:
            search_service.set_mode(DENSE)
            hub.broadcast(DEGRADED, degraded_payload("recovered", DENSE))
            log.info("gpu-шлюз восстановлен после %s отказов подряд", was)

    def _fail(self, reason):
        self.failures += 1
        if self.failures >= C.WATCHDOG_FAILURES and search_service.mode != LEXICAL:
            search_service.set_mode(LEXICAL)
            hub.broadcast(DEGRADED, degraded_payload("gateway", LEXICAL, reason))
            log.error("gpu-шлюз недоступен (%s отказа подряд): %s -> %s",
                      self.failures, reason, LEXICAL)

    async def _loop(self):
        while True:
            try:
                await self.check_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:                              # noqa: BLE001
                log.warning("сторож споткнулся: %s", e)
                self._fail(str(e))
            await asyncio.sleep(C.WATCHDOG_INTERVAL)

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

    def reset(self):
        """После смены адреса шлюза счётчик отказов относится к прежнему адресу."""
        self.failures = 0


watchdog = Watchdog()
