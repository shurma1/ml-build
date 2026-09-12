# -*- coding: utf-8 -*-
"""Состояние процесса, которое видят и роуты, и middleware.

Главное здесь — `fingerprint_mismatch`. Отпечаток эмбеддера сверяется при старте:
если индекс собран одной моделью, а запросы кодирует другая, поиск продолжит
возвращать правдоподобные списки, и заметить подмену по выдаче невозможно.
Поэтому несовпадение — это отказ обслуживать (503 index_fingerprint_mismatch),
а не «поищем как-нибудь».

Отказ именно такой, а не падение процесса: сервис обязан остаться на ногах,
чтобы отвечать на `/v1/version` и принимать `POST /v1/admin/gateway` — иначе
чинить рассинхрон будет нечем.
"""
import logging

log = logging.getLogger("core.state")


class AppState:
    def __init__(self):
        self.fingerprint_mismatch = None    # None = всё сходится
        self.index_fingerprint = None
        self.gateway_fingerprint = None
        self.started_at = None

    def set_mismatch(self, index_fp, gateway_fp):
        self.index_fingerprint = index_fp
        self.gateway_fingerprint = gateway_fp
        self.fingerprint_mismatch = (
            f"индекс собран другой моделью: индекс «{index_fp}», "
            f"шлюз отвечает «{gateway_fp}»")
        log.error("ОТКАЗ ОБСЛУЖИВАТЬ: %s", self.fingerprint_mismatch)

    def clear_mismatch(self):
        if self.fingerprint_mismatch:
            log.info("отпечаток эмбеддера сошёлся, обслуживание возобновлено")
        self.fingerprint_mismatch = None


state = AppState()
