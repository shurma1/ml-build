# -*- coding: utf-8 -*-
"""Конверт ошибок core-api: {"error": {code, message, retryable, request_id}}.

Коды — ровно те, что перечислены в core-api.yaml. Новый код заводится правкой
контракта, а не строкой в обработчике: клиент разбирает именно code, а не текст.
"""
from fastapi.responses import JSONResponse

CODES = {
    "unauthorized": 401,
    "session_not_found": 404,
    "service_not_found": 404,
    "gpu_unavailable": 503,
    "index_fingerprint_mismatch": 503,
    "schema_validation_failed": 400,
    "audio_format_unsupported": 400,
    "rate_limited": 429,
    "internal": 500,
}

RETRYABLE = {"gpu_unavailable", "rate_limited", "internal"}


class ApiError(Exception):
    def __init__(self, code, message, status=None, retryable=None):
        self.code = code
        self.message = message
        self.status = status or CODES.get(code, 500)
        self.retryable = code in RETRYABLE if retryable is None else retryable
        super().__init__(message)

    def body(self, request_id=None):
        return error_body(self.code, self.message, self.retryable, request_id)

    def response(self, request_id=None):
        return JSONResponse(status_code=self.status, content=self.body(request_id))


def error_body(code, message, retryable=None, request_id=None):
    err = {"code": code, "message": message,
           "retryable": bool(code in RETRYABLE if retryable is None else retryable)}
    if request_id:
        err["request_id"] = request_id
    return {"error": err}


def not_found(kind, ident):
    code = "session_not_found" if kind == "session" else "service_not_found"
    return ApiError(code, f"{kind} {ident} не найден")
