# -*- coding: utf-8 -*-
"""Клиент к vLLM: извлечение фактов, вопрос по документу, озвучивание вердикта.

Ходим в vLLM сырым HTTP, а не через SDK: нужен полный контроль над телом
запроса. Структурированный вывод в vLLM за версии переезжал между `guided_json`
и `response_format: json_schema`, поэтому режим определяется один раз опытным
путём при первом вызове и дальше не гадается.

Чего здесь НЕТ и не должно появиться: решения о праве на услугу. Рантайм-проверка
права силами модели измерена на этом корпусе и отвергнута — 14% самопротиворечий,
17% ложных отказов, 2.7 с на услугу. Вердикт считает таблица предикатов на
основном сервере; сюда он приходит на вход, в explain().
"""
import json
import re
import time

import httpx

from . import config as C
from . import prompts
from .vendor.extraction_schema import parse_llm_output, JSON_SCHEMA, validate

# Режим структурированного вывода: определяется при первом вызове.
_STRUCT_MODE = None          # None -> не выяснено, "guided" | "response_format" | "none"


def _struct_body(mode, schema):
    if mode == "guided":
        return {"guided_json": schema, "guided_decoding_backend": "xgrammar"}
    if mode == "response_format":
        return {"response_format": {"type": "json_schema", "json_schema": {
            "name": "facts", "schema": schema, "strict": True}}}
    return {}


class LlmClient:
    def __init__(self):
        self.http = httpx.AsyncClient(base_url=C.VLLM_URL, timeout=C.LLM_TIMEOUT)
        self.model = C.LLM_MODEL
        self.ready = False
        self.struct_mode = None

    async def close(self):
        await self.http.aclose()

    async def probe(self):
        """Узнать имя модели у самого vLLM и выяснить, как он принимает схему.

        Имя модели не зашиваем: vLLM отдаёт его тем, под которым реально загрузил,
        а несовпадение имени — 400 на каждом запросе.
        """
        r = await self.http.get("/v1/models")
        r.raise_for_status()
        data = r.json().get("data") or []
        if data:
            self.model = data[0]["id"]
        self.struct_mode = await self._detect_struct_mode()
        self.ready = True
        return {"model": self.model, "struct_mode": self.struct_mode}

    async def _detect_struct_mode(self):
        tiny = {"type": "object", "additionalProperties": False,
                "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
        msgs = [{"role": "user", "content": "Верни {\"ok\": true}"}]
        for mode in ("guided", "response_format"):
            body = {"model": self.model, "messages": msgs, "max_tokens": 16,
                    "temperature": 0, **_struct_body(mode, tiny)}
            try:
                r = await self.http.post("/v1/chat/completions", json=body)
                if r.status_code == 200:
                    return mode
            except Exception:
                pass
        # Схему движок не принимает — работаем без грамматики.
        # validate() на приёме остаётся и ловит всё вне enum, но дрейф значений
        # вырастет с 7% до 54%: это авария, а не рабочий режим.
        return "none"

    # --- извлечение фактов ---------------------------------------------------

    async def extract(self, turns):
        if self.struct_mode is None:
            await self.probe()
        body = {
            "model": self.model,
            "messages": prompts.extract_messages(turns),
            "max_tokens": C.LLM_MAX_TOKENS_EXTRACT,
            "temperature": C.LLM_TEMPERATURE,
            **_struct_body(self.struct_mode, JSON_SCHEMA),
        }
        t0 = time.perf_counter()
        r = await self.http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        raw = data["choices"][0]["message"]["content"]
        facts = parse_llm_output(raw)

        # Что модель прислала сверх схемы — отдельный сигнал, а не мусор:
        # рост этого числа означает, что грамматика отвалилась или схема разъехалась.
        dropped = _dropped_keys(raw, facts)
        usage = data.get("usage") or {}
        return {
            "facts": facts,
            "raw_dropped": dropped,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_prefix_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        }

    # --- вопрос по документу -------------------------------------------------

    async def ask_stream(self, document, question, history=None):
        """Отдаёт куски текста по мере генерации; цитаты сверяются в конце."""
        body = {"model": self.model,
                "messages": prompts.ask_messages(document, question, history),
                "max_tokens": C.LLM_MAX_TOKENS_ASK,
                "temperature": C.LLM_TEMPERATURE, "stream": True}
        parts = []
        async with self.http.stream("POST", "/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content")
                except Exception:
                    continue
                if delta:
                    parts.append(delta)
                    yield {"type": "delta", "text": delta}
        answer = "".join(parts)
        yield {"type": "done", "answer": answer,
               "citations": verify_citations(answer, document)}

    async def ask(self, document, question, history=None):
        out = {"answer": "", "citations": []}
        async for ev in self.ask_stream(document, question, history):
            if ev["type"] == "done":
                out = {"answer": ev["answer"], "citations": ev["citations"]}
        return out

    # --- озвучивание готового вердикта ---------------------------------------

    async def explain(self, title, status, reasons, facts=None):
        body = {"model": self.model,
                "messages": prompts.explain_messages(title, status, reasons, facts),
                "max_tokens": 220, "temperature": C.LLM_TEMPERATURE}
        r = await self.http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        # Переписанная цитата хуже отсутствующей: оператор показывает её посетителю
        # как строку регламента. Всё, чего нет в основаниях дословно, вырезаем.
        allowed = [x.get("quote") for x in reasons if x.get("quote")]
        return {"text": _strip_invented_quotes(text, allowed)}

    def info(self):
        return {"model": self.model, "struct_mode": self.struct_mode, "ready": self.ready}


# --- проверка цитат ----------------------------------------------------------

_QUOTE_RE = re.compile(r"«([^»]{8,400})»")


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").replace("ё", "е")).strip().lower()


def verify_citations(answer, document):
    """Цитата засчитывается, только если дословно есть в документе.

    Смещения считаются в ИСХОДНОМ тексте поля — их же использует подсветка
    в карточке на стороне core-api.
    """
    out = []
    for m in _QUOTE_RE.finditer(answer or ""):
        q = m.group(1).strip()
        nq = _norm(q)
        for field, value in document.items():
            if not value:
                continue
            text = str(value)
            pos = text.find(q)
            if pos < 0:
                # мягкий проход: модель могла нормализовать пробелы внутри цитаты
                pos = _norm(text).find(nq)
                if pos < 0:
                    continue
            out.append({"quote": q, "field": field, "start": pos, "end": pos + len(q)})
            break
    return out


def _strip_invented_quotes(text, allowed):
    keep = {_norm(a) for a in allowed}
    def sub(m):
        return m.group(0) if _norm(m.group(1)) in keep else ""
    return _QUOTE_RE.sub(sub, text).replace("  ", " ").strip()


def _dropped_keys(raw, kept):
    try:
        m = re.search(r"\{.*\}", re.sub(r"(?s)<think>.*?</think>", "", raw or ""), re.S)
        got = json.loads(m.group(0)) if m else {}
    except Exception:
        return ["<не разобран JSON>"]
    if not isinstance(got, dict):
        return ["<не объект>"]
    return sorted(k for k, v in got.items()
                  if v not in (None, "", [], {}) and k not in kept)
