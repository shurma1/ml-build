# -*- coding: utf-8 -*-
"""Шина событий сессии: один источник, много подписчиков (WS оператора).

Порядок гарантируется полем `seq`, монотонным в пределах сессии. Клиент обязан
пережить пропуск `transcript.utterance` — сегмент мог быть отброшен как
устаревший, — но не обязан переживать перестановку, поэтому номер ставится
здесь, под общим замком, а не в месте отправки.
"""
import asyncio
import logging
from collections import defaultdict

log = logging.getLogger("core.events")

READY = "session.ready"
UTTERANCE = "transcript.utterance"
STATE_UPDATED = "state.updated"
RESULTS_UPDATED = "results.updated"
DEGRADED = "degraded"
ERROR = "error"

# Что видит оператор, когда отваливается подсистема. Текст исключения сюда
# не попадает: «ConnectError: [Errno 61] Connection refused» на рабочем месте
# в МФЦ — это не сообщение, а шум. Техническая строка уходит в `detail`
# и в лог, где её и будут читать.
DEGRADED_TEXT = {
    "gateway": "Распознавание речи недоступно — вводите реплики текстом. "
               "Поиск работает, список может быть беднее.",
    "llm": "Разбор речи недоступен — признаки берутся по ключевым словам. "
           "Проверьте факты о посетителе вручную.",
    "asr": "Микрофон недоступен — вводите реплики текстом.",
    "recovered": "Связь с сервером моделей восстановлена.",
}


def degraded_payload(kind, mode, detail=None):
    return {"reason": DEGRADED_TEXT.get(kind, DEGRADED_TEXT["gateway"]),
            "kind": kind, "mode": mode, "detail": str(detail) if detail else None}


class Hub:
    def __init__(self):
        self._subs = defaultdict(set)
        self._seq = defaultdict(int)
        self._lock = asyncio.Lock()

    def subscribe(self, session_id):
        q = asyncio.Queue(maxsize=256)
        self._subs[str(session_id)].add(q)
        return q

    def unsubscribe(self, session_id, q):
        s = self._subs.get(str(session_id))
        if s:
            s.discard(q)
            if not s:
                self._subs.pop(str(session_id), None)
                self._seq.pop(str(session_id), None)

    def subscribers(self, session_id):
        return len(self._subs.get(str(session_id), ()))

    def emit(self, session_id, type_, payload=None):
        sid = str(session_id)
        self._seq[sid] += 1
        ev = {"type": type_, "seq": self._seq[sid]}
        if payload:
            ev.update(payload)
        for q in list(self._subs.get(sid, ())):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                # Оператор не успевает читать. Терять свежее событие хуже, чем
                # старое: выбрасываем голову очереди и кладём новое.
                try:
                    q.get_nowait()
                    q.put_nowait(ev)
                except Exception:                               # noqa: BLE001
                    log.warning("очередь событий сессии %s переполнена", sid)
        return ev

    def broadcast(self, type_, payload=None):
        """Во все живые сессии — так расходится `degraded`."""
        for sid in list(self._subs):
            self.emit(sid, type_, payload)


hub = Hub()
