# -*- coding: utf-8 -*-
"""Корпус в памяти процесса: карточки, типы, исходные поля, предикаты, филиалы.

Грузится один раз при старте и дальше только читается. 732 карточки — это
единицы мегабайт, а обращаться к ним нужно на каждом ходу каждого диалога:
держать их в базе и ходить за ними по сети значит платить круговой обход
за подсветку цитаты, которая и так считается подстрокой.

Исходные (raw) поля карточки нужны отдельно от обработанных: подсветка
адресует смещения именно в них — `serviceRecipients`, `rejectReasonsText`
и прочие поля выгрузки, а не в укороченные `who`/`reject` из corpus.py.
"""
import json
import logging
import sys
import threading

from . import config as C

log = logging.getLogger("core.catalog")

# Поля исходной карточки, по которым имеет смысл искать цитату и рисовать подсветку.
RAW_FIELDS = ("serviceTitleText", "serviceRecipients", "documentsText",
              "serviceResultText", "rejectReasonsText", "timeTermText",
              "paymentInfoText", "serviceOrderingText")


def _v2():
    if C.V2_PATH not in sys.path:
        sys.path.insert(0, C.V2_PATH)


class Catalog:
    def __init__(self):
        self.cards = []
        self.types = {}            # type_id -> тип из build_types
        self.by_key = {}           # type_key -> тип
        self.raw = {}              # service_id -> исходная карточка выгрузки
        self.card_by_id = {}       # service_id -> обработанная карточка
        self.src = {}              # type_id -> текст для eligibility.check
        self.version = {}
        self._views = {}            # service_id -> собранная карточка-шпаргалка
        self._lock = threading.Lock()

    def load(self, path=None):
        _v2()
        from corpus import load_cards, build_types
        from canon import term_days, is_free
        path = path or C.DATA
        cards = load_cards(path)
        types = build_types(cards)          # проставляет type_id на карточки
        raw = {r["id"]: r for r in json.load(open(path, encoding="utf-8"))}

        with self._lock:
            self._views.clear()
            self.cards = cards
            self.types = {t["type_id"]: t for t in types}
            self.by_key = {t["type_key"]: t for t in types}
            self.raw = raw
            self.card_by_id = {c["id"]: c for c in cards}
            # ровно тот же источник, что собирает DialogSearch.__init__
            self.src = {
                t["type_id"]: ((raw[t["instances"][0]["id"]].get("serviceRecipients") or "")
                               + "\n"
                               + (raw[t["instances"][0]["id"]].get("rejectReasonsText") or ""))
                for t in types}
            self.term_days = term_days
            self.is_free = is_free
            self.version = {
                "cards": len(cards), "types": len(types),
                "source_sha": _sha(path),
                "built_at": None,
            }
        log.info("корпус загружен: %s карточек -> %s типов", len(cards), len(types))
        return self

    # --- доступ -------------------------------------------------------------

    def type_of_service(self, service_id):
        c = self.card_by_id.get(service_id)
        return self.types.get(c["type_id"]) if c else None

    def raw_fields(self, service_id):
        r = self.raw.get(service_id) or {}
        return {k: r[k] for k in RAW_FIELDS if isinstance(r.get(k), str) and r[k].strip()}

    def view(self, service_id):
        """Карточка-шпаргалка. Корпус неизменяем, поэтому разбор делается один
        раз на услугу и дальше отдаётся из памяти: он идёт и в карточку, и в
        каждую строку выдачи, то есть по нескольку раз на ход диалога."""
        v = self._views.get(service_id)
        if v is not None:
            return v
        card = self.card_by_id.get(service_id)
        if not card:
            return None
        from . import cardview
        v = cardview.build(card, self.raw.get(service_id) or {},
                           self.types.get(card["type_id"]))
        with self._lock:
            self._views[service_id] = v
        return v

    def structural(self, service_id):
        """Срок и плата — из canon, а не из текста наугад."""
        r = self.raw.get(service_id) or {}
        return {"term_days": self.term_days(r.get("timeTermText") or ""),
                "is_free": self.is_free(r.get("paymentInfoText") or "")}

    def type_structural(self, type_id):
        t = self.types.get(type_id)
        if not t:
            return {"term_days": None, "is_free": None}
        return self.structural(t["instances"][0]["id"])


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


catalog = Catalog()
