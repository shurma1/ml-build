# -*- coding: utf-8 -*-
"""Причины вердикта о праве — в вид, пригодный для чтения вслух.

`eligibility.check()` собирает `why` подстановкой значения предиката в шаблон,
а значение там бывает списком альтернатив. В результате наружу уходил
питоновский repr: «услуга для категории: ['военнослужащий', 'СВО']» — оператор
читает это посетителю.

Правка сделана здесь, а не в `embending/v2/eligibility.py`, сознательно: там
живёт измеренная логика вердикта, и трогать её ради оформления нельзя. Вердикт
и состав причин остаются ровно теми же, меняется только запись.
"""
import re

# Виды заявителя оператор называет по-русски, а не кодом схемы
RECIPIENT_WORDS = {
    "person": "физическое лицо",
    "ip": "индивидуальный предприниматель",
    "organization": "организация",
}

_LIST_RX = re.compile(r"\[([^\[\]]*)\]")
_QUOTED_RX = re.compile(r"['\"]([^'\"]*)['\"]")


def _unlist(m):
    items = [x.strip() for x in _QUOTED_RX.findall(m.group(1))]
    if not items:
        items = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
    return ", ".join(RECIPIENT_WORDS.get(x, x) for x in items)


def humanize(text):
    if not text:
        return text
    out = _LIST_RX.sub(_unlist, str(text))
    # одиночное значение вида recipient_type тоже приходит кодом
    for code, word in RECIPIENT_WORDS.items():
        out = re.sub(rf"(?<![а-яёa-z]){code}(?![а-яёa-z])", word, out)
    return re.sub(r"\s+", " ", out).strip()


def humanize_reasons(reasons):
    """Не меняет ни статус, ни состав причин — только их запись."""
    out = []
    for r in reasons or []:
        d = dict(r)
        if d.get("why"):
            d["why"] = humanize(d["why"])
        alts = d.get("alternatives")
        if isinstance(alts, list):
            d["alternatives"] = [
                ", ".join(RECIPIENT_WORDS.get(str(x), str(x)) for x in a) if isinstance(a, list)
                else RECIPIENT_WORDS.get(str(a), a)
                for a in alts
            ]
        out.append(d)
    return out
