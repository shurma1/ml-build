# -*- coding: utf-8 -*-
"""Подсветка условий в карточке услуги. Без LLM — поиском подстроки.

Почему это вообще возможно: при генерации таблицы предикатов цитата
отбрасывалась, если её не находили в исходнике дословно (`generate_predicates.py`
сверял нормализованное вхождение). Значит у каждого из 363 предикатов цитата
физически присутствует в тексте карточки, и задача сводится к «найти смещения»,
а не «попросить модель показать пальцем».

Нормализация обязательна и ровно та же, что была при генерации: нижний регистр
и схлопнутые пробелы. Исходник свёрстан переносами и двойными пробелами, поэтому
наивный `text.find(quote)` промахивается на большинстве цитат. Здесь строится
карта «смещение в нормализованном тексте -> смещение в исходном», и наружу
отдаются смещения ИСХОДНОГО текста — именно их подсвечивает интерфейс.

Цитата могла быть обрезана генератором до 220 символов, поэтому при промахе
полной строки пробуются её префиксы: лучше подсветить первую фразу условия,
чем не подсветить ничего.
"""
import logging
import re

log = logging.getLogger("core.highlights")

MIN_PREFIX = 40          # до такой длины режем длинную цитату, если целиком не нашлась
MIN_QUOTE = 12           # короче не ищем вовсе: случайное совпадение дороже пропуска.
                         # 12 символов покрывает самые короткие реальные цитаты таблицы
                         # («члены их семей», «физические лица») и отсекает обрывки слов.

_TERM_RX = re.compile(r"(?i)(\d{1,3}\s*(?:рабочих|календарных)\s+дн\w+"
                      r"|\d{1,3}\s*р\.?\s?д\.?(?![а-я])"
                      r"|\d{1,3}\s*к\.?\s?д\.?(?![а-я]))")
_FREE_RX = re.compile(r"(?i)(бесплатн\w*)")
_COST_RX = re.compile(r"(?i)(\d[\d\s]{2,}\s*руб\w*|госпошлин\w+[^.;\n]{0,60})")


def _norm_map(s):
    """-> (нормализованный текст, карта позиций в исходный текст)."""
    out, idx = [], []
    for i, ch in enumerate(s):
        c = ch.lower()
        if c.isspace():
            if out and out[-1] == " ":
                continue
            out.append(" ")
        else:
            out.append(c)
        idx.append(i)
    return "".join(out), idx


def _norm_query(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def find_quote(fields, quote):
    """Найти цитату в полях карточки. -> (field, start, end) или None.

    `fields` — исходные поля карточки: {имя поля: текст}. Порядок важен,
    поэтому поля перебираются в порядке словаря, а он у нас задан RAW_FIELDS.
    """
    q = _norm_query(quote)
    if len(q) < MIN_QUOTE:
        return None
    # «…» и хвостовое многоточие модель дописывала сама — в исходнике их нет
    q = q.strip("«»\"' ").rstrip(".…")
    candidates = [q]
    for cut in (220, 120, 60, MIN_PREFIX):
        if len(q) > cut:
            candidates.append(q[:cut])
    # обрезать лучше по границе слова, иначе префикс кончается посреди слова
    candidates = [re.sub(r"\S*$", "", c).strip() if len(c) < len(q) else c
                  for c in candidates]
    seen = set()
    for cand in candidates:
        if len(cand) < MIN_QUOTE or cand in seen:
            continue
        seen.add(cand)
        for field, text in fields.items():
            if not text:
                continue
            norm, idx = _norm_map(text)
            pos = norm.find(cand)
            if pos < 0 and "ё" in norm or "ё" in cand:
                pos = norm.replace("ё", "е").find(cand.replace("ё", "е"))
            if pos < 0:
                continue
            start = idx[pos]
            end = idx[min(pos + len(cand) - 1, len(idx) - 1)] + 1
            return field, start, end
    return None


def _kind_of_group(group, facts, holds):
    """Вердикт по группе предикатов одного поля — теми же правилами, что в
    eligibility.check: предикаты одного поля это альтернативы (ИЛИ).

    Считается здесь заново не ради удобства, а ради согласованности: подсветка
    обязана говорить то же самое, что статус в списке. Разошлись — оператор
    видит «не положено» и зелёную строку под ним.
    """
    from eligibility import _FACT_OF
    fact_key = _FACT_OF.get(group[0]["field"])
    if not fact_key:
        return "unknown", None, fact_key
    got = facts.get(fact_key)
    if got in (None, "", []):
        return "unknown", None, fact_key
    results = [holds(p["op"], p["value"], got) for p in group]
    if any(r is True for r in results):
        return "matching", results, fact_key
    if all(r is False for r in results):
        return "blocking", results, fact_key
    return "unknown", results, fact_key


def for_type(type_row, fields, facts):
    """Подсветки для одного типа услуги в полях конкретной карточки."""
    from eligibility import load_predicates, _holds, _WHY

    out = []
    preds = load_predicates().get(type_row.get("type_key"), []) if type_row else []
    by_field = {}
    for p in preds:
        by_field.setdefault(p["field"], []).append(p)

    for field, group in by_field.items():
        kind, results, fact_key = _kind_of_group(group, facts, _holds)
        for i, p in enumerate(group):
            if not p.get("quote"):
                continue
            # в подошедшей группе зелёным помечается та альтернатива, что сработала
            pk = kind
            if kind == "matching" and results and results[i] is not True:
                continue
            why = _WHY.get((p["field"], p["op"]), "условие: {v}").format(v=p["value"])
            loc = find_quote(fields, p["quote"])
            if not loc:
                continue
            f, start, end = loc
            out.append({"field": f, "start": start, "end": end, "kind": pk,
                        "reason": why})

    out.extend(_structural(fields))
    return _dedup(out)


def _structural(fields):
    """Срок и плата. Это не право, а то, что оператор проговаривает вслух
    почти всегда, поэтому подсвечивается отдельными видами."""
    out = []
    term = fields.get("timeTermText") or ""
    m = _TERM_RX.search(term)
    if m:
        out.append({"field": "timeTermText", "start": m.start(1), "end": m.end(1),
                    "kind": "term", "reason": "срок предоставления услуги"})
    pay = fields.get("paymentInfoText") or ""
    m = _COST_RX.search(pay) or _FREE_RX.search(pay)
    if m:
        free = bool(_FREE_RX.match(m.group(1)))
        out.append({"field": "paymentInfoText", "start": m.start(1), "end": m.end(1),
                    "kind": "payment",
                    "reason": "услуга бесплатна" if free else "услуга платная"})
    return out


def _dedup(hs):
    """Один и тот же отрезок мог прийти от двух предикатов. Приоритет —
    blocking > unknown > matching: оператору важнее увидеть препятствие."""
    rank = {"blocking": 0, "unknown": 1, "matching": 2, "term": 3, "payment": 4}
    best = {}
    for h in hs:
        key = (h["field"], h["start"], h["end"])
        cur = best.get(key)
        if cur is None or rank.get(h["kind"], 9) < rank.get(cur["kind"], 9):
            best[key] = h
    return sorted(best.values(), key=lambda h: (h["field"], h["start"]))
