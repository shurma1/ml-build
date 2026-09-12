# -*- coding: utf-8 -*-
"""Право на услугу: находим ВСЁ, помечаем что не положено, цитируем почему.

Принцип: право проверяется ПОСЛЕ поиска, никогда вместо него. Отфильтрованная на
этапе retrieval услуга теряется навсегда, а оператору нужно уметь сказать «услуга
есть, но вам не положена вот по этой строке регламента».

Предикаты берутся из v2/eval/predicates.jsonl — типизированной таблицы, собранной
LLM оффлайн по одному разу на тип. Регулярками это делать нельзя: они не различают,
к кому относится условие. Реальный провал прежней версии — «ребёнка в возрасте до
17 лет» срабатывало на возраст ЗАЯВИТЕЛЯ. Поэтому у предиката есть subject, и
проверяются только те, чей subject — applicant.

Три статуса:
    eligible — ни одно проверяемое условие не нарушено
    blocked  — нарушено; отдаём причину и дословную цитату
    unknown  — условие есть, факта о клиенте нет; это вход для наводящего вопроса
"""
import json, os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval', 'predicates.jsonl')
_TABLE = None

# Проверяем только условия к самому заявителю. Условия к ребёнку, супругу или
# объекту недвижимости оператор видит справочно — по ним у нас нет фактов.
CHECKABLE_SUBJECT = 'applicant'

_FACT_OF = {'age': 'age', 'children': 'children', 'category': 'categories',
            'recipient_type': 'recipient'}
_WHY = {
    ('age', '>='): 'требуется возраст не менее {v}',
    ('age', '<='): 'услуга предоставляется в возрасте до {v}',
    ('age', '=='): 'требуется возраст ровно {v}',
    ('children', '>='): 'требуется не менее {v} детей',
    ('children', '<='): 'не более {v} детей',
    ('category', 'in'): 'услуга для категории: {v}',
    ('category', 'has'): 'услуга для категории: {v}',
    ('recipient_type', '=='): 'услуга только для заявителей вида: {v}',
    ('recipient_type', 'in'): 'услуга только для заявителей вида: {v}',
}


def load_predicates(path=None):
    global _TABLE
    if _TABLE is None:
        _TABLE = {}
        p = path or _PATH
        if os.path.exists(p):
            for line in open(p, encoding='utf-8'):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                _TABLE[d['type_key']] = [x for x in d.get('predicates') or []
                                         if x.get('subject') == CHECKABLE_SUBJECT]
    return _TABLE


def _as_list(v):
    return v if isinstance(v, list) else [v]


def _holds(op, need, got):
    if op in ('>=', '<=', '==') and isinstance(need, (int, float)):
        if not isinstance(got, (int, float)):
            return None
        return got >= need if op == '>=' else got <= need if op == '<=' else got == need
    if op in ('in', 'has', '=='):
        want = {str(x).lower() for x in _as_list(need)}
        have = {str(x).lower() for x in _as_list(got)}
        return bool(want & have)
    return None


def check(type_row, facts, source_text=None):
    """-> (status, [{'why','quote','field'}]). facts — что известно о клиенте."""
    preds = load_predicates().get(type_row.get('type_key'), [])
    if not preds:
        # таблица не покрывает тип — честно говорим «не знаем», а не «положено»
        return ('eligible', []) if not type_row.get('recipients') else \
               _fallback_recipient(type_row, facts)
    # Предикаты ОДНОГО поля — альтернативы (льгота для военнослужащих ИЛИ участников СВО),
    # разных полей — совокупность (возраст И категория). Иначе услуга с двумя
    # допустимыми категориями оказывается недоступна вообще никому.
    by_field = {}
    for p in preds:
        by_field.setdefault(p['field'], []).append(p)

    blocked, unknown = [], []
    for field, group in by_field.items():
        fact_key = _FACT_OF.get(field)
        if not fact_key:
            continue
        got = facts.get(fact_key)
        whys = [_WHY.get((field, p['op']), 'условие: {v}').format(v=p['value']) for p in group]
        if got in (None, '', []):
            unknown.append({'why': ' или '.join(dict.fromkeys(whys)), 'need_fact': fact_key,
                            'quote': group[0].get('quote')})
            continue
        results = [_holds(p['op'], p['value'], got) for p in group]
        if any(r is True for r in results):
            continue                                   # хотя бы одна альтернатива подошла
        if all(r is False for r in results):
            blocked.append({'why': ' или '.join(dict.fromkeys(whys)), 'field': field,
                            'quote': group[0].get('quote'),
                            'alternatives': [p['value'] for p in group]})
    if blocked:
        return 'blocked', blocked
    if unknown:
        return 'unknown', unknown
    return 'eligible', []


def _fallback_recipient(type_row, facts):
    """Пока предикатов нет — проверяем хотя бы вид заявителя из структурного поля."""
    need = type_row.get('recipients') or []
    got = facts.get('recipient')
    if not need:
        return 'eligible', []
    if got is None:
        return 'unknown', [{'why': 'услуга только для заявителей вида: ' + ', '.join(need),
                            'need_fact': 'recipient', 'quote': None}]
    if got not in need:
        return 'blocked', [{'why': 'услуга только для заявителей вида: ' + ', '.join(need),
                            'quote': None, 'field': 'recipient_type'}]
    return 'eligible', []
