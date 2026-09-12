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

# Поля, по которым несовпадение НЕ является отказом.
#
# Список категорий посетителя заведомо неполон: он назвал то, что вспомнил или о
# чём спросили, а не всё, что имеет. Замерено на корпусе: произнесённое «я
# пенсионер» переводит 61 услугу из 318 в blocked, а следующая фраза «и инвалид»
# возвращает 11 из них обратно. Вердикт, который переворачивается от того, что
# посетитель успел упомянуть, — это не вердикт.
#
# Возраст, число детей и вид заявителя устроены иначе: это одно значение, и если
# оно названо, несовпадение окончательно. Их отказ остаётся отказом.
INCOMPLETE_FACTS = {'category'}

# Значения, которых извлечение не может выдать физически: JSON_SCHEMA отдаётся
# движку как guided_json, и enum в ней жёсткий. Предикат, требующий значения вне
# enum, не станет истинным НИКОГДА — а поскольку blocked объявляется, когда все
# альтернативы поля ложны, такой предикат давал гарантированный ложный отказ.
# На текущей таблице это 12 значений category из 20 («мобилизованный»,
# «реабилитированные лица», «малоимущие семьи»…) и 4 из 7 значений
# recipient_type («citizen», «foreign citizen», «patient»…), а всего пять типов,
# по которым отказ получал любой посетитель, назвавший хоть какую-то категорию,
# — включая социальную помощь малоимущим и выплату при рождении ребёнка.
#
# Сводить их к enum — работа для сборки таблицы, а не для рантайма. Здесь мы
# лишь отказываемся выносить по ним вердикт: непроверяемое условие даёт unknown
# и наводящий вопрос, а не отказ.
def _reachable():
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from extraction_schema import CATEGORIES, RECIPIENTS
    return {'category': {c.lower() for c in CATEGORIES},
            'recipient_type': {r.lower() for r in RECIPIENTS}}


_REACHABLE = None


def reachable_values(field):
    """Значения поля, которые извлечение способно выдать. None = поле числовое."""
    global _REACHABLE
    if _REACHABLE is None:
        _REACHABLE = _reachable()
    return _REACHABLE.get(field)
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


def _values(group):
    return {str(x).lower() for p in group for x in _as_list(p['value'])}


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
        why = ' или '.join(dict.fromkeys(whys))
        allowed = reachable_values(field)
        if allowed is not None and not (_values(group) & allowed):
            # Условие сформулировано значениями, которых извлечение не выдаёт.
            # Проверить его нечем — говорим «не знаем», а не «не положено».
            unknown.append({'why': why, 'need_fact': fact_key,
                            'quote': group[0].get('quote'), 'unverifiable': True})
            continue
        if got in (None, '', []):
            unknown.append({'why': why, 'need_fact': fact_key,
                            'quote': group[0].get('quote')})
            continue
        results = [_holds(p['op'], p['value'], got) for p in group]
        if any(r is True for r in results):
            continue                                   # хотя бы одна альтернатива подошла
        if all(r is False for r in results):
            row = {'why': why, 'field': field, 'quote': group[0].get('quote'),
                   'alternatives': [p['value'] for p in group]}
            if field in INCOMPLETE_FACTS:
                # Ни одна из НАЗВАННЫХ категорий не подошла. Это не отказ:
                # посетитель мог не упомянуть ту, по которой услуга положена.
                row['need_fact'] = fact_key
                row['not_among_stated'] = True
                unknown.append(row)
            else:
                blocked.append(row)
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
