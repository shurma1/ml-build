# -*- coding: utf-8 -*-
"""Наводящий вопрос = тот, что сильнее всего режет текущий список кандидатов.

Никакого вызова LLM здесь нет и не нужно: выбор считается по ≤30 кандидатам
за десятки микросекунд. LLM нужна была раньше — чтобы разметить услуги гранями
(offline, один раз). Формулировки вопросов заранее написаны человеком, поэтому
ответ оператору отдаётся мгновенно.

Метрика — прирост информации. Грань, делящая кандидатов пополам, снимает 1 бит,
то есть вдвое сокращает список. Грань, по которой все кандидаты одинаковы,
не снимает ничего, и спрашивать про неё — трата времени посетителя.
"""
import math, re
from collections import Counter

# Грани, по которым имеет смысл спрашивать. value_fn -> хешируемое значение или None.
# Порядок роли не играет: выбирает энтропия, а не позиция в списке.
FACETS = [
    dict(key='recipient', q='Услуга нужна вам как физическому лицу, ИП или организации?',
         value=lambda t: tuple(sorted(t['recipients'])) or None),
    dict(key='life', q='К какой ситуации это относится: {options}?',
         value=lambda t: tuple(sorted(t['life'])) or None, show_options=True),
    dict(key='municipality', q='В каком муниципальном образовании вы прописаны?',
         value=lambda t: len(t['municipalities']) or None),
]

# Доменные бинарные грани: ключ -> (регулярка по названию, вопрос)
TOPIC = [
    ('выплата',     r'(?i)выплат|пособи|компенсац|субсиди|материальн\w+ помощ',
     'Речь о денежной выплате или о документе/справке?'),
    ('документ',    r'(?i)выдача|оформлени|замена|удостоверени|паспорт|справк|свидетельств|дубликат',
     'Вам нужно получить документ на руки?'),
    ('недвижимость', r'(?i)земельн|жил\w+ помещени|объект\w* капитальн|строительств|градостроительн|адрес|перепланиров|снос|ввод объект',
     'Это связано с землёй, домом или квартирой?'),
    ('ребенок',     r'(?i)ребен|дет\w|многодетн|опек|усыновл|школ|дошкольн',
     'Это связано с детьми?'),
    ('транспорт',   r'(?i)транспортн|водительск|автомоб|такси|парковк',
     'Это связано с транспортом или правами?'),
    ('бизнес',      r'(?i)предпринимател|юридическ\w+ лиц|реестр|лицензи|разрешени\w+ на (торговл|деятельн)',
     'Вы обращаетесь по делам бизнеса?'),
    ('льгота',      r'(?i)ветеран|инвалид|чернобыл|сво\b|вбд|малоимущ|многодетн|пенсионер',
     'У вас есть льготная категория — ветеран, инвалид, многодетный, участник СВО?'),
    ('первично',    r'(?i)впервые|первичн|постановка на учет|регистрац',
     'Вы оформляете это впервые или меняете уже имеющееся?'),
]


def _entropy(counts):
    n = sum(counts)
    return -sum(c / n * math.log2(c / n) for c in counts if c) if n else 0.0


def _gain(values):
    """Прирост информации от грани: сколько бит снимает ответ на вопрос."""
    known = [v for v in values if v is not None]
    if len(known) < 2:
        return 0.0
    groups = Counter(known)
    if len(groups) < 2:
        return 0.0                      # все кандидаты одинаковы — вопрос бесполезен
    before = math.log2(len(values))
    # после ответа останется одна группа; ждём её по доле
    after = sum(c / len(known) * math.log2(c) for c in groups.values())
    unknown_penalty = (len(values) - len(known)) / len(values)
    return max(0.0, (before - after) * (1 - unknown_penalty * 0.5))


def suggest(candidates, asked=(), top_n=3, min_gain=0.25):
    """candidates — список типов (dict из build_types). -> вопросы, лучший первым."""
    if len(candidates) < 2:
        return []
    out = []
    for f in FACETS:
        if f['key'] in asked:
            continue
        vals = [f['value'](t) for t in candidates]
        g = _gain(vals)
        if g < min_gain:
            continue
        q = f['q']
        if f.get('show_options'):
            opts = sorted({x for t in candidates for x in t['life']})[:4]
            if len(opts) < 2:
                continue
            q = q.format(options=', '.join(opts))
        out.append({'key': f['key'], 'question': q, 'gain_bits': round(g, 2),
                    'splits': len({v for v in vals if v is not None})})
    for key, rx, q in TOPIC:
        if key in asked:
            continue
        vals = [bool(re.search(rx, t['title'])) for t in candidates]
        share = sum(vals) / len(vals)
        if not 0.2 <= share <= 0.8:      # грань не делит — не спрашиваем
            continue
        g = _entropy([sum(vals), len(vals) - sum(vals)])
        out.append({'key': key, 'question': q, 'gain_bits': round(g, 2),
                    'splits': 2, 'yes': sum(vals), 'no': len(vals) - sum(vals)})
    out.sort(key=lambda x: -x['gain_bits'])
    return out[:top_n]


def matches(t, key, answer):
    """Совпал ли кандидат с ответом клиента: True / False / None (признак не заполнен).

    None принципиально отличается от False: у половины услуг жизненная ситуация или
    список заявителей просто не заполнены, и наказывать их за это нельзя.
    """
    for k, rx, _ in TOPIC:
        if k == key:
            return bool(re.search(rx, t['title'])) == bool(answer)
    if key == 'recipient':
        return answer in t['recipients'] if t['recipients'] else None
    if key == 'life':
        return answer in t['life'] if t['life'] else None
    if key == 'municipality':
        return answer in t['municipalities'] if t['municipalities'] else None
    return None


def rescore(candidates, scores, key, answer, weight=0.35):
    """Мягкое переранжирование ответом клиента. Возвращает новые оценки.

    Именно мягкое, а не фильтрация. Замерено на 690 запросах:
        жёсткий фильтр  2 вопроса -> R@1 0.775
        переранжирование 2 вопроса -> R@1 0.810,  4 вопроса -> 0.839
    Фильтр выбрасывает цель навсегда, стоит признаку услуги оказаться незаполненным
    или регулярке не сработать на её названии. Переранжирование ошибается обратимо:
    следующий вопрос может вернуть кандидата наверх.
    """
    out = dict(scores)
    for t in candidates:
        m = matches(t, key, answer)
        if m is True:
            out[t['type_id']] = out.get(t['type_id'], 0.0) + weight
        elif m is False:
            out[t['type_id']] = out.get(t['type_id'], 0.0) - weight
    return out


def apply_answer(candidates, key, answer):
    """Жёсткая фильтрация. Оставлена для явного сужения оператором («точно не это»),
    в автоматическом цикле уточнения использовать rescore()."""
    for k, rx, _ in TOPIC:
        if k == key:
            want = bool(answer)
            return [t for t in candidates if bool(re.search(rx, t['title'])) == want]
    if key == 'recipient':
        return [t for t in candidates if not t['recipients'] or answer in t['recipients']]
    if key == 'life':
        return [t for t in candidates if not t['life'] or answer in t['life']] or candidates
    if key == 'municipality':
        return [t for t in candidates
                if not t['municipalities'] or answer in t['municipalities']]
    return candidates
