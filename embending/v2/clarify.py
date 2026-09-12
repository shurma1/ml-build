# -*- coding: utf-8 -*-
"""Наводящий вопрос = тот, что сильнее всего режет текущий список кандидатов.

Никакого вызова LLM здесь нет и не нужно: выбор считается по ≤30 кандидатам
за десятки микросекунд. LLM нужна была раньше — чтобы разметить услуги гранями
(offline, один раз). Формулировки вопросов заранее написаны человеком, поэтому
ответ оператору отдаётся мгновенно.

Метрика — прирост информации: сколько бит снимает ОТВЕТ. Грань, делящая
кандидатов пополам, снимает 1 бит; грань, по которой все кандидаты одинаковы,
не снимает ничего, и спрашивать про неё — трата времени посетителя.

Два правила, без которых вопросы превращаются в шум (замерено на 12 типовых
обращениях: 34 вопроса на 12 диалогов, ни одного диалога без вопроса):

**Не спрашиваем то, что уже прозвучало.** «У вас есть льготная категория?»
при `categories=['многодетный']` — не уточнение, а сообщение посетителю, что
его не слушали. Отсюда `answered()`: грань, закрытая фактом диалога, выбывает
вместе с уже заданными.

**Считаем бит по ответу, а не по разнообразию значений.** Прежняя мера брала
энтропию НАБОРОВ: у каждой услуги свой кортеж жизненных ситуаций, все кортежи
разные — и вопрос «К какой ситуации это относится?» получал 3.43 бита там, где
ответ не отсекал никого. Теперь считается прямо: сколько кандидатов останется
после каждого варианта ответа. Услуга с незаполненным признаком не отсеивается
никогда — иначе вопрос «выигрывал» бы за счёт дыр в выгрузке.
"""
import math, re
from collections import Counter

RECIPIENTS = ('person', 'ip', 'organization')

# Грани, по которым имеет смысл спрашивать. sets -> множества значений услуги,
# options -> варианты ответа. Порядок роли не играет: выбирает энтропия.
FACETS = [
    dict(key='recipient', q='Услуга нужна вам как физическому лицу, ИП или организации?',
         sets=lambda t: set(t['recipients'] or ()),
         options=lambda c: ['person', 'ip', 'organization']),
    dict(key='life', q='К какой ситуации это относится: {options}?',
         sets=lambda t: set(t['life'] or ()),
         options=lambda c: _top_values(c, 'life', 3), show_options=True),
    dict(key='municipality', q='В каком муниципальном образовании вы прописаны?',
         sets=lambda t: set(t['municipalities'] or ()),
         options=lambda c: _top_values(c, 'municipalities', 4)),
]

# Жизненная ситуация сама отвечает на часть граней: «Утрата документов» — это
# про документ, «Открытие своего дела» — про бизнес. Ключи — значения из
# справочника extraction_schema.LIFE_SITUATIONS, только их и может вернуть
# извлечение; чего в справочнике нет, то просто не закрывает ничего.
SITUATION_ANSWERS = {
    'Рождение ребенка': {'ребенок'},
    'Многодетная семья': {'ребенок'},
    'Детские пособия': {'ребенок', 'выплата'},
    'Услуги опеки': {'ребенок'},
    'Утрата документов': {'документ'},
    'Перемена имени': {'документ'},
    'Открытие своего дела': {'бизнес'},
    'Индивидуальное жилищное строительство': {'недвижимость'},
    'Сделки с недвижимостью': {'недвижимость'},
    'Чернобыльские выплаты': {'выплата', 'льгота'},
    'Меры поддержки СВО': {'льгота'},
    'Меры поддержки военнослужащих/ветеранов боевых действий': {'льгота'},
}

# Чего не хватило проверке права -> вопрос человеческими словами. Это самые
# честные вопросы из всех: они не про список вообще, а про конкретную услугу
# наверху выдачи, вердикт по которой без этого факта не считается.
NEED_QUESTIONS = {
    'age': 'Сколько вам полных лет?',
    'children': 'Сколько у вас детей?',
    'categories': 'Есть ли льготная категория?',
    'recipient': 'Обращение от себя, от ИП или от организации?',
    'municipality': 'В каком муниципальном образовании вы прописаны?',
}

# Форма ответа на вопрос о факте. Числовые поля интерфейс показывает полем ввода,
# остальные — кнопками.
NEED_FORM = {
    'age':      dict(kind='number', min=0, max=120),
    'children': dict(kind='number', min=0, max=15),
    'categories': dict(kind='choice'),
    'recipient':  dict(kind='choice'),
    'municipality': dict(kind='choice'),
}

# Факт закрывает грань: спрашивать об одном и том же двумя вопросами незачем
FACET_OF_FACT = {'categories': 'льгота', 'children': 'ребенок',
                 'recipient': 'recipient', 'municipality': 'municipality'}

# --- чем отвечают на вопрос -------------------------------------------------
#
# Вопрос обязан сам описывать свои варианты ответа. Раньше наружу уходили только
# `key` и текст, и интерфейс не мог отрисовать кнопку: чтобы узнать, что «К какой
# ситуации это относится: A, B, C?» ждёт одно из трёх значений, пришлось бы
# разбирать русскую фразу регуляркой. Ровно поэтому вопросы в интерфейсе и
# остались подписями, а ветка переранжирования по ответу не работала.
#
# `fact` отвечает на второй вопрос: заполняет ли ответ факт о посетителе. Ответ
# на «Сколько вам полных лет?» обязан попасть в состояние диалога, иначе он не
# делает вообще ничего — `matches()` ключа 'age' не знает, и rescore по нему
# пустой. Ответ на «Это связано с детьми?» только двигает список.
RECIPIENT_LABELS = {'person': 'Физлицо', 'ip': 'ИП', 'organization': 'Организация'}
YES_NO = [{'value': True, 'label': 'Да'}, {'value': False, 'label': 'Нет'}]
CATEGORY_OPTIONS = ['ВБД', 'СВО', 'ЧАЭС', 'многодетный', 'инвалид',
                    'пенсионер', 'малоимущий', 'военнослужащий']


def _opts(values, labels=None):
    labels = labels or {}
    return [{'value': v, 'label': labels.get(v, str(v))} for v in values]

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
     'Есть ли льготная категория: ветеран, инвалид, многодетный, '
     'пенсионер, участник СВО?'),
    ('первично',    r'(?i)впервые|первичн|постановка на учет|регистрац',
     'Вы оформляете это впервые или меняете уже имеющееся?'),
]


def _entropy(counts):
    n = sum(counts)
    return -sum(c / n * math.log2(c / n) for c in counts if c) if n else 0.0


def _top_values(candidates, field, n):
    """Самые частые значения признака среди кандидатов — по частоте, не по алфавиту.

    Алфавит здесь был прямой ошибкой: в вариантах ответа оказывались ситуации,
    к которым относилась одна услуга из двадцати, а массовая не попадала вовсе.
    """
    c = Counter(x for t in candidates for x in (t.get(field) or ()))
    return [v for v, _ in c.most_common(n)]


def _gain(sets, options, n_total):
    """Сколько бит снимет ответ на вопрос с такими вариантами.

    Кандидат остаётся, если вариант ему подходит ИЛИ признак у него не заполнен:
    незаполненность — дыра в выгрузке, а не отказ, и половина корпуса без неё
    вылетала бы из выдачи. Вклад варианта взвешен тем, как часто он и есть
    верный ответ среди кандидатов.
    """
    if n_total < 2 or len(options) < 2:
        return 0.0
    remains, weights = [], []
    for o in options:
        hit = sum(1 for s in sets if o in s)
        if not hit:
            continue
        remains.append(hit + sum(1 for s in sets if not s))
        weights.append(hit)
    total = sum(weights)
    if total == 0 or len(remains) < 2:
        return 0.0                      # все кандидаты одинаковы — вопрос бесполезен
    after = sum(w / total * math.log2(r) for w, r in zip(weights, remains))
    return max(0.0, math.log2(n_total) - after)


def answered(feats):
    """Грани, о которых спрашивать уже нечего: факт прозвучал в диалоге.

    Спрашивать про известное — худший вид наводящего вопроса: посетитель только
    что это сказал, а система переспрашивает. Берём именно признаки поиска, а не
    состояние диалога: муниципалитет, например, приходит из филиала оператора,
    и в состоянии его может не быть.
    """
    if feats is None:
        return set()
    facts = getattr(feats, 'facts', None) or {}
    situation = getattr(feats, 'life_situation', None)
    out = set()
    if getattr(feats, 'recipient', None):
        out.add('recipient')
    if situation:
        out.add('life')
    if getattr(feats, 'municipality', None):
        out.add('municipality')
    categories = facts.get('categories') or []
    if categories:
        out.add('льгота')
    if facts.get('age') is not None:
        out.add('age')
    if facts.get('children') is not None or 'многодетный' in categories:
        out.add('ребенок')
    if getattr(feats, 'recipient', None) in ('ip', 'organization'):
        out.add('бизнес')
    return out | set(SITUATION_ANSWERS.get(situation, ()))


# Отрыв top1-top2, выше которого список считается решённым и уточнять нечего.
# Замерено на 690 запросах: отрыв уже посчитан ретривером и отлично предсказывает
# правоту верхней строки —
#     отрыв < 0.005 (25% запросов)  точность top-1 = 0.446
#     0.005 .. 0.010 (17%)                          0.517
#     0.010 .. 0.020 (25%)                          0.836
#     0.020 .. 0.050 (29%)                          0.980
#     >= 0.050        (4%)                          1.000
# Гейт на 0.02 снимает треть вопросов, не трогая ни одного случая, где они нужны:
# связка «центроид + 4 вопроса» даёт R@1 0.904 и с гейтом, и без — но 308 вопросов
# вместо 482. Шкала косинусная и к модели привязана: при смене эмбеддера порог
# надо перемерить, абсолютный косинус между моделями не переносится.
CERTAIN_MARGIN = 0.02


def suggest(candidates, asked=(), top_n=2, min_gain=0.8, feats=None, needed=(),
            margin=None, certain_margin=CERTAIN_MARGIN):
    """candidates — список типов (dict из build_types). -> вопросы, лучший первым.

    `feats` — то, что о посетителе уже известно: эти грани выбывают.
    `needed` — факты, без которых не считается вердикт по услугам НАВЕРХУ выдачи
    (`need_fact` из проверки права). Такой вопрос идёт первым: он про конкретную
    услугу на экране, а не про список вообще.

    `margin` — отрыв top1-top2 у ретривера. Когда он велик, список уже решён, и
    вопросы «на сужение» (FACETS, TOPIC) отпадают. Вопросы из `needed` гейт НЕ
    трогает: они спрашивают не «какая из услуг», а «положена ли вам вот эта»,
    и остаются нужны, даже когда услуга определена однозначно.

    Пустой список — нормальный и частый исход. Вопрос ради вопроса стоит
    посетителю времени, а оператору — доверия.
    """
    if len(candidates) < 2:
        return []
    asked = set(asked) | answered(feats)
    out = []
    for key in dict.fromkeys(needed or ()):
        q = NEED_QUESTIONS.get(key)
        facet = FACET_OF_FACT.get(key, key)
        if not q or facet in asked:
            continue
        asked.add(facet)
        form = dict(NEED_FORM.get(key) or {'kind': 'text'})
        if key == 'categories':
            form['options'] = _opts(CATEGORY_OPTIONS) + [{'value': None, 'label': 'Нет льгот'}]
        elif key == 'recipient':
            form['options'] = _opts(RECIPIENTS, RECIPIENT_LABELS)
        elif key == 'municipality':
            form['options'] = _opts(_top_values(candidates, 'municipalities', 6))
            if len(form['options']) < 2:
                form = {'kind': 'text'}
        out.append({'key': facet, 'question': q, 'gain_bits': None, 'splits': 0,
                    # ответ на этот вопрос заполняет ФАКТ о посетителе, а не только
                    # двигает список: без записи в состояние ответ на «сколько вам
                    # лет» не делает ничего — rescore ключа 'age' не знает.
                    'fact': key, **form})

    if margin is not None and margin >= certain_margin:
        # Список решён: сужать нечего. Остаются только вопросы про вердикт.
        return out[:top_n]

    graded = []
    for f in FACETS:
        if f['key'] in asked:
            continue
        sets = [f['sets'](t) for t in candidates]
        options = f['options'](candidates)
        g = _gain(sets, options, len(candidates))
        if g < min_gain:
            continue
        q = f['q']
        if f.get('show_options'):
            if len(options) < 2:
                continue
            q = q.format(options=', '.join(options))
        if f['key'] == 'recipient':
            opts = _opts(RECIPIENTS, RECIPIENT_LABELS)
        else:
            opts = _opts(options)
        graded.append({'key': f['key'], 'question': q, 'gain_bits': round(g, 2),
                       'splits': len(options), 'kind': 'choice', 'options': opts,
                       'fact': f['key'] if f['key'] in ('recipient', 'municipality') else None})
    for key, rx, q in TOPIC:
        if key in asked:
            continue
        vals = [bool(re.search(rx, t['title'])) for t in candidates]
        yes = sum(vals)
        g = _entropy([yes, len(vals) - yes])
        if g < min_gain:                 # грань не делит — не спрашиваем
            continue
        graded.append({'key': key, 'question': q, 'gain_bits': round(g, 2),
                       'splits': 2, 'yes': yes, 'no': len(vals) - yes,
                       'kind': 'boolean', 'options': list(YES_NO), 'fact': None})
    graded.sort(key=lambda x: -x['gain_bits'])
    return (out + graded)[:top_n]


def needed_facts(results, top=None):
    """Факты, без которых не считается вердикт по показанным услугам.

    `top` ограничивает верхом выдачи: факт, решающий судьбу восьмой строки,
    посетителю сейчас не нужен, а вопрос о нём выглядит как придирка.
    """
    rows = results[:top] if top else results
    return sorted({d['need_fact'] for r in rows for d in (r.get('reasons') or ())
                   if r.get('status') == 'unknown' and 'need_fact' in d})


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


# Шкала здесь КОСИНУСНАЯ, и это единственное, что определяет величину веса.
# Замерено: разброс косинуса по всему пулу из 30 кандидатов — 0.058, медианный
# отрыв соседних позиций — 0.013. Прежние 0.35 были в шесть раз больше всего
# разброса, то есть ответ не двигал кандидата, а перекладывал список в три блока
# («совпало» / «признак не заполнен» / «не совпало»), и оценка ретривера внутри
# блоков уже ничего не решала.
#
# При безошибочных ответах разницы между 0.02 и 0.35 нет (R@1 0.816 у обоих).
# Разница появляется там, где ответ приходит из речи через ASR и извлечение:
#     доля ошибочных ответов     0%      15%     30%
#     weight = 0.35            0.816   0.623   0.462
#     weight = 0.02            0.816   0.692   0.578
#     не спрашивать вообще     0.732   0.732   0.732
# При 0.35 уже 15% ошибок опускают систему НИЖЕ уровня «не задавать вопросов».
# 0.02 деградирует полого и цену ошибки оставляет обратимой — ради чего
# переранжирование и выбрано вместо фильтра.
RESCORE_WEIGHT = 0.02


def rescore(candidates, scores, key, answer, weight=RESCORE_WEIGHT):
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
