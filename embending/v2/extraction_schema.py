# -*- coding: utf-8 -*-
"""Жёсткая схема извлечения признаков + валидация + монотонное состояние диалога.

Зачем: описание enum'ов словами в промпте LLM игнорирует. Замерено на 132 репликах —
68% значений приходили вне схемы («life_situation: worker on production», «categories:
выплата»), и каждый ход модель выдумывала новую формулировку. Отсюда 61% дрейфа поля.

Валидация по enum убирает 89% дрейфа, монотонное слияние — остаток:
    свободный JSON        исчезло 6.8%   заменено 60.6%
    + валидация           исчезло 0.8%   заменено  6.8%
    + монотонность        исчезло 0.0%   заменено  6.8%

JSON_SCHEMA ниже отдаётся движку как guided_json / grammar — тогда невалидное
значение физически не может быть сгенерировано, а validate() остаётся страховкой.
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LIFE_SITUATIONS = [
    "Рождение ребенка", "Многодетная семья", "Детские пособия", "Выход на пенсию",
    "Утрата документов", "Смена места жительства", "Перемена имени",
    "Индивидуальное жилищное строительство", "Сделки с недвижимостью",
    "Открытие своего дела", "Услуги опеки", "Юридические услуги", "Агентские услуги",
    "Чернобыльские выплаты", "Меры поддержки СВО",
    "Меры поддержки военнослужащих/ветеранов боевых действий",
    "Первичный въезд иностранного гражданина",
]
RECIPIENTS = ["person", "ip", "organization"]
CATEGORIES = ["ВБД", "СВО", "ЧАЭС", "многодетный", "инвалид", "пенсионер",
              "малоимущий", "военнослужащий"]
MUNICIPALITIES = [
    "город Тула", "город Алексин", "город Донской", "город Ефремов",
    "город Новомосковск", "р.п. Новогуровский", "р.п. Славный", "Арсеньевский район",
    "Белевский район", "Богородицкий район", "Веневский район", "Воловский район",
    "Дубенский район", "Заокский район", "Каменский район", "Кимовский район",
    "Киреевский район", "Куркинский район", "Одоевский район", "Плавский район",
    "Суворовский район", "Тепло-Огаревский район", "Узловский район",
    "Чернский район", "Щекинский район", "Ясногорский район",
]

# Отдаётся движку: vLLM guided_json / xgrammar / outlines / llama.cpp grammar.
JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intents"],
    "properties": {
        "intents": {"type": "array", "minItems": 1, "maxItems": 4,
                    "items": {"type": "string", "maxLength": 90}},
        "life_situation": {"type": ["string", "null"], "enum": LIFE_SITUATIONS + [None]},
        "recipient": {"type": ["string", "null"], "enum": RECIPIENTS + [None]},
        "municipality": {"type": ["string", "null"], "enum": MUNICIPALITIES + [None]},
        "categories": {"type": "array", "maxItems": 4,
                       "items": {"type": "string", "enum": CATEGORIES}},
        "age": {"type": ["integer", "null"], "minimum": 0, "maximum": 120},
        "children": {"type": ["integer", "null"], "minimum": 0, "maximum": 15},
        "documents": {"type": "array", "maxItems": 6,
                      "items": {"type": "string", "maxLength": 40}},
        "unresolved": {"type": "array", "maxItems": 3,
                       "items": {"type": "string", "maxLength": 80}},
    },
}

_ENUMS = {'life_situation': set(LIFE_SITUATIONS), 'recipient': set(RECIPIENTS),
          'municipality': set(MUNICIPALITIES)}
_LIST_ENUMS = {'categories': set(CATEGORIES)}
_INTS = {'age': (0, 120), 'children': (0, 15)}


def validate(raw):
    """Страховка на приёме: всё вне схемы отбрасывается молча.

    Нужна даже при constrained decoding — движок может быть без поддержки грамматик,
    может отвалиться на таймауте, может прийти ответ из кэша старой версии схемы.
    """
    d = raw if isinstance(raw, dict) else {}
    out = {}
    for k, allowed in _ENUMS.items():
        v = d.get(k)
        if isinstance(v, str) and v in allowed:
            out[k] = v
        elif k == 'municipality' and isinstance(v, str) and v.strip():
            # LLM может вернуть разговорную форму («Щекино», «в Туле») — приводим
            # тем же газеттиром, что и поиск, вместо того чтобы молча терять факт
            from facets import find_municipality
            mo = find_municipality(v)
            if mo:
                out[k] = mo
    for k, allowed in _LIST_ENUMS.items():
        v = [x for x in (d.get(k) or []) if isinstance(x, str) and x in allowed]
        if v:
            out[k] = sorted(set(v))
    for k, (lo, hi) in _INTS.items():
        v = d.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and lo <= v <= hi:
            out[k] = int(v)
    ints = [s.strip() for s in (d.get('intents') or [])
            if isinstance(s, str) and 3 <= len(s.strip()) <= 90]
    if ints:
        out['intents'] = ints[:4]
    for k in ('documents', 'unresolved'):
        v = [s.strip() for s in (d.get(k) or []) if isinstance(s, str) and s.strip()]
        if v:
            out[k] = v[:6]
    return out


def parse_llm_output(text):
    """Достать JSON из ответа модели, даже если она обрамила его текстом."""
    if isinstance(text, dict):
        return validate(text)
    t = re.sub(r'(?s)<think>.*?</think>', '', text or '')
    m = re.search(r'\{.*\}', t, re.S)
    if not m:
        return {}
    try:
        return validate(json.loads(m.group(0)))
    except Exception:
        return {}


class DialogState:
    """Состояние диалога. Только накапливается — заполненное поле не обнуляется.

    Это отдельное от валидации требование: даже валидный ответ может не содержать
    ранее найденный факт (LLM видит окно, а не весь диалог). Молчание не означает
    отмену. Явная отмена возможна только через drop().
    """

    # intents переформулируются каждый ход по замыслу — в стабильное состояние не входят
    VOLATILE = {'intents', 'unresolved'}

    def __init__(self):
        self.state, self.volatile, self.history = {}, {}, []

    def update(self, raw):
        d = validate(raw)
        changed = {}
        for k, v in d.items():
            if k in self.VOLATILE:
                self.volatile[k] = v
                continue
            old = self.state.get(k)
            if old is None:
                self.state[k] = v; changed[k] = (None, v)
            elif isinstance(old, list) and isinstance(v, list):
                merged = sorted(set(old) | set(v))
                if merged != old:
                    self.state[k] = merged; changed[k] = (old, merged)
            elif v != old:
                # противоречие: клиент поправил себя. Берём новое, но помечаем.
                self.state[k] = v; changed[k] = (old, v)
        self.history.append(changed)
        return changed

    def drop(self, key):
        self.state.pop(key, None)

    def as_features(self):
        d = dict(self.state)
        d.update(self.volatile)
        d['facts'] = {k: d.pop(k) for k in ('age', 'children', 'categories') if k in d}
        return d


EXTRACTION_PROMPT = """Извлеки факты о посетителе МФЦ из разговора.
Верни JSON по схеме. Значения перечислимых полей бери ТОЛЬКО из списков.
Чего в разговоре нет — ставь null или пустой список. Ничего не додумывай.

Разговор:
{dialog}"""
