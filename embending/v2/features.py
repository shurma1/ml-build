# -*- coding: utf-8 -*-
"""Признаки, которые LLM вытаскивает из распознанного диалога оператор↔клиент.

Ключевое архитектурное решение: LLM НЕ формулирует поисковый запрос строкой.
Она заполняет типизированную структуру. Из неё поиск сам строит:
  * 1-4 коротких канонических формулировки намерения -> плотный поиск (батчем);
  * фасеты -> жёсткие фильтры (муниципалитет, тип заявителя);
  * факты о клиенте -> проверка права на услугу ПОСЛЕ поиска, не вместо него.

Разделение обязательно: услугу, на которую клиент не имеет права, всё равно надо
найти и показать с причиной отказа. Фильтр по праву на этапе поиска её потеряет.
"""
from dataclasses import dataclass, field
from typing import Any

RECIPIENTS = ('person', 'ip', 'organization')

# Схема, которую отдаёт LLM. Держим её маленькой: чем меньше полей,
# тем стабильнее извлечение и тем дешевле вызов.
EXTRACTION_SCHEMA = {
    "intents": "1-4 коротких формулировки того, что человеку нужно, по 3-8 слов, "
               "деловым языком, без канцелярита. Если намерений несколько — перечисли все.",
    "life_situation": "одна из: Рождение ребенка, Многодетная семья, Детские пособия, "
                      "Выход на пенсию, Утрата документов, Смена места жительства, "
                      "Перемена имени, Индивидуальное жилищное строительство, "
                      "Сделки с недвижимостью, Открытие своего дела, Услуги опеки, "
                      "Юридические услуги, Агентские услуги, Чернобыльские выплаты, "
                      "Меры поддержки СВО, Меры поддержки военнослужащих/ветеранов, "
                      "Первичный въезд иностранного гражданина — или null",
    "recipient": "person | ip | organization | null",
    "municipality": "муниципальное образование Тульской области или null",
    "attributes": "ключевые сущности из диалога: документы, объекты, статусы. 0-8 штук",
    "facts": {
        "age": "возраст клиента, число или null",
        "children": "число детей или null",
        "has_property": "true/false/null",
        "categories": "льготные категории: ВБД, СВО, ЧАЭС, многодетный, инвалид, пенсионер, малоимущий",
    },
    "unresolved": "чего в диалоге НЕ прозвучало, но это важно для выбора услуги",
}


@dataclass
class QueryFeatures:
    intents: list = field(default_factory=list)
    life_situation: Any = None
    recipient: Any = None
    municipality: Any = None
    attributes: list = field(default_factory=list)
    documents: list = field(default_factory=list)   # что назвал посетитель; в поиск не идёт
    facts: dict = field(default_factory=dict)
    unresolved: list = field(default_factory=list)
    raw_text: str = ''          # ручной ввод или последняя реплика — запасной путь

    @classmethod
    def from_llm(cls, d):
        """Признаки из извлечения.

        `attributes` здесь останется пустым, и это не оплошность вызывающего:
        поля `attributes` в `extraction_schema.JSON_SCHEMA` нет, а схема отдаётся
        движку как guided_json — модель физически не может его вернуть. Живо
        только для `from_text()`, где список заполняет не LLM.

        Названия документов модель возвращает в `documents` — раньше они
        доходили до `DialogState` и там терялись: ни поиск, ни состояние
        диалога их не читали, а токены вывода на них тратились каждый ход.
        Теперь они хотя бы доходят до оператора (`render_state`).

        В энкодер документы СОЗНАТЕЛЬНО не идут. Отдельной поисковой строкой они
        встали бы в центроид наравне с намерением, то есть при одном намерении
        забрали бы половину запроса, — а измерить это не на чем: в эталоне из 690
        запросов извлечения нет. Это кандидат на замер по живому логу, не правка
        вслепую.
        """
        d = d or {}
        f = d.get('facts') or {}
        return cls(
            intents=[s.strip() for s in (d.get('intents') or []) if s and s.strip()][:4],
            life_situation=d.get('life_situation') or None,
            recipient=d.get('recipient') if d.get('recipient') in RECIPIENTS else None,
            municipality=d.get('municipality') or None,
            attributes=[s.strip().lower() for s in (d.get('attributes') or []) if s][:8],
            documents=[s.strip() for s in (d.get('documents') or []) if s and s.strip()][:6],
            facts={k: v for k, v in f.items() if v not in (None, '', [], {})},
            unresolved=list(d.get('unresolved') or []),
            raw_text=d.get('raw_text') or '',
        )

    @classmethod
    def from_text(cls, text):
        """Ручной ввод: одна строка — одно намерение. Фасеты добираются газеттиром."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from facets import find_municipality, find_recipient
        return cls(intents=[text.strip()], municipality=find_municipality(text),
                   recipient=find_recipient(text), raw_text=text)

    def search_strings(self):
        """Строки, которые пойдут в энкодер. Батчем — одна прогонка вместо N."""
        out = list(self.intents)
        if self.attributes:
            out.append(' '.join(self.attributes))
        if not out and self.raw_text:
            out.append(self.raw_text)
        if self.life_situation:
            out = [f'{s}. {self.life_situation}' if i == 0 else s for i, s in enumerate(out)]
        return out[:5]


EXTRACTION_PROMPT = """Ты разбираешь расшифровку разговора оператора МФЦ с посетителем.
Верни СТРОГО JSON по схеме ниже, без пояснений.

Схема:
{schema}

Правила:
- intents: то, зачем человек пришёл, ДЕЛОВЫМ языком, но без канцелярита.
  Если из разговора следует несколько разных услуг — перечисли все, не выбирай одну.
- Не додумывай факты, которых в разговоре нет: ставь null.
- unresolved: чего оператор НЕ спросил, но без этого услугу не выбрать.

Разговор:
{dialog}"""
