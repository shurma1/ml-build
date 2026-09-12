# -*- coding: utf-8 -*-
"""Тесты того, что ломается молча.

Здесь нет проверок «сервис отвечает 200» — это видно и так. Здесь закреплены
три инварианта, нарушение которых не вызывает ни ошибки, ни падения, а просто
тихо ухудшает продукт на десятки процентов:

  1. промпт извлечения только ДОПИСЫВАЕТСЯ (иначе рассыпается кэш префикса
     и возвращается нестабильность от перестановки фактов);
  2. префикс эмбеддера входит в отпечаток (иначе индекс и запрос разъезжаются);
  3. выдуманная цитата не доходит до оператора (он показывает её посетителю
     как строку регламента).

Запуск:  python3 -m pytest gpu/tests -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpu.gateway import prompts


TURNS = [
    {"seq": 1, "text": "Здравствуйте, чем могу помочь?", "speaker": None},
    {"seq": 2, "text": "Я ветеран труда, хочу оформить выплату", "speaker": None},
    {"seq": 3, "text": "Мне шестьдесят три, живу в Щёкино", "speaker": None},
]


def test_system_prompt_stable_across_turns():
    """Системное сообщение не меняется от хода к ходу — оно и есть кэшируемый префикс."""
    a = prompts.extract_messages(TURNS[:2])
    b = prompts.extract_messages(TURNS)
    assert a[0]["content"] == b[0]["content"]


def test_dialog_is_append_only():
    """Каждый следующий ход — строгое расширение предыдущего, без перезаписи.

    Если этот тест упал, значит кто-то начал пересобирать историю (суммаризация,
    окно, переупорядочивание). Префилл на ход вырастет с ~30 токенов до ~1200.
    """
    prev = ""
    for n in range(1, len(TURNS) + 1):
        cur = prompts.extract_messages(TURNS[:n])[1]["content"]
        assert cur.startswith(prev), f"ход {n} переписал историю, а не дописал"
        prev = cur


def test_schema_enums_are_in_the_prompt():
    """Перечисления должны быть в промпте дословно: 68% значений приходили вне
    схемы, когда enum'ы описывались словами."""
    for value in ("Рождение ребенка", "Щекинский район", "ВБД"):
        assert value in prompts.SYSTEM_EXTRACT


def test_question_is_not_a_fact_rule_present():
    """Единственная защита от ошибки атрибуции ролей на уровне промпта.
    Диаризации нет; если это правило пропадёт, вопросы оператора станут фактами."""
    assert "Вопрос — не факт" in prompts.SYSTEM_EXTRACT


def test_document_precedes_question_in_ask():
    """Документ раньше вопроса — иначе второй вопрос по той же карточке
    не попадёт в кэш и будет стоить как первый."""
    msgs = prompts.ask_messages({"documentsText": "паспорт, СНИЛС"}, "а нужен ли ИНН?")
    doc_at = next(i for i, m in enumerate(msgs) if "паспорт, СНИЛС" in m["content"])
    q_at = next(i for i, m in enumerate(msgs) if m["content"] == "а нужен ли ИНН?")
    assert doc_at < q_at


def test_embed_fingerprint_covers_prefix():
    """Смена префикса обязана менять отпечаток. Иначе индекс, собранный со
    старым префиксом, молча считается совместимым."""
    from gpu.gateway import config as C
    from gpu.gateway.embed import fingerprint
    original = C.QUERY_PREFIX
    before = fingerprint()
    try:
        C.QUERY_PREFIX = "Instruct: совсем другая задача\nQuery: "
        assert fingerprint() != before
    finally:
        C.QUERY_PREFIX = original
    assert fingerprint() == before


def test_invented_quote_is_stripped():
    """Цитата, которой нет в основаниях, не доходит до оператора.

    Он показывает её посетителю как строку регламента — пересказ под видом
    цитаты хуже, чем отсутствие цитаты.
    """
    from gpu.gateway.llm import _strip_invented_quotes
    real = "принимающие участие в специальной военной операции"
    text = f"Услуга не положена: «{real}». Также «этого в регламенте не было»."
    out = _strip_invented_quotes(text, [real])
    assert real in out
    assert "этого в регламенте не было" not in out


def test_citation_offsets_point_into_the_field():
    """Смещения цитаты считаются в исходном тексте поля — их же использует
    подсветка в карточке на стороне core-api."""
    from gpu.gateway.llm import verify_citations
    doc = {"documentsText": "Нужны паспорт, СНИЛС и справка о доходах за 3 месяца."}
    cits = verify_citations("Понадобится «справка о доходах за 3 месяца».", doc)
    assert len(cits) == 1
    c = cits[0]
    assert c["field"] == "documentsText"
    assert doc["documentsText"][c["start"]:c["end"]] == c["quote"]
