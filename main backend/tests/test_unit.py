# -*- coding: utf-8 -*-
"""Проверки, которым не нужен ни сервер, ни шлюз, ни база.

Здесь заперты инварианты, которые ломаются молча: проверка адреса шлюза,
сравнение секрета за константное время, вырезание токена из логов и
согласованность подсветки с вердиктом о праве.
"""
import logging
import os
import sys

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(ROOT, "embending", "v2"))
os.environ.setdefault("ADMIN_SECRET", "x" * 40)

from core import config as C                                     # noqa: E402
from core.gateway import validate_base_url, GatewayConfigError    # noqa: E402
from core.logging_conf import RedactFilter, redact, remember_secret, mask  # noqa: E402


# --- адрес шлюза ------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://abc-10100.proxy.runpod.net",
    "https://gpu.mfc.example.ru",
    "http://localhost:10100",
    "http://127.0.0.1:8901",
])
def test_base_url_принимается(url):
    assert validate_base_url(url) == url.rstrip("/")


@pytest.mark.parametrize("url,почему", [
    ("http://evil.example.com", "http только для localhost"),
    ("https://x.example.com/api", "путь запрещён"),
    ("https://x.example.com?token=1", "параметры запрещены"),
    ("ftp://x.example.com", "чужая схема"),
    ("https://user:pass@x.example.com", "учётные данные"),
    ("", "пусто"),
    (None, "не строка"),
])
def test_base_url_отвергается(url, почему):
    with pytest.raises(GatewayConfigError):
        validate_base_url(url)


# --- секрет -----------------------------------------------------------------

def test_секрет_сравнивается_за_константное_время():
    """Не стилистика: обычное == выходит на первом несовпавшем байте,
    и по времени ответа секрет подбирается посимвольно."""
    import ast
    import inspect
    from core.routers import admin
    src = inspect.getsource(admin._authorized)
    assert "compare_digest" in src
    # Проверяем дерево, а не текст: в тексте «==» встречается в комментарии.
    import textwrap
    дерево = ast.parse(textwrap.dedent(src))
    сравнения = [n for n in ast.walk(дерево) if isinstance(n, ast.Compare)
                 and any(isinstance(o, (ast.Eq, ast.NotEq)) for o in n.ops)]
    assert not сравнения, "секрет сравнивается оператором, а не compare_digest"


def test_короткий_admin_secret_не_проходит(monkeypatch):
    monkeypatch.setenv("ADMIN_SECRET", "короткий")
    with pytest.raises(C.ConfigError):
        C.admin_secret()


def test_пустой_admin_secret_не_проходит(monkeypatch):
    monkeypatch.delenv("ADMIN_SECRET", raising=False)
    with pytest.raises(C.ConfigError):
        C.admin_secret()


# --- логи -------------------------------------------------------------------

TOKEN = "gw-live-token-9f1c2a3b4d"


def _capture(logger_name, msg, *args):
    import io
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.addFilter(RedactFilter())
    lg = logging.getLogger(logger_name)
    lg.handlers = [h]
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    lg.debug(msg, *args)
    return buf.getvalue()


def test_токен_не_попадает_в_лог_ни_одним_путём():
    remember_secret(TOKEN)
    случаи = [
        ("websockets.client", "= connection open: %s",
         (f"wss://gpu.example/v1/asr/stream?session_id=1&token={TOKEN}",)),
        ("httpx", "HTTP Request: POST %s headers=%s",
         ("https://gpu.example/v1/embed", {"Authorization": f"Bearer {TOKEN}"})),
        ("core", "конфигурация: %s", ({"token": TOKEN, "base_url": "https://x"},)),
        ("core", f"токен {TOKEN} в самом сообщении", ()),
        ("core", "вложено: %s", ([{"headers": {"authorization": f"Bearer {TOKEN}"}}],)),
    ]
    for name, msg, args in случаи:
        out = _capture(name, msg, *args)
        assert TOKEN not in out, f"токен протёк: {name} -> {out}"


def test_редактирование_не_ломает_access_лог_uvicorn():
    """Форматтер access-лога uvicorn распаковывает ровно пять аргументов.
    Очистка args сломала бы каждую строку доступа — и это уже случалось."""
    from uvicorn.logging import AccessFormatter
    remember_secret(TOKEN)
    rec = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 1, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1", "GET", f"/v1/x?token={TOKEN}", "1.1", 200), None)
    assert RedactFilter().filter(rec)
    assert len(rec.args) == 5 and rec.args[4] == 200
    line = AccessFormatter(use_colors=False).format(rec)
    assert TOKEN not in line and "200" in line


def test_маска_показывает_только_четыре_символа():
    assert mask(TOKEN) == TOKEN[:4] + "***"
    assert mask("abc") == "***"
    assert mask("") is None


def test_пароль_базы_вырезается():
    assert "emb" not in redact("host=h port=5434 password=emb dbname=d").split("password=")[1]


# --- подсветка --------------------------------------------------------------

@pytest.fixture(scope="module")
def каталог():
    from core.catalog import catalog
    if not catalog.types:
        catalog.load()
    return catalog


def test_все_цитаты_предикатов_находятся_в_карточке(каталог):
    """Цитата отбрасывалась при генерации, если не встречалась в исходнике
    дословно. Значит подсветка — это поиск подстроки, и он обязан находить всё."""
    from core import highlights as H
    from eligibility import load_predicates
    всего = найдено = 0
    for key, preds in load_predicates().items():
        t = каталог.by_key.get(key)
        if not t:
            continue
        fields = каталог.raw_fields(t["instances"][0]["id"])
        for p in preds:
            if not p.get("quote"):
                continue
            всего += 1
            найдено += bool(H.find_quote(fields, p["quote"]))
    assert всего > 300
    assert найдено == всего, f"не найдено {всего - найдено} цитат из {всего}"


def test_смещения_подсветки_указывают_на_настоящий_текст(каталог):
    from core import highlights as H
    t = каталог.by_key["zабота выдача карты"]
    fields = каталог.raw_fields(t["instances"][0]["id"])
    for h in H.for_type(t, fields, {"categories": ["СВО"], "recipient": "person"}):
        текст = fields[h["field"]]
        assert 0 <= h["start"] < h["end"] <= len(текст)
        assert текст[h["start"]:h["end"]].strip()


def test_подсветка_не_противоречит_вердикту(каталог):
    """Оператор не должен видеть «не положено» и зелёную строку под ним."""
    from core import highlights as H
    from eligibility import check
    t = каталог.by_key["zабота выдача карты"]
    fields = каталог.raw_fields(t["instances"][0]["id"])
    src = каталог.src[t["type_id"]]
    for facts in ({"categories": ["СВО"], "recipient": "person"},
                  {"categories": ["пенсионер"], "recipient": "person"},
                  {"recipient": "person"}):
        статус, _ = check(t, facts, src)
        виды = {h["kind"] for h in H.for_type(t, fields, facts)}
        if статус == "blocked":
            assert "blocking" in виды and "matching" not in виды
        elif статус == "eligible":
            assert "blocking" not in виды


# --- состояние диалога ------------------------------------------------------

def test_состояние_только_накапливается():
    from core.sessions import new_dialog_state, render_state, restore_state
    st = new_dialog_state()
    st.update({"intents": ["пособие"], "recipient": "person", "age": 30})
    st.update({"intents": ["выплата"]})                 # факты не пришли
    assert st.state["recipient"] == "person" and st.state["age"] == 30
    снимок = render_state(st, {"age"})
    assert снимок["facts"]["age"] == 30 and снимок["pinned"] == ["age"]
    # снимок восстанавливается обратно без потерь
    назад = restore_state(снимок)
    assert назад.state["recipient"] == "person" and назад.state["age"] == 30


def test_закреплённое_поле_не_отдаётся_извлечению():
    """Поле, выставленное оператором, извлечение не трогает: иначе следующая же
    реплика затрёт правку, ради которой оператор и вмешался."""
    from core.sessions import new_dialog_state
    st = new_dialog_state()
    st.state["age"] = 41
    pinned = {"age"}
    raw = {"age": 70, "children": 2}
    st.update({k: v for k, v in raw.items() if k not in pinned})
    assert st.state["age"] == 41 and st.state["children"] == 2


def test_снятый_факт_остаётся_на_экране_но_из_поиска_уходит():
    """Нажатие по чипу не стирает факт, а снимает его с работы.

    Стереть нельзя: модель назовёт его на следующей же реплике снова, и оператор
    будет снимать одно и то же по кругу. Поэтому значение остаётся в состоянии —
    оператор видит зачёркнутым, что уже отверг, — а из признаков поиска уходит.
    """
    from core.sessions import (new_dialog_state, render_state, restore_dismissed,
                               strip_dismissed, without_dismissed)
    from features import QueryFeatures
    st = new_dialog_state()
    st.update({"intents": ["пособие"], "municipality": "город Тула",
               "categories": ["пенсионер", "многодетный"]})
    снятое = {"municipality": {"город Тула"}, "categories": {"пенсионер"}}

    снимок = render_state(st, set(), None, снятое)
    assert снимок["municipality"] == "город Тула", "факт пропал с экрана"
    assert снимок["dismissed"] == {"municipality": ["город Тула"],
                                   "categories": ["пенсионер"]}

    f = strip_dismissed(QueryFeatures.from_llm(st.as_features()), снятое)
    assert f.municipality is None
    assert f.facts["categories"] == ["многодетный"], "снята вся категория целиком"

    # извлечение назвало снятое снова — в состояние оно не возвращается
    assert without_dismissed({"municipality": "город Тула", "categories": ["пенсионер"]},
                             снятое) == {}
    # и переживает обрыв WS вместе со снимком
    assert restore_dismissed(снимок) == снятое


def test_снимается_значение_а_не_поле():
    """Клиент, поправивший себя, должен быть услышан: блокируется ровно то
    значение, от которого отказались, — иначе одно нажатие глушит поле."""
    from core.sessions import strip_dismissed, without_dismissed
    from features import QueryFeatures
    снятое = {"municipality": {"город Тула"}}
    assert without_dismissed({"municipality": "Щекинский район"}, снятое) \
        == {"municipality": "Щекинский район"}
    f = strip_dismissed(QueryFeatures.from_llm({"municipality": "Щекинский район"}), снятое)
    assert f.municipality == "Щекинский район"


# --- наводящие вопросы -------------------------------------------------------

def _тип(tid, title, life=(), recipients=(), municipalities=()):
    return {"type_id": tid, "title": title, "life": list(life),
            "recipients": list(recipients), "municipalities": list(municipalities)}


def _признаки(**kw):
    from features import QueryFeatures
    return QueryFeatures(**kw)


def test_не_спрашиваем_про_то_что_уже_прозвучало():
    """Худший наводящий вопрос — про факт, который посетитель только что назвал:
    это не уточнение, а сообщение, что его не слушали."""
    from clarify import suggest
    cands = [_тип(1, "Пособие многодетным", ["Многодетная семья"], ["person"]),
             _тип(2, "Выплата ветеранам", ["Меры поддержки СВО"], ["person"]),
             _тип(3, "Регистрация ИП", ["Открытие своего дела"], ["ip"]),
             _тип(4, "Справка о составе семьи", [], ["person"])]
    голые = {q["key"] for q in suggest(cands, top_n=9)}
    assert голые, "без известных фактов вопросы должны находиться"

    знаем = _признаки(life_situation="Многодетная семья", recipient="person",
                      municipality="город Тула", facts={"categories": ["многодетный"]})
    ключи = {q["key"] for q in suggest(cands, feats=знаем, top_n=9)}
    assert not ключи & {"life", "recipient", "municipality", "льгота", "ребенок"}, ключи


def test_вопрос_обязан_резать_список():
    """Грань, по которой все кандидаты одинаковы, не снимает ничего.

    Прежняя мера брала энтропию НАБОРОВ значений: у каждой услуги свой кортеж
    ситуаций, все кортежи разные — и вопрос получал 3.4 бита, не отсекая никого.
    """
    from clarify import suggest
    одинаковые = [_тип(i, f"Выдача справки №{i}", ["Утрата документов"], ["person"])
                  for i in range(1, 9)]
    assert [q for q in suggest(одинаковые, top_n=9) if q["key"] == "life"] == []

    разные = ([_тип(i, f"Выплата №{i}", ["Детские пособия"], ["person"]) for i in range(1, 5)]
              + [_тип(i, f"Справка №{i}", ["Выход на пенсию"], ["person"]) for i in range(5, 9)])
    жизнь = [q for q in suggest(разные, top_n=9) if q["key"] == "life"]
    assert жизнь and жизнь[0]["gain_bits"] >= 0.8


def test_нехватка_факта_для_вердикта_спрашивается_первой():
    """Самый честный вопрос — не про список вообще, а про услугу наверху выдачи,
    вердикт по которой без этого факта не считается."""
    from clarify import suggest
    cands = [_тип(1, "Выплата многодетным", ["Детские пособия"], ["person"]),
             _тип(2, "Выдача паспорта", ["Утрата документов"], ["person"])]
    q = suggest(cands, needed=["age"], top_n=2)
    assert q[0]["key"] == "age" and q[0]["gain_bits"] is None
    assert "лет" in q[0]["question"]
    # и не дублируется гранью про то же самое
    q = suggest(cands, needed=["categories"], top_n=9)
    assert sum(1 for x in q if x["key"] == "льгота") == 1


def test_вопросов_может_не_быть_вовсе():
    """Вопрос ради вопроса стоит посетителю времени, а оператору — доверия."""
    from clarify import suggest
    один = [_тип(1, "Выдача справки", ["Утрата документов"], ["person"])]
    assert suggest(один) == []
    одинаковые = [_тип(i, f"Выдача справки №{i}", ["Утрата документов"], ["person"])
                  for i in range(1, 9)]
    знаем = _признаки(life_situation="Утрата документов", recipient="person",
                      municipality="город Тула", facts={"children": 0, "categories": ["СВО"]})
    assert suggest(одинаковые, feats=знаем) == []


# --- то, что уходит в extract ------------------------------------------------

def _сессия():
    from core.sessions import Session
    return Session({"session_id": "00000000-0000-0000-0000-000000000001",
                    "branch_id": "b", "operator_id": "o", "municipality": None,
                    "started_at": None, "closed_at": None})


def test_в_extract_уходит_вся_хронология_только_дописыванием():
    """Порядок фиксирован, окно не режется, реплики не переставляются.

    На этом держится попадание в prefix cache vLLM (префилл падает с ~1200
    токенов на ход до ~30) и устойчивость модели: у 9B измерено 14% смены
    вердикта от одной лишь перестановки фактов.
    """
    s = _сессия()
    снимки = []
    for i, (текст, пусто) in enumerate([("здравствуйте, мне нужна выплата", False),
                                        ("ага", True),
                                        ("я многодетная мать", False),
                                        ("понятно", True),
                                        ("трое детей", False)], start=1):
        s.seq = i
        s.turns.append({"seq": i, "text": текст, "speaker": "client", "empty": пусто})
        снимки.append(s.llm_turns())

    последний = снимки[-1]
    # хронология не нарушена
    assert [t["seq"] for t in последний] == sorted(t["seq"] for t in последний)
    # пустые сегменты не попадают — извлекать из них нечего
    assert all("ага" not in t["text"] and "понятно" not in t["text"] for t in последний)
    # и главное: каждый следующий запрос — расширение предыдущего, а не новый
    for раньше, позже in zip(снимки, снимки[1:]):
        assert позже[:len(раньше)] == раньше, "префикс изменился — кэш сломан"


def test_пустые_сегменты_не_ломают_дописывание():
    """Исключение «ага» допустимо именно потому, что сохраняет префикс."""
    s = _сессия()
    s.turns = [{"seq": 1, "text": "первая", "speaker": None, "empty": False},
               {"seq": 2, "text": "ага", "speaker": None, "empty": True},
               {"seq": 3, "text": "вторая", "speaker": None, "empty": False}]
    assert [t["seq"] for t in s.llm_turns()] == [1, 3]


# --- разбор регламента на сервере -------------------------------------------

@pytest.fixture(scope="module")
def карточки(каталог):
    return [(c, каталог.raw[c["id"]], каталог.types.get(c["type_id"]))
            for c in каталог.cards]


def test_смещения_пунктов_точны(карточки):
    """Каждый пункт несёт смещения в ИСХОДНОЕ поле.

    Это и есть то, что делает подсветку цитат возможной: интерфейс волен
    показывать короткую форму пункта, а подсветку накладывать на исходный
    текст по этим смещениям. Разъедутся — подсветка встанет не на ту строку.
    """
    from core import cardview as CV
    проверено = 0
    for _, raw, t in карточки[:200]:
        v = CV.build(_, raw, t)
        пункты = list(v["documents"]["always"]) + list(v["documents"]["situational"])
        for r in v["rejects"]:
            пункты.append(r)
            пункты.extend(r["also"])          # схлопнутые дубли — со своими смещениями
        for пункт in пункты:
            поле = raw.get(пункт["field"]) or ""
            assert поле[пункт["start"]:пункт["end"]] == пункт["fullText"], пункт
            проверено += 1
    assert проверено > 400, f"проверено всего {проверено} пунктов"


def test_срок_в_карточке_совпадает_с_canon(карточки):
    """`decisionDays` обязан быть тем же числом, что `ServiceCard.term_days`.

    Иначе оператор видит два разных срока на одном экране — в строке выдачи
    и в карточке, — и оба выглядят одинаково достоверно.
    """
    from core import cardview as CV
    from canon import term_days
    for _, raw, __ in карточки:
        canon_v = term_days(raw.get("timeTermText") or "")
        card_v = CV.parse_terms(raw.get("timeTermText")).get("decisionDays")
        if canon_v is not None:
            assert card_v == canon_v, f"{raw['id']}: карточка {card_v} против canon {canon_v}"


def test_бесплатность_считается_одним_источником(карточки):
    """`/бесплатно/` без оглядки на сумму в рублях — то, из-за чего карточка
    с госпошлиной показывалась как бесплатная."""
    from core import cardview as CV
    from canon import is_free
    for _, raw, t in карточки:
        assert CV.build(_, raw, t)["isFree"] == is_free(raw.get("paymentInfoText") or "")


def test_разбор_покрывает_корпус_а_не_образцы(карточки):
    """Сторож против возврата правил, написанных по пяти карточкам.

    Пороги взяты с запасом от измеренного: срок 87%, пустой чек-лист 1%,
    формула «кому положено» — у всех, у кого заполнено структурное поле.
    """
    from core import cardview as CV
    n = len(карточки)
    срок = пусто = без_опоры = 0
    for c, raw, t in карточки:
        v = CV.build(c, raw, t)
        if v["terms"]["decisionDays"] is not None:
            срок += 1
        if v["documents"]["total"] == 0 and (raw.get("documentsText") or "").strip():
            пусто += 1
        if not v["recipients"]["categories"] and not v["recipients"]["recipientIds"]:
            без_опоры += 1
    assert срок / n > 0.80, f"срок найден только у {срок}/{n}"
    assert пусто / n < 0.05, f"пустой чек-лист у {пусто}/{n}"
    assert без_опоры / n < 0.05, f"формула без опоры у {без_опоры}/{n}"


def test_формула_кому_положено_не_спорит_с_вердиктом(каталог):
    """Формула строится из тех же предикатов, что дают вердикт о праве,
    поэтому названная в ней категория обязана давать eligible."""
    from core import cardview as CV
    from eligibility import check
    t = каталог.by_key["zабота выдача карты"]
    sid = t["instances"][0]["id"]
    v = CV.build(каталог.card_by_id[sid], каталог.raw[sid], t)
    категории = v["recipients"]["categories"]
    assert категории, "категории не извлеклись"
    for кат in категории:
        статус, _ = check(t, {"categories": [кат], "recipient": "person"},
                          каталог.src[t["type_id"]])
        assert статус == "eligible", f"{кат} есть в формуле, но вердикт {статус}"
