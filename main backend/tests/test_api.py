# -*- coding: utf-8 -*-
"""Проверки живого API. Требуют поднятого core-api и дублёра шлюза.

    python3 tools/stub_gateway.py --port 10100 --token dev-token-abc &
    ADMIN_SECRET=… GW_BASE_URL=http://127.0.0.1:10100 GW_TOKEN=dev-token-abc python3 app.py &
    python3 -m pytest tests/test_api.py -q

Без них набор пропускается целиком, а не падает: это интеграционные проверки,
и их отсутствие не должно выглядеть поломкой кода.
"""
import itertools
import os
import random

import httpx
import pytest

CORE = os.getenv("CORE_URL", "http://127.0.0.1:8080")
STUB = os.getenv("STUB_URL", "http://127.0.0.1:10100")
STUB2 = os.getenv("STUB2_URL", "http://127.0.0.1:10101")
# Сервер может быть поднят с закрытым публичным доступом — набор обязан
# проходить в обеих конфигурациях, иначе «зелёные тесты» ничего не значат
# ровно там, где стоит прод. Объявлено ДО _жив(): его зовёт pytestmark
# на импорте модуля.
API_TOKEN = os.getenv("API_TOKEN", "")
AUTH = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
SECRET = os.getenv("ADMIN_SECRET", "0123456789012345678901234567890123")
H = {"X-Admin-Secret": SECRET}

# Начало со случайного места: окно лимита — минута, и два прогона набора подряд
# не должны делить корзины попыток.
_корзина = itertools.count(random.randint(0, 10 ** 6))


def адрес():
    """Своя корзина лимита попыток на каждый тест админ-роута.

    Лимит «10 в минуту с одного адреса» — рабочее ограничение, и обходить его
    в тестах нельзя. Но тесты и не должны мешать друг другу, поэтому каждый
    приходит со своего адреса. Работает только когда сервер поднят с
    ADMIN_TRUST_PROXY=1; отдельная проверка на сам лимит — в конце файла.
    """
    n = next(_корзина)
    return {**H, "X-Forwarded-For": f"203.0.{n // 250 % 250}.{n % 250 + 1}"}


def _жив(url):
    try:
        return httpx.get(f"{url}/v1/health" if "8080" in url else f"{url}/readyz",
                         headers=AUTH if "8080" in url else None,
                         timeout=3).status_code == 200
    except httpx.HTTPError:
        # Именно сетевая недоступность. Всё остальное (опечатка, NameError)
        # обязано падать громко: иначе весь набор молча «пропускается»
        # и выглядит зелёным.
        return False


pytestmark = pytest.mark.skipif(not _жив(CORE), reason="core-api не поднят")


GW_TOKEN = os.getenv("GW_TOKEN", "dev-token-abc")
STUB2_TOKEN = os.getenv("STUB2_TOKEN", "second-token-xyz")


@pytest.fixture(scope="module")
def c():
    return httpx.Client(base_url=CORE, timeout=60, headers=AUTH)


@pytest.fixture(scope="module", autouse=True)
def шлюз_на_месте(c):
    """Набор обязан начинаться с заведомо рабочего шлюза и на нём же кончаться.

    Адрес шлюза живёт в базе и переживает перезапуск — значит проверка, которая
    его меняет, способна испортить и следующий прогон, и соседний тест. Без
    возврата на место ошибка выглядит как «извлечение фактов сломалось»,
    хотя на деле шлюзу просто подставили чужой токен.
    """
    def поставить():
        r = c.post("/v1/admin/gateway", headers=адрес(),
                   json={"base_url": STUB, "token": GW_TOKEN})
        return r.status_code
    if _жив(STUB):
        assert поставить() == 200, "исходный дублёр шлюза не принимается"
    yield
    if _жив(STUB):
        поставить()


@pytest.fixture(scope="module")
def филиал(c):
    br = c.get("/v1/branches").json()
    return next(b for b in br if b["municipality"] == "Щекинский район")


@pytest.fixture
def сессия(c, филиал):
    s = c.post("/v1/sessions", json={"branch_id": филиал["branch_id"],
                                     "operator_id": "pytest"}).json()
    yield s
    c.post(f"/v1/sessions/{s['session_id']}/close", json={"outcome": "abandoned"})


# --- справочники ------------------------------------------------------------

def test_филиалы_отдаются_с_муниципалитетом(c):
    br = c.get("/v1/branches").json()
    assert len(br) == 114
    assert sum(1 for b in br if b["municipality"]) > 100


def test_версия_несёт_отпечаток_и_режим(c):
    v = c.get("/v1/version").json()
    assert v["api"] and v["corpus"]["cards"] == 732 and v["corpus"]["types"] == 318
    assert v["mode"] in ("dense", "degraded_lexical")
    assert v["gpu_gateway"] in ("up", "down", "degraded")


# --- сессия -----------------------------------------------------------------

def test_муниципалитет_берётся_из_филиала(c, сессия):
    assert сессия["municipality"] == "Щекинский район"
    assert сессия["ws_url"].startswith("ws")


def test_ход_диалога_даёт_состояние_и_список(c, сессия):
    sid = сессия["session_id"]
    r = c.post(f"/v1/sessions/{sid}/turns",
               json={"text": "я участник СВО, нужна карта Забота"})
    assert r.status_code == 200
    d = r.json()
    assert d["turn"]["source"] == "operator_typed" and not d["turn"]["empty"]
    assert d["search"]["results"] and d["search"]["mode"] in ("dense", "degraded_lexical")
    for x in d["search"]["results"]:
        assert x["status"] in ("eligible", "blocked", "unknown")
        assert "highlights" in x and "rank" in x


def test_пустой_сегмент_не_дёргает_llm(c, сессия):
    sid = сессия["session_id"]
    c.post(f"/v1/sessions/{sid}/turns", json={"text": "нужен загранпаспорт"})
    r = c.post(f"/v1/sessions/{sid}/turns", json={"text": "Ага"}).json()
    assert r.get("skipped") == "empty_segment"
    assert r["turn"]["empty"] is True


def test_состояние_накапливается_между_ходами(c, сессия):
    sid = сессия["session_id"]
    c.post(f"/v1/sessions/{sid}/turns", json={"text": "я многодетная мать"})
    r = c.post(f"/v1/sessions/{sid}/turns", json={"text": "живу в Щекино"}).json()
    кат = r["state"]["facts"]["categories"]
    assert "многодетный" in кат, "категория пропала на следующем ходу"


def test_закреплённый_факт_переживает_извлечение(c, сессия):
    sid = сессия["session_id"]
    c.post(f"/v1/sessions/{sid}/turns", json={"text": "хочу карту Забота"})
    r = c.post(f"/v1/sessions/{sid}/facts",
               json={"set": {"age": 41, "categories": ["СВО"]}}).json()
    assert set(r["state"]["pinned"]) == {"age", "categories"}
    r = c.post(f"/v1/sessions/{sid}/turns",
               json={"text": "я пенсионер, мне 70 лет"}).json()
    assert r["state"]["facts"]["age"] == 41, "извлечение перезаписало закреплённое"
    assert r["state"]["facts"]["categories"] == ["СВО"]
    # снятие закрепления возвращает поле извлечению
    c.post(f"/v1/sessions/{sid}/facts", json={"unpin": ["age", "categories"]})
    r = c.post(f"/v1/sessions/{sid}/turns",
               json={"text": "я пенсионер, мне 70 лет"}).json()
    assert r["state"]["facts"]["age"] == 70


def test_ответ_на_вопрос_переранжирует_а_не_фильтрует(c, сессия):
    """Жёсткий фильтр измерен и хуже: R@1 0.775 против 0.810."""
    sid = сессия["session_id"]
    r = c.post(f"/v1/sessions/{sid}/turns",
               json={"text": "нужна выплата на ребенка"}).json()
    вопросы = r["search"]["questions"]
    if not вопросы:
        pytest.skip("вопросов не предложено")
    было = len(r["search"]["results"])
    r2 = c.post(f"/v1/sessions/{sid}/answers",
                json={"answers": [{"key": вопросы[0]["key"], "value": True}]}).json()
    assert len(r2["results"]) >= было, "ответ выбросил кандидатов — это фильтрация"


def test_снимок_сессии_восстанавливает_ui(c, сессия):
    sid = сессия["session_id"]
    c.post(f"/v1/sessions/{sid}/turns", json={"text": "потерял паспорт"})
    s = c.get(f"/v1/sessions/{sid}").json()
    assert s["session_id"] == sid and s["turns"] and s["state"] and s["search"]


def test_закрытие_отдаёт_204_без_тела(c, филиал):
    s = c.post("/v1/sessions", json={"branch_id": филиал["branch_id"],
                                     "operator_id": "pytest"}).json()
    r = c.post(f"/v1/sessions/{s['session_id']}/close", json={"outcome": "served"})
    assert r.status_code == 204 and not r.content


def test_неизвестная_сессия_даёт_404(c):
    r = c.get("/v1/sessions/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "session_not_found"


# --- поиск и карточка -------------------------------------------------------

def test_разовый_поиск(c):
    r = c.post("/v1/search", json={"text": "хочу зарегистрировать брак", "k": 5}).json()
    assert len(r["results"]) <= 5 and r["results"]
    assert any("брак" in x["title"].lower() for x in r["results"])


def test_поиск_проверяет_право_после_поиска_а_не_вместо(c):
    """Услуга, на которую клиент не имеет права, обязана быть НАЙДЕНА
    и показана со статусом blocked и цитатой регламента."""
    r = c.post("/v1/search", json={
        "text": "выплата участникам СВО",
        "features": {"intents": ["выплата участникам СВО"], "recipient": "person",
                     "facts": {"categories": ["пенсионер"], "age": 70}},
        "k": 10}).json()
    blocked = [x for x in r["results"] if x["status"] == "blocked"]
    assert blocked, "ни одна неподходящая услуга не показана — их отфильтровали"
    for x in blocked:
        assert x["reasons"], "статус blocked без причины"
        assert any(rr.get("quote") for rr in x["reasons"]), "нет дословной цитаты"


def test_карточка_услуги_с_подсветкой(c):
    r = c.post("/v1/search", json={"text": "карта Забота", "k": 8}).json()
    sid = next(x["service_id"] for x in r["results"] if x["service_id"])
    card = c.get(f"/v1/services/{sid}").json()
    assert card["fields"] and card["title"]
    for h in card["highlights"]:
        текст = card["fields"][h["field"]]
        assert 0 <= h["start"] < h["end"] <= len(текст)
        assert h["kind"] in ("blocking", "matching", "unknown", "term", "payment")


def test_неизвестная_услуга_даёт_404(c):
    r = c.get("/v1/services/нет-такой")
    assert r.status_code == 404 and r.json()["error"]["code"] == "service_not_found"


def test_вопрос_по_документу_отдаёт_sse(c):
    if not _жив(STUB):
        pytest.skip("дублёр шлюза не поднят")
    r = c.post("/v1/search", json={"text": "карта Забота", "k": 8}).json()
    sid = next((x["service_id"] for x in r["results"] if x["service_id"]), None)
    if not sid:
        pytest.skip("в выдаче нет карточки с определённым муниципалитетом")
    with httpx.stream("POST", f"{CORE}/v1/services/{sid}/ask", headers=AUTH,
                      json={"question": "кому положена услуга?"}, timeout=60) as resp:
        assert resp.status_code == 200
        события = [l for l in resp.iter_lines() if l.startswith("event:")]
    assert any("delta" in e for e in события)
    assert any("done" in e or "citation" in e for e in события)


# --- админ-роут -------------------------------------------------------------

def test_админ_роут_скрыт_из_публичной_схемы(c):
    пути = c.get("/openapi.json").json()["paths"]
    assert not any("admin" in p for p in пути)


def test_неверный_секрет_отвергается(c):
    # значение заголовка обязано быть latin-1, поэтому секрет тут ASCII
    r = c.post("/v1/admin/gateway", headers={**адрес(), "X-Admin-Secret": "wrong-secret"},
               json={"base_url": STUB})
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"


@pytest.mark.parametrize("url", ["http://evil.example.com", "https://x.ru/api",
                                 "https://x.ru?a=1"])
def test_плохой_адрес_отвергается_до_сети(c, url):
    r = c.post("/v1/admin/gateway", headers=адрес(), json={"base_url": url})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "schema_validation_failed"


def test_мёртвый_адрес_даёт_502_и_ничего_не_меняет(c):
    a = адрес()
    было = c.get("/v1/admin/gateway", headers=a).json()["current"]["base_url"]
    r = c.post("/v1/admin/gateway", headers=a,
               json={"base_url": "http://localhost:59999", "token": "x"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "gpu_unavailable"
    assert c.get("/v1/admin/gateway", headers=a).json()["current"]["base_url"] == было


def test_токен_шлюза_никогда_не_отдаётся_целиком(c):
    d = c.get("/v1/admin/gateway", headers=адрес()).json()
    for место in (d["current"]["token"], d["stored"]["token"]):
        if место:
            assert место.endswith("***") and len(место) <= 7


def test_смена_адреса_без_перезапуска(c):
    if not _жив(STUB2):
        pytest.skip("второй дублёр шлюза не поднят")
    a = адрес()
    поколение = c.get("/v1/admin/gateway", headers=a).json()["current"]["generation"]
    r = c.post("/v1/admin/gateway", headers=a,
               json={"base_url": STUB2, "token": STUB2_TOKEN})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] and d["fingerprint_match"]
    новое = c.get("/v1/admin/gateway", headers=a).json()["current"]
    assert новое["base_url"] == STUB2.rstrip("/")
    assert новое["generation"] > поколение, "пулы клиентов не пересобрались"
    # поиск продолжает работать на новом адресе
    assert c.post("/v1/search", json={"text": "потерял паспорт", "k": 1}).json()["results"]
    # Вернуть ИМЕННО исходную пару «адрес + токен». Восстановить адрес, приложив
    # к нему токен соседнего шлюза, — значит оставить систему в тихой деградации.
    r = c.post("/v1/admin/gateway", headers=a, json={"base_url": STUB, "token": GW_TOKEN})
    assert r.status_code == 200, "шлюз не вернулся на исходный адрес"


def test_чужой_отпечаток_даёт_409(c):
    """Индекс собран одной моделью, запросы кодировались бы другой. Выдача
    осталась бы правдоподобной — поэтому отказ, а не предупреждение."""
    bad = os.getenv("STUB_BAD_URL", "http://127.0.0.1:10102")
    if not _жив(bad):
        pytest.skip("дублёр с чужим отпечатком не поднят")
    a = адрес()
    было = c.get("/v1/admin/gateway", headers=a).json()["current"]["base_url"]
    r = c.post("/v1/admin/gateway", headers=a, json={"base_url": bad, "token": "t2"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "index_fingerprint_mismatch"
    assert c.get("/v1/admin/gateway", headers=a).json()["current"]["base_url"] == было, \
        "адрес сохранился, несмотря на чужой отпечаток"


def test_лимит_попыток_к_админ_роуту(c):
    """Не более 10 попыток в минуту с одного адреса, дальше 429."""
    if not os.getenv("ADMIN_TRUST_PROXY", "1").lower() in ("1", "true", "yes"):
        pytest.skip("сервер не доверяет X-Forwarded-For — корзину не изолировать")
    # свежая корзина на каждый прогон: окно лимита — минута, и повторный
    # запуск набора в пределах неё не должен спотыкаться об остаток прошлого
    свой = {**H, "X-Forwarded-For": f"198.51.100.{random.randint(1, 250)}",
            "X-Admin-Secret": "wrong-secret"}
    коды = [c.post("/v1/admin/gateway", headers=свой,
                   json={"base_url": STUB}).status_code for _ in range(13)]
    assert 429 in коды, f"лимит не сработал: {коды}"
    assert коды.index(429) >= 10, f"лимит сработал слишком рано: {коды}"


def test_нечитаемый_отпечаток_не_проходит_как_совпавший(c):
    """Шлюз ответил на /readyz, но /v1/models отверг токен — отпечатка нет.

    Неизвестный отпечаток это НЕ совпавший: сохранить адрес здесь значит
    начать кодировать запросы неизвестно чем при живом индексе. Выдача
    останется правдоподобной, и заметить подмену будет нечем.
    """
    bad = os.getenv("STUB_BAD_URL", "http://127.0.0.1:10102")
    if not _жив(bad):
        pytest.skip("дублёр с чужим отпечатком не поднят")
    a = адрес()
    было = c.get("/v1/admin/gateway", headers=a).json()["current"]["base_url"]
    # токен НЕ передаём: /readyz открыт, а /v1/models ответит 401
    r = c.post("/v1/admin/gateway", headers=a, json={"base_url": bad, "token": ""})
    assert r.status_code == 409, f"адрес принят без сверки отпечатка: {r.status_code}"
    assert r.json()["error"]["code"] == "index_fingerprint_mismatch"
    assert c.get("/v1/admin/gateway", headers=a).json()["current"]["base_url"] == было
