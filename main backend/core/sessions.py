# -*- coding: utf-8 -*-
"""Сессия обслуживания: реплики -> факты -> состояние -> поиск -> лог.

Здесь живут три инварианта из четырёх, которые ломают продукт молча.

**Вся хронология в extract, только дописывание.** В `POST /v1/llm/extract`
уходят все содержательные реплики сессии с начала, в исходном порядке. Не окно,
не пересказ, не перестановка. На этом держится попадание в prefix cache vLLM
(префилл падает с ~1200 токенов на ход до ~30) и устойчивость модели: у 9B
измерено 14% смены вердикта от одной лишь перестановки фактов.

**Состояние только накапливает.** Молчание модели не отменяет ранее найденный
факт — за это отвечает `DialogState` из v2. Поле, выставленное оператором
руками, помечается `pinned` и в извлечение не отдаётся вовсе: иначе следующая
же реплика затрёт правку, ради которой оператор и вмешался.

**Право считается после поиска.** Отбора по праву на этапе retrieval здесь нет
и быть не может: услугу, на которую клиент не имеет права, нужно найти и
показать со статусом `blocked` и строкой регламента.
"""
import asyncio
import logging
import sys
import time
import uuid

from . import config as C
from . import db
from .errors import ApiError
from .events import hub, DEGRADED, RESULTS_UPDATED, STATE_UPDATED, UTTERANCE, degraded_payload
from .gateway import gateway, StaleSegment
from .search_service import search_service, DENSE

log = logging.getLogger("core.sessions")

FLAT_FACTS = ("age", "children", "categories")
STATE_FIELDS = ("life_situation", "recipient", "municipality") + FLAT_FACTS
VOLATILE = ("intents", "unresolved")


def _v2():
    if C.V2_PATH not in sys.path:
        sys.path.insert(0, C.V2_PATH)


def new_dialog_state():
    _v2()
    from extraction_schema import DialogState
    return DialogState()


def render_state(st, pinned, changed=None):
    """DialogState -> схема DialogState из core-api.yaml."""
    d = st.as_features()
    facts = d.get("facts") or {}
    return {
        "life_situation": d.get("life_situation"),
        "recipient": d.get("recipient"),
        "municipality": d.get("municipality"),
        "facts": {"age": facts.get("age"), "children": facts.get("children"),
                  "categories": facts.get("categories") or []},
        "intents": d.get("intents") or [],
        "unresolved": d.get("unresolved") or [],
        "pinned": sorted(pinned or ()),
        "changed": changed or {},
    }


def restore_state(snapshot):
    """Обратная операция: снимок из fact_state -> живой DialogState.

    Нужна после перезапуска процесса и после обрыва WS: состояние диалога это
    то, ради чего сессия вообще существует, и терять его из-за рестарта нельзя.
    """
    st = new_dialog_state()
    snap = snapshot or {}
    facts = snap.get("facts") or {}
    for k in ("life_situation", "recipient", "municipality"):
        if snap.get(k) is not None:
            st.state[k] = snap[k]
    for k in FLAT_FACTS:
        v = facts.get(k)
        if v not in (None, [], ""):
            st.state[k] = v
    for k in VOLATILE:
        if snap.get(k):
            st.volatile[k] = snap[k]
    return st


class Session:
    """Живое состояние одной сессии в памяти процесса. Истина — в базе;
    здесь кэш, чтобы не собирать DialogState заново на каждой реплике."""

    def __init__(self, row):
        self.id = str(row["session_id"])
        self.branch_id = row.get("branch_id")
        self.operator_id = row.get("operator_id")
        self.municipality = row.get("municipality")
        self.started_at = row.get("started_at")
        self.closed_at = row.get("closed_at")
        self.state = new_dialog_state()
        self.pinned = set()
        self.answers = []                 # [(ключ вопроса, ответ)] за весь диалог
        self.seq = 0                      # номер последней реплики
        self.last_changed = {}
        self.last_search = None
        self.turns = []                   # [{seq, text, speaker, source, empty, t0, t1, asr_ms}]
        self.lock = asyncio.Lock()

    @property
    def closed(self):
        return self.closed_at is not None

    def llm_turns(self):
        """То, что уходит в extract: ВСЕ содержательные реплики, в хронологии.

        Пустые сегменты («ага», «понятно») исключены — извлекать из них нечего,
        и это единственный фильтр вызовов, переживший замеры. Важно, что их
        исключение сохраняет дописывание: префикс запроса на ходу N остаётся
        префиксом запроса на ходу N+1, а значит кэш не рушится.
        """
        return [{"seq": t["seq"], "text": t["text"], "speaker": t.get("speaker")}
                for t in self.turns if not t.get("empty")]

    def render(self):
        return render_state(self.state, self.pinned, self.last_changed)


class SessionManager:
    def __init__(self):
        self._live = {}
        self._limiter = None

    def limiter(self):
        if self._limiter is None:
            _v2()
            from ratelimit import CallLimiter
            self._limiter = CallLimiter(max_concurrent=C.LLM_MAX_CONCURRENT,
                                        stale_after=C.LLM_STALE_AFTER)
        return self._limiter

    # --- жизненный цикл -----------------------------------------------------

    async def create(self, branch_id, operator_id, window=None):
        muni = None
        if branch_id:
            row = await db.fetchrow(
                "SELECT branch_id, municipality FROM branch WHERE branch_id = %s", (branch_id,))
            if not row:
                raise ApiError("schema_validation_failed", f"филиал {branch_id} неизвестен")
            muni = row["municipality"]
        sid = uuid.uuid4()
        row = await db.fetchrow(
            'INSERT INTO session (session_id, branch_id, operator_id, "window", municipality, '
            "purge_after) VALUES (%s, %s, %s, %s, %s, now() + make_interval(days => %s)) "
            "RETURNING session_id, branch_id, operator_id, municipality, started_at, closed_at",
            (sid, branch_id, operator_id, window, muni, C.PURGE_DAYS))
        s = Session(row)
        self._live[s.id] = s
        log.info("сессия %s открыта: филиал=%s оператор=%s МО=%s",
                 s.id, branch_id, operator_id, muni)
        return s

    async def get(self, session_id, restore=True):
        sid = str(session_id)
        s = self._live.get(sid)
        if s is not None:
            return s
        try:
            uuid.UUID(sid)
        except ValueError:
            raise ApiError("session_not_found", f"сессия {sid} не найдена")
        row = await db.fetchrow(
            "SELECT session_id, branch_id, operator_id, municipality, started_at, closed_at "
            "FROM session WHERE session_id = %s", (sid,))
        if not row:
            raise ApiError("session_not_found", f"сессия {sid} не найдена")
        s = Session(row)
        if restore:
            await self._restore(s)
        self._live[sid] = s
        return s

    async def _restore(self, s):
        turns = await db.fetch(
            "SELECT seq, t0, t1, speaker, source, text, empty, asr_ms FROM turn "
            "WHERE session_id = %s ORDER BY seq", (s.id,))
        s.turns = [dict(t) for t in turns]
        s.seq = max((t["seq"] for t in s.turns), default=0)
        fs = await db.fetchrow(
            "SELECT state, changed, pinned FROM fact_state WHERE session_id = %s "
            "ORDER BY seq DESC LIMIT 1", (s.id,))
        if fs:
            s.state = restore_state(fs["state"])
            s.pinned = set(fs["pinned"] or ())
            s.last_changed = fs["changed"] or {}
        sl = await db.fetchrow(
            "SELECT features, results, questions, mode, ms, ms_embed FROM search_log "
            "WHERE session_id = %s ORDER BY seq DESC LIMIT 1", (s.id,))
        if sl:
            s.answers = [(a["key"], a["value"])
                         for a in ((sl["features"] or {}).get("answers") or [])]
        return s

    def forget(self, session_id):
        self._live.pop(str(session_id), None)

    async def close(self, session_id, chosen_service_id=None, outcome=None, note=None):
        s = await self.get(session_id)
        await db.execute(
            "UPDATE session SET closed_at = now(), outcome = %s, chosen_service_id = %s, "
            "operator_note = %s WHERE session_id = %s",
            (outcome, chosen_service_id, (note or "")[:500] or None, s.id))
        s.closed_at = time.time()
        log.info("сессия %s закрыта: исход=%s услуга=%s", s.id, outcome, chosen_service_id)
        self.forget(s.id)
        return s

    # --- обработка одного хода ---------------------------------------------

    async def add_turn(self, s, text, speaker=None, source="asr",
                       t0=None, t1=None, asr_ms=None, emit=True):
        """Полный конвейер одного хода. Возвращает {turn, state, search}."""
        _v2()
        from ratelimit import is_empty_segment

        text = (text or "").strip()
        if not text:
            raise ApiError("schema_validation_failed", "пустая реплика")
        async with s.lock:
            s.seq += 1
            seq = s.seq
            empty = bool(is_empty_segment(text))
            turn = {"seq": seq, "t0": t0, "t1": t1, "speaker": speaker, "source": source,
                    "text": text[:2000], "empty": empty, "asr_ms": asr_ms}
            s.turns.append(turn)
        await db.execute(
            "INSERT INTO turn (session_id, seq, t0, t1, speaker, source, text, empty, asr_ms) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (session_id, seq) DO NOTHING",
            (s.id, seq, t0, t1, speaker, source, turn["text"], empty, asr_ms))
        if emit:
            hub.emit(s.id, UTTERANCE, {"utterance": _utterance(turn)})

        if empty:
            # «ага», «понятно» — извлекать нечего, список не трогаем.
            return {"turn": _turn(turn), "state": s.render(),
                    "search": s.last_search, "skipped": "empty_segment"}

        state_payload, llm_ok = await self._extract(s, seq)
        if emit and state_payload:
            hub.emit(s.id, STATE_UPDATED, state_payload)

        search = await self._run_search(s, seq, fallback_text=None if llm_ok else text)
        if emit and search:
            hub.emit(s.id, RESULTS_UPDATED, search)
        return {"turn": _turn(turn), "state": s.render(), "search": search}

    async def _extract(self, s, seq):
        """-> (payload для state.updated, признак что LLM отработала)."""
        turns = s.llm_turns()
        if not turns:
            return None, True
        t0 = time.perf_counter()
        facts, dropped, ok = {}, [], True
        try:
            data = await self.limiter().run(
                s.id, lambda: gateway.extract(s.id, turns))
            if data is None:
                # Диалог ушёл вперёд, пока запрос стоял в очереди. Не ошибка.
                return None, True
            payload, _meta = data
            facts = payload.get("facts") or {}
            dropped = payload.get("raw_dropped") or []
        except StaleSegment:
            return None, True
        except Exception as e:                                   # noqa: BLE001
            ok = False
            log.warning("извлечение фактов недоступно (%s): работаем на газеттире", e)
            hub.emit(s.id, DEGRADED, degraded_payload("llm", search_service.mode, e))

        llm_ms = int((time.perf_counter() - t0) * 1000)
        async with s.lock:
            # Закреплённое оператором извлечением не перезаписывается — вообще.
            raw = {k: v for k, v in (facts or {}).items() if k not in s.pinned}
            changed = s.state.update(raw) if raw else {}
            s.last_changed = {k: list(v) for k, v in changed.items()}
            snapshot = s.render()
            pinned = sorted(s.pinned)
        await db.execute(
            "INSERT INTO fact_state (session_id, seq, state, changed, pinned, dropped, llm_ms) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (session_id, seq) DO UPDATE SET "
            "state = EXCLUDED.state, changed = EXCLUDED.changed, pinned = EXCLUDED.pinned, "
            "dropped = EXCLUDED.dropped, llm_ms = EXCLUDED.llm_ms",
            (s.id, seq, _json(snapshot), _json(s.last_changed), pinned, list(dropped), llm_ms))
        return {"state": snapshot, "changed": s.last_changed,
                "missing_facts": (s.last_search or {}).get("missing_facts", [])}, ok

    async def _run_search(self, s, seq, fallback_text=None):
        feats = self._features(s, fallback_text)
        if not feats.search_strings():
            return s.last_search
        r = await search_service.search(
            feats, k=C.SEARCH_K,
            asked=tuple(k for k, _ in s.answers),
            answers=list(s.answers) or None,
            municipality=s.municipality)
        r = await enrich(r, s)
        s.last_search = r
        await self._log_search(s, seq, feats, r)
        return r

    def _features(self, s, fallback_text=None):
        """QueryFeatures для поиска.

        Обычный путь — из накопленного состояния. Если LLM отвалилась, признаки
        берёт газеттир из текста реплики, а накопленные факты подставляются
        сверху: список станет беднее, но право по-прежнему проверяется по тому,
        что уже известно о клиенте.
        """
        _v2()
        from features import QueryFeatures

        d = dict(s.state.as_features())
        last = next((t["text"] for t in reversed(s.turns) if not t.get("empty")), "")
        d["raw_text"] = last
        if fallback_text:
            f = QueryFeatures.from_text(fallback_text)
            f.life_situation = f.life_situation or d.get("life_situation")
            f.municipality = f.municipality or d.get("municipality")
            f.recipient = f.recipient or d.get("recipient")
            f.facts = {**(d.get("facts") or {}), **(f.facts or {})}
            return f
        return QueryFeatures.from_llm(d)

    async def _log_search(self, s, seq, feats, r):
        """search_log — на каждом ходу. Это не отладка, а единственный способ
        узнать реальное качество: все нынешние метрики получены на синтетике."""
        features = {
            "intents": feats.intents, "life_situation": feats.life_situation,
            "recipient": feats.recipient, "municipality": feats.municipality,
            "facts": feats.facts, "unresolved": feats.unresolved,
            "answers": [{"key": k, "value": v} for k, v in s.answers],
            "branch_id": s.branch_id,
        }
        results = [{"type_id": x["type_id"], "rank": i + 1, "score": x.get("score"),
                    "status": x.get("status"), "service_id": x.get("service_id")}
                   for i, x in enumerate(r.get("results") or [])]
        try:
            await db.execute(
                "INSERT INTO search_log (session_id, seq, features, results, questions, mode, "
                "ms, ms_embed, corpus_version) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (session_id, seq) DO UPDATE SET features = EXCLUDED.features, "
                "results = EXCLUDED.results, questions = EXCLUDED.questions, "
                "mode = EXCLUDED.mode, ms = EXCLUDED.ms, ms_embed = EXCLUDED.ms_embed",
                (s.id, seq, _json(features), _json(results), _json(r.get("questions") or []),
                 r.get("mode") or DENSE, int(r.get("ms") or 0), int(r.get("ms_embed") or 0),
                 (await corpus_version_str())))
        except Exception as e:                                   # noqa: BLE001
            log.error("лог поиска не записан для сессии %s ход %s: %s", s.id, seq, e)

    # --- правки оператором --------------------------------------------------

    async def set_facts(self, s, setter=None, unpin=()):
        """Ручная правка. Выставленное здесь закрепляется и извлечением не трогается.

        Это главная страховка от ошибки атрибуции ролей: диаризации нет, роль
        говорящего выводит LLM, и «у вас двое детей?» от оператора она может
        записать клиенту.
        """
        setter = setter or {}
        async with s.lock:
            for k in unpin or ():
                s.pinned.discard(k)
            changed = {}
            for k, v in setter.items():
                if k not in STATE_FIELDS:
                    continue
                old = s.state.state.get(k)
                if v is None:
                    s.state.state.pop(k, None)
                    s.pinned.discard(k)
                else:
                    s.state.state[k] = v
                    s.pinned.add(k)
                if old != v:
                    changed[k] = [old, v]
            s.last_changed = changed
            snapshot = s.render()
            pinned = sorted(s.pinned)
            seq = s.seq
        await db.execute(
            "INSERT INTO fact_state (session_id, seq, state, changed, pinned) "
            "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (session_id, seq) DO UPDATE SET "
            "state = EXCLUDED.state, changed = EXCLUDED.changed, pinned = EXCLUDED.pinned",
            (s.id, seq, _json(snapshot), _json(changed), pinned))
        hub.emit(s.id, STATE_UPDATED, {"state": snapshot, "changed": changed,
                                       "missing_facts": (s.last_search or {}).get("missing_facts", [])})
        search = await self._run_search(s, seq)
        if search:
            hub.emit(s.id, RESULTS_UPDATED, search)
        return {"state": snapshot, "search": search}

    async def answer(self, s, answers):
        """Ответ на наводящий вопрос ПЕРЕРАНЖИРУЕТ, а не фильтрует.

        Жёсткий фильтр измерен и хуже: R@1 0.775 против 0.810. Он навсегда
        выбрасывает цель, у которой признак услуги просто не заполнен, —
        а незаполненных признаков в этом корпусе половина.
        """
        async with s.lock:
            by_key = dict(s.answers)
            for a in answers:
                by_key[a["key"]] = a["value"]
            s.answers = list(by_key.items())
            seq = s.seq
        search = await self._run_search(s, seq)
        if search:
            hub.emit(s.id, RESULTS_UPDATED, search)
        return search

    async def mark_opened(self, session_id, service_id):
        """Какую карточку оператор реально открыл. Это и есть та самая неявная
        разметка релевантности, ради которой лог вообще ведётся."""
        try:
            await db.execute(
                "UPDATE search_log SET opened = array_append(opened, %s) "
                "WHERE session_id = %s AND seq = (SELECT max(seq) FROM search_log "
                "WHERE session_id = %s) AND NOT (%s = ANY(opened))",
                (service_id, str(session_id), str(session_id), service_id))
        except Exception as e:                                   # noqa: BLE001
            log.warning("не отмечено открытие карточки %s: %s", service_id, e)


# --- обогащение выдачи ------------------------------------------------------

async def enrich(r, session=None):
    """Дополнить результаты тем, что нужно интерфейсу: подсветка, срок, плата.

    Подсветка считается здесь, а не в v2: это отдельная задача UI, и тянуть её
    в измеренный модуль поиска незачем.
    """
    from .catalog import catalog
    from . import highlights as H
    from .reasons import humanize_reasons

    facts = {}
    if session is not None:
        d = session.state.as_features()
        facts = dict(d.get("facts") or {})
        if d.get("recipient"):
            facts["recipient"] = d["recipient"]

    for i, x in enumerate(r.get("results") or []):
        x["rank"] = i + 1
        # Причина уходит оператору в руки: списка в питоновском синтаксисе
        # там быть не должно
        x["reasons"] = humanize_reasons(x.get("reasons"))
        t = catalog.types.get(x["type_id"])
        sid = x.get("service_id") or (t["instances"][0]["id"] if t else None)
        x.setdefault("needs_municipality", not x.get("service_id"))
        if t and sid:
            fields = catalog.raw_fields(sid)
            x["highlights"] = H.for_type(t, fields, facts)
            x.update(catalog.structural(sid))
            # Сводка для строки выдачи. Считается здесь, а не на клиенте:
            # разбор регламента в двух реализациях неизбежно расходится, и
            # оператор получает «Бесплатно» там, где в карточке госпошлина.
            x["card"] = _summary(catalog.view(sid))
        else:
            x["highlights"] = []
            x.setdefault("term_days", None)
            x.setdefault("is_free", None)
            x["card"] = None
        if x.get("needs_municipality") and "available_in" not in x and t:
            x["available_in"] = t.get("municipalities") or []
        x.setdefault("available_in", [])
        x.setdefault("department", None)
    return r


def _summary(view):
    """То немногое, по чему услугу выбирают в списке. Полная карточка — по клику."""
    if not view:
        return None
    return {
        "title": view["title"],
        "department": view["departmentName"],
        "isFree": view["isFree"],
        "decisionDays": view["terms"]["decisionDays"],
        "documentsTotal": view["documents"]["total"],
        "channels": [c["label"] for c in view["channels"]],
        "mfcCount": view["mfcCount"],
        "recipientsFormula": view["recipients"]["formula"],
    }


async def corpus_version_str():
    from .catalog import catalog
    v = catalog.version or {}
    return f"{v.get('cards')}c/{v.get('types')}t/{v.get('source_sha')}"


def _utterance(t):
    return {"seq": t["seq"], "t0": t.get("t0"), "t1": t.get("t1"), "text": t["text"],
            "speaker": t.get("speaker"), "source": t.get("source"), "asr_ms": t.get("asr_ms")}


def _turn(t):
    return {**_utterance(t), "empty": bool(t.get("empty"))}


def _json(v):
    from psycopg.types.json import Jsonb
    return Jsonb(v)


manager = SessionManager()
