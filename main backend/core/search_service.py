# -*- coding: utf-8 -*-
"""Поиск для core-api: пул синхронных DialogSearch + аварийный лексический режим.

Почему пул, а не один экземпляр: `DialogSearch` из embending/v2 синхронный и
держит собственное соединение к Postgres. Запросы там написаны на чистом SQL и
измерены — переписывать их в async ради единообразия значит трогать то, что
работает. Поэтому поиск исполняется в потоках, а экземпляры разбираются из
очереди: соединение psycopg не потокобезопасно, и один экземпляр на всех
означал бы либо гонку, либо глобальную блокировку на 6 мс каждого хода.

Корпус (карточки, типы, исходники) у экземпляров ОБЩИЙ по ссылке: он неизменяем,
а 732 карточки в четырёх копиях — это мегабайты на пустом месте.

Лексический режим — авария, а не улучшение. Гибрид «всегда включён» измерен и
отвергнут: R@1 0.631 -> 0.586. Он включается только когда шлюз не отвечает,
и тогда его задача не быть лучше, а не дать оператору пустой экран.
"""
import asyncio
import logging
import re
import sys
import time

from . import config as C
from .embed_client import EmbedUnavailable

log = logging.getLogger("core.search")

DENSE = "dense"
LEXICAL = "degraded_lexical"


def _v2():
    if C.V2_PATH not in sys.path:
        sys.path.insert(0, C.V2_PATH)


class SearchService:
    def __init__(self):
        self._pool = None
        self._instances = []
        self.mode = DENSE
        self._on_degraded = None

    async def start(self, workers=None):
        _v2()
        from dialog import DialogSearch
        n = workers or C.SEARCH_WORKERS
        loop = asyncio.get_running_loop()

        def build(share_from=None):
            d = DialogSearch(data=C.DATA)
            if share_from is not None:
                # корпус неизменяем — делим по ссылке, а не копируем на каждый поток
                d.cards, d.types, d.src = share_from.cards, share_from.types, share_from.src
            return d

        first = await loop.run_in_executor(None, build)
        self._instances = [first]
        for _ in range(n - 1):
            self._instances.append(await loop.run_in_executor(None, build, first))
        self._pool = asyncio.Queue()
        for d in self._instances:
            self._pool.put_nowait(d)
        log.info("поиск готов: %s экземпляров, %s типов", n, len(first.types))
        return self

    async def stop(self):
        for d in self._instances:
            try:
                d.conn.close()
            except Exception:                                   # noqa: BLE001
                pass
        self._instances = []
        self._pool = None

    def on_degraded(self, cb):
        self._on_degraded = cb

    @property
    def ready(self):
        return self._pool is not None

    def set_mode(self, mode):
        if mode != self.mode:
            log.warning("режим поиска: %s -> %s", self.mode, mode)
            self.mode = mode

    # --- основной вход ------------------------------------------------------

    async def search(self, features, k=None, asked=(), answers=None, municipality=None):
        """-> SearchResponse-словарь. Ошибка шлюза превращается в лексический
        режим, а не в 500: оператор обязан продолжать работать."""
        if not self.ready:
            raise RuntimeError("поиск не инициализирован")
        k = k or C.SEARCH_K
        if municipality and not features.municipality:
            features.municipality = municipality

        d = await self._pool.get()
        try:
            loop = asyncio.get_running_loop()
            if self.mode == DENSE:
                try:
                    r = await loop.run_in_executor(
                        None, self._dense, d, features, k, asked, answers)
                    r["mode"] = DENSE
                    return r
                except EmbedUnavailable as e:
                    # Шлюз отвалился между опросами сторожа — не ждём его вердикта.
                    log.warning("плотный поиск недоступен (%s), уходим в лексику", e)
                    self.set_mode(LEXICAL)
                    if self._on_degraded:
                        self._on_degraded(str(e))
            r = await loop.run_in_executor(None, self._lexical, d, features, k, asked, answers)
            r["mode"] = LEXICAL
            return r
        finally:
            self._pool.put_nowait(d)

    # --- плотный путь -------------------------------------------------------

    @staticmethod
    def _dense(d, features, k, asked, answers):
        """Ровно v2: мультизапрос -> RRF -> право -> вопрос. Ничего своего."""
        if answers:
            r = d.narrow(features, list(answers), k=k)
        else:
            r = d.search(features, k=k, pool=C.SEARCH_POOL, asked=tuple(asked))
        return r

    # --- аварийный лексический путь ----------------------------------------

    @staticmethod
    def _lexical(d, features, k, asked, answers):
        """tsvector + pg_trgm вместо векторов. Всё, что ПОСЛЕ поиска — то же самое:
        право проверяется той же таблицей предикатов, вопрос выбирается той же
        энтропией. Деградирует именно retrieval, а не вердикт о праве."""
        _v2()
        from eligibility import check as check_eligibility
        from clarify import suggest, rescore
        from facets import find_municipality
        from aliases import expand_query

        t0 = time.perf_counter()
        strings = features.search_strings() or ([features.raw_text] if features.raw_text else [])
        query = " ".join(s for s in strings if s).strip()
        if not query:
            return {"results": [], "questions": [], "missing_facts": [],
                    "municipality": features.municipality, "n_intents": 0,
                    "ms": 0.0, "ms_embed": 0.0}

        pool_n = C.SEARCH_POOL
        runs, best, titles = [], {}, {}
        expanded = expand_query(query)

        ts = _ts_rows(d.conn, expanded, pool_n)
        if ts:
            runs.append([r[0] for r in ts])
            for tid, title, sc in ts:
                titles[tid] = title
                best[tid] = max(best.get(tid, 0.0), float(sc))
        tg = _trgm_rows(d.conn, query, pool_n)
        if tg:
            runs.append([r[0] for r in tg])
            for tid, title, sc in tg:
                titles.setdefault(tid, title)
                best[tid] = max(best.get(tid, 0.0), float(sc))

        order = _rrf(runs, w=[3.0, 1.0][:len(runs)])[:pool_n]

        muni = features.municipality or find_municipality(features.raw_text or query)
        facts = dict(features.facts)
        if features.recipient:
            facts["recipient"] = features.recipient

        cands = [d.types[t] for t in order if t in d.types]
        scores = {t: best.get(t, 0.0) for t in order}
        if answers:
            for key, ans in answers:
                scores = rescore(cands, scores, key, ans)
            order = sorted(order, key=lambda t: -scores.get(t, 0.0))
            asked = set(asked) | {key for key, _ in answers}

        out = []
        for tid in order[:k]:
            t = d.types.get(tid)
            if not t:
                continue
            status, detail = check_eligibility(t, facts, d.src.get(tid))
            out.append({"type_id": tid, "title": t["title"],
                        "score": round(float(scores.get(tid, best.get(tid, 0.0))), 4),
                        "status": status, "reasons": detail,
                        **d._resolve(tid, muni)})
        need = sorted({x["need_fact"] for r in out for x in r["reasons"]
                       if r["status"] == "unknown" and "need_fact" in x})
        return {"results": out, "questions": suggest(cands[:20], asked=tuple(asked)),
                "missing_facts": need, "municipality": muni, "n_intents": len(strings),
                "ms": round((time.perf_counter() - t0) * 1000, 1), "ms_embed": 0.0}


# --- лексические каналы -----------------------------------------------------

_STOP = set('''и в во не что он на как а то все она так его но да ты к у же вы за бы по только ее мне
было вот от меня еще нет о из ему когда если уже или ни быть был него до вас нибудь ли
для мы тебя их чем была сам чтоб без будет где есть надо ней там этот того этого какой при
про них мой тем чтобы нее при над нас это мне мной ими под чем чей как-то нужно хочу надо'''.split())


def _ts_rows(conn, query, k):
    """OR-семантика. websearch_to_tsquery склеивает термины через AND, и тогда
    расширение синонимами даёт ноль строк — дизъюнкция строится явно."""
    terms = [t for t in re.findall(r"[\w-]{3,}", query.lower()) if t not in _STOP]
    if not terms:
        return []
    tsq = " | ".join(terms[:40])
    try:
        return conn.execute(
            "SELECT type_id, title, ts_rank_cd(tsv, q) AS score "
            "FROM service_type, to_tsquery('russian', %s) q "
            "WHERE tsv @@ q ORDER BY score DESC LIMIT %s", (tsq, k)).fetchall()
    except Exception as e:                                      # noqa: BLE001
        log.warning("лексический канал tsvector отказал: %s", e)
        return []


def _trgm_rows(conn, query, k):
    """Триграммы вытягивают опечатки и обрывки слов, на которых tsvector молчит."""
    q = query[:200]
    try:
        return conn.execute(
            "SELECT type_id, title, similarity(title, %s) AS score FROM service_type "
            "WHERE similarity(title, %s) > 0.06 ORDER BY score DESC LIMIT %s",
            (q, q, k)).fetchall()
    except Exception as e:                                      # noqa: BLE001
        log.warning("лексический канал pg_trgm отказал: %s", e)
        return []


def _rrf(rankings, k=60, w=None):
    w = w or [1.0] * len(rankings)
    sc = {}
    for wi, r in zip(w, rankings):
        for pos, d in enumerate(r):
            sc[d] = sc.get(d, 0.0) + wi / (k + pos + 1)
    return sorted(sc, key=sc.get, reverse=True)


search_service = SearchService()
