# -*- coding: utf-8 -*-
"""Диалоговый поиск: признаки -> мультизапрос -> право на услугу -> наводящий вопрос.

Бюджет одного хода (CPU, тёплый кэш):
    признаки от LLM ............ вне этого модуля, доминирует в общем времени
    эмбеддинг 1-5 строк ........ ОДИН батч, ~110 мс (не 5×104: батч почти бесплатен)
    HNSW × N строк ............. ~3 мс каждый
    слияние RRF ................ <1 мс
    проверка права ............. <1 мс на 30 кандидатов
    наводящий вопрос ........... <1 мс, чистая арифметика по энтропии
Повторный поиск в том же диалоге почти весь уходит в кэш: намерения повторяются.
"""
import os, sys, time, functools, json
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import QueryFeatures
from extraction_schema import DialogState, validate, parse_llm_output
from clarify import suggest, apply_answer, rescore
from eligibility import check as check_eligibility
from corpus import load_cards, build_types
from facets import find_municipality

DB = os.getenv("EMB_DB", "host=localhost port=5434 dbname=embeddings user=emb password=emb")
MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-large-instruct")
_INSTR = ("Given a Russian citizen's question about government services, "
          "retrieve the matching official service description")
QUERY_PREFIX = os.getenv("EMB_QUERY_PREFIX",
                         f"Instruct: {_INSTR}\nQuery: " if "instruct" in MODEL else "")


def _rrf(rankings, k=60, w=None):
    w = w or [1.0] * len(rankings)
    sc = {}
    for wi, r in zip(w, rankings):
        for pos, d in enumerate(r):
            sc[d] = sc.get(d, 0.0) + wi / (k + pos + 1)
    return sorted(sc, key=sc.get, reverse=True)


class DialogSearch:
    def __init__(self, db=DB, model=MODEL, data=None, embedder=None):
        # Модель эмбеддера живёт на арендованной видеокарте, а не в этом процессе:
        # сюда она приходит по HTTP через POST /v1/embed у gpu-шлюза. Префикс
        # запроса ставит ШЛЮЗ по полю kind — здесь его добавлять нельзя (см. embed()).
        self.conn = psycopg.connect(db, autocommit=True)
        register_vector(self.conn)
        if embedder is None:
            _core = os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), 'main backend')
            if _core not in sys.path:
                sys.path.insert(0, _core)
            from core.embed_client import GatewayEmbedder
            embedder = GatewayEmbedder(kind='query')
        self.model = embedder
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cards = load_cards(data or os.path.join(root, 'data', 'services_for_llm.json'))
        self.types = {t['type_id']: t for t in build_types(self.cards)}
        raw = {r['id']: r for r in json.load(
            open(data or os.path.join(root, 'data', 'services_for_llm.json'), encoding='utf-8'))}
        self.src = {}
        for t in self.types.values():
            r = raw[t['instances'][0]['id']]
            self.src[t['type_id']] = ((r.get('serviceRecipients') or '') + '\n' +
                                      (r.get('rejectReasonsText') or ''))
        self._cache = {}

    # --- один батч на все строки запроса: 5 строк стоят почти как одна ---
    def embed(self, strings):
        """QUERY_PREFIX здесь НЕ добавляется: префикс ставит шлюз по полю kind.
        Добавить его ещё и тут значит закодировать запрос дважды префиксованным,
        а индекс — одинарным; выдача останется правдоподобной, и заметить это
        по результатам невозможно. Цена измерена: -19 п.п. R@1."""
        miss = [s for s in strings if s not in self._cache]
        if miss:
            v = self.model.encode(miss)
            self._cache.update(zip(miss, v))
        return [self._cache[s] for s in strings]

    def _ann(self, vec, k):
        return self.conn.execute(
            "SELECT type_id, title, n_instances, 1-(embedding <=> %s) AS score "
            "FROM service_type ORDER BY embedding <=> %s LIMIT %s", (vec, vec, k)).fetchall()

    def search(self, feats, k=8, pool=30, asked=()):
        t0 = time.perf_counter()
        if isinstance(feats, str):
            feats = QueryFeatures.from_text(feats)
        strings = feats.search_strings()
        if not strings:
            return {'results': [], 'questions': [], 'ms': 0}

        vecs = self.embed(strings)
        t_emb = time.perf_counter()

        runs, best = [], {}
        for v in vecs:
            rows = self._ann(v, pool)
            runs.append([r[0] for r in rows])
            for tid, title, n, sc in rows:
                if sc > best.get(tid, (0,))[0]:
                    best[tid] = (sc, title, n)
        # первое намерение весомее остальных: LLM ставит главное первым
        order = _rrf(runs, w=[3.0] + [1.0] * (len(runs) - 1))[:pool]

        muni = feats.municipality or find_municipality(feats.raw_text or '')
        facts = dict(feats.facts)
        if feats.recipient:
            facts['recipient'] = feats.recipient

        cands = [self.types[t] for t in order if t in self.types]
        out = []
        for tid in order[:k]:
            t = self.types.get(tid)
            if not t:
                continue
            status, detail = check_eligibility(t, facts, self.src[tid])
            out.append({
                'type_id': tid, 'title': t['title'], 'score': round(best[tid][0], 4),
                'status': status,                       # eligible | blocked | unknown
                'reasons': detail,
                **self._resolve(tid, muni),
            })
        questions = suggest(cands[:20], asked=asked)
        # факты, которых не хватило для проверки права, — тоже повод спросить
        need = sorted({d['need_fact'] for r in out for d in r['reasons']
                       if r['status'] == 'unknown' and 'need_fact' in d})
        return {'results': out, 'questions': questions, 'missing_facts': need,
                'municipality': muni, 'n_intents': len(strings),
                'ms': round((time.perf_counter() - t0) * 1000, 1),
                'ms_embed': round((t_emb - t0) * 1000, 1)}

    def _resolve(self, type_id, muni):
        rs = self.conn.execute(
            "SELECT id, municipality, department FROM service WHERE type_id=%s "
            "ORDER BY municipality NULLS FIRST", (type_id,)).fetchall()
        if muni:
            hit = [r for r in rs if r[1] == muni]
            if hit:
                return {'service_id': hit[0][0], 'department': hit[0][2]}
        if len(rs) == 1:
            return {'service_id': rs[0][0], 'department': rs[0][2]}
        return {'service_id': None, 'needs_municipality': True}

    def narrow(self, feats, answers, k=8, pool=20):
        """answers — [(ключ_вопроса, ответ), ...] за весь диалог. Переранжирование,
        а не фильтрация: ответ не выбрасывает кандидата, а двигает его."""
        r = self.search(feats, k=pool, pool=pool)
        cands = [self.types[x['type_id']] for x in r['results'] if x['type_id'] in self.types]
        scores = {x['type_id']: x['score'] for x in r['results']}
        asked = set()
        for key, ans in answers:
            asked.add(key)
            scores = rescore(cands, scores, key, ans)
        by_id = {x['type_id']: x for x in r['results']}
        ranked = sorted(cands, key=lambda t: -scores[t['type_id']])
        r['results'] = [by_id[t['type_id']] for t in ranked][:k]
        r['questions'] = suggest(ranked, asked=asked)
        r['asked'] = sorted(asked)
        return r
