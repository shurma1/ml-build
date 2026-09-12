# -*- coding: utf-8 -*-
"""v2 search: dense over service TYPES + lexical rescue + municipality facet.

    from v2.search import Searcher
    s = Searcher()
    s.search("хочу расписаться в щекино")
"""
import os, re, sys, time, functools
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facets import find_municipality, find_recipient
from aliases import expand_query

DB    = os.getenv("EMB_DB", "host=localhost port=5434 dbname=embeddings user=emb password=emb")
MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-large-instruct")
_INSTR = ("Given a Russian citizen's question about government services, "
          "retrieve the matching official service description")
QUERY_PREFIX = os.getenv("EMB_QUERY_PREFIX",
                         f"Instruct: {_INSTR}\nQuery: " if "instruct" in MODEL else "")

# Dense is the primary channel. The lexical channel and the reranker are RESCUE stages:
# running them unconditionally costs accuracy and latency (measured on 618 citizen
# queries: always-on RRF 1:1 drops R@1 0.631 -> 0.586). The gate is the top1-top2
# MARGIN, not the absolute score — absolute cosine is not comparable across models
# (e5-instruct sits at 0.88-0.94, bge-m3 at 0.53-0.73), the margin is.
RESCUE_MARGIN = float(os.getenv("EMB_RESCUE_MARGIN", "0.005"))   # ~ the 25th pct of the margin
RERANK = os.getenv("EMB_RERANK", "")            # e.g. BAAI/bge-reranker-v2-m3; "" = off
RERANK_TOPK = int(os.getenv("EMB_RERANK_TOPK", "10"))


class Searcher:
    def __init__(self, db=DB, model=MODEL):
        from sentence_transformers import SentenceTransformer
        self.pool = psycopg.connect(db, autocommit=True)
        register_vector(self.pool)
        self.model = SentenceTransformer(model, trust_remote_code="bge" not in model.lower())
        self.model.max_seq_length = 512
        self.reranker = None
        if RERANK:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK, max_length=512)

    # ---- query embedding, memoised: MFC query logs are extremely head-heavy ----
    @functools.lru_cache(maxsize=4096)
    def embed(self, q):
        return self.model.encode([QUERY_PREFIX + q], normalize_embeddings=True,
                                 convert_to_numpy=True)[0]

    def _dense(self, v, k):
        return self.pool.execute(
            "SELECT type_id, title, n_instances, 1-(embedding <=> %s) AS score "
            "FROM service_type ORDER BY embedding <=> %s LIMIT %s", (v, v, k)).fetchall()

    def _lexical(self, q, k):
        """OR-semantics tsquery. websearch_to_tsquery ANDs its terms, which makes any
        synonym expansion match zero rows — build the disjunction explicitly instead."""
        terms = [t for t in re.findall(r'[\w-]{3,}', q.lower()) if t not in _STOP]
        if not terms:
            return []
        tsq = ' | '.join(terms)
        return self.pool.execute(
            "SELECT type_id, title, n_instances, ts_rank_cd(tsv, q) AS score "
            "FROM service_type, to_tsquery('russian', %s) q "
            "WHERE tsv @@ q ORDER BY score DESC LIMIT %s", (tsq, k)).fetchall()

    @functools.lru_cache(maxsize=512)
    def branch_municipality(self, branch_id):
        r = self.pool.execute("SELECT municipality FROM branch WHERE branch_id = %s",
                              (branch_id,)).fetchone()
        return r[0] if r else None

    def search(self, query, k=5, pool=30, municipality=None, recipient=None, branch_id=None):
        # приоритет: явный параметр > топоним в запросе > район филиала оператора
        t0 = time.perf_counter()
        muni = (municipality or find_municipality(query)
                or (self.branch_municipality(branch_id) if branch_id else None))
        recip = recipient or find_recipient(query)

        v = self.embed(query)
        dense = self._dense(v, pool)
        t_dense = time.perf_counter()

        unsure = len(dense) > 1 and (dense[0][3] - dense[1][3]) < RESCUE_MARGIN
        used_rescue = used_rerank = False
        if unsure:
            lex = self._lexical(expand_query(query), pool)
            if lex:
                used_rescue = True
                ranked = _rrf([[r[0] for r in dense], [r[0] for r in lex]], w=(5.0, 1.0))
                # keep the DENSE cosine as the reported score; fusion only reorders
                meta = {r[0]: r for r in lex}
                meta.update({r[0]: r for r in dense})
                dense = [meta[t] for t in ranked][:pool]
            if self.reranker is not None:
                head = dense[:RERANK_TOPK]
                docs = {r[0]: r[1] for r in head}
                life = dict(self.pool.execute(
                    "SELECT type_id, life FROM service_type WHERE type_id = ANY(%s)",
                    ([r[0] for r in head],)).fetchall())
                pairs = [(query, docs[t] + ('. ' + life[t] if life.get(t) else ''))
                         for t, *_ in head]
                sc = self.reranker.predict(pairs, show_progress_bar=False)
                head = [h for _, h in sorted(zip(sc, head), key=lambda x: -x[0])]
                dense = head + dense[RERANK_TOPK:]
                used_rerank = True

        out = []
        for type_id, title, n_inst, score in dense[:k]:
            out.append({'type_id': type_id, 'title': title, 'score': float(score),
                        'n_instances': n_inst,
                        **self._resolve(type_id, muni, recip)})
        return {'query': query, 'municipality': muni, 'municipality_from':
                ('параметр' if municipality else 'запрос' if find_municipality(query)
                 else 'филиал' if muni else None), 'recipient': recip,
                'rescue': used_rescue, 'reranked': used_rerank, 'results': out,
                'ms': round((time.perf_counter() - t0) * 1000, 1),
                'ms_embed_and_ann': round((t_dense - t0) * 1000, 1)}

    def _resolve(self, type_id, muni, recip):
        """type -> the concrete card. Ambiguity is returned, never guessed away."""
        rs = self.pool.execute(
            "SELECT id, title, municipality, department FROM service WHERE type_id=%s "
            "ORDER BY municipality NULLS FIRST", (type_id,)).fetchall()
        if muni:
            hit = [r for r in rs if r[2] == muni]
            if hit:
                return {'service_id': hit[0][0], 'municipality_of_card': hit[0][2],
                        'department': hit[0][3], 'needs_municipality': False}
        if len(rs) == 1:
            return {'service_id': rs[0][0], 'municipality_of_card': rs[0][2],
                    'department': rs[0][3], 'needs_municipality': False}
        return {'service_id': None, 'needs_municipality': True,
                'available_in': [r[2] for r in rs if r[2]]}


_STOP = set('''и в во не что он на как а то все она так его но да ты к у же вы за бы по только ее мне
было вот от меня еще нет о из ему когда если уже или ни быть был него до вас нибудь ли
для мы тебя их чем была сам чтоб без будет где есть надо ней там этот того этого какой при
про них мой тем чтобы нее при над нас это мне мной ими под чем чей как-то нужно хочу надо'''.split())


def _rrf(rankings, k=60, w=None):
    w = w or [1.0] * len(rankings)
    sc = {}
    for wi, r in zip(w, rankings):
        for pos, d in enumerate(r):
            sc[d] = sc.get(d, 0.0) + wi / (k + pos + 1)
    return sorted(sc, key=sc.get, reverse=True)


if __name__ == "__main__":
    import json
    s = Searcher()
    qs = sys.argv[1:] or ["хочу расписаться", "потерял паспорт что делать",
                          "разрешение на строительство дома в щекинском районе",
                          "скидочная карта забота алексин"]
    for q in qs:
        r = s.search(q, k=3)
        print(f"\n### {q}   [{r['ms']}ms, muni={r['municipality']}, "
              f"rescue={r['rescue']}, rerank={r['reranked']}]")
        for x in r['results']:
            tail = (f"-> {x['service_id']}" if x['service_id']
                    else f"-> выбери МО ({len(x.get('available_in', []))} шт.)")
            print(f"  {x['score']:.3f}  {x['title'][:78]}  {tail}")
