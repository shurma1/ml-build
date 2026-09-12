# -*- coding: utf-8 -*-
"""Evaluate a retrieval config against the citizen-query set.

    python3 v2/eval/run_eval.py                       # default model
    python3 v2/eval/run_eval.py BAAI/bge-m3 Qwen/Qwen3-Embedding-0.6B

Metric is TYPE-level: the answer is right when the correct service TYPE is returned.
Doc-level R@1 is meaningless on this corpus — 440 of 732 cards are the same service
replicated across 26 municipalities, and which replica ranks first is decided by a
cosine gap of ~0.002. The municipality is resolved afterwards, from a facet.
"""
import os, sys, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from corpus import load_cards, build_types, search_text
from aliases import doc_aliases_for

DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'services_for_llm.json')
INSTR = ("Given a Russian citizen's question about government services, "
         "retrieve the matching official service description")


def prefixes(model_id):
    if 'instruct' in model_id.lower():
        return '', f'Instruct: {INSTR}\nQuery: '
    if 'e5' in model_id.lower():
        return 'passage: ', 'query: '
    return '', ''


def metrics(ranks):
    r = np.array(ranks, dtype=float)
    return dict(n=len(r), R1=(r <= 1).mean(), R3=(r <= 3).mean(), R5=(r <= 5).mean(),
                R10=(r <= 10).mean(), MRR=np.mean(1 / r))


def bootstrap_ci(ranks, seed=0, n=5000):
    rng = np.random.default_rng(seed); r = np.array(ranks, dtype=float)
    s = [(r[rng.integers(0, len(r), len(r))] <= 1).mean() for _ in range(n)]
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def main(models, with_aliases=True):
    from sentence_transformers import SentenceTransformer
    cards = load_cards(DATA)
    types = build_types(cards)
    pos = {t['type_id']: i for i, t in enumerate(types)}
    texts = [search_text(t, doc_aliases_for(t['title'], t.get('type_key')) if with_aliases else '')
             for t in types]

    # gold is resolved through the CARD ids, which are stable; type_id is derived
    card2type = {c['id']: c['type_id'] for c in cards}
    qs = []
    for line in open(os.path.join(HERE, 'citizen_queries.jsonl'), encoding='utf-8'):
        d = json.loads(line)
        gold = {card2type[i] for i in d['ids'] if i in card2type}
        if len(gold) != 1:
            continue          # the service was re-grouped since generation; skip
        t = gold.pop()
        for q in d['queries']:
            qs.append((q, t))
    print(f'corpus: {len(cards)} cards -> {len(types)} types | eval: {len(qs)} queries\n')

    for mid in models:
        dp, qp = prefixes(mid)
        m = SentenceTransformer(mid, trust_remote_code='bge' not in mid.lower())
        m.max_seq_length = 512
        E = m.encode([dp + t for t in texts], batch_size=8, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
        Q = m.encode([qp + q for q, _ in qs], batch_size=32, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
        order = np.argsort(-(Q @ E.T), axis=1)
        ranks = [int(np.where(order[i] == pos[t])[0][0]) + 1 for i, (_, t) in enumerate(qs)]
        mm = metrics(ranks); lo, hi = bootstrap_ci(ranks)
        print(f"{mid:46s} dim={E.shape[1]:5d} R@1={mm['R1']:.3f} R@3={mm['R3']:.3f} "
              f"R@5={mm['R5']:.3f} R@10={mm['R10']:.3f} MRR={mm['MRR']:.3f} "
              f"CI95(R@1)=[{lo:.3f},{hi:.3f}]", flush=True)
        del m


if __name__ == '__main__':
    main(sys.argv[1:] or ['intfloat/multilingual-e5-large-instruct'])
