# -*- coding: utf-8 -*-
"""Приёмка №2: метрика на 690 запросах через СЕТЕВОЙ эмбеддер.

Отличие от `v2/eval/run_eval.py` — не в метрике, а в том, откуда берутся векторы.
Там модель поднимается в процессе и корпус кодируется заново; здесь запрос уходит
в `POST /v1/embed` gpu-шлюза, а ранжирование идёт по индексу, который уже лежит
в Postgres. Это ровно тот путь, которым ходит рабочий сервер, поэтому проверяется
именно он: замена локальной модели на сетевую не должна была ничего сдвинуть.

Порог: R@1 >= 0.732, R@10 >= 0.977.

    GW_BASE_URL=… GW_TOKEN=… python3 tools/eval_http.py
"""
import json
import os
import sys
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(ROOT, "embending", "v2"))

from core import config as C                                   # noqa: E402
from core.embed_client import GatewayEmbedder                  # noqa: E402
from corpus import load_cards, build_types                     # noqa: E402

QUERIES = os.path.join(ROOT, "embending", "v2", "eval", "citizen_queries.jsonl")
FLOOR_R1, FLOOR_R10 = 0.732, 0.977
BATCH = 64


def load_queries():
    cards = load_cards(C.DATA)
    build_types(cards)                  # именно он проставляет type_id на карточки
    card2type = {c["id"]: c["type_id"] for c in cards}
    out = []
    for line in open(QUERIES, encoding="utf-8"):
        d = json.loads(line)
        gold = {card2type[i] for i in d["ids"] if i in card2type}
        if len(gold) != 1:
            continue                    # тип перегруппировался с момента генерации
        t = gold.pop()
        for q in d["queries"]:
            out.append((q, t))
    return out


def main(use_cache=False):
    qs = load_queries()
    print(f"запросов: {len(qs)}", flush=True)

    emb = GatewayEmbedder(kind="query", use_cache=use_cache)
    conn = psycopg.connect(C.DB, autocommit=True)
    register_vector(conn)

    t0 = time.perf_counter()
    vecs = []
    for i in range(0, len(qs), BATCH):
        vecs.extend(emb.encode([q for q, _ in qs[i:i + BATCH]]))
        print(f"  закодировано {min(i + BATCH, len(qs))}/{len(qs)}", end="\r", flush=True)
    t_emb = time.perf_counter() - t0
    print(f"\nэмбеддинг: {t_emb:.1f} с ({t_emb / len(qs) * 1000:.1f} мс/запрос)", flush=True)

    ranks = []
    for (q, gold), v in zip(qs, vecs):
        rows = conn.execute(
            "SELECT type_id FROM service_type ORDER BY embedding <=> %s LIMIT 100",
            (np.asarray(v, dtype=np.float32),)).fetchall()
        ids = [r[0] for r in rows]
        ranks.append(ids.index(gold) + 1 if gold in ids else 10_000)

    r = np.array(ranks, dtype=float)
    m = {"n": len(r), "R@1": (r <= 1).mean(), "R@3": (r <= 3).mean(),
         "R@5": (r <= 5).mean(), "R@10": (r <= 10).mean(), "MRR": float(np.mean(1 / r))}
    print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                   for k, v in m.items()), flush=True)

    # Порог задан с точностью до трёх знаков (0.732 это 505/690 = 0.73188…),
    # поэтому и сравнение идёт в той же точности, а не по сырому float.
    bad = []
    if round(m["R@1"], 3) < FLOOR_R1:
        bad.append(f'R@1 {m["R@1"]:.4f} < {FLOOR_R1}')
    if round(m["R@10"], 3) < FLOOR_R10:
        bad.append(f'R@10 {m["R@10"]:.4f} < {FLOOR_R10}')
    if bad:
        print("ПОРОГ НЕ ВЗЯТ: " + "; ".join(bad))
        return 1
    print(f"порог взят: R@1 >= {FLOOR_R1}, R@10 >= {FLOOR_R10}")
    return 0


if __name__ == "__main__":
    sys.exit(main(use_cache="--cache" in sys.argv))
