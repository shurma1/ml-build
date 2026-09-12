import json
import numpy as np
import torch
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DB = "host=localhost port=5434 dbname=embeddings user=emb password=emb"
GIGA = "ai-sage/Giga-Embeddings-instruct-480M-0826"
E5 = "intfloat/multilingual-e5-large-instruct"
INSTR = ("Given a Russian citizen's question about government services, "
         "retrieve the matching official service description")

CASES = [
    ("во сколько лет меняют паспорт и какие документы для этого нужны", "10000000611"),
    ("хочу выехать за границу как оформить паспорт для поездок за рубеж", "10000008589"),
    ("мне 24 года родила второго ребёнка положена ли мне какая-то выплата", "710000050"),
    ("служил в армии имею ли право на вторую пенсию нужна справка из военкомата", "voenkomat2"),
    ("у меня нет прав как получить водительское удостоверение", "10000001193"),
    ("хотим подать заявление чтобы расписаться", "rast_brak"),
    ("как официально оформить брак", "reg_brak"),
    ("нужна справка для визы о размере моей пенсии", "Srv00054"),
    ("хочу получить скидочную карту забота в алексине", "Zabota_Aleksin_1"),
    ("потерял российский паспорт надо восстановить", "10000000611"),
    ("какие налоги платит ип и куда обратиться за информированием по налогам", "10001760860"),
    ("родился третий ребёнок до какого возраста выплата молодой маме", "710000050"),
    ("оформить биометрический загранпаспорт нового поколения", "10000008589"),
    ("хочу развестись если у нас нет общих детей", "rast_brak"),
    ("хочу узнать свой размер пенсии справка", "Srv00054"),
    ("обменять водительские права после окончания срока действия", "10000001193"),
]


def metrics(rank_fn):
    ranks = [rank_fn(q, true) for q, true in CASES]
    n = len(ranks)
    r1 = sum(r <= 1 for r in ranks) / n
    r3 = sum(r <= 3 for r in ranks) / n
    r5 = sum(r <= 5 for r in ranks) / n
    mrr = sum(1 / r for r in ranks) / n
    return r1, r3, r5, mrr, ranks


def main():
    with torch.device("mps" if torch.backends.mps.is_available() else "cpu"):
        pass
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    # corpus from DB (exact texts used at index time)
    with psycopg.connect(DB) as conn:
        register_vector(conn)
        rows = conn.execute("SELECT id, content, embedding::text FROM services").fetchall()
    ids = np.array([r[0] for r in rows])
    texts = [r[1] for r in rows]
    giga_emb = np.asarray([[float(x) for x in r[2][1:-1].split(",")] for r in rows], dtype=np.float32)

    def giga_ranker(model, with_instr):
        def rank(q, true):
            t = f"Instruct: {INSTR}\nQuery: {q}" if with_instr else q
            v = model.encode([t], normalize_embeddings=True)[0].astype(np.float32)
            sims = giga_emb @ v
            order = np.argsort(-sims)
            return int(np.where(ids[order] == true)[0][0]) + 1
        return rank

    m = SentenceTransformer(GIGA, trust_remote_code=True, device=dev)
    m.max_seq_length = 512
    for name, wi in [("Giga-480M (instruct)", True), ("Giga-480M (no instruct)", False)]:
        r1, r3, r5, mrr, ranks = metrics(giga_ranker(m, wi))
        print(f"{name:26s} R@1={r1:.2f} R@3={r3:.2f} R@5={r5:.2f} MRR={mrr:.3f}")
        for q, true, r in zip([c[0] for c in CASES], [c[1] for c in CASES], ranks):
            if r > 1:
                print(f"    miss rank={r:3d} {true:>15s}  {q[:60]}")
    del m
    torch.mps.empty_cache()

    e5 = SentenceTransformer(E5, device=dev)
    e5.max_seq_length = 512
    cache = "/tmp/e5_corpus.npz"
    try:
        d = np.load(cache)
        e5_emb = d["emb"]
    except Exception:
        e5_emb = e5.encode(texts, batch_size=16, normalize_embeddings=True,
                           convert_to_numpy=True,
                           prompt_name=None).astype(np.float32)
        np.savez_compressed(cache, emb=e5_emb)

    def e5_rank(q, true):
        v = e5.encode([f"Instruct: {INSTR}\nQuery: {q}"], normalize_embeddings=True)[0].astype(np.float32)
        order = np.argsort(-(e5_emb @ v))
        return int(np.where(ids[order] == true)[0][0]) + 1

    r1, r3, r5, mrr, ranks = metrics(e5_rank)
    print(f"{'mE5-large-instruct (0.56B)':26s} R@1={r1:.2f} R@3={r3:.2f} R@5={r5:.2f} MRR={mrr:.3f}")
    for (q, true), r in zip(CASES, ranks):
        if r > 1:
            print(f"    miss rank={r:3d} {true:>15s}  {q[:60]}")


if __name__ == "__main__":
    main()
