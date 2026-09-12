import numpy as np
import torch
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from eval_bench import CASES, INSTR

DB = "host=localhost port=5434 dbname=embeddings user=emb password=emb"


def run(model_id: str, max_len: int = 512):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    use_instruct = "bge" not in model_id.lower()
    m = SentenceTransformer(model_id, trust_remote_code=use_instruct, device=dev)
    m.max_seq_length = max_len
    with psycopg.connect(DB) as conn:
        register_vector(conn)
        rows = conn.execute("SELECT id, content FROM services").fetchall()
    ids = np.array([r[0] for r in rows])
    texts = [r[1] for r in rows]
    d = m.encode(texts, batch_size=8, normalize_embeddings=True,
                 convert_to_numpy=True)
    ranks = []
    for q, true in CASES:
        qt = f"Instruct: {INSTR}\nQuery: {q}" if use_instruct else q
        v = m.encode([qt], normalize_embeddings=True, convert_to_numpy=True)[0]
        order = np.argsort(-(d @ v))
        ranks.append(int(np.where(ids[order] == true)[0][0]) + 1)
    n = len(ranks)
    print(f"{model_id:45s} dim={d.shape[1]:5d} R@1={sum(r<=1 for r in ranks)/n:.2f} "
          f"R@3={sum(r<=3 for r in ranks)/n:.2f} R@5={sum(r<=5 for r in ranks)/n:.2f} "
          f"MRR={sum(1/r for r in ranks)/n:.3f}", flush=True)
    for (q, true), r in zip(CASES, ranks):
        if r > 1:
            print(f"    miss rank={r:3d} {true:>15s}  {q[:60]}")
    del m
    if dev == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    import sys
    for mid in sys.argv[1:]:
        run(mid)
