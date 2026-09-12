import json
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DB = "host=localhost port=5434 dbname=embeddings user=emb password=emb"
MODEL = "BAAI/bge-m3"
DATA = "/Users/dmitriy/Desktop/sber embending/data/services_for_llm.json"


def compose(row: dict) -> str:
    parts = [
        row.get("serviceTitleText") or "",
        row.get("departmentName") or "",
        row.get("serviceRecipients") or "",
        row.get("serviceOrderingText") or "",
        row.get("paymentInfoText") or "",
        row.get("timeTermText") or "",
        row.get("serviceResultText") or "",
        ", ".join(row.get("lifeSituationNames") or []),
    ]
    return "\n".join(p for p in parts if p)


def main():
    with open(DATA, encoding="utf-8") as f:
        rows = json.load(f)

    model = SentenceTransformer(MODEL)
    model.max_seq_length = 512

    with psycopg.connect(DB) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE services")
            batch, size = [], 32
            for r in rows:
                batch.append(r)
                if len(batch) == size:
                    flush(cur, model, batch)
                    batch = []
            if batch:
                flush(cur, model, batch)
            cur.execute("SELECT count(*), count(embedding) FROM services")
            print("inserted rows:", cur.fetchone())
        conn.commit()


def flush(cur, model, batch):
    texts = [compose(r) for r in batch]
    embs = model.encode(texts, batch_size=4, normalize_embeddings=True,
                        show_progress_bar=False)
    cur.executemany(
        "INSERT INTO services (id, title, department, content, embedding) "
        "VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding",
        [(r["id"], r["serviceTitleText"][:300], r.get("departmentName"),
          compose(r)[:4000], e) for r, e in zip(batch, embs)],
    )
    print("batch ok", len(batch), flush=True)


if __name__ == "__main__":
    main()
