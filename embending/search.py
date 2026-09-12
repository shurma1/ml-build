import sys
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DB = "host=localhost port=5434 dbname=embeddings user=emb password=emb"
MODEL = "BAAI/bge-m3"
_model = None


def search(query: str, k: int = 5):
    q = _model.encode([query], normalize_embeddings=True)[0]
    with psycopg.connect(DB) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, department, 1 - (embedding <=> %s) AS score "
                "FROM services ORDER BY embedding <=> %s LIMIT %s",
                (q, q, k))
            return cur.fetchall()


def main():
    query = " ".join(sys.argv[1:]) or "Как получить паспорт в МФЦ?"
    global _model
    _model = SentenceTransformer(MODEL)
    _model.max_seq_length = 512
    for r in search(query):
        print(f"{r[3]:.4f}  [{r[0]}] {r[1][:90]} | {r[2][:40] if r[2] else ''}")


if __name__ == "__main__":
    main()
