# -*- coding: utf-8 -*-
"""Build the v2 index: 320 service types (vector + tsvector) + 732 instances."""
import json, os, sys, time
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import load_cards, build_types, search_text, type_body
from aliases import doc_aliases_for
from branches import load_branches, service_branches

DB    = os.getenv("EMB_DB", "host=localhost port=5434 dbname=embeddings user=emb password=emb")
MODEL = os.getenv("EMB_MODEL", "BAAI/bge-m3")
DOC_PREFIX = os.getenv("EMB_DOC_PREFIX", "")      # e5 family needs "passage: "
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.getenv("EMB_DATA", os.path.join(_ROOT, "data", "services_for_llm.json"))
CLASSIFICATION = os.path.join(_ROOT, "data", "flat_classification_clean.json")
KNOWLEDGE      = os.path.join(_ROOT, "data", "db_knowledge_clean.json")
HERE  = os.path.dirname(os.path.abspath(__file__))


def main():
    cards = load_cards(DATA)
    types = build_types(cards)
    print(f"cards={len(cards)} types={len(types)}", flush=True)

    model = SentenceTransformer(MODEL, trust_remote_code="bge" not in MODEL.lower())
    model.max_seq_length = 512
    dim = model.get_sentence_embedding_dimension()

    rows, texts = [], []
    for t in types:
        al = doc_aliases_for(t['title'], t.get('type_key'))
        st = search_text(t, al)
        texts.append(DOC_PREFIX + st)
        rows.append((t['type_id'], t['title'], ', '.join(t['life']), al, type_body(t), st, len(t['instances'])))

    t0 = time.time()
    embs = model.encode(texts, batch_size=8, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    print(f"encoded {len(texts)} types in {time.time()-t0:.1f}s (dim={dim})", flush=True)

    ddl = open(os.path.join(HERE, "schema.sql")).read().replace("%(DIM)s", str(dim))
    with psycopg.connect(DB) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(ddl)
            cur.executemany(
                "INSERT INTO service_type (type_id,title,life,aliases,body,search_text,"
                "n_instances,embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                [r + (e,) for r, e in zip(rows, embs)])
            cur.executemany(
                "INSERT INTO service (id,type_id,title,municipality,department,recipients,payload) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                [(c['id'], c['type_id'], c['raw_title'][:500], c['municipality'],
                  c['department_full'], c['recipients'], json.dumps(c, ensure_ascii=False))
                 for c in cards])
            if os.path.exists(CLASSIFICATION) and os.path.exists(KNOWLEDGE):
                br = load_branches(CLASSIFICATION)
                cur.executemany(
                    "INSERT INTO branch (branch_id,code,name,name_full,municipality,address,"
                    "chief,windows,area,schedule) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [(b['branch_id'], b['code'], b['name'], b['name_full'], b['municipality'],
                      json.dumps(b['address'], ensure_ascii=False), b['chief'], b['windows'],
                      b['area'], b['schedule']) for b in br])
                known = {b['branch_id'] for b in br}
                ids = {c['id'] for c in cards}
                link = [(sid, bid) for sid, bids in service_branches(KNOWLEDGE).items()
                        if sid in ids for bid in bids if bid in known]
                cur.executemany("INSERT INTO service_branch (id,branch_id) VALUES (%s,%s) "
                                "ON CONFLICT DO NOTHING", link)
                print(f"branch: {len(br)}  service_branch: {len(link)}")
            cur.execute("ANALYZE service_type; ANALYZE service; ANALYZE branch; ANALYZE service_branch;")
            print("service_type:", cur.execute("SELECT count(*) FROM service_type").fetchone()[0],
                  " service:", cur.execute("SELECT count(*) FROM service").fetchone()[0])
        conn.commit()


if __name__ == "__main__":
    main()
