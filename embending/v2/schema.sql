-- ---------------------------------------------------------------------------
-- v2 schema: service TYPE is the retrieval unit, municipality is a facet.
-- 732 cards collapse into ~322 types; the type is what a citizen searches for,
-- the municipality is what decides which card they are finally shown.
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

DROP TABLE IF EXISTS service_branch CASCADE;
DROP TABLE IF EXISTS branch CASCADE;
DROP TABLE IF EXISTS service CASCADE;
DROP TABLE IF EXISTS service_type CASCADE;

CREATE TABLE service_type (
    type_id      bigint PRIMARY KEY,
    title        text NOT NULL,            -- canonical, municipality-free name
    life         text NOT NULL DEFAULT '',
    aliases      text NOT NULL DEFAULT '', -- "как это называют люди"
    body         text NOT NULL DEFAULT '', -- кому / результат / документы
    search_text  text NOT NULL,            -- EXACTLY what was embedded
    n_instances  int  NOT NULL DEFAULT 1,
    embedding    vector(%(DIM)s),
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('russian', title),   'A') ||
        setweight(to_tsvector('russian', aliases), 'A') ||
        setweight(to_tsvector('russian', life),    'B') ||
        setweight(to_tsvector('russian', body),    'C')
    ) STORED
);

CREATE TABLE service (
    id           text PRIMARY KEY,
    type_id      bigint NOT NULL REFERENCES service_type(type_id) ON DELETE CASCADE,
    title        text NOT NULL,        -- original card title (with municipality)
    municipality text,                 -- NULL = region-wide service
    department   text,
    recipients   text[] NOT NULL DEFAULT '{}',
    payload      jsonb NOT NULL
);

CREATE TABLE branch (
    branch_id    text PRIMARY KEY,
    code         text,
    name         text NOT NULL,
    name_full    text,
    municipality text,                 -- выведен из адреса: контекст оператора по умолчанию
    address      jsonb,
    chief        text,
    windows      int,
    area         int,
    schedule     text[] NOT NULL DEFAULT '{}'
);

-- Связь «услуга ↔ где её получить». Почти полный граф (медиана 708 услуг на филиал),
-- поэтому для ранжирования бесполезна — нужна для карточки и для маршрутизации.
CREATE TABLE service_branch (
    id        text NOT NULL REFERENCES service(id) ON DELETE CASCADE,
    branch_id text NOT NULL REFERENCES branch(branch_id) ON DELETE CASCADE,
    PRIMARY KEY (id, branch_id)
);

-- ANN over 322 rows: HNSW is overkill but costs nothing and keeps the plan stable
CREATE INDEX service_type_emb_idx ON service_type
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX service_type_tsv_idx ON service_type USING gin (tsv);
CREATE INDEX service_type_trgm_idx ON service_type USING gin (title gin_trgm_ops);
CREATE INDEX service_type_idx ON service (type_id);
CREATE INDEX service_muni_idx ON service (municipality);
CREATE INDEX service_recip_idx ON service USING gin (recipients);
CREATE INDEX branch_muni_idx ON branch (municipality);
CREATE INDEX service_branch_b_idx ON service_branch (branch_id);
