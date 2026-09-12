-- ---------------------------------------------------------------------------
-- Дополнение к embending/v2/schema.sql: рантайм основного сервера.
--
-- v2/schema.sql описывает КОРПУС (service_type, service, branch, service_branch) —
-- он перестраивается целиком при смене данных. Здесь — то, что копится в работе
-- и переживает перестройку индекса: сессии, реплики, состояние, лог поиска.
--
-- Главная ценность этих таблиц не в работе приложения, а в том, что они
-- производят живой лог. Сейчас его нет, и поэтому реальный R@1 известен только
-- полосой 0.70-0.80: синтетические запросы порождены из тех же полей, что
-- индексируются (завышают), а 62% хвостовых промахов - кривая разметка эталона
-- (занижает). Пара «признаки диалога -> услуга, которую оператор в итоге открыл»
-- закрывает этот вопрос за несколько недель пилота.
-- ---------------------------------------------------------------------------

-- Защита от рассинхрона индекса и кодировщика. Проверяется при старте core-api:
-- несовпадение отпечатка = отказ обслуживать, а не «как-нибудь поищем».
CREATE TABLE IF NOT EXISTS meta (
    key        text PRIMARY KEY,
    value      jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
-- meta['embed_fingerprint'] = {"model":"intfloat/multilingual-e5-large-instruct",
--                              "dim":1024,"max_seq":512,"normalize":true,
--                              "query_prefix_sha":"9f1c2a"}
-- meta['corpus_version']    = {"cards":732,"types":318,"built_at":"...","source_sha":"..."}

CREATE TABLE IF NOT EXISTS session (
    session_id    uuid PRIMARY KEY,
    branch_id     text REFERENCES branch(branch_id),
    operator_id   text NOT NULL,
    "window"      int,                    -- window — зарезервированное слово PG, отсюда кавычки
    municipality  text,                    -- по умолчанию из адреса филиала
    started_at    timestamptz NOT NULL DEFAULT now(),
    closed_at     timestamptz,
    outcome       text,                    -- served | not_found | wrong_branch | refused | abandoned
    chosen_service_id text REFERENCES service(id),
    operator_note text,
    purge_after   timestamptz NOT NULL     -- см. «Хранение ПДн» ниже
        DEFAULT now() + interval '30 days'
);

-- Реплика диалога. text — персональные данные: чистится по purge_after,
-- агрегаты в search_log переживают чистку.
CREATE TABLE IF NOT EXISTS turn (
    session_id uuid NOT NULL REFERENCES session(session_id) ON DELETE CASCADE,
    seq        int  NOT NULL,
    t0         real,
    t1         real,
    speaker    text,                       -- client | operator | NULL (диаризации нет)
    source     text NOT NULL DEFAULT 'asr',-- asr | operator_typed
    text       text NOT NULL,
    empty      boolean NOT NULL DEFAULT false,  -- «ага», «понятно» — LLM не вызывалась
    asr_ms     int,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, seq)
);

-- Снимок DialogState после каждого хода. Нужен и для восстановления UI после
-- обрыва WS, и для замера дрейфа фактов на живых диалогах — той же метрикой,
-- что в eval/measure_churn.py.
CREATE TABLE IF NOT EXISTS fact_state (
    session_id uuid NOT NULL REFERENCES session(session_id) ON DELETE CASCADE,
    seq        int  NOT NULL,
    state      jsonb NOT NULL,
    changed    jsonb NOT NULL DEFAULT '{}'::jsonb,   -- поле -> [было, стало]
    pinned     text[] NOT NULL DEFAULT '{}',         -- выставлено оператором вручную
    dropped    text[] NOT NULL DEFAULT '{}',         -- отброшено validate() — сигнал дрейфа схемы
    llm_ms     int,
    PRIMARY KEY (session_id, seq)
);

-- Лог поиска. Это и есть будущий эталон релевантности.
CREATE TABLE IF NOT EXISTS search_log (
    session_id uuid NOT NULL REFERENCES session(session_id) ON DELETE CASCADE,
    seq        int  NOT NULL,
    features   jsonb NOT NULL,             -- QueryFeatures, ушедшие в энкодер
    results    jsonb NOT NULL,             -- top-k: type_id, rank, score, status
    questions  jsonb NOT NULL DEFAULT '[]'::jsonb,
    mode       text NOT NULL DEFAULT 'dense',   -- dense | degraded_lexical
    opened     text[] NOT NULL DEFAULT '{}',    -- какие карточки оператор реально открыл
    ms         int,
    ms_embed   int,
    corpus_version text,
    PRIMARY KEY (session_id, seq)
);

-- Кэш векторов запросов. Эмбеддер живёт только на GPU-боксе, поэтому кэш
-- обязан пережить его перезапуск: намерения в диалогах повторяются, и по
-- замерам тёплый кэш это 7-9 мс против 120 мс холодного.
CREATE TABLE IF NOT EXISTS query_cache (
    text_sha    bytea PRIMARY KEY,
    text        text NOT NULL,
    fingerprint text NOT NULL,             -- при смене отпечатка кэш инвалидируется целиком
    embedding   vector(1024) NOT NULL,
    hits        int NOT NULL DEFAULT 0,
    last_seen   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS session_branch_idx  ON session (branch_id, started_at DESC);
CREATE INDEX IF NOT EXISTS session_purge_idx   ON session (purge_after);
CREATE INDEX IF NOT EXISTS turn_session_idx    ON turn (session_id, seq);
CREATE INDEX IF NOT EXISTS search_log_mode_idx ON search_log (mode, seq);
CREATE INDEX IF NOT EXISTS query_cache_fp_idx  ON query_cache (fingerprint);

-- Хранение ПДн. Расшифровка разговора с посетителем — персональные данные.
-- Ежесуточный джоб: удалить turn.text и fact_state.state по истёкшему purge_after,
-- оставив search_log (в нём нет прямой речи — только признаки, ранги и статусы).
-- Так живой лог для метрики копится бессрочно, а сырая речь не хранится дольше
-- срока, на который получено согласие.
