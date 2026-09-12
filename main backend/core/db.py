# -*- coding: utf-8 -*-
"""Асинхронный доступ к Postgres + таблица `meta`.

Две разные связи с одной базой живут рядом и это сознательно:
  * здесь — async-пул psycopg 3 для рантайм-таблиц (session, turn, fact_state,
    search_log, query_cache, meta);
  * в embending/v2 — свои синхронные соединения под поиск. Запросы там написаны
    на чистом SQL и измерены; переписывать их в async ради единообразия значит
    трогать то, что работает. Поиск исполняется в пуле потоков (см. search_service).

`meta` — это и есть то, что делает смену адреса шлюза переживающей перезапуск:
конфигурация лежит в базе, а не в памяти процесса и не в окружении.
"""
import json
import logging
import os

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async

from . import config as C

log = logging.getLogger("core.db")

_pool = None
RUNTIME_DDL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "api", "schema_runtime.sql")


async def _configure(conn):
    await register_vector_async(conn)


async def open_pool():
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            C.DB, min_size=C.DB_POOL_MIN, max_size=C.DB_POOL_MAX,
            kwargs={"autocommit": True, "row_factory": dict_row},
            configure=_configure, open=False,
        )
        await _pool.open(wait=True, timeout=30)
    return _pool


async def close_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool():
    if _pool is None:
        raise RuntimeError("пул к базе ещё не открыт")
    return _pool


async def fetch(sql, params=()):
    async with pool().connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchall()


async def fetchrow(sql, params=()):
    async with pool().connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchone()


async def execute(sql, params=()):
    async with pool().connection() as conn:
        cur = await conn.execute(sql, params)
        return cur.rowcount


async def apply_runtime_schema():
    """Идемпотентный CREATE TABLE IF NOT EXISTS из api/schema_runtime.sql."""
    path = RUNTIME_DDL
    if not os.path.exists(path):
        log.warning("schema_runtime.sql не найден по пути %s — схема не применена", path)
        return False
    ddl = open(path, encoding="utf-8").read()
    async with pool().connection() as conn:
        await conn.execute(ddl)
    return True


# --- meta -------------------------------------------------------------------

async def meta_get(key, default=None):
    row = await fetchrow("SELECT value FROM meta WHERE key = %s", (key,))
    return row["value"] if row else default


async def meta_set(key, value):
    await execute(
        "INSERT INTO meta (key, value, updated_at) VALUES (%s, %s, now()) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
        (key, json.dumps(value, ensure_ascii=False)))
    return value


async def meta_all():
    rows = await fetch("SELECT key, value, updated_at FROM meta")
    return {r["key"]: r["value"] for r in rows}
