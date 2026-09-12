# -*- coding: utf-8 -*-
"""Справочники и состояние системы."""
import logging

from fastapi import APIRouter

from .. import config as C
from .. import db
from ..catalog import catalog
from ..search_service import search_service
from ..watchdog import watchdog

log = logging.getLogger("core.routers.reference")
router = APIRouter(prefix="/v1", tags=["reference"])


@router.get("/branches")
async def list_branches():
    rows = await db.fetch(
        "SELECT branch_id, name, municipality, address, windows, schedule "
        "FROM branch ORDER BY name")
    return [{"branch_id": r["branch_id"], "name": r["name"],
             "municipality": r["municipality"], "address": r["address"] or {},
             "windows": r["windows"], "schedule": list(r["schedule"] or [])} for r in rows]


@router.get("/version")
async def version():
    """`embed_fingerprint` обязан совпадать с тем, чем собран индекс.
    Несовпадение означает, что индекс и запросы кодируются разными моделями —
    это молчаливая деградация на 19 п.п. R@1, и сервис её не терпит."""
    from ..app_state import state
    stored = await db.meta_get("embed_fingerprint")
    fp = stored.get("fingerprint") if isinstance(stored, dict) else stored
    return {"api": C.API_VERSION,
            "corpus": {**(catalog.version or {}),
                       **(await db.meta_get("corpus_version") or {})},
            "embed_fingerprint": fp,
            "gpu_gateway": watchdog.status,
            "mode": search_service.mode,
            "index_fingerprint_mismatch": state.fingerprint_mismatch}


@router.get("/health")
async def health():
    return {"status": "alive"}
