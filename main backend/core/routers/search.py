# -*- coding: utf-8 -*-
"""Разовый поиск без сессии: строка поиска в UI и регрессионные прогоны."""
import logging
import sys

from fastapi import APIRouter

from .. import config as C
from ..errors import ApiError
from ..search_service import search_service
from ..sessions import enrich

log = logging.getLogger("core.routers.search")
router = APIRouter(prefix="/v1", tags=["search"])


@router.post("/search")
async def search(body: dict):
    if C.V2_PATH not in sys.path:
        sys.path.insert(0, C.V2_PATH)
    from features import QueryFeatures

    body = body or {}
    text = (body.get("text") or "").strip()
    raw_features = body.get("features") or {}
    if not text and not raw_features:
        raise ApiError("schema_validation_failed", "нужен text или features")
    k = body.get("k") or C.SEARCH_K
    if not isinstance(k, int) or not 1 <= k <= 30:
        raise ApiError("schema_validation_failed", "k: целое 1..30")

    if raw_features:
        feats = QueryFeatures.from_llm({**raw_features, "raw_text": text})
        if not feats.search_strings() and text:
            feats = QueryFeatures.from_text(text)
    else:
        feats = QueryFeatures.from_text(text)

    muni = None
    branch_id = body.get("branch_id")
    if branch_id:
        from .. import db
        row = await db.fetchrow("SELECT municipality FROM branch WHERE branch_id = %s",
                                (branch_id,))
        if not row:
            raise ApiError("schema_validation_failed", f"филиал {branch_id} неизвестен")
        muni = row["municipality"]

    r = await search_service.search(feats, k=k, municipality=muni)
    return await enrich(r)
