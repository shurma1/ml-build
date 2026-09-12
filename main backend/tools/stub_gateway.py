# -*- coding: utf-8 -*-
"""Дублёр gpu-шлюза для локальной проверки core-api. НЕ ПРОДАКШН.

Настоящий шлюз (`gpu/`, образ reassel/mfc-gpu-gateway) требует видеокарты: GigaAM,
mE5-large-instruct и Qwen через vLLM. Здесь то же сетевое поведение — те же пути,
тот же конверт `{data, meta}`, тот же Bearer, тот же алгоритм отпечатка, —
но за ним:

  * `/v1/embed`   — НАСТОЯЩАЯ модель mE5-large-instruct через sentence-transformers
                    на CPU. Именно поэтому приёмка (R@1 0.732) на нём осмысленна:
                    числа те же, что у боевого шлюза, меняется только транспорт.
  * `/v1/llm/*`   — заглушка на газеттире и регулярках, НЕ языковая модель.
                    Годится проверить конвейер «реплики -> факты -> поиск», но не
                    качество извлечения.
  * `/v1/asr/*`   — распознавания нет; сегменты берутся из сценария (?script=…),
                    чтобы можно было прогнать WS-путь без микрофона и без GPU.

Запуск:
    ADMIN_SECRET=… python3 tools/stub_gateway.py --port 10100 --token dev-token
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "embending", "v2"))

EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-large-instruct")
EMB_MAX_SEQ = int(os.getenv("EMB_MAX_SEQ", "512"))
_INSTR = ("Given a Russian citizen's question about government services, "
          "retrieve the matching official service description")
QUERY_PREFIX = os.getenv("EMB_QUERY_PREFIX",
                         f"Instruct: {_INSTR}\nQuery: " if "instruct" in EMB_MODEL else "")
DOC_PREFIX = os.getenv("EMB_DOC_PREFIX", "")
TOKEN = os.getenv("GW_TOKEN", "")

STATE = {"model": None, "dim": None, "ready": False, "fail_readyz": False, "calls": 0}
# --no-model: поднять только /readyz и /v1/models, без весов. Нужен, чтобы
# проверить сверку отпечатков, не тратя минуту и гигабайты на вторую копию модели.
NO_MODEL = False
FP_OVERRIDE = os.getenv("FP_OVERRIDE", "")


def fingerprint():
    """Ровно тот же расчёт, что в gpu/gateway/embed.py — иначе сверка бессмысленна."""
    if FP_OVERRIDE:
        return FP_OVERRIDE
    h = hashlib.sha256(
        f"{EMB_MODEL}|{EMB_MAX_SEQ}|norm|{QUERY_PREFIX}|{DOC_PREFIX}".encode()).hexdigest()[:8]
    return f"{EMB_MODEL}/{EMB_MAX_SEQ}/norm/{h}"


def load_model():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB_MODEL, trust_remote_code="bge" not in EMB_MODEL.lower())
    m.max_seq_length = EMB_MAX_SEQ
    v = m.encode(["прогрев"], normalize_embeddings=True, convert_to_numpy=True)
    STATE["model"], STATE["dim"], STATE["ready"] = m, int(v.shape[1]), True
    print(f"stub-gateway: {EMB_MODEL} загружен, dim={STATE['dim']}, fp={fingerprint()}",
          flush=True)


app = FastAPI(title="stub gpu-gateway")
OPEN = {"/healthz", "/readyz", "/metrics", "/_admin/stats", "/_admin/readyz_fail"}


@app.on_event("startup")
async def _startup():
    if NO_MODEL:
        STATE["ready"], STATE["dim"] = True, 1024
        print(f"stub-gateway: без модели, fp={fingerprint()}", flush=True)
        return
    await asyncio.get_running_loop().run_in_executor(None, load_model)


@app.middleware("http")
async def auth(request: Request, call_next):
    if request.url.path not in OPEN and TOKEN:
        if request.headers.get("authorization", "") != f"Bearer {TOKEN}":
            return JSONResponse(status_code=401, content={"error": {
                "code": "unauthorized", "message": "нужен Bearer-токен", "retryable": False}})
    return await call_next(request)


def ok(data, t0, **meta):
    return {"data": data, "meta": {"ms": round((time.perf_counter() - t0) * 1000, 1), **meta}}


def fail(code, message, status=503, retryable=True):
    return JSONResponse(status_code=status,
                        content={"error": {"code": code, "message": message,
                                           "retryable": retryable}})


@app.get("/healthz")
async def healthz():
    return {"status": "alive"}


@app.get("/readyz")
async def readyz():
    if STATE["fail_readyz"] or not STATE["ready"]:
        return JSONResponse(status_code=503, content={
            "asr": False, "embed": STATE["ready"], "llm": False, "ready": False})
    return {"asr": True, "embed": True, "llm": True, "ready": True,
            "vram_used_mb": 0, "vram_total_mb": 0}


@app.get("/v1/models")
async def models():
    return {"asr": {"repo": "stub", "variant": "none"},
            "embed": {"model": EMB_MODEL, "dim": STATE["dim"], "fingerprint": fingerprint()},
            "llm": {"model": "stub-rules", "quantization": None, "max_model_len": 0},
            "prompt_version": "stub", "vendor_sha256": {}}


@app.get("/_admin/stats")
async def stats():
    """Счётчики вызовов — чтобы тест мог доказать, куда реально ушёл трафик."""
    return {"calls": STATE["calls"], "fingerprint": fingerprint(), "no_model": NO_MODEL}


@app.post("/_admin/readyz_fail")
async def readyz_fail(body: dict):
    """Только для тестов: погасить /readyz, не убивая процесс."""
    STATE["fail_readyz"] = bool(body.get("fail"))
    return {"fail_readyz": STATE["fail_readyz"]}


# --- эмбеддер: настоящая модель ---------------------------------------------

@app.post("/v1/embed")
async def embed(body: dict):
    if NO_MODEL:
        return fail("not_ready", "дублёр поднят без модели (--no-model)")
    if not STATE["ready"]:
        return fail("not_ready", "эмбеддер ещё грузится")
    texts, kind = body.get("texts") or [], body.get("kind")
    if kind not in ("query", "passage"):
        return fail("schema_validation_failed", "kind: query | passage", 400, False)
    if not texts or len(texts) > 256:
        return fail("schema_validation_failed", "texts: от 1 до 256 строк", 400, False)
    t0 = time.perf_counter()
    pref = QUERY_PREFIX if kind == "query" else DOC_PREFIX
    loop = asyncio.get_running_loop()
    vecs = await loop.run_in_executor(None, lambda: STATE["model"].encode(
        [pref + t for t in texts], normalize_embeddings=True, convert_to_numpy=True,
        batch_size=max(1, len(texts))))
    STATE["calls"] += 1
    return ok({"vectors": [[round(float(x), 6) for x in v] for v in vecs],
               "dim": STATE["dim"], "fingerprint": fingerprint()}, t0, model=EMB_MODEL)


# --- LLM: заглушка на правилах, а не модель ---------------------------------

_CATS = [("СВО", r"(?i)\bсво\b|специальн\w+ военн\w+ операц"),
         ("ВБД", r"(?i)ветеран\w* боевых|\bвбд\b"),
         ("ЧАЭС", r"(?i)чернобыл|\bчаэс\b"),
         ("многодетный", r"(?i)многодетн"),
         ("инвалид", r"(?i)инвалид"),
         ("пенсионер", r"(?i)пенсионер|на пенсии"),
         ("малоимущий", r"(?i)малоимущ|малообеспеч"),
         ("военнослужащий", r"(?i)военнослужащ|по контракту|мобилизованн")]
_LIFE = [("Рождение ребенка", r"(?i)родил|рождени\w* ребен|новорожд"),
         ("Многодетная семья", r"(?i)многодетн"),
         ("Детские пособия", r"(?i)пособи\w* на ребен|детск\w* пособи"),
         ("Выход на пенсию", r"(?i)пенси"),
         ("Утрата документов", r"(?i)потер\w|утрат\w|укра\w+ паспорт"),
         ("Индивидуальное жилищное строительство", r"(?i)\bижс\b|стро\w+ дом|разрешени\w* на строительств"),
         ("Сделки с недвижимостью", r"(?i)купл\w*-продаж|сделк\w* с недвижим|росреестр"),
         ("Открытие своего дела", r"(?i)открыть \w*\s*ип|регистрац\w+ ип|своё дело|свое дело"),
         ("Меры поддержки СВО", r"(?i)\bсво\b|специальн\w+ военн\w+ операц")]
_NUM = {"один": 1, "одного": 1, "два": 2, "двое": 2, "двоих": 2, "три": 3, "трое": 3,
        "троих": 3, "четыре": 4, "четверо": 4, "пять": 5, "пятеро": 5}


def _rule_extract(turns):
    """Не языковая модель, а правила: газеттир + регулярки. Хватает, чтобы
    прогнать конвейер целиком; качества извлечения здесь мерить нельзя."""
    from facets import find_municipality, find_recipient
    client = " ".join(t.get("text") or "" for t in turns
                      if (t.get("speaker") or "client") != "operator")
    whole = " ".join(t.get("text") or "" for t in turns)
    out = {}
    cats = [c for c, rx in _CATS if re.search(rx, client)]
    if cats:
        out["categories"] = sorted(set(cats))
    life = next((l for l, rx in _LIFE if re.search(rx, client)), None)
    if life:
        out["life_situation"] = life
    mo = find_municipality(whole)
    if mo:
        out["municipality"] = mo
    rec = find_recipient(client)
    out["recipient"] = rec or "person"
    m = re.search(r"(?i)мне\s+(\d{1,3})\s*(?:год|лет|года)", client)
    if m:
        out["age"] = int(m.group(1))
    m = re.search(r"(?i)(\d{1,2}|" + "|".join(_NUM) + r")\s*(?:ребен|детей|дет)", client)
    if m:
        g = m.group(1).lower()
        out["children"] = int(g) if g.isdigit() else _NUM.get(g)
    # намерение = самые содержательные реплики клиента, обрезанные до 90 символов
    sents = [s.strip() for s in re.split(r"[.!?\n]", client) if len(s.strip()) > 12]
    sents.sort(key=len, reverse=True)
    out["intents"] = [s[:90] for s in sents[:3]] or [client[:90] or "консультация"]
    return out


@app.post("/v1/llm/extract")
async def llm_extract(body: dict):
    from extraction_schema import validate
    turns = body.get("turns") or []
    if not turns:
        return fail("schema_validation_failed", "turns не может быть пустым", 400, False)
    t0 = time.perf_counter()
    raw = _rule_extract(turns)
    facts = validate(raw)
    dropped = sorted(set(raw) - set(facts))
    return ok({"facts": facts, "raw_dropped": dropped}, t0, model="stub-rules",
              prompt_tokens=sum(len(t.get("text") or "") // 4 for t in turns),
              cached_prefix_tokens=0)


@app.post("/v1/llm/ask")
async def llm_ask(body: dict):
    doc, q = body.get("document") or {}, (body.get("question") or "").strip()
    if not doc or not q:
        return fail("schema_validation_failed", "нужны document и question", 400, False)
    # Совпадение по основе слова (первые 5 букв): «не положено» найдёт
    # «положении», «потерял» — «утраты» не найдёт никогда, но найдёт тему.
    stems = [w[:5] for w in re.findall(r"[а-яёa-z]{4,}", q.lower())]
    best = None                                   # (score, field, quote, start)
    for field, text in doc.items():
        if not isinstance(text, str):
            continue
        for sent in re.split(r"(?<=[.;\n])", text):
            low = sent.lower()
            score = sum(1 for s in stems if s in low)
            if score and (best is None or score > best[0]):
                quote = sent.strip()[:220]
                best = (score, field, quote, text.find(quote))
    if best:
        hit_field, quote, hit_start = best[1], best[2], best[3]
        answer = f"В регламенте по этому вопросу сказано: «{quote}»"
    else:
        # Отказа нет: даже без совпадения оператор получает опору — название
        # услуги и её заявителей, — а не строчку «не указано».
        anchor = next((doc[k] for k in ("serviceRecipients", "serviceTitleText")
                       if isinstance(doc.get(k), str) and doc[k].strip()), None)
        if anchor is None:
            anchor = next((v for v in doc.values() if isinstance(v, str) and v.strip()), "")
        quote = anchor.strip().split("\n")[0][:220].rstrip(".")
        answer = (f"Вопрос шире формулировок регламента, вот опора: «{quote}». "
                  "Уточните формулировку — подскажу по карточке.")
        hit_field = next((k for k, v in doc.items()
                          if isinstance(v, str) and quote in v), "serviceTitleText")
        hit_start = str(doc[hit_field]).find(quote)

    async def sse():
        for chunk in re.findall(r".{1,40}(?:\s|$)", answer):
            yield f"event: delta\ndata: {json.dumps({'type':'delta','text':chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.005)
        if quote:
            cit = {"type": "citation", "quote": quote, "field": hit_field,
                   "start": hit_start, "end": hit_start + len(quote)}
            yield f"event: citation\ndata: {json.dumps(cit, ensure_ascii=False)}\n\n"
        yield f"event: done\ndata: {json.dumps({'type':'done'}, ensure_ascii=False)}\n\n"

    if not body.get("stream", True):
        return ok({"answer": answer, "citations": []}, time.perf_counter())
    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/v1/llm/explain")
async def llm_explain(body: dict):
    title, status = body.get("title"), body.get("status")
    if not title or status not in ("eligible", "blocked", "unknown"):
        return fail("schema_validation_failed", "нужны title и status", 400, False)
    reasons = body.get("reasons") or []
    why = "; ".join(r.get("why") or "" for r in reasons if r.get("why"))
    text = {"eligible": f"«{title}» — вам подходит.",
            "blocked": f"«{title}» — не положено: {why}." if why else f"«{title}» — не положено.",
            "unknown": f"«{title}» — нужно уточнить: {why}." if why else f"«{title}» — нужны уточнения."}[status]
    return ok({"text": text, "alternatives_hint": None}, time.perf_counter(), model="stub-rules")


@app.post("/v1/asr/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    await audio.read()
    return ok({"segments": []}, time.perf_counter(), model="stub")


@app.websocket("/v1/asr/stream")
async def asr_stream(ws: WebSocket):
    """Распознавания нет. Сегменты берутся из ?script=реплика|реплика — это
    позволяет прогнать WS-путь core-api целиком, не имея ни GPU, ни микрофона."""
    if TOKEN and ws.query_params.get("token") != TOKEN:
        await ws.close(code=4401)
        return
    await ws.accept()
    script = [s for s in (ws.query_params.get("script") or "").split("|") if s.strip()]
    every = int(ws.query_params.get("every_bytes") or 3200)
    seq, got = 0, 0
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text"):
                if json.loads(msg["text"]).get("type") == "close":
                    break
                continue
            got += len(msg.get("bytes") or b"")
            while script and got >= every:
                got -= every
                seq += 1
                await ws.send_json({"type": "utterance", "seq": seq, "t0": seq * 1.0,
                                    "t1": seq * 1.0 + 0.9, "text": script.pop(0),
                                    "speaker": None, "ms": 12})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=10100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--token", default=TOKEN)
    ap.add_argument("--no-model", action="store_true",
                    help="не грузить веса: только /readyz и /v1/models")
    ap.add_argument("--fingerprint", default=FP_OVERRIDE,
                    help="подменить отпечаток — для проверки сверки")
    a = ap.parse_args()
    TOKEN = a.token
    NO_MODEL = a.no_model
    FP_OVERRIDE = a.fingerprint
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
