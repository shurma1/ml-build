# -*- coding: utf-8 -*-
"""gpu-gateway: три модели за одним портом.

Сервис не хранит состояния и не знает бизнес-логики: ни сессий, ни диалогов,
ни корпуса услуг. Инстанс vast.ai может быть вытеснен в любую минуту — потеря
этого сервиса обязана быть деградацией, а не порчей данных. Всё, что переживает
рестарт, живёт в core-api на основном сервере.

Наружу смотрит один порт. Шлюз поднимается ПЕРВЫМ, до всех моделей, и сам
запускает vLLM дочерним процессом: иначе первые двадцать минут жизни инстанса
наружу не отвечает ничего и непонятно, идёт загрузка или всё встало.
Страница статуса — на `/`.
"""
import asyncio
import io
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse, HTMLResponse

from . import config as C
from . import boot as B
from .asr import AsrWorker, pcm16_to_float32
from .embed import EmbedWorker, fingerprint
from .llm import LlmClient
from .prompts import PROMPT_VERSION

STARTED = time.time()
asr = AsrWorker()
emb = EmbedWorker()
llm = LlmClient()
boot = B.Boot()
vllm = None
METRICS = {"asr_segments": 0, "asr_dropped": 0, "embed_texts": 0,
           "llm_extract": 0, "llm_ask": 0, "llm_dropped_keys": 0, "errors": 0}
STATE = {"asr": False, "embed": False, "llm": False}
# Сколько видеопамяти реально заняли ASR и эмбеддер — против того, сколько
# под них зарезервировано. Расхождение видно на странице статуса, и резерв
# можно сузить переменной RESERVE_GB, освободив память под KV-кэш.
VRAM = {"ours_mb": None, "reserve_mb": None}


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


async def _sequence():
    """Порядок задан видеопамятью, а не удобством.

    vLLM профилирует память при старте и забирает свою долю ПЕРВЫМ; GigaAM
    и эмбеддер грузятся в остаток. Поменять местами — получить OOM у vLLM
    на ровном месте.
    """
    global vllm
    loop = asyncio.get_running_loop()

    try:
        boot.begin("gpu")
        boot.gpu = B.gpu_info()
        if boot.gpu.get("error"):
            boot.fail("gpu", boot.gpu["error"]); return
        cc = boot.gpu.get("compute_capability", "—")
        boot.done("gpu", f"{boot.gpu.get('name')} · sm_{cc.replace('.', '')} · "
                         f"torch {boot.gpu.get('torch')} / CUDA {boot.gpu.get('cuda')}")

        # --- какая модель ---
        p = boot.begin("download_llm", "ищу репозиторий")
        model, tried = await loop.run_in_executor(None, B.resolve_llm_model, C.LLM_MODEL)
        if not model:
            boot.fail("download_llm", "не найден ни один кандидат: " + ", ".join(tried))
            return
        C.LLM_MODEL = llm.model = model
        p.detail = model
        boot.done("download_llm", await loop.run_in_executor(None, B.download, p, model))

        p = boot.begin("download_asr", f"{C.ASR_REPO} @ {C.ASR_VARIANT}")
        boot.done("download_asr", await loop.run_in_executor(
            None, lambda: B.download(p, C.ASR_REPO, revision=C.ASR_VARIANT)))

        p = boot.begin("download_embed", C.EMB_MODEL)
        boot.done("download_embed", await loop.run_in_executor(None, B.download, p, C.EMB_MODEL))

        # --- vLLM первым за памятью ---
        p = boot.begin("vllm", "старт процесса")
        vllm = B.VllmProcess(boot, model)
        secs, note = await vllm.run(p)
        boot.done("vllm", f"поднялся за {secs} с · {note}")

        # Замер собственного потребления: снимок ДО наших моделей, когда
        # vLLM уже забрал свою долю. Разница и есть то, подо что резервируем.
        _vram_before = (B.gpu_info().get("vram_used_mb") or 0)
        p = boot.begin("load_asr", f"{C.ASR_REPO} / {C.ASR_VARIANT}")
        dev = _device()
        await loop.run_in_executor(None, asr.load, dev)
        STATE["asr"] = True
        boot.done("load_asr", f"на {dev}, прогрет")

        p = boot.begin("load_embed", C.EMB_MODEL)
        await loop.run_in_executor(None, emb.load, dev)
        STATE["embed"] = True
        boot.done("load_embed", f"dim={emb.dim}, {fingerprint()}")

        ours = (B.gpu_info().get("vram_used_mb") or 0) - _vram_before
        VRAM["ours_mb"] = max(0, ours)
        VRAM["reserve_mb"] = int(B.RESERVE_GB * 1024)

        boot.begin("probe_llm")
        info = await llm.probe()
        STATE["llm"] = True
        if info["struct_mode"] == "none":
            # Не падаем, но и не делаем вид, что всё хорошо: без грамматики
            # дрейф значений 54% вместо 7%. Это авария, а не рабочий режим.
            boot.fail("probe_llm", "движок не принял JSON-схему ни одним способом")
        else:
            boot.done("probe_llm", f"{info['model']} · режим {info['struct_mode']}")
    except Exception as e:                        # noqa: BLE001
        current = next((k for k in boot.order if boot.phases[k].state == "running"), "gpu")
        boot.fail(current, f"{type(e).__name__}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_sequence())
    yield
    task.cancel()
    if vllm:
        vllm.stop()
    await llm.close()


app = FastAPI(title="mfc gpu-gateway", version="1.0.0", lifespan=lifespan)

OPEN_PATHS = {"/", "/healthz", "/readyz", "/metrics"}


@app.middleware("http")
async def envelope(request: Request, call_next):
    rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
    if request.url.path not in OPEN_PATHS and C.TOKEN:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {C.TOKEN}":
            return JSONResponse(status_code=401, content={"error": {
                "code": "unauthorized", "message": "нужен Bearer-токен",
                "retryable": False, "request_id": rid}})
    try:
        resp = await call_next(request)
    except HTTPException:
        raise
    except Exception as e:                        # noqa: BLE001
        METRICS["errors"] += 1
        return JSONResponse(status_code=500, content={"error": {
            "code": "internal", "message": f"{type(e).__name__}: {e}",
            "retryable": True, "request_id": rid}})
    resp.headers["X-Request-Id"] = rid
    return resp


def ok(data, t0, **meta):
    return {"data": data, "meta": {"ms": round((time.perf_counter() - t0) * 1000, 1), **meta}}


def fail(code, message, status=503, retryable=True):
    return JSONResponse(status_code=status, content={"error": {
        "code": code, "message": message, "retryable": retryable}})


# --- эксплуатация ------------------------------------------------------------

@app.get("/healthz")
async def healthz():
    return {"status": "alive", "uptime_s": round(time.time() - STARTED)}


@app.get("/readyz")
async def readyz():
    body = {**STATE, "ready": boot.ready, "error": boot.error}
    g = B.gpu_info()
    body["vram_used_mb"], body["vram_total_mb"] = g.get("vram_used_mb"), g.get("vram_total_mb")
    return JSONResponse(status_code=200 if boot.ready else 503, content=body)


@app.get("/v1/status")
async def status():
    """Всё, что рисует страница статуса: фазы, прогресс, память, счётчики."""
    boot.gpu.update({k: v for k, v in B.gpu_info().items() if k.startswith("vram")})
    d = boot.dict(models=await _models_dict())
    d["metrics"] = dict(METRICS)
    d["vram_ours_mb"] = VRAM["ours_mb"]
    d["vram_reserve_mb"] = VRAM["reserve_mb"]
    d["vllm_alive"] = bool(boot.vllm_proc and boot.vllm_proc.poll() is None)
    return d


@app.get("/v1/logs")
async def logs(name: str = "vllm", tail: int = 200):
    if name not in ("vllm", "gateway", "supervisord"):
        return fail("schema_validation_failed", "name: vllm | gateway | supervisord", 400, False)
    tail = max(1, min(tail, 2000))
    if name == "vllm" and boot.vllm_tail:
        # Кольцевой буфер свежее файла: строка попадает в него до сброса на диск
        lines = list(boot.vllm_tail)[-tail:]
    else:
        lines = B.read_log(name, tail)
    return {"name": name, "lines": lines}


async def _models_dict():
    vendor = {}
    man = os.path.join(os.path.dirname(__file__), "vendor", "MANIFEST")
    if os.path.exists(man):
        for line in open(man, encoding="utf-8"):
            parts = line.split()
            if len(parts) == 2:
                vendor[parts[0]] = parts[1]
    return {"asr": asr.info(), "embed": emb.info(), "llm": llm.info(),
            "prompt_version": PROMPT_VERSION, "vendor_sha256": vendor}


@app.get("/v1/models")
async def models():
    return await _models_dict()


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    lines = [f"gw_{k} {v}" for k, v in METRICS.items()]
    lines += [f"gw_ready_{k} {int(bool(v))}" for k, v in STATE.items()]
    lines.append(f"gw_ready {int(boot.ready)}")
    lines.append(f"gw_uptime_seconds {round(time.time() - STARTED)}")
    g = B.gpu_info()
    if g.get("vram_total_mb"):
        lines.append(f"gw_vram_used_mb {g['vram_used_mb']}")
        lines.append(f"gw_vram_total_mb {g['vram_total_mb']}")
    return "\n".join(lines) + "\n"


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Страница статуса. Открыта без токена — это только разметка;
    за данными она ходит в /v1/status с Bearer, который вводит оператор."""
    path = os.path.join(os.path.dirname(__file__), "ui.html")
    return HTMLResponse(open(path, encoding="utf-8").read())


# --- ASR ---------------------------------------------------------------------

@app.websocket("/v1/asr/stream")
async def asr_stream(ws: WebSocket):
    token = ws.query_params.get("token") or ""
    if C.TOKEN and token != C.TOKEN:
        await ws.close(code=4401); return
    if not STATE["asr"]:
        await ws.accept()
        await ws.send_json({"type": "error", "code": "not_ready",
                            "message": "модель ещё грузится"})
        await ws.close(); return

    await ws.accept()
    sr = int(ws.query_params.get("sample_rate") or C.SAMPLE_RATE)
    if sr != C.SAMPLE_RATE:
        await ws.send_json({"type": "error", "code": "audio_format_unsupported",
                            "message": f"нужен {C.SAMPLE_RATE} Гц, пришло {sr}"})
        await ws.close(); return

    stream = asr.stream()
    loop = asyncio.get_running_loop()
    seq = 0
    pending = 0
    lock = asyncio.Lock()

    async def emit(t0, t1, wav, n):
        nonlocal pending
        try:
            async with lock:                  # GPU всё равно один — очередь честная
                text, ms = await loop.run_in_executor(asr.pool, asr.transcribe, wav)
            METRICS["asr_segments"] += 1
            if text:
                await ws.send_json({"type": "utterance", "seq": n, "t0": t0, "t1": t1,
                                    "text": text, "speaker": None, "ms": ms})
        except Exception:                     # noqa: BLE001
            # Оператор закрыл окно, пока сегмент считался. Это норма, а не сбой:
            # задача фоновая, и её падение не должно уронить соединение.
            METRICS["errors"] += 1
        finally:
            pending -= 1

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text"):
                cmd = json.loads(msg["text"]).get("type")
                if cmd == "close":
                    break
                continue
            raw = msg.get("bytes")
            if not raw:
                continue
            for t0, t1, wav in stream.feed(pcm16_to_float32(raw)):
                seq += 1
                if pending >= C.MAX_PENDING:
                    # Очередь глубже, чем пауза в речи: пока сегмент ждал,
                    # посетитель сказал следующее. Такой ответ уже не нужен.
                    METRICS["asr_dropped"] += 1
                    await ws.send_json({"type": "dropped", "seq": seq, "reason": "stale"})
                    continue
                pending += 1
                asyncio.create_task(emit(t0, t1, wav, seq))
    except WebSocketDisconnect:
        pass
    finally:
        stream.close()


@app.post("/v1/asr/transcribe")
async def asr_file(audio: UploadFile = File(...)):
    if not STATE["asr"]:
        return fail("not_ready", "модель ещё грузится")
    import soundfile as sf
    t0 = time.perf_counter()
    wav, sr = sf.read(io.BytesIO(await audio.read()), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != C.SAMPLE_RATE:
        try:
            import soxr
            wav = soxr.resample(wav, sr, C.SAMPLE_RATE)
        except ImportError:
            return fail("audio_format_unsupported",
                        f"нужен {C.SAMPLE_RATE} Гц или установленный soxr", 400, False)
    wav = np.concatenate([wav, np.zeros(int(0.6 * C.SAMPLE_RATE), np.float32)])

    loop = asyncio.get_running_loop()
    stream = asr.stream()
    segs, seq = [], 0
    for a, b, w in stream.feed(wav.astype(np.float32)):
        seq += 1
        text, ms = await loop.run_in_executor(asr.pool, asr.transcribe, w)
        if text:
            segs.append({"seq": seq, "t0": a, "t1": b, "text": text, "ms": ms})
    stream.close()
    return ok({"segments": segs}, t0, model=C.ASR_REPO)


# --- эмбеддер ----------------------------------------------------------------

@app.post("/v1/embed")
async def embed(body: dict):
    if not STATE["embed"]:
        return fail("not_ready", "эмбеддер ещё грузится")
    texts = body.get("texts") or []
    kind = body.get("kind")
    if kind not in ("query", "passage"):
        return fail("schema_validation_failed", "kind: query | passage", 400, False)
    if not texts or len(texts) > 256:
        return fail("schema_validation_failed", "texts: от 1 до 256 строк", 400, False)

    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    vecs, ms, cached = await loop.run_in_executor(None, emb.encode, texts, kind)
    METRICS["embed_texts"] += len(texts)
    return ok({"vectors": vecs, "dim": emb.dim, "fingerprint": fingerprint()},
              t0, model=C.EMB_MODEL, cached=cached)


# --- LLM ---------------------------------------------------------------------

@app.post("/v1/llm/extract")
async def llm_extract(body: dict):
    if not STATE["llm"]:
        return fail("not_ready", "vLLM ещё поднимается")
    turns = body.get("turns") or []
    if not turns:
        return fail("schema_validation_failed", "turns не может быть пустым", 400, False)
    t0 = time.perf_counter()
    r = await llm.extract(turns)
    METRICS["llm_extract"] += 1
    METRICS["llm_dropped_keys"] += len(r["raw_dropped"])
    return ok({"facts": r["facts"], "raw_dropped": r["raw_dropped"]}, t0,
              model=llm.model, prompt_version=PROMPT_VERSION,
              prompt_tokens=r["prompt_tokens"],
              cached_prefix_tokens=r["cached_prefix_tokens"])


@app.post("/v1/llm/ask")
async def llm_ask(body: dict):
    if not STATE["llm"]:
        return fail("not_ready", "vLLM ещё поднимается")
    doc, q = body.get("document") or {}, (body.get("question") or "").strip()
    if not doc or not q:
        return fail("schema_validation_failed", "нужны document и question", 400, False)
    METRICS["llm_ask"] += 1
    if not body.get("stream", True):
        t0 = time.perf_counter()
        return ok(await llm.ask(doc, q, body.get("history")), t0, model=llm.model)

    async def sse():
        try:
            async for ev in llm.ask_stream(doc, q, body.get("history")):
                yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:                    # noqa: BLE001
            payload = {"code": "internal", "message": str(e)}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/v1/llm/explain")
async def llm_explain(body: dict):
    if not STATE["llm"]:
        return fail("not_ready", "vLLM ещё поднимается")
    title, status = body.get("title"), body.get("status")
    if not title or status not in ("eligible", "blocked", "unknown"):
        return fail("schema_validation_failed",
                    "нужны title и status: eligible|blocked|unknown", 400, False)
    t0 = time.perf_counter()
    r = await llm.explain(title, status, body.get("reasons") or [], body.get("facts"))
    return ok(r, t0, model=llm.model)


@app.post("/v1/chat/completions")
async def passthrough(body: dict):
    """Сырой проброс в vLLM — для отладки и оффлайн-генераторов в eval/.

    Продуктовый код сюда не ходит: промпт и схема должны версионироваться
    на сервере, а не расползаться по вызывающим.
    """
    if not STATE["llm"]:
        return fail("not_ready", "vLLM ещё поднимается")
    body.setdefault("model", llm.model)
    r = await llm.http.post("/v1/chat/completions", json=body)
    return JSONResponse(status_code=r.status_code, content=r.json())
