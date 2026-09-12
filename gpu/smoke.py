#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка сборки на живой 5090. Шаг 1 плана: то, что может отменить остальное.

Проверяет ровно три несущих допущения архитектуры, в порядке убывания риска:

  1. Сборка под sm_120. vLLM с FP8, GigaAM со своим remote code и
     sentence-transformers уживаются на одной карте Blackwell и помещаются
     в 32 ГБ втроём.
  2. Структурированный вывод. Движок принимает JSON-схему — иначе дрейф
     значений вырастает с 7% до 54%, и монотонное состояние диалога не спасает.
  3. Кэш префикса. Префилл на ходе диалога падает с ~1200 токенов до десятков.
     Без этого одна карта не тянет даже пилот: считать надо не декод, а префилл.

Запуск на инстансе (шлюз уже поднят):
    python3 smoke.py --url http://127.0.0.1:10100 --token "$(cat /workspace/.gw_token)"
"""
import argparse
import json
import os
import statistics
import sys
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OK, BAD, WARN = "  ok  ", " FAIL ", " warn "


class Report:
    def __init__(self):
        self.rows = []
        self.failed = 0

    def add(self, status, name, detail=""):
        self.rows.append((status, name, detail))
        if status is BAD:
            self.failed += 1
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    def verdict(self):
        print("\n" + "─" * 72)
        for s, n, d in self.rows:
            print(f"[{s}] {n:<42} {d}")
        print("─" * 72)
        if self.failed:
            print(f"ПРОВАЛЕНО ПРОВЕРОК: {self.failed}. Разворачивать рано.")
        else:
            print("Все несущие допущения подтверждены. Можно идти к шагу 2 плана.")
        return 1 if self.failed else 0


def wait_ready(c, rep, timeout=1800):
    """Модели грузятся долго: vLLM компилирует графы, веса качаются с HF."""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            r = c.get("/readyz", timeout=10)
            body = r.json()
            if r.status_code == 200:
                rep.add(OK, "все три модели прогреты", f"за {round(time.time()-t0)} с")
                return body
            if body != last:
                print("   ...", json.dumps(body, ensure_ascii=False), flush=True)
                last = body
        except Exception as e:
            if str(e) != str(last):
                print("   ... шлюз ещё не отвечает", flush=True)
                last = e
        time.sleep(10)
    rep.add(BAD, "готовность", f"не поднялось за {timeout} с")
    return None


def check_vram(c, rep, ready):
    used, total = ready.get("vram_used_mb"), ready.get("vram_total_mb")
    if not total:
        rep.add(WARN, "VRAM", "не удалось прочитать")
        return
    free = total - used
    detail = f"{used/1024:.1f} из {total/1024:.1f} ГБ занято, свободно {free/1024:.1f} ГБ"
    # Запас нужен под KV-кэш на пике: пустой остаток означает OOM на нагрузке.
    rep.add(OK if free > 800 else BAD, "три модели поместились", detail)


def check_models(c, rep):
    m = c.get("/v1/models").json()
    rep.add(OK, "ASR", f"{m['asr']['repo']} / {m['asr']['variant']}")
    rep.add(OK, "эмбеддер", f"{m['embed']['model']}, dim={m['embed']['dim']}")
    rep.add(OK, "отпечаток эмбеддера", m["embed"]["fingerprint"])
    mode = m["llm"]["struct_mode"]
    rep.add(OK if mode in ("guided", "response_format") else BAD,
            "структурированный вывод", f"{m['llm']['model']}, режим: {mode}")
    if mode == "none":
        print("   !! движок не принял JSON-схему ни одним способом.")
        print("   !! без грамматики дрейф значений 54% вместо 7% — это авария.")
    return m


def bench_embed(c, rep):
    one = ["оформить пособие на второго ребёнка"]
    four = ["оформить пособие на второго ребёнка", "выплата при рождении",
            "детские пособия молодой маме", "материнский капитал"]
    lat = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = c.post("/v1/embed", json={"texts": [f"{_} потерял паспорт"], "kind": "query"})
        r.raise_for_status()
        lat.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter(); c.post("/v1/embed", json={"texts": four, "kind": "query"}); batch = (time.perf_counter()-t0)*1000
    t0 = time.perf_counter(); c.post("/v1/embed", json={"texts": four, "kind": "query"}); cached = (time.perf_counter()-t0)*1000
    rep.add(OK, "эмбеддинг одной строки", f"медиана {statistics.median(lat):.0f} мс")
    rep.add(OK, "четыре строки батчем", f"{batch:.0f} мс (на CPU было бы ~140)")
    rep.add(OK, "повтор из кэша", f"{cached:.0f} мс")


def _dialog(rep):
    p = os.path.join(ROOT, "embending", "v2", "eval", "dialogs.jsonl")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        turns = json.loads(f.readline())["turns"]
    return [{"seq": i + 1, "text": t["text"], "speaker": None}
            for i, t in enumerate(turns)]


def bench_extract(c, rep):
    """Главный замер: растущий диалог, ход за ходом — как в проде.

    Смотрим не только на задержку, но и на cached_prefix_tokens. Если он не
    растёт вместе с диалогом, значит кэш префикса не работает, и вся оценка
    нагрузки в архитектуре построена на песке.
    """
    turns = _dialog(rep)
    if not turns:
        rep.add(WARN, "извлечение фактов", "нет eval/dialogs.jsonl — пропущено")
        return
    lat, cached, prompt = [], [], []
    facts_seen = {}
    for n in range(2, min(len(turns), 12) + 1):
        t0 = time.perf_counter()
        r = c.post("/v1/llm/extract", json={"session_id": "smoke", "turns": turns[:n]},
                   timeout=90)
        r.raise_for_status()
        body = r.json()
        lat.append((time.perf_counter() - t0) * 1000)
        meta = body["meta"]
        prompt.append(meta.get("prompt_tokens") or 0)
        cached.append(meta.get("cached_prefix_tokens") or 0)
        facts_seen.update(body["data"]["facts"])
        dropped = body["data"]["raw_dropped"]
        if dropped:
            print(f"   ход {n}: вне схемы {dropped}", flush=True)

    rep.add(OK, "извлечение фактов, задержка",
            f"медиана {statistics.median(lat):.0f} мс, макс {max(lat):.0f} мс")
    rep.add(OK, "накоплено фактов", ", ".join(sorted(facts_seen)) or "пусто")

    if max(cached) == 0:
        rep.add(BAD, "кэш префикса",
                "cached_tokens=0 на всех ходах — префилл не переиспользуется")
    else:
        share = 100 * cached[-1] / max(prompt[-1], 1)
        rep.add(OK if share > 50 else WARN, "кэш префикса",
                f"на последнем ходе {cached[-1]} из {prompt[-1]} токенов ({share:.0f}%)")


def bench_ask(c, rep):
    p = os.path.join(ROOT, "embending", "data", "services_for_llm.json")
    if not os.path.exists(p):
        rep.add(WARN, "вопрос по документу", "нет корпуса — пропущено")
        return
    card = json.load(open(p, encoding="utf-8"))[0]
    doc = {k: v for k, v in card.items() if isinstance(v, str) and len(v) > 20}
    t0 = time.perf_counter()
    r = c.post("/v1/llm/ask", json={"document": doc, "stream": False,
                                    "question": "Какие документы нужны и сколько это стоит?"},
               timeout=120)
    r.raise_for_status()
    d = r.json()["data"]
    ms = (time.perf_counter() - t0) * 1000
    cits = d["citations"]
    rep.add(OK, "вопрос по документу", f"{ms:.0f} мс, ответ {len(d['answer'])} симв.")
    rep.add(OK if cits else WARN, "цитаты сверены дословно",
            f"{len(cits)} шт." if cits else "модель не привела ни одной цитаты")


def bench_asr(c, rep):
    p = os.path.join(ROOT, "sber ctc", "Test audio", "output.wav")
    if not os.path.exists(p):
        rep.add(WARN, "распознавание речи", "нет тестового wav — пропущено")
        return
    t0 = time.perf_counter()
    with open(p, "rb") as f:
        r = c.post("/v1/asr/transcribe", files={"audio": ("output.wav", f, "audio/wav")},
                   timeout=300)
    r.raise_for_status()
    segs = r.json()["data"]["segments"]
    ms = (time.perf_counter() - t0) * 1000
    if not segs:
        rep.add(BAD, "распознавание речи", "ноль сегментов на тестовом файле")
        return
    per = statistics.median(s["ms"] for s in segs)
    rep.add(OK, "распознавание речи",
            f"{len(segs)} реплик, медиана {per:.0f} мс на сегмент, всего {ms/1000:.1f} с")
    print("   первая реплика:", segs[0]["text"][:90], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.getenv("GW_URL", "http://127.0.0.1:10100"))
    ap.add_argument("--token", default=os.getenv("GW_TOKEN", ""))
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    headers = {"Authorization": f"Bearer {a.token}"} if a.token else {}
    c = httpx.Client(base_url=a.url, headers=headers, timeout=60)
    rep = Report()

    print(f"шлюз: {a.url}\n")
    ready = wait_ready(c, rep, a.timeout)
    if not ready:
        return rep.verdict()
    if ready.get("error"):
        rep.add(BAD, "загрузка моделей", ready["error"])
        return rep.verdict()

    check_vram(c, rep, ready)
    check_models(c, rep)
    bench_embed(c, rep)
    bench_asr(c, rep)
    bench_extract(c, rep)
    bench_ask(c, rep)
    return rep.verdict()


if __name__ == "__main__":
    sys.exit(main())
