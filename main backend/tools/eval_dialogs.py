# -*- coding: utf-8 -*-
"""Приёмка №3: диалог из eval/dialogs.jsonl проходит целиком.

Проверяется не качество поиска, а связность конвейера и одно свойство, которое
ломается молча: **состояние накапливается монотонно**. Заполненный факт не
должен исчезать оттого, что на следующей реплике модель о нём промолчала, —
именно этот дефект давал 61% дрейфа полей до введения DialogState.

Гоняется через живой HTTP-API, а не через модули напрямую: смысл проверки
в том, что конвейер собран, а не в том, что части существуют.

    CORE_URL=http://127.0.0.1:8080 python3 tools/eval_dialogs.py [сколько диалогов]
"""
import json
import os
import sys

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DIALOGS = os.path.join(ROOT, "embending", "v2", "eval", "dialogs.jsonl")
CORE = os.getenv("CORE_URL", "http://127.0.0.1:8080")

FACT_KEYS = ("age", "children", "categories")
FLAT_KEYS = ("life_situation", "recipient", "municipality")


def flatten(state):
    d = {k: state.get(k) for k in FLAT_KEYS}
    d.update({k: (state.get("facts") or {}).get(k) for k in FACT_KEYS})
    return {k: v for k, v in d.items() if v not in (None, [], "")}


def regressions(prev, cur, pinned):
    """Что пропало или изменилось не в сторону накопления."""
    bad = []
    for k, v in prev.items():
        if k in pinned:
            continue
        if k not in cur:
            bad.append(f"{k}: {v!r} исчез")
        elif isinstance(v, list) and isinstance(cur[k], list):
            lost = set(v) - set(cur[k])
            if lost:
                bad.append(f"{k}: потеряно {sorted(lost)}")
    return bad


def run_one(c, dlg, branch_id):
    s = c.post("/v1/sessions", json={"branch_id": branch_id,
                                     "operator_id": "eval"}).json()
    sid = s["session_id"]
    prev, steps, bad, searched = {}, 0, [], 0
    for t in dlg["turns"]:
        r = c.post(f"/v1/sessions/{sid}/turns",
                   json={"text": t["text"][:2000], "speaker": t.get("speaker")})
        if r.status_code != 200:
            bad.append(f"ход {steps}: HTTP {r.status_code} {r.text[:120]}")
            break
        d = r.json()
        steps += 1
        if d.get("skipped"):
            continue
        cur = flatten(d["state"])
        bad += [f"ход {steps}: {m}" for m in regressions(prev, cur, d["state"].get("pinned") or [])]
        prev = cur
        if d.get("search") and d["search"].get("results"):
            searched += 1
    snap = c.get(f"/v1/sessions/{sid}").json()
    if len(snap.get("turns") or []) != steps:
        bad.append(f"снимок вернул {len(snap.get('turns') or [])} реплик вместо {steps}")
    if flatten(snap["state"]) != prev:
        bad.append("состояние в снимке не совпало с последним ходом")
    c.post(f"/v1/sessions/{sid}/close", json={"outcome": "served"})
    return {"session_id": sid, "turns": steps, "searches": searched,
            "state": prev, "problems": bad}


def main(limit=None):
    dialogs = [json.loads(l) for l in open(DIALOGS, encoding="utf-8")]
    if limit:
        dialogs = dialogs[:limit]
    c = httpx.Client(base_url=CORE, timeout=120)
    branches = c.get("/v1/branches").json()
    branch_id = next((b["branch_id"] for b in branches if b["municipality"]), branches[0]["branch_id"])

    total_bad, rows = 0, []
    for i, dlg in enumerate(dialogs, 1):
        r = run_one(c, dlg, branch_id)
        total_bad += len(r["problems"])
        rows.append(r)
        mark = "OK  " if not r["problems"] else "FAIL"
        print(f"[{mark}] {i:2d}/{len(dialogs)} реплик {r['turns']:2d} "
              f"поисков {r['searches']:2d}  {dlg['title'][:46]}")
        print(f"        состояние: {json.dumps(r['state'], ensure_ascii=False)}")
        for p in r["problems"]:
            print(f"        ! {p}")

    ok = sum(1 for r in rows if not r["problems"])
    print(f"\n{ok}/{len(rows)} диалогов прошли целиком, "
          f"нарушений монотонности: {total_bad}")
    print(f"средне реплик на диалог: {sum(r['turns'] for r in rows) / max(1, len(rows)):.1f}, "
          f"поисков: {sum(r['searches'] for r in rows) / max(1, len(rows)):.1f}")
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else None))
