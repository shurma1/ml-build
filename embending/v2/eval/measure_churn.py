# -*- coding: utf-8 -*-
"""Замер дрейфа состояния при извлечении признаков.

Обосновывает переход на constrained decoding. Эталон — полное извлечение на КАЖДОЙ
реплике накопленного диалога; смотрим, как часто уже найденный факт исчезает или
подменяется другим значением, и насколько это лечится валидацией по enum.

    python3 v2/eval/measure_churn.py        # собрать эталон (нужна локальная LLM)
    python3 v2/eval/measure_churn.py score  # свести результат
"""
import os, sys, json, re, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mlx_lm

DLG = os.path.join(HERE, 'dialogs.jsonl')
OUT = os.path.join(HERE, 'gate_truth.jsonl')

SCHEMA = ('{"life_situation": str|null, "recipient": "person|ip|organization"|null, '
          '"municipality": str|null, "age": int|null, "children": int|null, '
          '"categories": [str], "documents": [str], "intent": str|null}')
PROMPT = """Извлеки факты о посетителе МФЦ из разговора. Только то, что прозвучало явно.
Верни СТРОГО JSON по схеме, без пояснений:
{schema}

Разговор:
{dialog}"""


def extract(model, tok, dialog_text):
    msgs = [{'role': 'user', 'content': PROMPT.format(schema=SCHEMA, dialog=dialog_text)}]
    try:
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                    enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    out = mlx_lm.generate(model, tok, prompt=p, max_tokens=220, verbose=False)
    out = re.sub(r'(?s)<think>.*?</think>', '', out)
    m = re.search(r'\{.*\}', out, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {}
    return {k: v for k, v in d.items() if v not in (None, '', [], {}, 'null')}


def state_key(d):
    return json.dumps({k: sorted(v) if isinstance(v, list) else v
                       for k, v in sorted(d.items()) if k != 'intent'}, ensure_ascii=False)


def main(limit=None):
    dialogs = [json.loads(l) for l in open(DLG, encoding='utf-8')]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)['title'] for l in open(OUT, encoding='utf-8')}
    dialogs = [d for d in dialogs if d['title'] not in done]
    if limit:
        dialogs = dialogs[:limit]
    print(f'диалогов к обработке: {len(dialogs)} (готово {len(done)})', flush=True)
    if not dialogs:
        return
    model, tok = mlx_lm.load(os.getenv('ALIAS_LLM', 'Qwen/Qwen3.5-4B'))
    f = open(OUT, 'a', encoding='utf-8')
    t0 = time.time()
    for di, d in enumerate(dialogs):
        acc, prev, rows = [], '{}', []
        for ti, t in enumerate(d['turns']):
            acc.append(('Оператор' if t['speaker'] == 'operator' else 'Клиент') + ': ' + t['text'])
            st = extract(model, tok, '\n'.join(acc))
            key = state_key(st)
            rows.append({'i': ti, 'speaker': t['speaker'], 'text': t['text'],
                         'informative': key != prev, 'state': st})
            prev = key
        f.write(json.dumps({'title': d['title'], 'turns': rows}, ensure_ascii=False) + '\n')
        f.flush()
        inf = sum(r['informative'] for r in rows)
        print(f'{di+1}/{len(dialogs)}  реплик {len(rows)}, информативных {inf}  '
              f'{time.time()-t0:.0f}s', flush=True)
    f.close()


def score():
    """Дрейф состояния на трёх уровнях строгости."""
    from extraction_schema import validate

    d = [json.loads(l) for l in open(OUT, encoding='utf-8')]

    def churn(fn, monotonic=False):
        n = van = flip = 0
        for x in d:
            prev = {}
            for r in x['turns']:
                st = fn({k: v for k, v in (r.get('state') or {}).items() if k != 'intent'})
                if monotonic:
                    merged = dict(prev); merged.update(st); st = merged
                n += 1
                for k, v in prev.items():
                    if k not in st:
                        van += 1
                    elif st[k] != v and not (isinstance(v, list) and isinstance(st[k], list)
                                             and set(v) <= set(st[k])):
                        flip += 1
                prev = st
        return n, van, flip

    print(f'диалогов: {len(d)}, реплик: {sum(len(x["turns"]) for x in d)}\n')
    print(f'{"уровень строгости":34s} {"факт исчез":>12s} {"значение заменено":>20s}')
    for lab, fn, mono in [('свободный JSON, как пишет LLM', lambda s: s, False),
                          ('+ validate() по enum', validate, False),
                          ('+ DialogState (монотонность)', validate, True)]:
        n, v, f = churn(fn, mono)
        print(f'{lab:34s} {v:6d} ({v/n*100:4.1f}%) {f:10d} ({f/n*100:5.1f}%)')
    print('\nЗначения вне схемы — корень дрейфа: каждый ход модель выдумывает')
    print('новую формулировку, и поле «меняется», хотя факт тот же.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'score':
        score()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
