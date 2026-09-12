"""Generate a realistic citizen-language eval set with a local LLM (Qwen3.5-4B via MLX)."""
import sys, os, json, re, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load
import mlx_lm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_queries.jsonl')
SYS = ("Ты имитируешь обычного жителя Тульской области, который пришёл на сайт МФЦ. "
       "Пиши так, как пишут в поисковую строку живые люди: разговорно, коротко, без канцелярита, "
       "без официальных названий услуг, иногда с опечатками и без знаков препинания.")

TMPL = """Ниже {n} услуг МФЦ.

{block}

Для КАЖДОЙ услуги придумай 3 поисковых запроса гражданина, которому нужна ИМЕННО эта услуга:
1) вопрос своими словами, 8-14 слов, разговорно, без канцелярита;
2) короткий запрос из 2-4 ключевых слов;
3) описание жизненной ситуации, 10-16 слов, из-за которой человек пришёл именно за этой услугой.

Жёсткие правила:
- запрос обязан однозначно указывать на эту услугу, а не на соседнюю;
- не выдумывай факты, которых нет в описании услуги;
- перефразируй, не копируй официальное название целиком;
- строчные буквы, без кавычек, без номеров и кодов, без слов «услуга», «МФЦ», «госуслуга».

Ответ строго в формате, без пояснений и без вступления:
{fmt}"""

def main(limit=None, seed=0):
    rows = load()
    # one representative per service TYPE -> queries are about the type, not the municipality
    seen, reps = set(), []
    for r in rows:
        if r['type_id'] in seen: continue
        seen.add(r['type_id']); reps.append(r)
    random.Random(seed).shuffle(reps)
    if limit: reps = reps[:limit]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)['type_id'])
    reps = [r for r in reps if r['type_id'] not in done]
    print(f'types to generate: {len(reps)} (already done {len(done)})', flush=True)

    model, tok = mlx_lm.load('Qwen/Qwen3.5-4B')
    B = 4
    f = open(OUT, 'a', encoding='utf-8')
    t0 = time.time()
    for bi in range(0, len(reps), B):
        chunk = reps[bi:bi + B]
        def desc(i, r):
            t = f"[{i+1}] Название: {r['title'][:220]}"
            if r['life']: t += f"\n    Жизненная ситуация: {', '.join(r['life'][:3])}"
            if r['result']: t += f"\n    Что человек получает: {r['result'][:160]}"
            if r['who']: t += f"\n    Кто может обратиться: {r['who'][:120]}"
            return t
        block = '\n'.join(desc(i, r) for i, r in enumerate(chunk))
        fmt = '\n'.join(f'[{i+1}]\n1) ...\n2) ...\n3) ...' for i in range(len(chunk)))
        msgs = [{'role': 'system', 'content': SYS},
                {'role': 'user', 'content': TMPL.format(n=len(chunk), block=block, fmt=fmt)}]
        try:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = mlx_lm.generate(model, tok, prompt=p, max_tokens=90 * len(chunk), verbose=False)
        out = re.sub(r'(?s)<think>.*?</think>', '', out)
        blocks = re.split(r'\n?\[(\d+)\]', out)
        parsed = {}
        for k in range(1, len(blocks) - 1, 2):
            parsed[int(blocks[k])] = blocks[k + 1]
        n = 0
        for i, r in enumerate(chunk):
            body = parsed.get(i + 1, '')
            qs = [re.sub(r'^\s*\d\)\s*', '', x).strip().strip('"«».') 
                  for x in body.strip().split('\n') if re.match(r'^\s*\d\)', x)]
            qs = [q.lower() for q in qs if 8 <= len(q) <= 160]
            if not qs: continue
            f.write(json.dumps({'type_id': r['type_id'], 'title': r['title'],
                                'ids': [x['id'] for x in rows if x['type_id'] == r['type_id']],
                                'queries': qs[:3]}, ensure_ascii=False) + '\n')
            n += 1
        f.flush()
        el = time.time() - t0
        print(f'{bi+len(chunk):4d}/{len(reps)}  +{n}  {el:.0f}s  eta {el/(bi+B)*(len(reps)-bi-B):.0f}s', flush=True)
    f.close()

if __name__ == '__main__':
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
