# -*- coding: utf-8 -*-
"""Сгенерировать «простые названия» услуги — как её называет человек, а не регламент.

Это НЕ поисковые запросы: это дополнительные имена сущности, которые кладутся
в отдельное поле и индексируются отдельно от канцелярского названия.
"""
import os, sys, json, re, time, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from corpus import load_cards, build_types
import mlx_lm

DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'services_for_llm.json')
OUT = os.path.join(HERE, 'aliases_generated.jsonl')
MODEL = os.getenv('ALIAS_LLM', 'Qwen/Qwen3.5-4B')

SYS = ("Ты редактор портала госуслуг. Твоя работа — переводить канцелярские названия услуг "
       "на человеческий язык, не искажая смысл.")

TMPL = """Ниже {n} услуг МФЦ с официальными названиями.

{block}

Для КАЖДОЙ услуги напиши 4 коротких «человеческих» названия — так, как эту услугу
назвал бы обычный человек, который пришёл её получать.

Требования:
- 2-6 слов, именная группа или короткое действие («сдать дом в эксплуатацию», «справка о пенсии»);
- разные по словарю: синоним, бытовое слово, аббревиатура/сокращение, формулировка через результат;
- НИЧЕГО не выдумывай: название должно означать ровно эту услугу, а не соседнюю;
- без слов «услуга», «МФЦ», «госуслуга», «предоставление», «осуществление»;
- строчными буквами, без кавычек и нумерации внутри строки.

Ответ строго в формате, без пояснений:
{fmt}"""


def main(limit=None, seed=0):
    types = build_types(load_cards(DATA))
    random.Random(seed).shuffle(types)
    if limit:
        types = types[:limit]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)['type_key'] for l in open(OUT, encoding='utf-8')}
    types = [t for t in types if t['type_key'] not in done]
    print(f'к генерации: {len(types)} (готово {len(done)})', flush=True)

    model, tok = mlx_lm.load(MODEL)
    B = 4
    f = open(OUT, 'a', encoding='utf-8')
    t0 = time.time()
    for bi in range(0, len(types), B):
        chunk = types[bi:bi + B]
        def desc(i, t):
            s = f"[{i+1}] {t['title'][:220]}"
            if t['life']:   s += f"\n    жизненная ситуация: {', '.join(t['life'][:3])}"
            if t['result']: s += f"\n    результат: {t['result'][:150]}"
            return s
        block = '\n'.join(desc(i, t) for i, t in enumerate(chunk))
        fmt = '\n'.join(f'[{i+1}]\n- ...\n- ...\n- ...\n- ...' for i in range(len(chunk)))
        msgs = [{'role': 'system', 'content': SYS},
                {'role': 'user', 'content': TMPL.format(n=len(chunk), block=block, fmt=fmt)}]
        try:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                        enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = mlx_lm.generate(model, tok, prompt=p, max_tokens=80 * len(chunk), verbose=False)
        out = re.sub(r'(?s)<think>.*?</think>', '', out)
        parts = re.split(r'\n?\[(\d+)\]', out)
        parsed = {int(parts[k]): parts[k + 1] for k in range(1, len(parts) - 1, 2)}
        n = 0
        for i, t in enumerate(chunk):
            body = parsed.get(i + 1, '')
            al = [re.sub(r'^\s*[-•*\d.)]+\s*', '', x).strip().strip('"«».').lower()
                  for x in body.strip().split('\n') if re.match(r'^\s*[-•*]', x)]
            al = [a for a in al if 6 <= len(a) <= 90 and not a.startswith(('услуга', 'мфц'))]
            al = list(dict.fromkeys(al))[:4]
            if not al:
                continue
            f.write(json.dumps({'type_key': t['type_key'], 'title': t['title'],
                                'ids': [c['id'] for c in t['instances']],
                                'aliases': al}, ensure_ascii=False) + '\n')
            n += 1
        f.flush()
        el = time.time() - t0
        print(f'{bi+len(chunk):4d}/{len(types)}  +{n}  {el:.0f}s  eta {el/(bi+B)*(len(types)-bi-B):.0f}s',
              flush=True)
    f.close()


if __name__ == '__main__':
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
