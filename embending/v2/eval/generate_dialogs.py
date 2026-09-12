# -*- coding: utf-8 -*-
"""Синтетические диалоги оператор↔клиент для замера гейта вызова LLM."""
import os, sys, json, re, random, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from corpus import load_cards, build_types
import mlx_lm

DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'services_for_llm.json')
OUT = os.path.join(HERE, 'dialogs.jsonl')

SYS = ("Ты пишешь реалистичные расшифровки разговоров в МФЦ Тульской области. "
       "Пиши как говорят живые люди: с паузами, переспросами, короткими репликами.")

TMPL = """Человек пришёл в МФЦ за услугой: «{title}»
{extra}

Напиши расшифровку разговора оператора и посетителя, 12-16 реплик.

Требования:
- посетитель НЕ называет услугу официально, он описывает свою ситуацию;
- факты о себе (возраст, дети, район, льготная категория, документы) посетитель
  выдаёт ПОСТЕПЕННО, по одному за реплику, и только когда оператор спросит;
- между содержательными репликами есть пустые: «ага», «да», «понятно», «секунду»;
- оператор задаёт уточняющие вопросы;
- без выдуманных номеров и сумм.

Формат строго построчно, ничего кроме него:
О: реплика оператора
К: реплика клиента
О: ...
"""

def main(n=14, seed=0):
    types = build_types(load_cards(DATA))
    types = [t for t in types if len(t['title']) > 30]
    random.Random(seed).shuffle(types)
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)['type_key'] for l in open(OUT, encoding='utf-8')}
    picks = [t for t in types if t['type_key'] not in done][:n]
    model, tok = mlx_lm.load(os.getenv('ALIAS_LLM', 'Qwen/Qwen3.5-4B'))
    f = open(OUT, 'a', encoding='utf-8')
    t0 = time.time()
    for i, t in enumerate(picks):
        extra = ''
        if t['life']:   extra += f"Жизненная ситуация: {', '.join(t['life'][:2])}\n"
        if t['who']:    extra += f"Кто может обратиться: {t['who'][:200]}\n"
        msgs = [{'role': 'system', 'content': SYS},
                {'role': 'user', 'content': TMPL.format(title=t['title'][:200], extra=extra)}]
        try:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                        enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = mlx_lm.generate(model, tok, prompt=p, max_tokens=700, verbose=False)
        out = re.sub(r'(?s)<think>.*?</think>', '', out)
        turns = []
        for line in out.split('\n'):
            m = re.match(r'\s*([ОOКK])\s*[:：]\s*(.+)', line.strip())
            if m:
                turns.append({'speaker': 'operator' if m.group(1) in 'ОO' else 'client',
                              'text': m.group(2).strip()})
        if len(turns) < 6:
            continue
        f.write(json.dumps({'type_key': t['type_key'], 'title': t['title'],
                            'ids': [c['id'] for c in t['instances']],
                            'turns': turns}, ensure_ascii=False) + '\n')
        f.flush()
        print(f'{i+1}/{len(picks)}  {len(turns)} реплик  {time.time()-t0:.0f}s', flush=True)
    f.close()

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
