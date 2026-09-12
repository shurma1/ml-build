# -*- coding: utf-8 -*-
"""Условия права на услугу -> типизированные предикаты. Оффлайн, один раз на 318 типов.

Зачем не регулярками на лету: регулярка не различает, к КОМУ относится условие.
Замеренный провал: «ребёнка в возрасте до 17 лет» срабатывало на возраст заявителя.
Поэтому у предиката обязательно есть поле subject, и обязательна цитата — оператор
показывает её клиенту, а не пересказывает.
"""
import os, sys, json, re, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from corpus import load_cards, build_types
from extraction_schema import CATEGORIES
import mlx_lm

ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, 'data', 'services_for_llm.json')
OUT = os.path.join(HERE, 'predicates.jsonl')

SUBJECTS = ['applicant', 'child', 'spouse', 'property', 'other']
FIELDS = ['age', 'children', 'category', 'recipient_type', 'residence', 'other']
OPS = ['>=', '<=', '==', 'in', 'has']

PROMPT = """Ниже условия получения услуги МФЦ. Выдели ПРОВЕРЯЕМЫЕ условия к заявителю.

Услуга: {title}
Кто может обратиться: {who}
Основания для отказа: {reject}

Верни JSON-массив. Каждый элемент:
{{"subject": один из {subjects},
  "field": один из {fields},
  "op": один из {ops},
  "value": число или строка,
  "quote": ДОСЛОВНАЯ цитата из текста выше, обосновывающая условие}}

Правила:
- subject — к кому относится условие. «ребёнка до 17 лет» -> subject "child", НЕ "applicant".
- category берётся только из списка: {cats}
- recipient_type: person / ip / organization
- НЕ выдумывай условий, которых нет в тексте. Нет проверяемых условий -> верни []
- quote обязана дословно встречаться в тексте выше.

Только JSON-массив, без пояснений."""


def main(limit=None):
    types = build_types(load_cards(DATA))
    raw = {r['id']: r for r in json.load(open(DATA, encoding='utf-8'))}
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding='utf-8'):
            try: done.add(json.loads(l)['type_key'])
            except Exception: pass
    todo = [t for t in types if t['type_key'] not in done]
    if limit:
        todo = todo[:limit]
    print(f'типов к обработке: {len(todo)} (готово {len(done)})', flush=True)
    if not todo:
        return
    model, tok = mlx_lm.load(os.getenv('ALIAS_LLM', 'Qwen/Qwen3.5-4B'))
    f = open(OUT, 'a', encoding='utf-8')
    t0 = time.time()
    for i, t in enumerate(todo):
        r = raw[t['instances'][0]['id']]
        who = (r.get('serviceRecipients') or '')[:900]
        rej = (r.get('rejectReasonsText') or '')[:700]
        src = t['title'] + '\n' + who + '\n' + rej
        msgs = [{'role': 'user', 'content': PROMPT.format(
            title=t['title'][:250], who=who or '—', reject=rej or '—',
            subjects=SUBJECTS, fields=FIELDS, ops=OPS, cats=CATEGORIES)}]
        try:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                        enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = mlx_lm.generate(model, tok, prompt=p, max_tokens=420, verbose=False)
        out = re.sub(r'(?s)<think>.*?</think>', '', out)
        m = re.search(r'\[.*\]', out, re.S)
        preds = []
        if m:
            try:
                for p_ in json.loads(m.group(0)):
                    if not isinstance(p_, dict):
                        continue
                    if p_.get('subject') not in SUBJECTS or p_.get('field') not in FIELDS:
                        continue
                    if p_.get('op') not in OPS or p_.get('value') in (None, '', []):
                        continue
                    q = (p_.get('quote') or '').strip()
                    # цитата обязана реально существовать в исходнике
                    norm = lambda s: re.sub(r'\s+', ' ', s.lower())
                    if not q or norm(q)[:60] not in norm(src):
                        continue
                    preds.append({'subject': p_['subject'], 'field': p_['field'],
                                  'op': p_['op'], 'value': p_['value'], 'quote': q[:220]})
            except Exception:
                pass
        f.write(json.dumps({'type_key': t['type_key'], 'title': t['title'],
                            'ids': [c['id'] for c in t['instances']],
                            'predicates': preds}, ensure_ascii=False) + '\n')
        f.flush()
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            el = time.time() - t0
            print(f'{i+1}/{len(todo)}  {el:.0f}s  eta {el/(i+1)*(len(todo)-i-1):.0f}s', flush=True)
    f.close()


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
