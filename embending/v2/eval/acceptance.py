# -*- coding: utf-8 -*-
"""Приёмочный прогон всего конвейера: схема -> поиск -> право -> наводящий вопрос.

Не метрика качества (для неё run_eval.py), а проверка, что части состыкованы
и что известные дефекты закрыты.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from extraction_schema import validate, DialogState, JSON_SCHEMA, LIFE_SITUATIONS
from eligibility import load_predicates, check
from corpus import load_cards, build_types
from ratelimit import is_empty_segment

OK, FAIL = '  OK  ', ' FAIL '
results = []


def t(name, cond, note=''):
    results.append((cond, name, note))
    print(f'[{OK if cond else FAIL}] {name}' + (f'   {note}' if note else ''), flush=True)


def main():
    types = build_types(load_cards(os.path.join(
        os.path.dirname(os.path.dirname(HERE)), 'data', 'services_for_llm.json')))
    by_key = {x['type_key']: x for x in types}

    # --- 1. схема извлечения ---
    dirty = {'life_situation': 'worker on production', 'recipient': 'физлицо',
             'categories': ['выплата', 'СВО'], 'age': 'тридцать', 'children': 2,
             'municipality': 'Щекино', 'intents': ['пособие'], 'мусор': 1}
    v = validate(dirty)
    t('мусор вне enum отброшен', 'life_situation' not in v and v.get('categories') == ['СВО'])
    t('разговорное МО приведено газеттиром', v.get('municipality') == 'Щекинский район')
    t('нечисловой возраст отброшен', 'age' not in v and v.get('children') == 2)
    t('все enum-значения схемы валидны сами по себе',
      all(validate({'life_situation': x}).get('life_situation') == x for x in LIFE_SITUATIONS))

    st = DialogState()
    st.update({'intents': ['пособие'], 'recipient': 'person'})
    st.update({'intents': ['пособие'], 'age': 22})
    st.update({'intents': ['выплата']})                      # факты не пришли
    t('состояние монотонно: факт не исчезает от молчания',
      st.state.get('recipient') == 'person' and st.state.get('age') == 22)
    t('intents не входят в стабильное состояние', 'intents' not in st.state)
    t('JSON_SCHEMA сериализуема для guided_json', bool(json.dumps(JSON_SCHEMA)))

    # --- 2. предикаты права ---
    tab = load_predicates()
    covered = sum(1 for v_ in tab.values() if v_)
    t('таблица предикатов загружена', len(tab) > 0, f'{covered} типов с условиями из {len(tab)}')
    allp = [p for line in open(os.path.join(HERE, 'predicates.jsonl'), encoding='utf-8')
            for p in json.loads(line).get('predicates', [])]
    subj = {}
    for p in allp:
        subj[p['subject']] = subj.get(p['subject'], 0) + 1
    t('условия к ребёнку отделены от условий к заявителю',
      subj.get('child', 0) > 0, f'subject: {subj}')
    t('в проверку идут только applicant-предикаты',
      all(p['subject'] == 'applicant' for v_ in tab.values() for p in v_))
    t('у каждого предиката есть проверенная цитата',
      all(p.get('quote') for p in allp), f'{len(allp)} предикатов')

    # альтернативы одного поля не должны блокировать друг друга
    alt = next((by_key[k] for k, v_ in tab.items()
                if k in by_key and sum(1 for p in v_ if p['field'] == 'category') > 1), None)
    if alt:
        vals = [p['value'] for p in tab[alt['type_key']] if p['field'] == 'category']
        flat = [x for v_ in vals for x in (v_ if isinstance(v_, list) else [v_])]
        sts = [check(alt, {'recipient': 'person', 'categories': [c]})[0] for c in flat]
        t('предикаты одного поля — альтернативы, а не совокупность',
          all(s == 'eligible' for s in sts), f'{alt["title"][:40]}: {flat} -> {sts}')

    # --- 3. отсев пустых сегментов ---
    t('пустые сегменты VAD отсеиваются',
      is_empty_segment('Ага') and is_empty_segment('Да, вот он'))
    t('короткие аббревиатуры не считаются мусором',
      not is_empty_segment('А вы не ИП?') and not is_empty_segment('СВО'))

    # --- 4. поиск end-to-end ---
    from dialog import DialogSearch
    from features import QueryFeatures
    d = DialogSearch()
    cases = [('сдать дом в эксплуатацию', 'ввод объект'),
             ('деньги за второго ребенка', 'рождении втор'),
             ('потерял паспорт', 'паспорт')]
    for q, want in cases:
        r = d.search(QueryFeatures.from_text(q), k=5)
        hit = any(want.lower() in x['title'].lower() for x in r['results'])
        t(f'поиск: «{q}»', hit, r['results'][0]['title'][:52] if r['results'] else '—')

    f = QueryFeatures.from_llm({'intents': ['выплата при рождении второго ребёнка'],
                                'recipient': 'person', 'municipality': 'Щекинский район',
                                'facts': {'age': 30, 'children': 2}})
    r = d.search(f, k=6)
    t('карточка резолвится по МО из признаков',
      any(x.get('service_id') for x in r['results']), f'МО={r["municipality"]}')
    t('статусы права проставлены',
      all(x['status'] in ('eligible', 'blocked', 'unknown') for x in r['results']),
      str({x['status'] for x in r['results']}))
    t('у заблокированных есть причина',
      all(x['reasons'] for x in r['results'] if x['status'] == 'blocked'))
    t('наводящие вопросы предлагаются', bool(r['questions']),
      r['questions'][0]['question'][:56] if r['questions'] else '—')

    ts = [d.search(f, k=8)['ms'] for _ in range(5)]
    warm = sorted(ts)[len(ts) // 2]
    t('тёплый поиск укладывается в 150 мс', warm < 150, f'p50 = {warm} мс')

    bad = sum(1 for ok, *_ in results if not ok)
    print(f'\n{len(results) - bad}/{len(results)} проверок пройдено')
    return bad


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
