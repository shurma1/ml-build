# -*- coding: utf-8 -*-
"""Перенос производных таблиц на новый ключ типа — по id карточек, а не по строке ключа.

Зачем понадобилось: `type_key` был одновременно ПРАВИЛОМ группировки и ВНЕШНИМ
КЛЮЧОМ для `eval/predicates.jsonl` и `eval/aliases_generated.jsonl`. Любая правка
правила молча осиротляла обе таблицы: при добавлении аббревиатур в ключ 57 строк
из 318 переставали находиться, то есть 60 типов разом теряли и проверку права,
и синонимы (а синонимы — это 5.2 п.п. R@1).

Здесь связь восстанавливается по тому, что действительно неизменно, — по id
карточек выгрузки. Старый ключ -> карточки -> новый ключ. Группа, которая
распалась надвое, отдаёт свою строку обеим половинам.

    python3 v2/tools/remap_type_keys.py --check    # показать, что изменится
    python3 v2/tools/remap_type_keys.py --apply    # переписать файлы (с .bak)
"""
import argparse
import collections
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
sys.path.insert(0, V2)
from corpus import load_cards, build_types, split_municipality       # noqa: E402

DATA = os.path.join(os.path.dirname(V2), 'data', 'services_for_llm.json')
TARGETS = [os.path.join(V2, 'eval', 'predicates.jsonl'),
           os.path.join(V2, 'eval', 'aliases_generated.jsonl')]
MAP_PATH = os.path.join(os.path.dirname(V2), 'data', 'type_map.json')


def legacy_key(title):
    """Ключ до правки: слова от четырёх букв, аббревиатуры терялись."""
    return ' '.join(sorted(set(re.findall(r'[а-яёa-z]{4,}', title.lower()))))


def build_bridge(cards, types):
    """-> (старый ключ -> [новые ключи]), в порядке убывания числа карточек."""
    new_of_card = {c['id']: t['type_key'] for t in types for c in t['instances']}
    by_old = collections.defaultdict(list)
    for c in cards:
        by_old[legacy_key(c['title'])].append(c['id'])
    bridge = {}
    for old, ids in by_old.items():
        counts = collections.Counter(new_of_card[i] for i in ids if i in new_of_card)
        bridge[old] = [k for k, _ in counts.most_common()]
    return bridge


def variant_of(title):
    """Суффикс-аббревиатура в хвосте заголовка: «… (СВО)» -> 'СВО'."""
    m = re.search(r'\(([А-ЯЁA-Z]{2,4})\)\s*$', title)
    return m.group(1) if m else None


def _norm(s):
    return re.sub(r'\s+', ' ', (s or '').replace('ё', 'е')).strip().lower()


def narrow_predicates(preds, source_text):
    """Оставить у половины распавшейся пары только её СОБСТВЕННЫЕ условия.

    Склеенный тип нёс условия обеих услуг вперемешку: у пары ВБД/СВО это условие
    «ветераны боевых действий», которое к карточке СВО отношения не имеет.
    Наследовать его целиком нельзя — это чужое основание с чужой цитатой.

    Правило отбора то же, которым таблица и порождалась: предикат остаётся, если
    его цитата ДОСЛОВНО есть в тексте этой карточки. Ничего не досочиняем — если
    после отбора условие по категории исчезло, тип останется без него, и
    `check()` вернёт unknown вместо выдуманного вердикта. Что нужно дописать
    руками, скрипт печатает списком.
    """
    src = _norm(source_text)
    if not src:
        return preds
    out = []
    for p in preds:
        q = _norm(p.get('quote'))
        if not q or q in src:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if not (a.apply or a.check):
        ap.error('нужен --check или --apply')

    cards = load_cards(DATA)
    types = build_types(cards)
    title_of = {t['type_key']: t['title'] for t in types}
    raw = {r['id']: r for r in json.load(open(DATA, encoding='utf-8'))}
    source_of = {}
    for t in types:
        r = raw[t['instances'][0]['id']]
        source_of[t['type_key']] = ((r.get('serviceRecipients') or '') + '\n' +
                                    (r.get('rejectReasonsText') or ''))
    bridge = build_bridge(cards, types)
    current = {t['type_key'] for t in types}
    print(f'карточек {len(cards)} -> типов {len(types)}')
    split = {o: n for o, n in bridge.items() if len(n) > 1}
    print(f'групп, распавшихся при смене правила: {len(split)}')
    for o, n in split.items():
        for k in n:
            print(f'    -> {title_of.get(k, k)[:78]}')

    for path in TARGETS:
        rows = [json.loads(l) for l in open(path, encoding='utf-8')]
        out, orphan, dup, pruned = [], [], 0, []
        for r in rows:
            new = bridge.get(r['type_key'])
            if not new:
                orphan.append(r['type_key'])
                continue
            for k in new:
                d = dict(r)
                d['type_key'] = k
                if len(new) > 1 and 'predicates' in d:
                    before = len(d['predicates'])
                    d['predicates'] = narrow_predicates(d['predicates'], source_of.get(k, ''))
                    if len(d['predicates']) != before:
                        pruned.append((title_of.get(k, k), before, len(d['predicates'])))
                out.append(d)
            dup += len(new) - 1
        covered = {d['type_key'] for d in out} & current
        name = os.path.basename(path)
        print(f'\n{name}: было {len(rows)} строк -> стало {len(out)} '
              f'(размножено {dup}, потеряно {len(orphan)})')
        print(f'  покрытие типов: {len(covered)} из {len(current)}')
        for k in orphan[:3]:
            print(f'  ПОТЕРЯНО: {k[:90]}')
        for title, b, aft in pruned:
            print(f'  чужие условия убраны ({b} -> {aft}): {title[:70]}')
            if aft == 0 or not any(x.get('field') == 'category'
                                   for d2 in out if d2.get('type_key') and False for x in []):
                pass
        if a.apply:
            shutil.copy(path, path + '.bak')
            with open(path, 'w', encoding='utf-8') as f:
                for d in out:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
            print(f'  записано, старое в {name}.bak')

    if a.apply:
        os.makedirs(os.path.dirname(MAP_PATH), exist_ok=True)
        m = {'_note': 'карточка -> ключ типа. Источник истины для группировки; '
                      'правится руками, порождается v2/tools/remap_type_keys.py',
             'map': {c['id']: t['type_key'] for t in types for c in t['instances']}}
        json.dump(m, open(MAP_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'\nхранимая таблица группировки: {MAP_PATH} ({len(m["map"])} карточек)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
