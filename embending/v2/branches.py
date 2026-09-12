# -*- coding: utf-8 -*-
"""Филиалы МФЦ: связь «услуга ↔ где её физически получить» + контекст оператора.

Почему это отдельно от поиска: филиал почти не фильтрует. Медианный филиал
оказывает 708 услуг из 732 — как признак для ранжирования он бесполезен.
Зато он решает другую задачу: филиал знает свой район, а район — это тот самый
фасет `municipality`, по которому карточка выбирается из группы дубликатов.
Оператор в отделении №28 в Щекино не должен ничего печатать про Щекино.
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facets import find_municipality

# Ленинский район вошёл в городской округ Тула (2014), но адреса остались старые
_FIXUP = [(re.compile(r'(?i)ленинск\w*\s*(р-н|район)'), 'город Тула')]


def branch_municipality(addr):
    """Адрес филиала -> муниципалитет. Это и есть контекст оператора по умолчанию."""
    probe = ' '.join(str((addr or {}).get(k) or '') for k in ('district', 'locality'))
    for rx, mo in _FIXUP:
        if rx.search(probe):
            return mo
    return find_municipality(probe)


def load_branches(classification_path, knowledge_path=None):
    fc = json.load(open(classification_path, encoding='utf-8'))
    by_branch = next(e for e in fc if e['id'] == 'by_branch')
    out = []
    for c in by_branch['categories']:
        addr = c.get('address') or {}
        out.append({
            'branch_id': c['id'],
            'code': c.get('code'),
            'name': re.split(r',\s*(?=ГБУ)', c['name'])[0].strip(),
            'name_full': c['name'],
            'municipality': branch_municipality(addr),
            'address': addr,
            'chief': c.get('chiefName'),
            'chief_post': c.get('chiefPost'),
            'windows': c.get('windowCount'),
            'area': c.get('areaSize'),
            'schedule': c.get('schedule') or [],
            'service_ids': c.get('ids') or [],
        })
    return out


def service_branches(knowledge_path):
    """id услуги -> список branch_id (поле mfcIds в db_knowledge)."""
    db = json.load(open(knowledge_path, encoding='utf-8'))
    return {r['id']: (r.get('mfcIds') or []) for r in db}


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    br = load_branches(os.path.join(root, 'data', 'flat_classification_clean.json'))
    sb = service_branches(os.path.join(root, 'data', 'db_knowledge_clean.json'))
    import collections, statistics as st
    print(f'филиалов: {len(br)}, из них с распознанным МО: {sum(1 for b in br if b["municipality"])}')
    print(f'уникальных отделений (по имени): {len(set(b["name"] for b in br))}')
    n = [len(b['service_ids']) for b in br]
    print(f'услуг на филиал: медиана {int(st.median(n))}, min {min(n)}, max {max(n)} из 732')
    m = collections.Counter(b['municipality'] for b in br)
    print(f'МО покрыто: {len(m)-(1 if None in m else 0)} из 26')
    print(f'услуг без единого филиала: {sum(1 for v in sb.values() if not v)}')
    b = next(x for x in br if x['municipality'] == 'Щекинский район')
    print(f'\nпример: {b["name"]}')
    print(f'  МО: {b["municipality"]}  ·  окон: {b["windows"]}  ·  услуг: {len(b["service_ids"])}')
    print(f'  адрес: {b["address"].get("locality")}, {b["address"].get("street")} {b["address"].get("building")}')
    print(f'  график: {b["schedule"][:3]}')
