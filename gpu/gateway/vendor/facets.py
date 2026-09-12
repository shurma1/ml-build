# -*- coding: utf-8 -*-
"""Structured facets pulled out of a free-text query. Pure Python, ~0.05 ms.

Municipality is the only strongly selective facet in this corpus (26 values on
445 of 732 cards). Everything else — branch, recipient — barely filters anything
(by_branch: every branch offers ~700 of 732 services).
"""
import re
from nltk.stem.snowball import SnowballStemmer

_ST = SnowballStemmer('russian')

# Unambiguous place roots.
STRONG = {
    'город Тула': ['тул', 'тульск'],
    'город Алексин': ['алексин', 'алексинск'],
    'город Донской': ['донск', 'донско'],
    'город Ефремов': ['ефремов'],
    'город Новомосковск': ['новомосковск', 'новомосков'],
    'р.п. Новогуровский': ['новогуров', 'новогуровск'],
    'р.п. Славный': ['славн'],
    'Арсеньевский район': ['арсеньев', 'арсеньевск'],
    'Белевский район': ['белев', 'белевск'],
    'Богородицкий район': ['богородиц', 'богородицк'],
    'Веневский район': ['венев', 'веневск'],
    'Воловский район': ['воловск'],
    'Дубенский район': ['дубен', 'дубенск'],
    'Заокский район': ['заок', 'заокск'],
    'Каменский район': ['каменск'],
    'Кимовский район': ['кимов', 'кимовск'],
    'Киреевский район': ['киреев', 'киреевск'],
    'Куркинский район': ['куркин', 'куркинск'],
    'Одоевский район': ['одоев', 'одоевск'],
    'Плавский район': ['плавск'],
    'Суворовский район': ['суворовск'],
    'Тепло-Огаревский район': ['теплоогарев', 'теплоогаревск', 'огарев'],
    'Узловский район': ['узловск', 'узлов'],
    'Чернский район': ['чернск'],
    'Щекинский район': ['щекин', 'щекинск'],
    'Ясногорский район': ['ясногор', 'ясногорск'],
}
# Roots that collide with ordinary words ("чёрный", "плавание", "каменный"):
# only accepted when a locality cue is present in the query.
WEAK = {
    'Воловский район': ['волов'], 'Каменский район': ['камен'], 'Плавский район': ['плав'],
    'Суворовский район': ['суворов'], 'Чернский район': ['черн'], 'город Ефремов': ['ефрем'],
}
CUE = re.compile(r'(?<![а-я])(район|р-н|рн|город|гор|мо|поселок|посёлок|рп|округ|адм\w*)')


def find_municipality(q):
    """-> canonical municipality name, or None when the query is region-wide/ambiguous."""
    ql = q.lower().replace('ё', 'е')
    ql = re.sub(r'тульск\w*\s+(обл\w*|региона?\w*)', ' ', ql)   # oblast != city of Tula
    toks = [_ST.stem(w) for w in re.findall(r'[а-яa-z]{3,}', ql)]
    hits = []

    def probe(table):
        for name, stems in table.items():
            for st in stems:
                if st in toks or any((t.startswith(st) and len(t) - len(st) <= 3) or
                                     (len(t) >= 4 and st.startswith(t) and len(st) - len(t) <= 3)
                                     for t in toks):
                    hits.append((len(st), name)); break

    probe(STRONG)
    if not hits and CUE.search(ql):
        probe(WEAK)
    if not hits:
        return None
    best = max(h[0] for h in hits)
    top = {n for l, n in hits if l == best}
    return top.pop() if len(top) == 1 else None


_RECIP = [(re.compile(r'(?<![а-я])(ип|индивидуальн\w* предпринимател\w*|самозанят\w*)'), 'ip'),
          (re.compile(r'(?<![а-я])(ооо|юр\w* лиц\w*|организац\w*|компани\w*|предприяти\w*)'), 'organization')]


def find_recipient(q):
    for rx, v in _RECIP:
        if rx.search(q.lower()):
            return v
    return None
