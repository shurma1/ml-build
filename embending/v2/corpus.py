# -*- coding: utf-8 -*-
"""Turn the raw MFC dump into the v2 corpus: service TYPES + INSTANCES.

The single most important transformation here: the municipality is pulled OUT of
the text that gets embedded and turned into a structured column. 445 of 732 cards
are the same municipal service replicated across 26 municipalities; leaving the
municipality inside the vector makes those 445 cards mutually indistinguishable
(intra-cluster cosine 0.83-0.88) and turns their ranking into a coin flip.
"""
import json, re, collections, hashlib

# --- 26 municipalities of Tula oblast -------------------------------------
MUNI_FULL = [
    'город Тула', 'город Алексин', 'город Донской', 'город Ефремов',
    'город Новомосковск', 'р.п. Новогуровский', 'р.п. Славный',
    'Арсеньевский район', 'Белевский район', 'Богородицкий район',
    'Веневский район', 'Воловский район', 'Дубенский район', 'Заокский район',
    'Каменский район', 'Кимовский район', 'Киреевский район', 'Куркинский район',
    'Одоевский район', 'Плавский район', 'Суворовский район',
    'Тепло-Огаревский район', 'Узловский район', 'Чернский район',
    'Щекинский район', 'Ясногорский район',
]
_MUNI_KEY = {}
for _f in MUNI_FULL:
    _k = re.sub(r'^(город|р\.п\.)\s+', '', _f)
    _MUNI_KEY[re.sub(r'\s+район$', '', _k).lower()] = _f
# short town forms that appear in card titles instead of the district name
_MUNI_KEY.update({
    'белев': 'Белевский район', 'венев': 'Веневский район', 'одоев': 'Одоевский район',
    'чернь': 'Чернский район', 'плавск': 'Плавский район', 'суворов': 'Суворовский район',
    'узловая': 'Узловский район', 'кимовск': 'Кимовский район', 'киреевск': 'Киреевский район',
    'богородицк': 'Богородицкий район', 'ясногорск': 'Ясногорский район',
    'дубна': 'Дубенский район', 'волово': 'Воловский район', 'теплое': 'Тепло-Огаревский район',
    'куркино': 'Куркинский район', 'арсеньево': 'Арсеньевский район', 'заокский': 'Заокский район',
    'архангельское': 'Каменский район', 'новогуровский': 'р.п. Новогуровский',
    'славный': 'р.п. Славный',
})

BOILER_PAY = re.compile(
    r'(?i)^\s*(государственная|муниципальная)?\s*услуга\s*(предоставляется)?\s*бесплатно\.?\s*$'
    r'|^\s*бесплатно\.?\s*$')


def split_municipality(row):
    """('Выдача разрешений …', 'город Алексин') — municipality out of the title."""
    t = (row.get('serviceTitleText') or '').strip()
    muni = None
    m = re.search(r'муниципального образования\s+(.+?)\s*$', row.get('departmentName') or '')
    if m and m.group(1).strip() in MUNI_FULL:
        muni = m.group(1).strip()
    m = re.search(r'\s*\(([^()]{1,40})\)\s*$', t)
    if m:
        key = re.sub(r'\s+район$', '', m.group(1).strip().lower())
        if key in _MUNI_KEY:
            t = t[:m.start()].strip()
            muni = muni or _MUNI_KEY[key]
    # "… на территории муниципального образования Узловский район" — the municipality
    # also leaks into the middle/end of the title, not only as a trailing "(Узловский)".
    def _strip_named(m):
        nonlocal muni
        name = m.group(1).strip()
        for full in MUNI_FULL:
            k = re.sub(r'^(город|р\.п\.)\s+', '', full)
            if name.lower().startswith(re.sub(r'\s+район$', '', k).lower()):
                muni = muni or full
                return ''
        return m.group(0)
    t = re.sub(r'\s+муниципального образования\s+([А-ЯЁ][^,.;]{2,40})', _strip_named, t)
    t = re.sub(r',?\s*расположенн\w+\s+на территории муниципального образования\s*$', '', t)
    t = re.sub(r',?\s*расположенн\w+\s+на территории\s*$', '', t)
    return re.sub(r'\s+', ' ', t).strip(' ,'), muni


def _short_list(s, limit=6, cap=400):
    """Bureaucratic enumeration -> first N items, joined. Keeps the signal, drops the tail."""
    if not s:
        return ''
    parts = [x.strip(' .;–-') for x in re.split(r'\n|(?<=[;.])\s*(?=\d+[)\.]\s)', s) if x.strip()]
    parts = [re.sub(r'^\d+[)\.]\s*', '', p) for p in parts if len(p) > 3]
    return '; '.join(parts[:limit])[:cap]


def load_cards(path):
    out = []
    for r in json.load(open(path, encoding='utf-8')):
        title, muni = split_municipality(r)
        dept = (r.get('departmentName') or '').strip()
        out.append({
            'id': r['id'],
            'raw_title': r.get('serviceTitleText') or '',
            'title': title,
            'municipality': muni,
            'department': 'Администрация муниципального образования'
                          if muni and 'муниципального образования' in dept else dept,
            'department_full': dept,
            'recipients': r.get('recipientIds') or [],
            'life': r.get('lifeSituationNames') or [],
            'who': _short_list(r.get('serviceRecipients'), 3),
            'result': _short_list(r.get('serviceResultText'), 4),
            'docs': _short_list(r.get('documentsText'), 8, 500),
            'term': (r.get('timeTermText') or '')[:300],
            'free': bool(BOILER_PAY.match(r.get('paymentInfoText') or '')),
            'payment': '' if BOILER_PAY.match(r.get('paymentInfoText') or '')
                       else (r.get('paymentInfoText') or '')[:300],
            'reject': (r.get('rejectReasonsText') or '')[:1000],
            'mfc_count': r.get('mfcCount'),
        })
    return out


def _type_key(title):
    return ' '.join(sorted(set(re.findall(r'[а-яёa-z]{4,}', title.lower()))))


def _type_id(key):
    """Stable across re-indexing and across changes to unrelated services.
    A positional index would renumber every type whenever the grouping shifts,
    silently invalidating every stored type_id (eval sets, analytics, bookmarks)."""
    return int(hashlib.blake2b(key.encode('utf-8'), digest_size=4).hexdigest(), 16)


def build_types(cards):
    g = collections.defaultdict(list)
    for c in cards:
        g[_type_key(c['title'])].append(c)
    types = []
    for key, inst in sorted(g.items()):
        tid = _type_id(key)
        rep = max(inst, key=lambda c: len(c['title']))
        for c in inst:
            c['type_id'] = tid
        types.append({
            'type_id': tid,
            'type_key': key,
            'title': rep['title'],
            'life': sorted({x for c in inst for x in c['life']}),
            'who': rep['who'], 'result': rep['result'], 'docs': rep['docs'],
            'municipalities': sorted({c['municipality'] for c in inst if c['municipality']}),
            'recipients': sorted({x for c in inst for x in c['recipients']}),
            'instances': inst,
        })
    return types


def type_body(t):
    """Everything except title/aliases that is worth indexing. Boilerplate excluded."""
    p = []
    if t['life']:   p.append('Жизненная ситуация: ' + ', '.join(t['life']))
    if t['who']:    p.append('Кому: ' + t['who'])
    if t['result']: p.append('Результат: ' + t['result'])
    if t['docs']:   p.append('Документы: ' + t['docs'])
    return '\n'.join(p)


def search_text(t, aliases=''):
    """Exactly what goes into the embedding. Title first — it carries most of the signal."""
    p = [t['title']]
    if aliases:
        p.append('Люди говорят: ' + aliases)
    body = type_body(t)
    if body:
        p.append(body)
    return '\n'.join(p)
