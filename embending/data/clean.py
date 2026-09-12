import json, re, html
from collections import defaultdict

SRC = '/Users/dmitriy/Desktop/Tulahack2026/Task'
OUT = f'{SRC}/clear'

db = json.load(open(f'{SRC}/db_knowledge.json'))
fc = json.load(open(f'{SRC}/flat_classification.json'))
fd = {e['id']: e['categories'] for e in fc}

def clean_html(s):
    if not isinstance(s, str):
        return s
    s = re.sub(r'(?i)<\s*br[^>]*>', '\n', s)
    s = re.sub(r'(?i)<\s*li[^>]*>', '- ', s)
    s = re.sub(r'(?i)</\s*(p|div|li|ul|ol|tr|table|h[1-6])\s*>', '\n', s)
    s = re.sub(r'<[^>]+>', '', s)
    s = html.unescape(s)
    s = s.replace('\u00a0', ' ').replace('\u200b', '').replace('\ufeff', '').replace('\u2028', '\n')
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r' ?\n ?', '\n', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

# ---------- 1. restore mapping cls-uuid -> db-id ----------
cls = defaultdict(lambda: defaultdict(set))
for dim, cats in fd.items():
    for c in cats:
        for i in c['ids']:
            cls[i][dim].add(c['id'])
dbs = {r['id']: {'by_department': {r['departmentId'] or ''},
                 'by_recipient': set(r['recipientIds']),
                 'by_life_situation': set(r['lifeSituationIds']),
                 'by_branch': set(r['mfcIds'])} for r in db}
def sig(s):
    return tuple((d, tuple(sorted(s[d]))) for d in ['by_department', 'by_recipient', 'by_life_situation'])
A, B = defaultdict(list), defaultdict(list)
for i, s in cls.items(): A[sig(s)].append(i)
for i, s in dbs.items(): B[sig(s)].append(i)
mapping = {}
for k in set(A) | set(B):
    ca, cb = A.get(k, []), B.get(k, [])
    if len(ca) == 1 and len(cb) == 1:
        mapping[ca[0]] = cb[0]
        continue
    if len(ca) != len(cb) or len(ca) > 64:
        continue
    adj = {x: {y for y in cb if cls[x]['by_branch'] <= dbs[y]['by_branch']} for x in ca}
    if any(not v for v in adj.values()):
        continue
    order = sorted(ca, key=lambda x: len(adj[x]))
    res = {}
    def bt(i, used):
        if i == len(order):
            return True
        x = order[i]
        for y in sorted(adj[x]):
            if y in used:
                continue
            used.add(y); res[x] = y
            if bt(i + 1, used):
                return True
            del res[x]; used.remove(y)
        return False
    if bt(0, set()):
        mapping.update(res)
print(f'[map] restored {len(mapping)}/732')

# ---------- 2. lookups from classification ----------
norm_ws = lambda s: re.sub(r'\s+', ' ', s).strip() if isinstance(s, str) else s
dept_name = {c['id']: norm_ws(c['name']) for c in fd['by_department']}
life_name = {c['id']: norm_ws(c['name']) for c in fd['by_life_situation']}
RECIPIENT = {'person': 'person', 'Organization': 'organization', 'IP': 'ip'}

# ---------- 3. clean db_knowledge ----------
TEXT = ['serviceTitleText', 'serviceOrderingText', 'serviceRecipients', 'documentsText',
        'paymentInfoText', 'timeTermText', 'serviceResultText', 'rejectReasonsText']
clean_db = []
for r in db:
    rec = {
        'id': r['id'],
        'departmentId': r['departmentId'] or None,
        'departmentName': dept_name.get(r['departmentId']),
        'recipientIds': sorted({RECIPIENT.get(x, x.lower()) for x in r['recipientIds']}),
        'lifeSituationIds': sorted(r['lifeSituationIds']),
        'lifeSituationNames': [life_name.get(x) for x in sorted(r['lifeSituationIds'])],
        'mfcIds': sorted(r['mfcIds']),
        'legalActsFiles': r['legalActsFiles'],
    }
    for f in TEXT:
        rec[f] = clean_html(r[f]) or None
    clean_db.append(rec)

# ---------- 4. clean flat_classification ----------
def parse_address(s):
    out = {'postal': None, 'region': None, 'district': None, 'locality': None,
           'street': None, 'building': None, 'raw': s}
    if not s:
        return out
    parts = [p.strip() for p in s.split(',') if p.strip()]
    rest = []
    for p in parts:
        if out['postal'] is None and re.fullmatch(r'\d{6}', p):
            out['postal'] = p; continue
        low = p.lower()
        if 'обл' in low and not out['region']: out['region'] = p; continue
        if re.search(r'р-н|район', low) and not out['district']: out['district'] = p; continue
        if re.search(r'^(?:д\.?\s|дом\s)', low) and not out['building']: out['building'] = p; continue
        if (re.search(r'^(?:ул|переул|пер|пр-кт|проспект|просп|ш\.?\s|б-р|наб|туп|мкр|пл|площадь|бульвар|проезд|тракт)', low)
                or re.search(r'(?:^|\s)(?:ул|улица)\s', low)) and not out['street']:
            out['street'] = p; continue
        if re.search(r'^(?:г\.?\s|город\s)|\sг\.?$|^(?:с\.?\s|деревня|село|пгт|дер|пос|д\.?\s)|\s(?:с|д|пос|дер|пгт)\.?$|^(?:хутор|х\.?\s|ст\.?\s|станция)', low) and not out['locality']:
            out['locality'] = p; continue
        rest.append(p)
    if rest and not out['locality']:
        out['locality'] = ', '.join(rest)
    return out

def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None

clean_fc = []
unmapped_total = 0
for e in fc:
    cats = []
    for c in e['categories']:
        nc = {'id': c['id'], 'name': norm_ws(c['name'])}
        if e['id'] == 'by_branch':
            nc.update({
                'code': c.get('code'),
                'address': parse_address(c.get('address')),
                'chiefName': c.get('chiefName'),
                'chiefPost': c.get('chiefPost'),
                'windowCount': to_int(c.get('windowCount')),
                'areaSize': to_int(c.get('areaSize')),
                'schedule': c.get('schedule') or [],
            })
        ids_db = sorted({mapping[i] for i in c['ids'] if i in mapping})
        unmapped = [i for i in c['ids'] if i not in mapping]
        unmapped_total += len(unmapped)
        nc['ids'] = ids_db
        if unmapped:
            nc['ids_unmapped_uuid'] = unmapped
        cats.append(nc)
    clean_fc.append({'id': e['id'], 'categories': cats})

# ---------- 5. write ----------
json.dump(clean_db, open(f'{OUT}/db_knowledge_clean.json', 'w'), ensure_ascii=False, indent=1)
json.dump(clean_fc, open(f'{OUT}/flat_classification_clean.json', 'w'), ensure_ascii=False, indent=1)

# ---------- 6. validate ----------
raw = json.dumps(clean_db, ensure_ascii=False)
print('[valid] records:', len(clean_db), '| html tags left:', len(re.findall(r'<[a-zA-Z/]', raw)))
print('[valid] nbsp left:', '\xa0' in raw, '| double spaces:', len(re.findall(r'  +', raw)))
print('[valid] null departmentId:', sum(1 for r in clean_db if not r['departmentId']))
print('[valid] recipients enum:', {x for r in clean_db for x in r['recipientIds']})
print('[valid] unmapped cls ids kept:', unmapped_total)
for e in clean_fc:
    print(f"[valid] {e['id']}: cats={len(e['categories'])}, ids={sum(len(c['ids']) for c in e['categories'])}")
