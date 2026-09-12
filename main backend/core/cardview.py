# -*- coding: utf-8 -*-
"""Карточка-шпаргалка: регламент -> то, что оператор читает вслух.

Раньше этот разбор жил на фронте (`cardModel.ts`) и был написан по пяти
карточкам из демо-набора. На полном корпусе из 732 он разваливался: срок
находился у 9% карточек, формула «кому положено» собиралась у 8%, 64% причин
отказа обрезались по седьмому слову. Перенос сюда — не вопрос вкуса: разбор
обязан жить там же, где корпус, иначе две реализации одного парсера расходятся
молча, и оператор произносит вслух то, чего в регламенте нет.

Главное решение здесь — **каждый производный пункт несёт смещения в исходное
поле** (`start`/`end`). Без этого подсветка условий права нежизнеспособна:
бэкенд отдаёт смещения в сыром `documentsText`, а интерфейс показывает
переписанный короткий текст, и они не совпадают ни на одном символе. Со смещениями
интерфейс волен показывать короткую форму, а подсветку накладывать на исходную.

Числовые факты берутся из `canon` — того же модуля, которым пользуется индекс.
Своих чисел здесь не считается: `decisionDays` это ровно `canon.term_days`,
дополненный словесными числительными там, где canon молчит. Разойтись с
`ServiceCard.term_days` эта карточка не может по построению.
"""
import logging
import re
import sys

from . import config as C

log = logging.getLogger("core.cardview")


def _v2():
    if C.V2_PATH not in sys.path:
        sys.path.insert(0, C.V2_PATH)


def _flat(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _cap(s):
    return s[:1].upper() + s[1:] if s else s


def plural(n, forms):
    m = abs(n) % 100
    d = m % 10
    if 10 < m < 20:
        return forms[2]
    if 1 < d < 5:
        return forms[1]
    if d == 1:
        return forms[0]
    return forms[2]


# --- сроки ------------------------------------------------------------------

NUMERALS = {
    "один": 1, "одного": 1, "два": 2, "двух": 2, "три": 3, "трех": 3, "трёх": 3,
    "четыре": 4, "четырех": 4, "четырёх": 4, "пять": 5, "пяти": 5, "шесть": 6,
    "шести": 6, "семь": 7, "семи": 7, "восемь": 8, "восьми": 8, "девять": 9,
    "девяти": 9, "десять": 10, "десяти": 10, "одиннадцать": 11, "двенадцать": 12,
    "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15, "двадцать": 20,
    "тридцать": 30, "сорок": 40, "шестьдесят": 60, "девяносто": 90, "сто": 100,
}
_NUM_RX = re.compile(
    r"(?i)(?<![а-яё])(" + "|".join(sorted(NUMERALS, key=len, reverse=True)) +
    r")\s+(?:рабоч|календарн)\w*\s+(?:дн|дня|дней)")
_DAYS_RX = re.compile(r"(?i)(\d{1,3})\s*(?:-?[а-яё]{1,2})?\s*(рабоч|календарн)\w*\s*дн")
_SHORT_RX = re.compile(r"(?i)(\d{1,3})\s*(р|к)\.?\s?д\.?(?![а-яё])")


def _days_near(text, scope, calendar_to_work=True):
    """Число дней в предложении, подходящем под `scope`. None, если не нашлось."""
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        s = _flat(sent)
        if not s or not re.search(scope, s, re.I):
            continue
        m = _DAYS_RX.search(s)
        if m:
            n = int(m.group(1))
            return round(n * 5 / 7) if (calendar_to_work and m.group(2).lower().startswith("календарн")) else n
        m = _SHORT_RX.search(s)
        if m:
            n = int(m.group(1))
            return round(n * 5 / 7) if (calendar_to_work and m.group(2).lower() == "к") else n
        m = _NUM_RX.search(s)
        if m:
            return NUMERALS[m.group(1).lower()]
    return None


def parse_terms(time_term_text):
    """-> {decisionDays, extensionDays, notifyDays, mfcDays}.

    `decisionDays` — это `canon.term_days`, и только если тот промолчал,
    подключаются словесные числительные («четыре рабочих дня»), которых canon
    не знает. Порядок именно такой: число в карточке обязано совпадать с числом
    в выдаче поиска, иначе оператор увидит два разных срока на одном экране.
    """
    _v2()
    from canon import term_days

    t = time_term_text or ""
    decision = term_days(t)
    if decision is None:
        m = _NUM_RX.search(t)
        if m:
            decision = NUMERALS[m.group(1).lower()]
    return {
        "decisionDays": decision,
        "extensionDays": _days_near(t, r"продлева|продлен|продлить"),
        "notifyDays": _days_near(t, r"информиру|уведомля|извеща|направля\w+ уведомл"),
        "mfcDays": _days_near(t, r"мфц|многофункциональн"),
    }


# --- каналы получения -------------------------------------------------------

def parse_channels(ordering_text):
    seen, out = set(), []
    for raw in (ordering_text or "").split("\n"):
        line = _flat(raw)
        if not line:
            continue
        if re.search(r"(?i)мфц|многофункциональн", line):
            ch = {"key": "mfc", "label": "МФЦ"}
        elif re.search(r"(?i)рпгу|региональн\w+ портал", line):
            ch = {"key": "rpgu", "label": "РПГУ"}
        elif re.search(r"(?i)епгу|единый портал|госуслуг", line):
            ch = {"key": "epgu", "label": "Госуслуги"}
        elif re.search(r"(?i)социальной защиты|соцзащит", line):
            ch = {"key": "agency", "label": "Соцзащита"}
        elif re.search(r"(?i)фнс|налогов\w+ службы", line):
            ch = {"key": "agency", "label": "УФНС"}
        elif re.search(r"(?i)администрац", line):
            ch = {"key": "agency", "label": "Администрация"}
        else:
            ch = {"key": "other", "label": " ".join(line.split()[:3])}
        if ch["label"] not in seen:
            seen.add(ch["label"])
            out.append(ch)
    return out


# --- результат --------------------------------------------------------------

# --- разбиение списка с сохранением смещений --------------------------------

# Маркер пункта: «1)», «1.», «а)», «-». Буквенные нумерации в регламентах
# встречаются наравне с цифровыми, и без них текст уходил в дробление по
# запятым, превращая одно предложение в шесть обрывков.
_ITEM_START = re.compile(
    r"(?:(?<=^)|(?<=\n)|(?<=[;:]))\s*(?:\d{1,2}[).]|[а-яё][).]\s|[-–—•])\s*")


def split_items(text, prose=True):
    """Пункты перечня -> [(текст, начало, конец)] в ИСХОДНЫХ смещениях.

    Смещения здесь не украшение: на них ложится подсветка цитат из таблицы
    предикатов. Поэтому перечень не нормализуется до разбиения — позиция
    каждого пункта в исходной строке сохраняется как есть.

    `prose=True` дополнительно разбирает перечисления прозой («предъявить
    паспорт, согласие…, а также документы…») — без этого у 106 карточек из 732
    чек-лист оказывался пустым, то есть у каждой седьмой оператор не видел
    ничего. Для причин отказа режим выключен: там пункт — это предложение
    целиком, и запятая внутри него не граница, а часть фразы.
    """
    t = text or ""
    if not t.strip():
        return []
    marks = [m.end() for m in _ITEM_START.finditer(t)]
    spans = []
    if marks:
        bounds = marks + [len(t)]
        # преамбула до первого маркера отбрасывается: это служебная подводка
        for a, b in zip(bounds, bounds[1:]):
            chunk = t[a:b]
            cut = _ITEM_START.search(chunk)
            end = a + (cut.start() if cut else len(chunk))
            spans.append((a, end))
    elif not prose:
        spans = [(0, len(t))]
    else:
        # перечисления прозой: режем по запятым/«а также» после глагола-указателя
        m = re.search(r"(?i)(предъяв\w+|предоставля\w+|представля\w+|необходим\w*|прилага\w+)\s*:?\s*",
                      t)
        start = m.end() if m else 0
        tail = t[start:]
        pos = start
        for part in re.split(r"(?i),\s*(?:а\s+также\s+)?|;\s*|\bа\s+также\s+", tail):
            if part is None:
                continue
            a = t.find(part, pos) if part.strip() else -1
            if a >= 0:
                spans.append((a, a + len(part)))
                pos = a + len(part)
    out = []
    for a, b in spans:
        raw = t[a:b]
        s = raw.strip()
        if len(s) < 4:
            continue
        if re.match(r"(?i)^(основани|отказ|вместе с заявлением|для предоставления|в случае)", s):
            continue
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        out.append((s, a + lead, b - trail))
    return out


# «Результатом предоставления услуги является:» — подводка, а не результат.
# Без её снятия в поле «Результат» уезжало «Результатом предоставлен…»,
# то есть половина служебной фразы вместо ответа на вопрос «что я получу».
_RESULT_LEAD = re.compile(
    r"(?i)^\s*(результат\w*\s+предоставления\s+(?:государственн\w+|муниципальн\w+)?\s*услуги"
    r"\s+(?:являет|являют)\w*|результат\w*\s*:)\s*:?\s*")


def parse_result(result_text):
    t = _RESULT_LEAD.sub("", result_text or "")
    # Первый содержательный пункт: тем же разбиением, что и перечни документов,
    # иначе в поле уезжает номер списка («1.») вместо самого результата.
    items = [x for x, _, _ in split_items(t, prose=False)]
    first = _flat(items[0]) if items else _flat(t.lstrip("-–— ").split("\n")[0])
    first = re.sub(r"(?i)^\s*(?:\d{1,2}[).]|[а-яё][).])\s*", "", first)
    if re.search(r"(?i)qr-?код|ор-?код", t):
        return {"value": "QR-код", "label": "уведомление с QR-кодом"}
    if re.search(r"(?i)справк", first):
        return {"value": "Справка", "label": first[:160]}
    if re.search(r"(?i)решение о назначении", t):
        return {"value": "Решение", "label": "решение о назначении или об отказе"}
    if re.search(r"(?i)^выдача\b", first):
        return {"value": "Выдача документа", "label": first[:160]}
    if re.search(r"(?i)уведомлени", first):
        return {"value": "Уведомление", "label": first[:160]}
    if not first:
        return {"value": "—", "label": ""}
    return {"value": _cap(first[:24]) + ("…" if len(first) > 24 else ""),
            "label": _cap(first[:160])}


# --- документы --------------------------------------------------------------

DOC_CONDITION_LABELS = {
    "foreign": "иностранное свидетельство",
    "representative": "через представителя",
    "no-registration": "нет прописки в области",
    "family": "для членов семьи",
    "if-available": "если есть",
    "on-request": "по запросу учреждения",
}
_DOC_CONDITIONS = [
    ("representative", r"(?i)представител|доверенност|полномочи"),
    ("foreign", r"(?i)иностранн"),
    ("no-registration", r"(?i)факт проживания|отсутстви\w+ регистраци|о пребывании"),
    ("family", r"(?i)членами семьи|член\w* семьи|родители участника|погибшего|умершего"),
    ("if-available", r"(?i)при наличии"),
    ("on-request", r"(?i)при необходимости"),
]
_CONDITION_CUTS = [" - при ", " – при ", "- при ", " (при ", " (в том числе",
                   ", в случае если", " в случае если", " при личном обращении"]
_DOC_REWRITES = [
    (r"(?i)паспорт либо иной документ, удостоверяющий личность( заявителя| получателя)?",
     "Паспорт (или иной документ, удостоверяющий личность)"),
    (r"(?i)документ,? удостоверяющий личность( заявителя| получателя)?",
     "Документ, удостоверяющий личность"),
    (r"(?i)подтверждающие факт проживания", "подтверждающие проживание"),
    (r"(?i)на территории Тульской области", "в Тульской области"),
    (r"(?i)справка организации, осуществляющей образовательную деятельность по очной форме обучения",
     "Справка из места учёбы (очная форма)"),
    (r"(?i), уполномоченного в установленном законодательством Российской Федерации порядке", ""),
    (r"(?i) в соответствии с законодательством Российской Федерации", ""),
    (r"(?i)получателя дополнительной меры социальной поддержки", "заявителя"),
    (r"(?i) в специальной военной операции, проводимой с 24 февраля 2022 года", " в СВО"),
    (r"(?i)сведения, подтверждающие участие гражданина в выполнении задач в СВО",
     "сведения, подтверждающие участие в СВО"),
]


def _shorten(raw, limit=90):
    """Короткая форма пункта. Исходный текст при этом НЕ выбрасывается —
    он остаётся в `fullText` вместе со смещениями."""
    main = raw
    cut = min((main.find(c) for c in _CONDITION_CUTS if main.find(c) >= 0), default=-1)
    if cut > 15:
        main = main[:cut]
    for rx, to in _DOC_REWRITES:
        main = re.sub(rx, to, main)
    main = _flat(main)
    paren = main.find(" (", 30)
    if paren > 0:
        main = main[:paren]
    main = re.sub(r"(?i)^о (рождении|заключении|расторжении|перемене|смерти|установлении)",
                  r"свидетельство о \1", main)
    main = re.sub(r"[.,;\s]+$", "", main)
    if len(main) > limit:
        cutw = main.rfind(" ", 0, limit)
        main = main[:cutw if cutw > 40 else limit].rstrip(" ,;") + "…"
    return _cap(main)


def parse_documents(documents_text):
    items = []
    for i, (raw, a, b) in enumerate(split_items(documents_text)):
        conds = [k for k, rx in _DOC_CONDITIONS if re.search(rx, raw)]
        items.append({"id": f"doc-{i}", "short": _shorten(raw), "fullText": raw,
                      "start": a, "end": b, "field": "documentsText",
                      "conditions": conds})
    return {"always": [d for d in items if not d["conditions"]],
            "situational": [d for d in items if d["conditions"]],
            "total": len(items)}


# --- причины отказа ---------------------------------------------------------

_REJECT_RULES = [
    ("Документы не донесены в срок", r"(?i)непредставление[\s\S]{0,80}в сроки, указанные", 1),
    ("Неполный пакет документов",
     r"(?i)неполн\w* (комплект|пакет)|неполном объеме|непредставление[\s\S]{0,60}документ", 1),
    ("Ошибки в заявлении",
     r"(?i)недостоверн|неполных данных|отсутствие подписи|отсутствие в запросе|не поддается прочтению|нечитаем", 2),
    ("Не удалось установить личность", r"(?i)неустановление личности|не установлена личность", 2),
    ("Нет полномочий у представителя", r"(?i)полномочи|уполномоченным представителем", 6),
    ("Нет подтверждения участия в СВО",
     r"(?i)подтверждающих (факт )?участия|факт участия граждан", 3),
    ("Заявитель не подходит под категорию",
     r"(?i)несоответствие заявителя|категориям граждан|требованиям пунктов|не относится к категор", 4),
    ("Сведения не совпали с данными ведомств", r"(?i)несоответствие сведений", 4),
    ("Мера уже назначена", r"(?i)ранее поданному|факта назначения|ранее предоставл", 5),
    ("Приговор суда", r"(?i)приговор", 7),
    ("Заявление отозвано", r"(?i)отзыв заявления|отказ заявителя", 6),
    ("Нет прописки в Тульской области",
     r"(?i)не прожива\w+ на территории|отсутствие регистрации по месту", 4),
]
# Служебная подводка, с которой начинается пункт: «для организации (ИП) - …»
_REJECT_LEAD = re.compile(
    r"(?i)^(для\s+(?:физическ\w+|юридическ\w+|организац\w+)[^-–—:]{0,60}[-–—:]\s*"
    r"|основани\w+[^:]{0,60}:\s*)")


def parse_rejects(reject_text):
    by_label, out = {}, []
    for i, (raw, a, b) in enumerate(split_items(reject_text, prose=False)):
        label, rank = None, 9
        for lab, rx, rk in _REJECT_RULES:
            if re.search(rx, raw):
                label, rank = lab, rk
                break
        if label is None:
            # Общий случай: снимаем служебную подводку и берём первое
            # содержательное предложение. Обрезка по седьмому слову, как было
            # на фронте, превращала 64% пунктов в бессмысленный огрызок.
            body = _REJECT_LEAD.sub("", raw)
            body = re.split(r"(?<=[.;])\s+", body)[0]
            label = _shorten(body, 70)
        item = {"id": f"reject-{i}", "short": label, "fullText": raw,
                "start": a, "end": b, "field": "rejectReasonsText", "rank": rank,
                "also": []}
        seen = by_label.get(label)
        if seen:
            # Несколько пунктов регламента сводятся к одной формулировке
            # («Ошибки в заявлении» ×3). Схлопываем их для показа, но дубликаты
            # кладём отдельно, СО СВОИМИ смещениями: дописывание чужого текста
            # в fullText разорвало бы связь пункта с его местом в исходнике,
            # а на ней держится подсветка.
            seen["also"].append({"fullText": raw, "start": a, "end": b,
                                 "field": "rejectReasonsText"})
            continue
        by_label[label] = item
        out.append(item)
    return sorted(out, key=lambda r: r["rank"])


# --- кому положено ----------------------------------------------------------

_RECIPIENT_LABELS = {"person": "физлица", "ip": "ИП", "organization": "организации"}
_CATEGORY_LABELS = {
    "СВО": "участники СВО", "ВБД": "ветераны боевых действий",
    "ЧАЭС": "пострадавшие от аварии на ЧАЭС", "многодетный": "многодетные семьи",
    "инвалид": "инвалиды", "пенсионер": "пенсионеры", "малоимущий": "малоимущие",
    "военнослужащий": "военнослужащие",
}


def parse_recipients(type_row, service_recipients, recipient_ids):
    """Формула «кому положено».

    Строится не регулярками по прозе, а из двух опор, которые есть почти
    у каждой карточки: льготные категории из таблицы предикатов (те самые, что
    дают вердикт о праве) и структурное поле `recipientIds`. Прозаический разбор
    собирал формулу у 8% карточек; этот — у всех, у кого поле заполнено,
    и главное — не расходится с вердиктом, потому что смотрит в тот же источник.
    """
    _v2()
    from eligibility import load_predicates

    cats = []
    if type_row:
        for p in load_predicates().get(type_row.get("type_key"), []):
            if p.get("field") == "category":
                for v in (p["value"] if isinstance(p["value"], list) else [p["value"]]):
                    if v not in cats:
                        cats.append(v)
    parts = [_CATEGORY_LABELS.get(c, c) for c in cats]
    kinds = [_RECIPIENT_LABELS[r] for r in (recipient_ids or []) if r in _RECIPIENT_LABELS]

    if parts:
        formula = _cap(", ".join(parts))
        if re.search(r"(?i)член\w*\s+(их\s+)?сем", service_recipients or ""):
            members = []
            if re.search(r"(?i)супруг", service_recipients): members.append("супруг(а)")
            if re.search(r"(?i)дет[ейи]|ребен|ребён", service_recipients): members.append("дети")
            if re.search(r"(?i)родител", service_recipients): members.append("родители")
            formula += " + члены семьи" + (f": {', '.join(members)}" if members else "")
    elif kinds:
        formula = _cap(", ".join(kinds))
    else:
        formula = _cap(_flat((service_recipients or "").split("\n")[0])[:120]) or "—"

    return {
        "formula": formula,
        "categories": cats,
        "recipientIds": list(recipient_ids or []),
        "fullText": service_recipients or "",
        "hasRepresentativeNote": bool(
            re.search(r"(?i)законн\w+ представител|их представители|обраща\w+ представител",
                      service_recipients or "")),
    }


# --- правовое основание -----------------------------------------------------

_LEGAL_RX = re.compile(
    r"(?i)(?<![а-яё])((?:приказ|указ|постановлени|распоряжени)\w*\s+[^\"()\n;]{5,140}?"
    r"от\s+\d{1,2}\s+[а-яё]+\s+\d{4}\s*(?:г(?:ода)?\.?)?\s*(?:N|№)\s*\d[\d\-осн/]*)")


def parse_legal_basis(*texts):
    for t in texts:
        m = _LEGAL_RX.search(t or "")
        if m:
            return (_flat(m.group(1))
                    .replace(" ", " ")
                    .replace(" N ", " № "))
    return None


# --- сборка -----------------------------------------------------------------

def build(card, raw, type_row=None):
    """card — обработанная карточка из corpus.load_cards, raw — сырая из выгрузки."""
    _v2()
    from canon import is_free

    title = card.get("title") or _flat(raw.get("serviceTitleText"))
    return {
        "title": title,
        "fullTitle": _flat(raw.get("serviceTitleText")),
        "department": card.get("department_full") or raw.get("departmentName"),
        "departmentName": raw.get("departmentName"),
        "municipality": card.get("municipality"),
        "channels": parse_channels(raw.get("serviceOrderingText")),
        "mfcCount": raw.get("mfcCount"),
        "lifeSituationNames": list(raw.get("lifeSituationNames") or []),
        "terms": parse_terms(raw.get("timeTermText")),
        "isFree": is_free(raw.get("paymentInfoText") or ""),
        "paymentText": _flat(raw.get("paymentInfoText")),
        "result": parse_result(raw.get("serviceResultText")),
        "recipients": parse_recipients(type_row, raw.get("serviceRecipients"),
                                       raw.get("recipientIds")),
        "documents": parse_documents(raw.get("documentsText")),
        "rejects": parse_rejects(raw.get("rejectReasonsText")),
        "legalBasis": parse_legal_basis(raw.get("rejectReasonsText"),
                                        raw.get("serviceOrderingText")),
    }
