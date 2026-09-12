# -*- coding: utf-8 -*-
"""Диалоговый поиск: признаки -> мультизапрос -> право на услугу -> наводящий вопрос.

Бюджет одного хода (CPU, тёплый кэш):
    признаки от LLM ............ вне этого модуля, доминирует в общем времени
    эмбеддинг 1-5 строк ........ ОДИН батч, ~110 мс (не 5×104: батч почти бесплатен)
    HNSW один раз .............. ~3 мс (намерения сливаются ДО поиска)
    проверка права ............. <1 мс на 30 кандидатов
    наводящий вопрос ........... <1 мс, чистая арифметика по энтропии
Повторный поиск в том же диалоге почти весь уходит в кэш: намерения повторяются.
"""
import os, sys, time, json
from collections import OrderedDict
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import QueryFeatures
from extraction_schema import DialogState, validate, parse_llm_output
from clarify import suggest, apply_answer, needed_facts, rescore
from eligibility import check as check_eligibility
from corpus import load_cards, build_types
from facets import find_municipality

DB = os.getenv("EMB_DB", "host=localhost port=5434 dbname=embeddings user=emb password=emb")
MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-large-instruct")
_INSTR = ("Given a Russian citizen's question about government services, "
          "retrieve the matching official service description")
QUERY_PREFIX = os.getenv("EMB_QUERY_PREFIX",
                         f"Instruct: {_INSTR}\nQuery: " if "instruct" in MODEL else "")


class DialogSearch:
    def __init__(self, db=DB, model=MODEL, data=None, embedder=None, cache_size=4096):
        # Модель эмбеддера живёт на арендованной видеокарте, а не в этом процессе:
        # сюда она приходит по HTTP через POST /v1/embed у gpu-шлюза. Префикс
        # запроса ставит ШЛЮЗ по полю kind — здесь его добавлять нельзя (см. embed()).
        self.conn = psycopg.connect(db, autocommit=True)
        register_vector(self.conn)
        if embedder is None:
            _core = os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), 'main backend')
            if _core not in sys.path:
                sys.path.insert(0, _core)
            from core.embed_client import GatewayEmbedder
            embedder = GatewayEmbedder(kind='query')
        self.model = embedder
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = data or os.path.join(root, 'data', 'services_for_llm.json')
        # Файл читается ОДИН раз: load_cards() разбирал его, а следом тот же
        # путь открывался повторно ради исходных полей — шесть мегабайт JSON
        # на каждый экземпляр пула, которых четыре.
        with open(path, encoding='utf-8') as f:
            rows = json.load(f)
        self.cards = load_cards(rows)
        self.types = {t['type_id']: t for t in build_types(self.cards)}
        raw = {r['id']: r for r in rows}
        self.src = {}
        for t in self.types.values():
            r = raw[t['instances'][0]['id']]
            self.src[t['type_id']] = ((r.get('serviceRecipients') or '') + '\n' +
                                      (r.get('rejectReasonsText') or ''))
        # Кэш векторов запроса. Ограничен намеренно: это обычный dict, живущий
        # столько же, сколько процесс, и на длинной смене он рос бы без предела —
        # 4 КБ на строку, четыре экземпляра в пуле. Намерения в диалогах
        # повторяются, поэтому попадания даёт и небольшой объём; за пределами
        # процесса есть второй кэш, в таблице query_cache.
        self._cache = OrderedDict()
        self._cache_cap = cache_size

    # --- один батч на все строки запроса: 5 строк стоят почти как одна ---
    def embed(self, strings):
        """QUERY_PREFIX здесь НЕ добавляется: префикс ставит шлюз по полю kind.
        Добавить его ещё и тут значит закодировать запрос дважды префиксованным,
        а индекс — одинарным; выдача останется правдоподобной, и заметить это
        по результатам невозможно.

        Цена рассогласования перемерена на нынешней конфигурации и невелика: двойной префикс -0.7 п.п. R@1, отсутствие префикса -0.9, чужой формат -1.0 — все три в пределах доверительного интервала. Стоявшая здесь цифра -19 п.п. относится к другому замеру (Giga-480M на плоском индексе из 732 карточек, embending/RESULTS.md), а не к mE5-large-instruct на 321 типе. Инвариант это не отменяет: он стоит дёшево, а рассогласование по выдаче не видно."""
        miss = [s for s in strings if s not in self._cache]
        if miss:
            v = self.model.encode(miss)
            for text, vec in zip(miss, v):
                self._cache[text] = vec
            while len(self._cache) > self._cache_cap:
                self._cache.popitem(last=False)
        out = []
        for s in strings:
            self._cache.move_to_end(s)
            out.append(self._cache[s])
        return out

    def _ann(self, vec, k):
        return self.conn.execute(
            "SELECT type_id, title, n_instances, 1-(embedding <=> %s) AS score "
            "FROM service_type ORDER BY embedding <=> %s LIMIT %s", (vec, vec, k)).fetchall()

    def search(self, feats, k=8, pool=30, asked=()):
        t0 = time.perf_counter()
        if isinstance(feats, str):
            feats = QueryFeatures.from_text(feats)
        strings = feats.search_strings()
        if not strings:
            return {'results': [], 'questions': [], 'ms': 0}

        vecs = self.embed(strings)
        t_emb = time.perf_counter()

        # Центроид намерений вместо слияния рангов. RRF переводит оценки в позиции
        # и величину сходства выбрасывает; у e5 все косинусы лежат в узкой полосе
        # 0.85-0.95, но различия ВНУТРИ неё информативны — их RRF и стирает.
        # Замерено на 230 группах по три формулировки: RRF 3:1:1 -> R@1 0.783,
        # центроид -> 0.857 (+7.4 п.п., ДИ [+3.5, +11.3]). На срезе самых непохожих
        # формулировок разрыв ещё шире: 0.638 против 0.828.
        #
        # Второе следствие важно не меньше: при нормированных документах
        # mean(q_j)·d = mean(q_j·d), поэтому N обращений к индексу схлопываются
        # в одно. И оценка в выдаче теперь та же, по которой список упорядочен, —
        # раньше порядок задавал RRF, а показывался максимум косинуса.
        q = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
        norm = float(np.linalg.norm(q))
        if norm:
            q = q / norm
        rows = self._ann(q, pool)
        order = [r[0] for r in rows]
        best = {tid: (sc, title, n) for tid, title, n, sc in rows}

        muni = feats.municipality or find_municipality(feats.raw_text or '')
        facts = dict(feats.facts)
        if feats.recipient:
            facts['recipient'] = feats.recipient

        cands = [self.types[t] for t in order if t in self.types]
        out = []
        # Карточки всех показываемых типов забираются ОДНИМ запросом. Раньше
        # _resolve() ходил в базу на каждую строку выдачи: при k=8 это восемь
        # последовательных обходов сети поверх одного ANN — половина времени
        # хода уходила на ожидание round-trip, а не на работу.
        cards = self._cards_for(order[:k])
        for tid in order[:k]:
            t = self.types.get(tid)
            if not t:
                continue
            status, detail = check_eligibility(t, facts, self.src[tid])
            out.append({
                'type_id': tid, 'title': t['title'], 'score': round(best[tid][0], 4),
                'status': status,                       # eligible | blocked | unknown
                'reasons': detail,
                **self._resolve(tid, muni, cards.get(tid, ())),
            })
        # факты, которых не хватило для проверки права, — тоже повод спросить,
        # причём самый весомый: вопрос будет про услугу наверху выдачи
        need = needed_facts(out)
        # Отрыв top1-top2 — мера уверенности ретривера, а не украшение выдачи:
        # по нему clarify решает, есть ли ещё что сужать (см. CERTAIN_MARGIN).
        margin = float(rows[0][3] - rows[1][3]) if len(rows) > 1 else None
        questions = suggest(cands[:20], asked=asked, feats=feats,
                            needed=needed_facts(out, top=3), margin=margin)
        return {'results': out, 'questions': questions, 'missing_facts': need,
                'municipality': muni, 'n_intents': len(strings),
                'margin': round(margin, 4) if margin is not None else None,
                'ms': round((time.perf_counter() - t0) * 1000, 1),
                'ms_embed': round((t_emb - t0) * 1000, 1)}

    def _resolve(self, type_id, muni, rows=None):
        """Тип -> конкретная карточка. Три исхода, и третий раньше терялся.

        `needs_municipality` («уточните МО») и «МО известен, а карточки для него
        нет» — разные состояния, а сваливались в одно. Оператор видел «уточните
        муниципалитет» там, где уточнять было нечего: район уже назван.

        Второе состояние отдаётся как `not_in_municipality`, и формулировать его
        нужно как пробел в выгрузке, а не как отказ. Данные этого не выдержат:
        частичное покрытие есть у 24 типов из 321, и среди них «Признание
        садового дома жилым» в 4 МО из 26 и «Выписка из похозяйственной книги»
        в 4 из 26 — процедуры, обязательные везде по федеральным актам. Каждое
        МО представлено в среднем 17 муниципальными типами из 34, ровно
        половиной. Это неполная выгрузка, а не карта услуг области.
        """
        rs = self._cards_for([type_id]).get(type_id, ()) if rows is None else rows
        if muni:
            hit = [r for r in rs if r[1] == muni]
            if hit:
                return {'service_id': hit[0][0], 'department': hit[0][2]}
        if len(rs) == 1:
            return {'service_id': rs[0][0], 'department': rs[0][2]}
        if muni:
            return {'service_id': None, 'needs_municipality': False,
                    'not_in_municipality': muni,
                    'available_in': sorted({r[1] for r in rs if r[1]})}
        return {'service_id': None, 'needs_municipality': True}

    def _cards_for(self, type_ids):
        """{type_id: [(id, municipality, department), ...]} одним запросом."""
        ids = list(type_ids)
        if not ids:
            return {}
        rows = self.conn.execute(
            "SELECT type_id, id, municipality, department FROM service "
            "WHERE type_id = ANY(%s) ORDER BY type_id, municipality NULLS FIRST",
            (ids,)).fetchall()
        out = {}
        for tid, sid, m, dep in rows:
            out.setdefault(tid, []).append((sid, m, dep))
        return out

    def narrow(self, feats, answers, k=8, pool=20):
        """answers — [(ключ_вопроса, ответ), ...] за весь диалог. Переранжирование,
        а не фильтрация: ответ не выбрасывает кандидата, а двигает его."""
        r = self.search(feats, k=pool, pool=pool)
        cands = [self.types[x['type_id']] for x in r['results'] if x['type_id'] in self.types]
        scores = {x['type_id']: x['score'] for x in r['results']}
        asked = set()
        for key, ans in answers:
            asked.add(key)
            scores = rescore(cands, scores, key, ans)
        by_id = {x['type_id']: x for x in r['results']}
        ranked = sorted(cands, key=lambda t: -scores[t['type_id']])
        r['results'] = [by_id[t['type_id']] for t in ranked][:k]
        # Гейт считается по ИСХОДНОЙ уверенности ретривера, а не по разведённым
        # ответами оценкам: rescore двигает кандидатов намеренно, и мерить по нему
        # «решён ли список» значило бы мерить силу собственной же правки.
        r['questions'] = suggest(ranked, asked=asked, feats=feats,
                                 needed=needed_facts(r['results'], top=3),
                                 margin=r.get('margin'))
        r['asked'] = sorted(asked)
        return r
