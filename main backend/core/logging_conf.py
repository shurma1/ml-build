# -*- coding: utf-8 -*-
"""Логи, из которых нельзя вытащить токен шлюза.

Требование не декоративное. Токен шлюза ходит двумя путями, и один из них —
query-параметр: `WS /v1/asr/stream?session_id=…&token=…`. На уровне DEBUG httpx
и websockets печатают URL целиком, то есть при обычной отладке токен оказался бы
в файле лога. Поэтому редактирование висит на ХЭНДЛЕРАХ корневого логгера:
фильтр на логгере ловит только свои записи, фильтр на хэндлере — все, что через
него проходят, включая чужие библиотеки.

Прячется три вещи:
  * текущий токен (и любой, что был активен раньше в этом процессе) — дословно;
  * `Bearer <...>` и `token=<...>` — на случай токена, о котором фильтр не знает,
    например чужого или ещё не сохранённого;
  * пароль из строки подключения к базе.
"""
import logging
import re
import threading

_lock = threading.Lock()
_secrets = set()

_PATTERNS = [
    (re.compile(r'(?i)(bearer\s+)([A-Za-z0-9._\-]{6,})'), r'\1***'),
    (re.compile(r'(?i)([?&](?:token|gw_token|access_token|secret)=)([^&\s"\']{4,})'), r'\1***'),
    (re.compile(r'(?i)("?(?:token|gw_token|admin_secret)"?\s*[:=]\s*"?)([^\s",}]{6,})'), r'\1***'),
    (re.compile(r'(?i)(password=)(\S+)'), r'\1***'),
]


def mask(token):
    """Как токен разрешено показывать наружу: первые 4 символа и звёздочки."""
    if not token:
        return None
    return (token[:4] + "***") if len(token) > 4 else "***"


def remember_secret(value):
    """Токен, который с этого момента вырезается из всех записей лога."""
    if value and len(value) >= 6:
        with _lock:
            _secrets.add(str(value))


def redact(text):
    if not text:
        return text
    s = str(text)
    with _lock:
        secrets = tuple(_secrets)
    for sec in secrets:
        if sec in s:
            s = s.replace(sec, mask(sec))
    for rx, repl in _PATTERNS:
        s = rx.sub(repl, s)
    return s


def _redact_value(v, depth=0):
    """Почистить аргумент записи, не сломав его тип.

    Секрет попадает в лог не только строкой: httpx на DEBUG кладёт заголовки
    СЛОВАРЁМ в один аргумент, и проверки `isinstance(arg, str)` для него мало.
    Поэтому контейнеры обходятся вглубь, а у прочих объектов проверяется их
    текстовое представление — и подменяется только если секрет там правда есть.
    Числа и прочее проходят насквозь: форматтер access-лога uvicorn ждёт в
    пятом аргументе целое и на строке сломается.
    """
    if depth > 4:
        return v
    if isinstance(v, str):
        return redact(v)
    if isinstance(v, dict):
        return {k: _redact_value(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return type(v)(_redact_value(x, depth + 1) for x in v)
    if isinstance(v, (int, float, bool, type(None))):
        return v
    try:
        text = str(v)
    except Exception:                           # noqa: BLE001
        return v
    cleaned = redact(text)
    return cleaned if cleaned != text else v


class RedactFilter(logging.Filter):
    """Вешается на хэндлер, а не на логгер: так через него проходят записи
    всех библиотек, включая httpx и websockets с их DEBUG-строками URL.

    Сообщение и аргументы чистятся ПО ОТДЕЛЬНОСТИ, а форма записи сохраняется.
    Схлопнуть их в одну строку нельзя: форматтер access-лога uvicorn ожидает
    в `args` кортеж ровно из пяти элементов и на пустом падает — а вместе с ним
    молча пропадает вся строка лога. Заодно это ровно то место, где мог бы
    протечь токен: и путь запроса, и заголовки приходят именно аргументами.
    """

    def filter(self, record):
        try:
            if isinstance(record.msg, str):
                record.msg = redact(record.msg)
            elif record.msg is not None:
                record.msg = _redact_value(record.msg)
            if record.args:
                record.args = _redact_value(record.args)
            if getattr(record, "exc_text", None):
                record.exc_text = redact(record.exc_text)
        except Exception:                       # noqa: BLE001
            record.msg = "<запись не поддалась редактированию и отброшена>"
            record.args = ()
        return True


def setup(level="INFO"):
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(h)
    flt = RedactFilter()
    for h in root.handlers:
        if not any(isinstance(f, RedactFilter) for f in h.filters):
            h.addFilter(flt)
    # uvicorn ставит свои хэндлеры мимо корневого — их тоже закрываем
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore", "websockets"):
        lg = logging.getLogger(name)
        for h in lg.handlers:
            if not any(isinstance(f, RedactFilter) for f in h.filters):
                h.addFilter(flt)
    return flt
