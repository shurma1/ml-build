# -*- coding: utf-8 -*-
"""Конфигурация core-api. Всё из окружения, молчаливых дефолтов у секретов нет.

Единственное место, где настройка НЕ берётся из окружения, — адрес gpu-шлюза:
арендованный под получает новый адрес при каждом пересоздании, поэтому он живёт
в `meta['gpu_gateway']` в базе и меняется через POST /v1/admin/gateway без
перезапуска процесса. Переменные GW_BASE_URL / GW_TOKEN остаются только как
первичный bootstrap: их читают ровно один раз, когда в meta ещё пусто.
"""
import os
import sys


def _int(name, default):
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _float(name, default):
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


API_VERSION = "1.0.0"

# --- база ---
DB = os.getenv("EMB_DB", "host=localhost port=5434 dbname=embeddings user=emb password=emb")
DB_POOL_MIN = _int("DB_POOL_MIN", 2)
DB_POOL_MAX = _int("DB_POOL_MAX", 10)

# --- корпус ---
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.getenv("EMB_DATA", os.path.join(_ROOT, "embending", "data", "services_for_llm.json"))
V2_PATH = os.getenv("V2_PATH", os.path.join(_ROOT, "embending", "v2"))

# --- gpu-шлюз (bootstrap; дальше правится через /v1/admin/gateway) ---
GW_BASE_URL = (os.getenv("GW_BASE_URL", "") or "").strip().rstrip("/")
GW_TOKEN = os.getenv("GW_TOKEN", "") or ""
GW_TIMEOUT = _float("GW_TIMEOUT", 20.0)
GW_EMBED_TIMEOUT = _float("GW_EMBED_TIMEOUT", 10.0)
GW_READY_TIMEOUT = _float("GW_READY_TIMEOUT", 10.0)   # в админ-роуте зафиксировано ТЗ: 10 с
GW_MAX_CONNECTIONS = _int("GW_MAX_CONNECTIONS", 32)

# --- слежение за шлюзом ---
WATCHDOG_INTERVAL = _float("WATCHDOG_INTERVAL", 15.0)
WATCHDOG_FAILURES = _int("WATCHDOG_FAILURES", 2)      # два отказа подряд -> degraded_lexical

# --- публичная авторизация ---
# Пусто = проверка выключена (локальная отладка). В отличие от ADMIN_SECRET это
# допустимо: рабочее место оператора живёт во внутренней сети за реверс-прокси.
API_TOKEN = os.getenv("API_TOKEN", "") or ""

# --- распознавание речи ---
# Пауза, которую VAD выжидает, прежде чем закрыть реплику. 0 = не переопределять,
# брать настройку шлюза (по умолчанию 200 мс).
#
# Это единственный параметр VAD, который стоит времени: чистое ожидание поверх
# уже сказанного, целиком входящее в задержку хода. Замерено на 37 с живого
# русского диалога: 200 мс -> 10 реплик, 100 мс -> 12 реплик при той же речи.
# Сэкономленные 100 мс покупаются ростом числа реплик на 20%, а каждая реплика —
# это вызов извлечения фактов, который на боевой карте занимал 5.2 с. Поэтому
# по умолчанию НЕ переопределяем, а параметр даём, чтобы можно было померить.
ASR_SILENCE_MS = _int("ASR_SILENCE_MS", 0)
ASR_SILENCE_LIMITS = (80, 1000)

# --- диалог ---
SEARCH_K = _int("SEARCH_K", 8)
SEARCH_POOL = _int("SEARCH_POOL", 30)
SEARCH_WORKERS = _int("SEARCH_WORKERS", 4)            # столько же соединений к PG держит v2
LLM_MAX_CONCURRENT = _int("LLM_MAX_CONCURRENT", 32)
LLM_STALE_AFTER = _float("LLM_STALE_AFTER", 6.0)

# --- ПДн ---
PURGE_INTERVAL = _float("PURGE_INTERVAL", 24 * 3600.0)
PURGE_DAYS = _int("PURGE_DAYS", 30)

# --- админ-роут ---
ADMIN_RATE_LIMIT = _int("ADMIN_RATE_LIMIT", 10)       # попыток в минуту с одного адреса
ADMIN_RATE_WINDOW = _float("ADMIN_RATE_WINDOW", 60.0)
# Откуда берётся «один адрес» для лимита. По умолчанию — адрес сокета.
# X-Forwarded-For учитывается только при явном ADMIN_TRUST_PROXY=1: заголовок
# ставит кто угодно, и слепое доверие к нему превращает лимит в декорацию —
# достаточно менять его значение на каждой попытке. За реверс-прокси все запросы
# придут с одного адреса, то есть лимит станет строже, а не слабее.
ADMIN_TRUST_PROXY = (os.getenv("ADMIN_TRUST_PROXY", "") or "").lower() in ("1", "true", "yes")
ADMIN_SECRET_MIN_LEN = 32

# Источники, которым разрешён доступ из браузера. Рабочее место оператора —
# отдельное SPA, оно ходит сюда с другого порта, и без этого списка браузер
# режет любой запрос. Звёздочка недопустима: сюда ходят с Bearer-токеном.
CORS_ORIGINS = [o.strip() for o in (os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173") or "").split(",") if o.strip()]

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
PUBLIC_WS_BASE = os.getenv("PUBLIC_WS_BASE", "") or ""   # напр. wss://mfc.example.ru/api


class ConfigError(RuntimeError):
    """Процесс не должен подниматься с таким окружением."""


def admin_secret():
    """ADMIN_SECRET обязателен и не короче 32 символов.

    Молчаливый дефолт здесь недопустим: этим секретом закрыт единственный роут,
    который умеет переписать адрес и токен шлюза в базе. Пустой или короткий
    секрет — это не «режим отладки», а открытая дверь, поэтому процесс не стартует.
    """
    s = os.getenv("ADMIN_SECRET", "") or ""
    if not s:
        raise ConfigError(
            "ADMIN_SECRET не задан. Роут /v1/admin/gateway меняет адрес и токен "
            "gpu-шлюза, дефолта у этого секрета быть не может. "
            f"Задайте строку длиной не менее {ADMIN_SECRET_MIN_LEN} символов.")
    if len(s) < ADMIN_SECRET_MIN_LEN:
        raise ConfigError(
            f"ADMIN_SECRET короче {ADMIN_SECRET_MIN_LEN} символов (сейчас {len(s)}).")
    return s


def check_env():
    """Вызывается при импорте приложения — до того, как открыт хоть один порт."""
    admin_secret()
    if not DB:
        raise ConfigError("EMB_DB не задан")
    return True


def fail_fast(exc):
    print(f"core-api: не стартует — {exc}", file=sys.stderr, flush=True)
    raise SystemExit(2)
