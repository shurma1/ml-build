# -*- coding: utf-8 -*-
"""Конфигурация шлюза. Всё — через окружение, ничего не зашито в код.

Порты выбраны из тех, что открывает шаблон vast.ai:
    10100 — шлюз, единственное, что смотрит наружу
    8901  — vLLM, только 127.0.0.1, наружу не выставляется
"""
import os

# --- сеть ---
PORT          = int(os.getenv("GW_PORT", "10100"))
VLLM_PORT     = int(os.getenv("VLLM_PORT", "8901"))
VLLM_URL      = os.getenv("VLLM_URL", f"http://127.0.0.1:{VLLM_PORT}")
TOKEN         = os.getenv("GW_TOKEN", "")          # пусто = авторизация выключена (только для локальной отладки)

# --- модели ---
ASR_REPO      = os.getenv("ASR_REPO", "ai-sage/GigaAM-v3")
ASR_VARIANT   = os.getenv("ASR_VARIANT", "e2e_rnnt")
EMB_MODEL     = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-large-instruct")
EMB_MAX_SEQ   = int(os.getenv("EMB_MAX_SEQ", "512"))
LLM_MODEL     = os.getenv("LLM_MODEL", "")         # проставляется onstart.sh после резолва репозитория

# --- префиксы эмбеддера ---
# ЖИВУТ ЗДЕСЬ И БОЛЬШЕ НИГДЕ. Если вызывающий код начнёт добавлять префикс сам,
# индекс и запрос однажды разъедутся молча.
# Цена рассогласования перемерена на нынешней конфигурации и невелика: двойной префикс -0.7 п.п.
# R@1, отсутствие префикса -0.9, чужой формат -1.0 — все три в пределах доверительного интервала.
# Стоявшая здесь цифра -19 п.п.
# относится к другому замеру (Giga-480M на плоском индексе из 732 карточек, embending/RESULTS.md), а не к mE5-large-instruct на 321 типе.
# Инвариант это не отменяет: он стоит дёшево, а рассогласование по выдаче не видно.
_INSTR = ("Given a Russian citizen's question about government services, "
          "retrieve the matching official service description")
QUERY_PREFIX = os.getenv("EMB_QUERY_PREFIX",
                         f"Instruct: {_INSTR}\nQuery: " if "instruct" in EMB_MODEL else "")
DOC_PREFIX   = os.getenv("EMB_DOC_PREFIX", "")

# --- аудио ---
SAMPLE_RATE   = 16000
BLOCK         = 512                      # 32 мс на шаг VAD
MIN_LEN       = int(0.15 * SAMPLE_RATE)  # сегменты короче отбрасываем
PREROLL       = int(0.12 * SAMPLE_RATE)
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "200"))
VAD_PAD_MS    = int(os.getenv("VAD_PAD_MS", "100"))
MAX_SEG_SEC   = float(os.getenv("MAX_SEG_SEC", "12"))
MAX_PENDING   = int(os.getenv("ASR_MAX_PENDING", "3"))   # глубже очереди — сегмент устарел

# --- LLM ---
LLM_MAX_TOKENS_EXTRACT = int(os.getenv("LLM_MAX_TOKENS_EXTRACT", "320"))
LLM_MAX_TOKENS_ASK     = int(os.getenv("LLM_MAX_TOKENS_ASK", "700"))
LLM_TIMEOUT            = float(os.getenv("LLM_TIMEOUT", "30"))
# Жадное декодирование. Не «на всякий случай»: на этом корпусе температура 0.7
# давала 17-18% расхождений между прогонами на одном и том же диалоге.
LLM_TEMPERATURE        = float(os.getenv("LLM_TEMPERATURE", "0"))
