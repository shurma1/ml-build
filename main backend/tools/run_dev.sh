#!/usr/bin/env bash
# Локальный стенд целиком: дублёр шлюза + core-api. Не для продакшна.
#
# Дублёр поднимает НАСТОЯЩИЙ эмбеддер на CPU, поэтому первый старт занимает
# около минуты и требует ~2 ГБ. LLM и ASR в дублёре — заглушки на правилах.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ADMIN_SECRET:?ADMIN_SECRET обязателен, не короче 32 символов}"
export GW_TOKEN="${GW_TOKEN:-dev-token-abc}"
export STUB2_TOKEN="${STUB2_TOKEN:-second-token-xyz}"
export EMB_DB="${EMB_DB:-host=localhost port=5434 dbname=embeddings user=emb password=emb}"
export ADMIN_TRUST_PROXY="${ADMIN_TRUST_PROXY:-1}"
LOG="${LOG_DIR:-/tmp/mfc}"; mkdir -p "$LOG"

echo "поднимаю дублёры шлюза…"
python3 tools/stub_gateway.py --port 10100 --token "$GW_TOKEN"     > "$LOG/stub.log"  2>&1 &
python3 tools/stub_gateway.py --port 10101 --token "$STUB2_TOKEN"  > "$LOG/stub2.log" 2>&1 &
python3 tools/stub_gateway.py --port 10102 --token t2 --no-model \
        --fingerprint "ЧУЖАЯ-МОДЕЛЬ/512/norm/deadbeef"             > "$LOG/stub_bad.log" 2>&1 &

echo "жду загрузки эмбеддера (до ~90 с)…"
for _ in $(seq 1 90); do
  curl -sf http://127.0.0.1:10100/readyz >/dev/null 2>&1 && break || sleep 1
done

export GW_BASE_URL="${GW_BASE_URL:-http://127.0.0.1:10100}"
echo "поднимаю core-api на :${PORT:-8080}…"
exec python3 app.py
