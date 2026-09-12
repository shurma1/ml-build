#!/usr/bin/env bash
# Запуск шлюза внутри контейнера. Ставится в On-start Script инстанса vast.
#
# Идемпотентен: если supervisord уже поднят, второй запуск ничего не делает.
# Возвращает управление сразу — vast ждёт завершения on-start скрипта,
# и висеть в нём нельзя.
set -euo pipefail

VENV="${MFC_VENV:-/opt/mfc/venv}"
ROOT="${MFC_ROOT:-/opt/mfc}"
LOGS="${MFC_LOGS:-/workspace/logs}"
PIDFILE=/workspace/supervisord.pid
mkdir -p "$LOGS"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "шлюз уже запущен (pid $(cat "$PIDFILE"))"
  exit 0
fi

# Токен: один на жизнь тома, чтобы core-api не перенастраивать после рестарта.
if [ -z "${GW_TOKEN:-}" ]; then
  if [ -f /workspace/.gw_token ]; then
    GW_TOKEN="$(cat /workspace/.gw_token)"
  else
    GW_TOKEN="$("$VENV/bin/python" -c 'import secrets;print(secrets.token_urlsafe(32))')"
    echo "$GW_TOKEN" > /workspace/.gw_token
    chmod 600 /workspace/.gw_token
  fi
fi

cat > /workspace/supervisord.conf <<CONF
[supervisord]
logfile=$LOGS/supervisord.log
pidfile=$PIDFILE
childlogdir=$LOGS

[program:gateway]
command=$VENV/bin/uvicorn gateway.app:app --host 0.0.0.0 --port ${GW_PORT:-10100} --ws-ping-interval 20
directory=$ROOT
autorestart=true
startsecs=10
stopasgroup=true
killasgroup=true
stdout_logfile=$LOGS/gateway.log
stderr_logfile=$LOGS/gateway.log
stdout_logfile_maxbytes=50MB
environment=HF_HOME="${HF_HOME:-/workspace/hf}",MFC_LOGS="$LOGS",GW_TOKEN="$GW_TOKEN",GW_PORT="${GW_PORT:-10100}",VLLM_PORT="${VLLM_PORT:-8901}",LLM_MODEL="${LLM_MODEL:-}",GPU_UTIL="${GPU_UTIL:-0.72}",MAX_LEN="${MAX_LEN:-16384}",MAX_SEQS="${MAX_SEQS:-32}",PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
CONF

# Только шлюз. vLLM он запускает сам дочерним процессом — так весь прогресс
# загрузки собирается в одном месте и виден на странице статуса.
"$VENV/bin/supervisord" -c /workspace/supervisord.conf

echo "шлюз поднимается на :${GW_PORT:-10100}"
echo "статус:  http://<host>:${GW_PORT:-10100}/"
echo "токен:   $GW_TOKEN"
