#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PID 1 контейнера. Площадка — RunPod: свой entrypoint вместо чужого.
#
# Прошлый образ наследовал entrypoint от vastai/base-image, и на RunPod он
# не работал: ждал окружения vast, а «Container start command» до выполнения
# не доходил вовсе, потому что CMD у образа не было.
# ---------------------------------------------------------------------------
set -euo pipefail

VENV="${MFC_VENV:-/opt/mfc/venv}"
ROOT="${MFC_ROOT:-/opt/mfc}"
LOGS="${MFC_LOGS:-/workspace/logs}"
mkdir -p "$LOGS" "${HF_HOME:-/workspace/hf}" /workspace/.cache

# Скомпилированные ядра — на том, а не на временный диск контейнера.
# FlashInfer кэширует результат JIT в ~/.cache; без этого компиляция
# повторялась бы при каждом запуске пода и стоила бы минуты простоя.
if [ ! -L /root/.cache ]; then
  [ -d /root/.cache ] && cp -a /root/.cache/. /workspace/.cache/ 2>/dev/null || true
  rm -rf /root/.cache
  ln -s /workspace/.cache /root/.cache
fi

# SSH: RunPod кладёт публичный ключ в PUBLIC_KEY. Без ключа sshd не поднимаем —
# демон с пустым authorized_keys на открытом порту не нужен никому.
if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
  ssh-keygen -A >/dev/null 2>&1 || true
  mkdir -p /run/sshd
  /usr/sbin/sshd
  echo "sshd поднят, ключ из PUBLIC_KEY принят"
fi

# Токен живёт на томе: он переживает перезапуск пода, и core-api
# не приходится перенастраивать после каждого рестарта.
if [ -z "${GW_TOKEN:-}" ]; then
  if [ -f /workspace/.gw_token ]; then
    GW_TOKEN="$(cat /workspace/.gw_token)"
  else
    GW_TOKEN="$("$VENV/bin/python" -c 'import secrets;print(secrets.token_urlsafe(32))')"
    echo "$GW_TOKEN" > /workspace/.gw_token
    chmod 600 /workspace/.gw_token
  fi
fi
export GW_TOKEN

cat > /workspace/supervisord.conf <<CONF
[supervisord]
nodaemon=true
logfile=$LOGS/supervisord.log
pidfile=/workspace/supervisord.pid
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
environment=HF_HOME="${HF_HOME:-/workspace/hf}",MFC_LOGS="$LOGS",GW_TOKEN="$GW_TOKEN",GW_PORT="${GW_PORT:-10100}",VLLM_PORT="${VLLM_PORT:-8901}",LLM_MODEL="${LLM_MODEL:-}",GPU_UTIL="${GPU_UTIL:-}",MAX_LEN="${MAX_LEN:-16384}",MAX_SEQS="${MAX_SEQS:-32}",PATH="$VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
CONF

echo "=================================================="
echo " статус:  http://<host>:${GW_PORT:-10100}/"
echo " токен:   $GW_TOKEN"
echo " логи:    $LOGS/"
echo "=================================================="

# Шлюз под supervisord, а не напрямую: его падение чинится перезапуском
# процесса за секунды, а не перезапуском всего пода.
exec "$VENV/bin/supervisord" -c /workspace/supervisord.conf
