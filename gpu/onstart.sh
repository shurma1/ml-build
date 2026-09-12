#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Развёртывание gpu-gateway БЕЗ собственного образа — поверх голого
# vastai/base-image. Путь для первого разведочного запуска: видно, где именно
# спотыкается сборка.
#
# Когда образ собран и опубликован (см. build_push.sh), этот скрипт не нужен:
# в шаблоне меняется Image Path, а On-start Script становится
#     entrypoint.sh; /opt/mfc/start.sh
#
# Запуск по ssh:  bash /workspace/mfc/gpu/onstart.sh
# Идемпотентен. Логи: /workspace/logs/. Наружу — только порт 10100.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${MFC_ROOT:-/workspace/mfc}"
VENV="${MFC_VENV:-/workspace/venv}"
LOGS="${MFC_LOGS:-/workspace/logs}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1 MFC_LOGS="$LOGS"

mkdir -p "$LOGS" "$HF_HOME"
say() { echo -e "\n=== $* ===" ; }

say "железо"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || {
  echo "nvidia-smi недоступен — это не GPU-инстанс"; exit 1; }

say "место на диске"
df -h /workspace | tail -1
FREE_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt 40 ]; then
  echo "!! свободно ${FREE_GB} ГБ, а нужно ~36 только под окружение и веса."
  echo "!! пересоздайте инстанс с --disk 100, иначе загрузка встанет на весах."
fi

# --- окружение ---------------------------------------------------------------
[ -d "$VENV" ] || { say "создаю venv в $VENV"; python3 -m venv "$VENV"; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel

# Фаза 1: vllm первым — он выбирает версию torch под нужную CUDA.
if ! python -c "import vllm" 2>/dev/null; then
  say "фаза 1: vllm (тянет torch под CUDA 12.8, это долго)"
  python -m pip install "vllm>=0.9.0"
fi
TORCH_BEFORE="$(python -c 'import torch; print(torch.__version__)')"
echo "torch: $TORCH_BEFORE"

# Фаза 2: остальное. Молчаливая подмена torch ломает ядра FP8 под sm_120,
# причём не при установке, а под нагрузкой — поэтому сверяем здесь.
say "фаза 2: остальные зависимости"
python -m pip install -q -r "$ROOT/gpu/requirements.txt"
TORCH_AFTER="$(python -c 'import torch; print(torch.__version__)')"
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "!! torch подменён: $TORCH_BEFORE -> $TORCH_AFTER, переустанавливаю vllm"
  python -m pip install --force-reinstall "vllm>=0.9.0"
fi
python -c "import vllm,torch,sentence_transformers,silero_vad,soxr;print('стек собран, torch',torch.__version__)"

# --- токен -------------------------------------------------------------------
if [ -z "${GW_TOKEN:-}" ]; then
  if [ -f /workspace/.gw_token ]; then GW_TOKEN="$(cat /workspace/.gw_token)"
  else GW_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
       echo "$GW_TOKEN" > /workspace/.gw_token; chmod 600 /workspace/.gw_token; fi
fi
export GW_TOKEN

# --- запуск ------------------------------------------------------------------
# Программа одна. Веса качает и vLLM запускает сам шлюз: так весь прогресс
# собирается в одном месте и виден на странице статуса с первой секунды,
# а не через двадцать минут молчания.
cat > /workspace/supervisord.conf <<CONF
[supervisord]
nodaemon=true
logfile=$LOGS/supervisord.log
pidfile=/workspace/supervisord.pid
childlogdir=$LOGS

[program:gateway]
command=$VENV/bin/uvicorn gateway.app:app --host 0.0.0.0 --port ${GW_PORT:-10100} --ws-ping-interval 20
directory=$ROOT/gpu
autorestart=true
startsecs=10
stopasgroup=true
killasgroup=true
stdout_logfile=$LOGS/gateway.log
stderr_logfile=$LOGS/gateway.log
stdout_logfile_maxbytes=50MB
environment=HF_HOME="$HF_HOME",MFC_LOGS="$LOGS",GW_TOKEN="$GW_TOKEN",GW_PORT="${GW_PORT:-10100}",VLLM_PORT="${VLLM_PORT:-8901}",LLM_MODEL="${LLM_MODEL:-}",GPU_UTIL="${GPU_UTIL:-0.72}",MAX_LEN="${MAX_LEN:-16384}",MAX_SEQS="${MAX_SEQS:-32}",PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
CONF

say "старт"
echo "статус: http://<host>:${GW_PORT:-10100}/   токен: $GW_TOKEN"
echo "веса начнут качаться через несколько секунд — прогресс на странице статуса"
exec "$VENV/bin/supervisord" -c /workspace/supervisord.conf
