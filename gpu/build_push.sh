#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Сборка и публикация образа в Docker Hub.
#
#   DOCKERHUB_USER=<логин> ./build_push.sh [тег]
#
# Целевая платформа — linux/amd64: машины vast.ai с RTX 5090 все x86-64,
# а колёса torch/vllm под CUDA существуют только для неё.
#
# На Apple Silicon сборка идёт через эмуляцию QEMU. Это работает, потому что
# тяжёлая часть — распаковка готовых колёс, а не компиляция, но занимает
# порядка часа. На x86-64 Linux — минут пятнадцать.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

USER_NAME="${DOCKERHUB_USER:-}"
TAG="${1:-cuda12.8-v1}"
IMAGE="${USER_NAME}/mfc-gpu-gateway"
PLATFORM="${PLATFORM:-linux/amd64}"

if [ -z "$USER_NAME" ]; then
  echo "Не задан DOCKERHUB_USER." >&2
  echo "  DOCKERHUB_USER=имя ./build_push.sh" >&2
  exit 2
fi

echo "образ:     $IMAGE:$TAG"
echo "платформа: $PLATFORM"
echo "хост:      $(uname -m)"
[ "$(uname -m)" = "arm64" ] && [ "$PLATFORM" = "linux/amd64" ] && \
  echo "внимание:  кросс-сборка через QEMU, ориентировочно час"

# Проверяем логин ДО часовой сборки, а не после неё.
# Флага --get-login у docker нет; имя лежит либо в хелпере учётных данных
# (на macOS это кейчейн), либо base64 в config.json.
hub_login() {
  local cfg="${DOCKER_CONFIG:-$HOME/.docker}/config.json"
  [ -f "$cfg" ] || return 0
  local store
  store="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('credsStore') or '')" "$cfg" 2>/dev/null)"
  if [ -n "$store" ] && command -v "docker-credential-$store" >/dev/null 2>&1; then
    echo "https://index.docker.io/v1/" | "docker-credential-$store" get 2>/dev/null \
      | python3 -c "import json,sys;print(json.load(sys.stdin).get('Username',''))" 2>/dev/null
  else
    python3 - "$cfg" <<'PYEOF' 2>/dev/null
import base64, json, sys
a = (json.load(open(sys.argv[1])).get("auths") or {}).get("https://index.docker.io/v1/") or {}
t = a.get("auth")
print(base64.b64decode(t).decode().split(":")[0] if t else (a.get("identitytoken") and "<token>") or "")
PYEOF
  fi
}

LOGGED="$(hub_login || true)"
if [ -z "$LOGGED" ]; then
  echo "Похоже, вы не залогинены в Docker Hub: docker login" >&2
  exit 2
fi
if [ "$LOGGED" != "$USER_NAME" ] && [ "$LOGGED" != "<token>" ]; then
  echo "Внимание: залогинен '$LOGGED', а публикуем в '$USER_NAME' — push упадёт по правам." >&2
  exit 2
fi
echo "логин:     $LOGGED"

# buildx с отдельным билдером: у стандартного драйвера нет кросс-платформенной
# сборки и кэша между запусками.
docker buildx inspect mfc >/dev/null 2>&1 || docker buildx create --name mfc --use --bootstrap
docker buildx use mfc

# Кэш в реестр удваивает заливку, а на первой сборке всё равно промахивается.
# Включается явно: PUSH_CACHE=1 ./build_push.sh
CACHE_ARGS=()
if [ "${PUSH_CACHE:-0}" = "1" ]; then
  CACHE_ARGS=(--cache-from "type=registry,ref=$IMAGE:buildcache"
              --cache-to   "type=registry,ref=$IMAGE:buildcache,mode=max")
fi

docker buildx build \
  --platform "$PLATFORM" \
  --tag "$IMAGE:$TAG" \
  --tag "$IMAGE:latest" \
  ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} \
  --provenance=false \
  --push \
  .

echo
echo "опубликовано: $IMAGE:$TAG"
echo
echo "В шаблон RunPod:"
echo "  Container image           $IMAGE:$TAG"
echo "  Container start command   (оставить пустым — ENTRYPOINT в образе)"
echo "  Template type / Compute   Pod / GPU"
echo "  Expose HTTP Ports         10100"
echo "  Expose TCP Ports          22"
echo "  Container Disk            30 ГБ"
echo "  Volume Disk               60 ГБ на /workspace"
echo "  Env                       GW_TOKEN=<длинная случайная строка>"
