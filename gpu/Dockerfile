# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# gpu-gateway: ASR + эмбеддер + vLLM в одном образе. Площадка — RunPod.
#
# База нейтральная, а не vastai/base-image: тот тянул Jupyter, Syncthing
# и Tensorboard, которых мы не используем, и навязывал свой entrypoint,
# на RunPod неработоспособный. CUDA 13 выбрана под наш стек — pip ставит
# torch с рантаймом cu130, и база должна ему соответствовать.
#
# Веса моделей в образ НЕ кладём: 14 лишних гигабайт слоёв превратили бы
# каждый перезапуск пода в ожидание, а смена модели требовала бы пересборки.
# Они качаются при первом старте в /workspace/hf — на том, который переживает
# перезапуск контейнера.
# ---------------------------------------------------------------------------
ARG BASE=nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04
FROM ${BASE}

LABEL org.opencontainers.image.title="mfc-gpu-gateway" \
      org.opencontainers.image.description="ASR (GigaAM-v3) + эмбеддер (mE5) + LLM (vLLM) за одним портом"

ENV DEBIAN_FRONTEND=noninteractive \
    MFC_ROOT=/opt/mfc \
    MFC_VENV=/opt/mfc/venv \
    HF_HOME=/workspace/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MFC_LOGS=/workspace/logs \
    GW_PORT=10100 \
    VLLM_PORT=8901 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# nvcc нужен не для сборки, а в рантайме: FlashInfer компилирует ядро под
# конкретную комбинацию (у нас sm_89 + KV в e4m3 + head_dim 256) прямо при
# старте, готового в колесе нет. База runtime компилятора не содержит —
# первый запуск падал на «/usr/local/cuda/bin/nvcc: not found».
# Ставим сам nvcc и заголовки CUDA, а не переходим на базу devel: так дешевле.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential \
      cuda-nvcc-13-0 cuda-cudart-dev-13-0 \
      curl ca-certificates git openssh-server \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/mfc/venv \
 && /opt/mfc/venv/bin/pip install -U pip wheel

# --- фаза 1: vllm СТРОГО первым ---------------------------------------------
# Он выбирает версию torch под нужную CUDA. Всё остальное ставится поверх
# и обязано с ней согласиться, а не наоборот.
RUN /opt/mfc/venv/bin/pip install "vllm>=0.9.0" \
 && /opt/mfc/venv/bin/python -c "import torch,pathlib;pathlib.Path('/opt/mfc/.torch').write_text(torch.__version__)"

# --- фаза 2: остальное, со сверкой что torch не подменили --------------------
# Молчаливая подмена версии ломает ядра FP8, и проявляется это не при установке,
# а падением под нагрузкой. Ловим здесь, на сборке.
COPY requirements.txt /opt/mfc/requirements.txt
RUN /opt/mfc/venv/bin/pip install -r /opt/mfc/requirements.txt \
 && BEFORE="$(cat /opt/mfc/.torch)" \
 && AFTER="$(/opt/mfc/venv/bin/python -c 'import torch;print(torch.__version__)')" \
 && if [ "$BEFORE" != "$AFTER" ]; then \
      echo "!! torch подменён: $BEFORE -> $AFTER, переустанавливаю vllm"; \
      /opt/mfc/venv/bin/pip install --force-reinstall "vllm>=0.9.0"; \
    fi \
 && /opt/mfc/venv/bin/python -c "import vllm,torch,sentence_transformers,silero_vad,soxr;print('стек собран, torch',torch.__version__)"

# --- код ---------------------------------------------------------------------
COPY gateway/     /opt/mfc/gateway/
COPY tests/       /opt/mfc/tests/
COPY smoke.py resolve_model.py sync_vendor.sh /opt/mfc/
COPY docker/entrypoint.sh /opt/mfc/entrypoint.sh
RUN chmod +x /opt/mfc/entrypoint.sh \
 && /opt/mfc/venv/bin/python -m compileall -q /opt/mfc/gateway \
 && mkdir -p /workspace/logs /workspace/hf

EXPOSE 10100
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${GW_PORT}/healthz" || exit 1

ENTRYPOINT ["/opt/mfc/entrypoint.sh"]
