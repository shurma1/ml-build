# -*- coding: utf-8 -*-
"""Загрузка: фазы, прогресс и надзор за vLLM.

Почему шлюз поднимается ПЕРВЫМ и сам запускает vLLM, а не стоит с ним рядом
под supervisord: иначе первые двадцать минут жизни инстанса наружу не смотрит
ничего. Веса качаются, графы компилируются, а оператор видит connection refused
и не знает, идёт работа или всё встало. Здесь порядок обратный — сначала
отвечает статус, потом всё остальное.

Порядок фаз задан не удобством, а видеопамятью: vLLM профилирует память при
старте и забирает свою долю первым. ASR и эмбеддер грузятся в остаток.
"""
import asyncio
import os
import threading
import re
import shutil
import subprocess
import time
from collections import deque

from . import config as C

LOG_DIR = os.getenv("MFC_LOGS", "/workspace/logs")


class Phase:
    __slots__ = ("key", "title", "state", "progress", "detail", "t0", "t1", "bytes", "total")

    def __init__(self, key, title):
        self.key, self.title = key, title
        self.state = "pending"            # pending | running | done | failed | skipped
        self.progress = 0.0               # 0..1, -1 = неизвестно
        self.detail = ""
        self.t0 = self.t1 = None
        self.bytes = self.total = 0

    def dict(self):
        d = {"key": self.key, "title": self.title, "state": self.state,
             "progress": round(self.progress, 4), "detail": self.detail}
        if self.total:
            d["bytes"], d["total_bytes"] = self.bytes, self.total
        if self.t0:
            d["elapsed_s"] = round((self.t1 or time.time()) - self.t0, 1)
        return d


class Boot:
    """Состояние загрузки. Читается из /v1/status, рисуется на странице статуса."""

    PLAN = [
        ("gpu",            "Проверка карты"),
        ("download_llm",   "Веса LLM"),
        ("download_asr",   "Веса распознавания речи"),
        ("download_embed", "Веса эмбеддера"),
        ("vllm",           "Запуск vLLM"),
        ("load_asr",       "Загрузка и прогрев GigaAM"),
        ("load_embed",     "Загрузка и прогрев эмбеддера"),
        ("probe_llm",      "Проверка схемы и модели"),
    ]

    def __init__(self):
        self.phases = {k: Phase(k, t) for k, t in self.PLAN}
        self.order = [k for k, _ in self.PLAN]
        self.started = time.time()
        self.error = None
        self.gpu = {}
        self.vllm_proc = None
        self.vllm_tail = deque(maxlen=400)

    # --- управление фазами ---
    def begin(self, key, detail=""):
        p = self.phases[key]
        p.state, p.t0, p.detail, p.progress = "running", time.time(), detail, 0.0
        return p

    def done(self, key, detail=""):
        p = self.phases[key]
        p.state, p.t1, p.progress = "done", time.time(), 1.0
        if detail:
            p.detail = detail

    def fail(self, key, detail):
        p = self.phases[key]
        p.state, p.t1, p.detail = "failed", time.time(), detail
        self.error = f"{p.title}: {detail}"

    def skip(self, key, detail=""):
        p = self.phases[key]
        p.state, p.detail, p.progress = "skipped", detail, 1.0

    @property
    def ready(self):
        return all(self.phases[k].state in ("done", "skipped") for k in self.order)

    def dict(self, models=None):
        return {
            "ready": self.ready,
            "error": self.error,
            "uptime_s": round(time.time() - self.started),
            "phases": [self.phases[k].dict() for k in self.order],
            "gpu": self.gpu,
            "models": models or {},
        }


# --- прогресс скачивания с HuggingFace ---------------------------------------

def _repo_size(repo, ignore_patterns=None, revision=None):
    """Сколько весит репозиторий на HuggingFace, с учётом исключённых путей."""
    import fnmatch
    from huggingface_hub import HfApi
    try:
        info = HfApi().model_info(repo, files_metadata=True, revision=revision, timeout=30)
    except Exception:
        return 0
    total = 0
    for f in (info.siblings or []):
        if any(fnmatch.fnmatch(f.rfilename, p) for p in (ignore_patterns or [])):
            continue
        total += f.size or 0
    return total


def _local_dir(repo):
    """Куда huggingface_hub кладёт файлы репозитория."""
    home = os.getenv("HF_HOME", "/workspace/hf")
    return os.path.join(home, "hub", "models--" + repo.replace("/", "--"))


def _dir_size(path, follow=False):
    """Размер папки. follow=False меряет ссылки как ссылки.

    Внутри hub-кэша реальные файлы лежат в blobs/, а snapshots/ — это ссылки
    на них. Для прогресса считаем весь кэш репозитория без follow (иначе
    удвоится), для итогового отчёта — тоже его, а не папку снимка: она из
    одних ссылок и давала честные, но бессмысленные 0.0 ГБ.
    """
    stat = os.stat if follow else os.lstat
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def _watch_download(phase: Phase, repo, stop):
    """Прогресс считаем по РАЗМЕРУ ПАПКИ НА ДИСКЕ, а не по прогресс-барам.

    Первая версия подменяла tqdm внутри snapshot_download — и показывала ноль
    всё время загрузки. Свой класс получает только внешний бар со счётчиком
    файлов; побайтовые бары отдельных файлов его не видят, а при включённом
    hf_transfer загрузка вообще идёт в Rust и рапортует мимо tqdm.
    Размер папки не зависит ни от версии библиотеки, ни от способа качать.
    """
    path = _local_dir(repo)
    while not stop.wait(2.0):
        done = _dir_size(path)
        phase.bytes = done
        if phase.total:
            phase.progress = min(0.999, done / phase.total)
            phase.detail = f"{done/2**30:.1f} из {phase.total/2**30:.1f} ГБ"
        else:
            phase.progress = -1.0
            phase.detail = f"{done/2**30:.1f} ГБ"


def download(phase: Phase, repo, **kw):
    phase.total = _repo_size(repo, kw.get("ignore_patterns"), kw.get("revision"))
    phase.detail = (f"{repo} · {phase.total/2**30:.1f} ГБ" if phase.total else repo)
    stop = threading.Event()
    threading.Thread(target=_watch_download, args=(phase, repo, stop), daemon=True).start()
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, max_workers=8, **kw)
    finally:
        stop.set()
    return f"{repo} · {_dir_size(_local_dir(repo))/2**30:.1f} ГБ на диске"


# Варианты GigaAM — это git-РЕВИЗИИ репозитория, а не подкаталоги.
# Значит лишнего не скачается само собой: snapshot_download с revision
# берёт только нужную ветку. Прежний фильтр по подкаталогам был пустышкой.
ASR_VARIANTS = ["e2e_rnnt", "e2e_ctc", "rnnt", "ctc"]


# --- vLLM как дочерний процесс ------------------------------------------------

_VLLM_MARKS = [
    (re.compile(r"Loading safetensors checkpoint shards:\s*(\d+)%"), lambda m: 0.15 + 0.55 * int(m.group(1)) / 100),
    (re.compile(r"Capturing CUDA graph|Capturing cudagraphs"), lambda m: 0.85),
    (re.compile(r"Starting vLLM API server|Uvicorn running"), lambda m: 0.95),
]


class VllmProcess:
    """vLLM подпроцессом. Владелец жизненного цикла — шлюз, а не supervisord.

    Так статус загрузки собирается в одном месте: сколько скачано, на какой
    стадии инициализация, что в последней строке лога. Падение процесса видно
    сразу и попадает на страницу статуса, а не только в файл.

    Необязательные флаги снимаются автоматически. Набор опций vLLM меняется
    от версии к версии (тот же --disable-log-requests успел стать
    --enable-log-requests), и проверка списка руками устаревает к следующему
    релизу. Лучше пережить незнакомый флаг с худшими настройками, чем не
    подняться вовсе на арендованной карте.
    """

    # (имя для сопоставления с текстом ошибки, аргументы)
    OPTIONAL = [
        # Модель мультимодальная, но картинки нам не нужны никогда. Дело не
        # в лишнем гигабайте весов: при старте vLLM профилирует пиковую память
        # на фиктивном запросе и для VL-модели подставляет туда картинку
        # максимального размера. Это раздувает оценку активаций и съедает
        # KV-кэш ещё до первого посетителя.
        ("--limit-mm-per-prompt",   ["--limit-mm-per-prompt", '{"image":0,"video":0}']),
        ("--kv-cache-dtype",        ["--kv-cache-dtype", os.getenv("KV_CACHE_DTYPE", "fp8")]),
        ("--enable-prefix-caching", ["--enable-prefix-caching"]),
        ("--disable-log-requests",  ["--disable-log-requests"]),
    ]
    # Порядок снятия, когда в ошибке НЕТ имени флага: движок принял аргументы
    # и умер позже, на инициализации. Первым идёт то, что чаще всего и ломает —
    # fp8-кэш тянет за собой JIT-компиляцию ядра под конкретную архитектуру.
    # --disable-log-requests сюда не входит: он не трогает ни память, ни ядра.
    RISK_ORDER = ["--kv-cache-dtype", "--limit-mm-per-prompt", "--enable-prefix-caching"]
    _BAD = re.compile(r"unrecognized arguments?:\s*(\S+)|error: argument\s+(\S+?)[:\s]|"
                      r"invalid choice.*?argument\s+(\S+)", re.I)

    def __init__(self, boot: Boot, model: str):
        self.boot, self.model = boot, model
        self.proc = None
        self.dropped = []
        self.log_path = os.path.join(LOG_DIR, "vllm.log")
        self.log = None

    def _cmd(self):
        util, why = auto_gpu_util()
        self.util_note = why
        exe = shutil.which("vllm") or "vllm"
        cmd = [exe, "serve", self.model,
               "--host", "127.0.0.1", "--port", str(C.VLLM_PORT),
               "--gpu-memory-utilization", str(util),
               "--max-model-len", os.getenv("MAX_LEN", "16384"),
               "--max-num-seqs", os.getenv("MAX_SEQS", "32")]
        for name, args in self.OPTIONAL:
            if name not in self.dropped:
                cmd += args
        return cmd

    def start(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        if self.log is None:
            self.log = open(self.log_path, "ab", buffering=0)
        cmd = self._cmd()
        self.boot.vllm_tail.append("$ " + " ".join(cmd))
        env = {**os.environ, "HF_HOME": os.getenv("HF_HOME", "/workspace/hf")}
        # Семплирование штатным PyTorch вместо FlashInfer. Тот компилирует свой
        # модуль при первом запуске, и это минуты простоя плюс лишний класс
        # отказов (нужны заголовки curand). На нашей нагрузке — 80 токенов
        # на извлечение фактов под жёсткой грамматикой — разницы в скорости нет.
        # Переопределяется переменной окружения, если захочется обратно.
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, bufsize=1,
                                     universal_newlines=True, env=env)
        self.boot.vllm_proc = self.proc
        return self.proc

    def failed_flag(self):
        """Какой флаг снять. Сначала ищем имя в тексте ошибки, иначе — по риску.

        Второй путь появился после реального отказа: vLLM принял
        --kv-cache-dtype fp8, а упал позже, когда FlashInfer полез
        компилировать ядро и не нашёл nvcc. Имени флага в трейсбеке не было,
        и прежняя логика, ждавшая слов unrecognized arguments, не сработала.
        """
        named = self._named_bad_flag()
        if named:
            return named
        for name in self.RISK_ORDER:
            if name not in self.dropped:
                return name
        return None

    def _named_bad_flag(self):
        """Флаг, прямо названный в сообщении об ошибке. None, если не назван."""
        for line in list(self.boot.vllm_tail)[-40:]:
            m = self._BAD.search(line)
            if not m:
                continue
            bad = next((g for g in m.groups() if g), "").strip().strip("'\"")
            for name, _ in self.OPTIONAL:
                if bad.startswith(name) or name.startswith(bad):
                    return name
        return None

    async def pump(self, phase: Phase):
        """Читать вывод vLLM: в файл, в кольцевой буфер для UI и в прогресс фазы."""
        loop = asyncio.get_running_loop()
        proc = self.proc
        while True:
            line = await loop.run_in_executor(None, proc.stdout.readline)
            if not line:
                break
            self.log.write(line.encode("utf-8", "replace"))
            self.boot.vllm_tail.append(line.rstrip("\n"))
            for rx, fn in _VLLM_MARKS:
                m = rx.search(line)
                if m:
                    phase.progress = max(phase.progress, fn(m))
                    phase.detail = line.strip()[:160]
                    break

    async def run(self, phase: Phase, timeout=1800):
        """Поднять vLLM, снимая необязательные флаги, если движок их не понял."""
        for attempt in range(len(self.OPTIONAL) + 1):
            self.start()
            asyncio.create_task(self.pump(phase))
            try:
                secs = await self.wait_healthy(phase, timeout)
                note = self.util_note
                if self.dropped:
                    note += " · сняты флаги: " + ", ".join(self.dropped)
                return secs, note
            except RuntimeError:
                bad = self.failed_flag()
                if not bad:
                    raise
                self.dropped.append(bad)
                phase.detail = f"движок не поднялся, снимаю {bad} и пробую снова"
                phase.progress = 0.0
                self.boot.vllm_tail.append(f"!! снят необязательный флаг {bad}")
        raise RuntimeError("vLLM не поднялся даже без необязательных флагов")

    async def wait_healthy(self, phase: Phase, timeout=1800):
        import httpx
        t0 = time.time()
        async with httpx.AsyncClient(base_url=C.VLLM_URL, timeout=5) as c:
            while time.time() - t0 < timeout:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"процесс vLLM завершился с кодом {self.proc.returncode}; "
                        f"последнее: {self.boot.vllm_tail[-1] if self.boot.vllm_tail else '—'}")
                try:
                    if (await c.get("/health")).status_code == 200:
                        return round(time.time() - t0)
                except Exception:
                    pass
                await asyncio.sleep(3)
        raise TimeoutError(f"vLLM не ответил за {timeout} с")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# Сколько видеопамяти оставить СЕБЕ, вне vLLM. Замеренный пик:
# GigaAM с активациями на 12-секундном сегменте ~2.5 ГБ, эмбеддер на батче
# из пяти строк ~1.8 ГБ, контекст CUDA процесса шлюза ~0.6 ГБ, фрагментация ~0.5.
# Сверху запас: OOM у vLLM случается не при старте, а на пике нагрузки.
RESERVE_GB = 7.0


def auto_gpu_util():
    """-> (доля, пояснение). Карта может быть и 24 ГБ, и 32 — фиксированное
    число здесь означало бы либо OOM на младшей, либо потерянный KV-кэш на старшей."""
    forced = os.getenv("GPU_UTIL", "").strip()
    if forced:
        return float(forced), f"задано явно: {forced}"
    total_mb = gpu_info().get("vram_total_mb")
    if not total_mb:
        return 0.70, "карта не определилась, взято безопасное 0.70"
    total = total_mb / 1024
    util = max(0.45, min(0.85, (total - RESERVE_GB) / total))
    return round(util, 2), (f"из {total:.0f} ГБ оставлено {RESERVE_GB:.0f} ГБ "
                            f"под ASR, эмбеддер и запас")


def gpu_info():
    out = {}
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info()
            out = {"name": p.name, "compute_capability": f"{p.major}.{p.minor}",
                   "vram_total_mb": int(total / 2**20),
                   "vram_used_mb": int((total - free) / 2**20),
                   "torch": torch.__version__, "cuda": torch.version.cuda}
        else:
            out = {"name": "CPU", "compute_capability": "—", "torch": torch.__version__}
    except Exception as e:                      # noqa: BLE001
        out = {"error": f"{type(e).__name__}: {e}"}
    return out


def read_log(name, tail=200):
    path = os.path.join(LOG_DIR, f"{name}.log")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        try:
            f.seek(-min(os.path.getsize(path), 256 * 1024), os.SEEK_END)
        except OSError:
            pass
        lines = f.read().decode("utf-8", "replace").splitlines()
    return lines[-tail:]


# --- какой репозиторий у модели -----------------------------------------------

# Точное имя FP8-чекпойнта заранее неизвестно, а ошибка в нём проявляется не при
# старте, а как 400 на каждом запросе — уже после того, как инстанс оплачен.
LLM_CANDIDATES = [
    # RedHatAI — бывшая Neural Magic, авторы llm-compressor и половины кода
    # квантования в самом vLLM. Официального FP8 у Qwen для 3.5-9B нет:
    # есть только bf16 на 18 ГБ, который в 24 ГБ не помещается.
    # У этого кванта 13 ГБ, а не 9.5: зрительная башня намеренно оставлена
    # в bf16 (см. ignore в quantization_config).
    "RedHatAI/Qwen3.5-9B-FP8-dynamic",
    "Qwen/Qwen3.5-9B",          # bf16, 18 ГБ — только для карт от 32 ГБ
    "Qwen/Qwen3.5-4B",          # запасной: меньше и, по замерам, устойчивее на извлечении
]


def resolve_llm_model(forced=""):
    """-> (repo_id, [что пробовали]). Пустая строка, если не нашлось ничего."""
    from huggingface_hub import HfApi
    api = HfApi()
    tried = []
    for repo in ([forced] if forced else LLM_CANDIDATES):
        tried.append(repo)
        try:
            api.model_info(repo, timeout=20)
            return repo, tried
        except Exception:
            continue
    return "", tried
