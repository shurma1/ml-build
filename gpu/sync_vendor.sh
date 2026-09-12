#!/usr/bin/env bash
# Схема извлечения — одна на весь проект. Здесь она КОПИЯ, а не форк.
#
# Почему копия, а не импорт: шлюз едет на арендованную машину без корпуса услуг
# и без кода поиска. Почему с манифестом: расхождение копии с оригиналом обязано
# быть видно, а не обнаруживаться по странным ответам модели через неделю.
# Контрольные суммы отдаются в /v1/models, core-api сверяет их при старте.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/../embending/v2" && pwd)"
DST="$(cd "$(dirname "$0")" && pwd)/gateway/vendor"
mkdir -p "$DST"
touch "$DST/__init__.py"
: > "$DST/MANIFEST"
for f in extraction_schema.py facets.py; do
  cp "$SRC/$f" "$DST/$f"
  shasum -a 256 "$DST/$f" | awk -v n="$f" '{print n" "$1}' >> "$DST/MANIFEST"
done
echo "синхронизировано из $SRC:"
cat "$DST/MANIFEST"
