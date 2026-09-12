#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI поверх gateway.boot.resolve_llm_model — для onstart.sh и ручной проверки.

    python3 resolve_model.py                      # перебрать кандидатов
    LLM_MODEL=org/repo python3 resolve_model.py   # проверить конкретный
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gateway.boot import resolve_llm_model


def main():
    repo, tried = resolve_llm_model(os.getenv("LLM_MODEL", "").strip())
    if repo:
        print(repo)
        return 0
    sys.stderr.write(
        "Не найден ни один кандидат: " + ", ".join(tried) + "\n"
        "Задайте LLM_MODEL=<org/repo> явно.\n"
        "Если официального FP8-чекпойнта нет — квантуйте сами:\n"
        "  pip install llmcompressor   (рецепт FP8 W8A8, ~30 мин на 5090)\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
