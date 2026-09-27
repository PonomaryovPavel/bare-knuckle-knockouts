#!/usr/bin/env bash
# Полный прогон: выгрузка википедии -> данные -> модели -> графики -> текст.
set -euo pipefail
cd "$(dirname "$0")"
python3 src/01_parse_wikitext.py
python3 src/02_build_panel.py
python3 src/03_model.py
python3 src/04_compare.py
python3 src/05_figures.py
python3 src/06_readme.py
python3 -m pytest tests -q
