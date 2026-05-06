#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 make_emnlp_polished_figures.py

pdflatex -interaction=nonstopmode insight_emnlp_polished.tex >/tmp/insight_emnlp_polished_pdflatex1.log
bibtex insight_emnlp_polished >/tmp/insight_emnlp_polished_bibtex.log
pdflatex -interaction=nonstopmode insight_emnlp_polished.tex >/tmp/insight_emnlp_polished_pdflatex2.log
pdflatex -interaction=nonstopmode insight_emnlp_polished.tex >/tmp/insight_emnlp_polished_pdflatex3.log

python3 - <<'PY'
from pathlib import Path
import fitz

pdf = Path("insight_emnlp_polished.pdf")
doc = fitz.open(pdf)
section_pages = {}
for index, page in enumerate(doc, start=1):
    text = page.get_text("text")
    for marker in (
        "Abstract",
        "Introduction",
        "Related Work",
        "Method",
        "Experiments",
        "Conclusion",
        "Limitations",
        "References",
        "Extended Analyses",
    ):
        if marker in text and marker not in section_pages:
            section_pages[marker] = index

print(f"POLISHED_DONE pages={doc.page_count}")
for marker, page in section_pages.items():
    print(f"{marker}: page {page}")
PY

ls -lh insight_emnlp_polished.pdf
