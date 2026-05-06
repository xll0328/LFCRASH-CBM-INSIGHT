#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode insight_emnlp_arr8.tex >/tmp/insight_emnlp_arr8_pdflatex1.log
bibtex insight_emnlp_arr8 >/tmp/insight_emnlp_arr8_bibtex.log
pdflatex -interaction=nonstopmode insight_emnlp_arr8.tex >/tmp/insight_emnlp_arr8_pdflatex2.log
pdflatex -interaction=nonstopmode insight_emnlp_arr8.tex >/tmp/insight_emnlp_arr8_pdflatex3.log

python3 - <<'PY'
from pathlib import Path

import fitz

pdf = Path("insight_emnlp_arr8.pdf")
doc = fitz.open(pdf)
section_pages = {}
for index, page in enumerate(doc, start=1):
    text = page.get_text("text")
    for marker in ("Abstract", "Introduction", "Related Work", "Method", "Experiments", "Conclusion", "Limitations", "References", "Extended Analyses"):
        if marker in text and marker not in section_pages:
            section_pages[marker] = index

print(f"ARR8_DONE pages={doc.page_count}")
for marker, page in section_pages.items():
    print(f"{marker}: page {page}")
PY

ls -lh insight_emnlp_arr8.pdf
