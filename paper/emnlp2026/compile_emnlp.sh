#!/bin/bash
cd /data/sony/LFCRASH/LFCRASH-CBM/paper/emnlp2026
rm -f insight_emnlp.aux insight_emnlp.bbl insight_emnlp.blg insight_emnlp.log insight_emnlp.out
pdflatex -interaction=batchmode insight_emnlp.tex </dev/null
bibtex insight_emnlp
pdflatex -interaction=batchmode insight_emnlp.tex </dev/null
pdflatex -interaction=batchmode insight_emnlp.tex </dev/null
pdflatex -interaction=batchmode insight_emnlp.tex </dev/null
python - <<'PY'
from pathlib import Path

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    from PyPDF2 import PdfReader, PdfWriter

src = Path("insight_emnlp.pdf")
dst = Path("insight_emnlp_first25.pdf")
reader = PdfReader(str(src))
writer = PdfWriter()
for page in reader.pages[:25]:
    writer.add_page(page)
with dst.open("wb") as f:
    writer.write(f)
print(f"FIRST25_DONE pages={min(25, len(reader.pages))}/{len(reader.pages)}")
PY
ls -lh insight_emnlp.pdf insight_emnlp_first25.pdf
printf 'COMPILE_DONE\n'
