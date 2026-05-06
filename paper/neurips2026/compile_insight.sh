#!/bin/bash
cd /data/sony/LFCRASH/LFCRASH-CBM/paper/neurips2026
pdflatex -interaction=batchmode insight_main.tex </dev/null
bibtex insight_main
pdflatex -interaction=batchmode insight_main.tex </dev/null
pdflatex -interaction=batchmode insight_main.tex </dev/null
ls -lh insight_main.pdf
echo 'COMPILE_DONE'
