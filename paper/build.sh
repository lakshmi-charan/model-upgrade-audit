#!/bin/bash
# Build the paper PDF (pandoc + pandoc-crossref + xelatex). Usage: bash paper/build.sh <pandoc-crossref binary>
set -e
cd "$(dirname "$0")"
XREF=${1:-pandoc-crossref}
printf '\n# References {-}\n\n::: {#refs}\n:::\n\n\\appendix\n\n' > /tmp/_refs_block.md
for f in 00_meta.md sections/0[1-7]_*.md /tmp/_refs_block.md sections/08_*.md sections/09_*.md; do cat "$f"; printf "\n\n"; done > paper_full.md
pandoc paper_full.md --columns=150 --filter "$XREF" --citeproc --bibliography refs.bib -M reference-section-title=References \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue -V citecolor=blue \
  -H header.tex -o paper.tex --standalone
sed -i "/usepackage{lmodern}/d" paper.tex; xelatex -interaction=nonstopmode paper.tex > /tmp/_xelatex.log 2>&1 || true
sed -i "/usepackage{lmodern}/d" paper.tex; xelatex -interaction=nonstopmode paper.tex > /tmp/_xelatex.log 2>&1 || true
grep -E "^! |undefined" /tmp/_xelatex.log | head -20 || true
ls -la paper.pdf
