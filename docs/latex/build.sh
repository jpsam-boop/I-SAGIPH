#!/usr/bin/env bash
# Rebuild the PDF guides from docs/*.md (needs pandoc + XeLaTeX with the Carlito and DejaVu fonts).
set -e
cd "$(dirname "$0")"
for n in code_guide statistical_models; do
  sed '1{/^# /d}' ../$n.md | sed 's/^## References$/## References {.unnumbered}/' > $n.md
  pandoc $n.md -f markdown+tex_math_dollars+pipe_tables+autolink_bare_uris -t latex -s --columns=60 \
    -V documentclass=article -V papersize=a4 -V geometry:margin=2.2cm -V fontsize=10.5pt \
    -V mainfont=Carlito -V sansfont=Carlito -V monofont="DejaVu Sans Mono" -V mathfont="DejaVu Math TeX Gyre" \
    -V linestretch=1.12 --include-in-header=header.tex --include-before-body=title_$n.tex -o $n.tex
  # The methods guide has a long reference list; compact only that section to
  # keep its final references from spilling onto a mostly empty extra page.
  if [ "$n" = "statistical_models" ]; then
    python -c 'from pathlib import Path; p=Path("statistical_models.tex"); s=p.read_text(); marker=r"\subsection*{References}"; s=s.replace(marker, "\\small\n"+marker, 1); p.write_text(s)'
  fi
  xelatex -interaction=nonstopmode $n.tex >/dev/null && xelatex -interaction=nonstopmode $n.tex >/dev/null
  cp $n.pdf ../
done
echo "PDFs written to docs/"
