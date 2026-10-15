#!/usr/bin/env bash
# Renders the two selection-miscalibration panels of the ACCV 2026 supplement
# (Robust Local Optimization Done Right, fig:supp-sel-miscal) into the landing
# assets, each in a light-theme and a dark-theme variant on a transparent
# background:
#
#   docs/src/assets/landing/sel_miscal_{median,maa}_{light,dark}.png
#
# The page shows one variant per theme (docs/src/assets/theme_images.css).
# The panels are pgfplots files produced by LocalOptimizationDoneRight's
# accv2026/scripts/fig1_sel_miscal.jl; this script only reads them. Re-run it
# after an accv2026 rerun regenerates them. Needs pdflatex (with pgfplots,
# adjustbox, preview) and pdftocairo (poppler). Run from anywhere:
#
#   docs/render_sel_miscal.sh path/to/LocalOptimizationDoneRight
set -euo pipefail

LODR=${1:?usage: docs/render_sel_miscal.sh path/to/LocalOptimizationDoneRight}
FIG=$LODR/accv2026/paper/figures
OUT=$(cd "$(dirname "$0")" && pwd)/src/assets/landing
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

for f in panel_sel_miscal panel_sel_miscal_maa fig_colors; do
    test -f "$FIG/$f.tex" || { echo "missing $FIG/$f.tex" >&2; exit 1; }
    cp "$FIG/$f.tex" "$WORK/"
done

# Both variants: no legend fill, so the page background shows through.
# `every axis post` is applied after the panel's own axis options, so it wins.
COMMON='\pgfplotsset{every axis post/.append style={legend style={fill=none}}}'

# Dark variant: light text, axes and ticks, a dim grid. The series keep the
# fig_colors.tex colours, except the two too dark to read on a dark page
# (contrast against #1f2424 below 3:1): SIMFIT++ navy (0.9:1) and Welsch
# purple (2.3:1), lightened to about 4.7:1.
DARK='\color{white!85!black}
\pgfplotsset{every axis post/.append style={
    axis line style={white!85!black}, tick style={white!85!black},
    grid style={white!30!black}, minor grid style={white!22!black},
    every axis label/.append style={white!85!black},
    tick label style={white!85!black}}}
\definecolor{ColorSIMFITsel}{HTML}{7B83EB}
\definecolor{ColorWELsel}{HTML}{C463E0}'

render() {  # $1 = panel file, $2 = output name, $3 = light|dark
    local theme=""
    [ "$3" = dark ] && theme=$DARK
    # The legend is defined ("legend to name") by the median panel only; the
    # mAA panel's is named `..._unused`. Typeset the median panel into a
    # discarded box first so every document can place that legend.
    local defs='\setbox0\hbox{\input{panel_sel_miscal}}'
    cat > "$WORK/$2_$3.tex" <<EOF
\\documentclass{article}
\\usepackage[active,tightpage]{preview}
\\setlength{\\PreviewBorder}{10pt}
\\usepackage{amsmath,amssymb}
\\usepackage{adjustbox}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.17}
\\usepgfplotslibrary{fillbetween,groupplots}
\\input{fig_colors}
% Legend entries cite the paper's bibliography; a web page has none.
\\renewcommand{\\cite}[1]{}
$COMMON
\\begin{document}
$theme
$defs
\\begin{preview}
\\begin{minipage}{9cm}
\\centering
\\input{$1}\\\\[4pt]
\\adjustbox{max width=\\linewidth}{\\pgfplotslegendfromname{selmiscallegend}}
\\end{minipage}
\\end{preview}
\\end{document}
EOF
    (cd "$WORK" && pdflatex -interaction=nonstopmode -halt-on-error "$2_$3.tex" >/dev/null) ||
        { tail -40 "$WORK/$2_$3.log" >&2; exit 1; }
    # The PDF page has no fill; -transp keeps it transparent in the PNG.
    pdftocairo -png -transp -r 300 -singlefile "$WORK/$2_$3.pdf" "$OUT/$2_$3"
    echo "wrote $OUT/$2_$3.png"
}

for v in light dark; do
    render panel_sel_miscal sel_miscal_median "$v"
    render panel_sel_miscal_maa sel_miscal_maa "$v"
done
