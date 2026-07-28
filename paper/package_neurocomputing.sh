#!/usr/bin/env bash
# =====================================================================
# package_neurocomputing.sh
# Build a minimal, self-contained submission package for Neurocomputing
# (Elsevier elsarticle build: main_els.tex).
#
# What it does:
#   1. Copies ONLY the files main_els.tex actually needs into a clean staging
#      dir (no CVPR/MDPI files, no logs, no build artifacts).
#   2. Does a fresh pdflatex+bibtex+pdflatex x2 build inside the staging dir to
#      prove the package compiles stand-alone.
#   3. Zips the staging dir into  dist/DPRNet_Neurocomputing_submission.zip
#
# Usage:
#   bash package_neurocomputing.sh
#
# Requires pdflatex/bibtex on PATH. If you use TinyTeX, run e.g.:
#   PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH" bash package_neurocomputing.sh
# =====================================================================
set -euo pipefail

# Resolve the paper/ directory (this script lives in it).
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$SRC/dist/DPRNet_Neurocomputing_submission"
ZIP="$SRC/dist/DPRNet_Neurocomputing_submission.zip"

echo "==> Source dir : $SRC"
echo "==> Staging dir: $STAGE"

# ---- 1. Clean staging ------------------------------------------------
rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE"

# ---- 2. Copy the minimal file set -----------------------------------
# Top-level build files.
cp "$SRC/main_els.tex"        "$STAGE/"
cp "$SRC/preamble_els.tex"    "$STAGE/"
cp "$SRC/references.bib"      "$STAGE/"
cp "$SRC/elsarticle.cls"      "$STAGE/"
cp "$SRC/elsarticle-num.bst"  "$STAGE/"

# Shared content directories (sections / tables / figures + image assets).
mkdir -p "$STAGE/sections" "$STAGE/tables" "$STAGE/figures"
cp "$SRC"/sections/*.tex "$STAGE/sections/"
cp "$SRC"/tables/*.tex   "$STAGE/tables/"
cp "$SRC"/figures/*.tex  "$STAGE/figures/"
# Raster assets used by fig7 (visual comparison) and fig8/fig5 (cluster/hist).
cp -R "$SRC/figures/fig7_assets"  "$STAGE/figures/fig7_assets"
cp -R "$SRC/figures/fig58_assets" "$STAGE/figures/fig58_assets"

# Supplementary submission items (uploaded separately in EM, kept here for convenience).
cp "$SRC/cover_letter_neurocomputing.txt" "$STAGE/" 2>/dev/null || true
cp "$SRC/highlights_els.txt"              "$STAGE/" 2>/dev/null || true

# Drop CSV lists / other assets accidentally matched: none copied, we were explicit.

# ---- 3. Fresh stand-alone compile to verify the package -------------
echo "==> Test-compiling the staged package..."
pushd "$STAGE" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main_els > build1.log 2>&1
bibtex main_els > build_bib.log 2>&1 || true
pdflatex -interaction=nonstopmode -halt-on-error main_els > build2.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_els > build3.log 2>&1

if grep -qiE "undefined (citation|reference)" build3.log; then
  echo "!! WARNING: undefined citations/references remain -- check build3.log"
fi
if [ ! -f main_els.pdf ]; then
  echo "!! ERROR: main_els.pdf was not produced. See build*.log"; popd >/dev/null; exit 1
fi
PAGES=$(grep -oE "Output written on main_els.pdf \([0-9]+ pages" build3.log | grep -oE "[0-9]+ pages" || echo "?")
echo "==> Stand-alone build OK: main_els.pdf ($PAGES)"

# Keep a copy of the compiled PDF at the package root, remove noisy intermediates.
rm -f build1.log build2.log build3.log build_bib.log \
      main_els.aux main_els.log main_els.out main_els.blg main_els.spl
# NOTE: main_els.bbl is intentionally kept -- Elsevier EM likes having the .bbl.
popd >/dev/null

# ---- 4. Zip ----------------------------------------------------------
echo "==> Zipping..."
( cd "$SRC/dist" && zip -rq "DPRNet_Neurocomputing_submission.zip" "DPRNet_Neurocomputing_submission" )

echo ""
echo "==================================================================="
echo " Package ready:"
echo "   $ZIP"
echo ""
echo " Contents (LaTeX source + assets + compiled PDF + cover letter +"
echo " highlights). Upload the .tex sources (NOT a PDF-only) to Editorial"
echo " Manager; upload cover_letter and highlights in their own EM fields."
echo "==================================================================="
