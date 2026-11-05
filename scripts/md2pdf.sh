#!/usr/bin/env bash
# markdown -> styled PDF.  usage: scripts/md2pdf.sh docs/MODEL_V2.md docs/MODEL_V2.pdf
set -euo pipefail
SRC=$1; OUT=$2
NODE="/Applications/DeepSeek Harness.app/Contents/Resources/runtime/primary-runtime/dependencies/node/bin/node"
CLI="/Applications/DeepSeek Harness.app/Contents/Resources/app.asar.unpacked/dsh/node_modules/@deepseek-ai/libreoffice-kit/lib/cli.js"
TMP=$(mktemp -d)
pandoc "$SRC" -o "$TMP/raw.docx" --from markdown+pipe_tables --resource-path=.
.venv/bin/python scripts/style_docx.py "$TMP/raw.docx" "$TMP/styled.docx"
rm -f "$OUT"
"$NODE" "$CLI" convert --input "$TMP/styled.docx" --output "$OUT"
head -c 5 "$OUT"; echo ""
ls -la "$OUT"
