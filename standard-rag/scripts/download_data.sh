#!/usr/bin/env bash
# Download TriviaQA (handled by HuggingFace datasets at runtime) and the
# DPR 100-word Wikipedia passage dump used by Contriever / Self-RAG.

set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
mkdir -p "${DATA_DIR}"

PASSAGES_URL="https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
PASSAGES_GZ="${DATA_DIR}/psgs_w100.tsv.gz"
PASSAGES_TSV="${DATA_DIR}/psgs_w100.tsv"

if [[ -f "${PASSAGES_TSV}" ]]; then
    echo "[skip] ${PASSAGES_TSV} already exists."
else
    echo "[download] Downloading Wikipedia passages (~9 GB) …"
    curl -C - -L "${PASSAGES_URL}" -o "${PASSAGES_GZ}"
    echo "[extract] Decompressing …"
    gunzip "${PASSAGES_GZ}"
    echo "[done] Passages saved to ${PASSAGES_TSV}"
fi

echo "[cache] Pre-downloading TriviaQA rc.wikipedia (validation split) …"
${PYTHON:-uv run python} -c "
from datasets import load_dataset
ds = load_dataset('trivia_qa', 'rc.wikipedia', split='validation', trust_remote_code=True)
print(f'TriviaQA validation: {len(ds):,} examples cached.')
"

echo ""
echo "All data ready in ${DATA_DIR}."
echo "Run 'make build-index' next."
