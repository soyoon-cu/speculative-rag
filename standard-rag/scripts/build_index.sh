#!/usr/bin/env bash
# Embed the DPR Wikipedia passage dump with Contriever-MSMARCO and write a
# FAISS flat-IP index to disk.

set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
PASSAGES_TSV="${DATA_DIR}/psgs_w100.tsv"
INDEX_PATH="${DATA_DIR}/faiss_contriever.index"
META_PATH="${DATA_DIR}/passages_meta.pkl"
PASSAGES_SUBSET="${PASSAGES_SUBSET:-}"  # empty → full corpus

if [[ ! -f "${PASSAGES_TSV}" ]]; then
    echo "[error] Passages file not found: ${PASSAGES_TSV}"
    echo "Run 'make download-data' first."
    exit 1
fi

SUBSET_FLAG=""
if [[ -n "${PASSAGES_SUBSET}" ]]; then
    SUBSET_FLAG="--passages-subset ${PASSAGES_SUBSET}"
    echo "[info] Building index over first ${PASSAGES_SUBSET} passages (subset mode)."
else
    echo "[info] Building full index (~21 M passages). Estimated time: 2–3 hr on p3.8xl."
fi

echo "[build] Starting FAISS index build …"
${PYTHON:-uv run python} -m rag.retrieval.index \
    --passages "${PASSAGES_TSV}" \
    --output "${INDEX_PATH}" \
    --meta "${META_PATH}" \
    ${SUBSET_FLAG}

echo ""
echo "[done] Index written to ${INDEX_PATH}"
echo "       Metadata written to ${META_PATH}"
echo "Run 'make eval-mistral' to start evaluation."
