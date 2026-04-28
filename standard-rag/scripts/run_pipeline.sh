#!/usr/bin/env bash
# Container entry point for Vertex AI custom job.
# Runs the full Standard RAG pipeline and uploads results to GCS.
set -euo pipefail

export PATH="/usr/local/bin:$PATH"

GCS_BUCKET="${GCS_BUCKET:?GCS_BUCKET env var is required}"
RESULTS_PATH="${RESULTS_PATH:-/app/results.json}"
RUN_FULL_INDEX="${RUN_FULL_INDEX:-false}"
INDEX_TARGET="build-index-subset"
[ "$RUN_FULL_INDEX" = "true" ] && INDEX_TARGET="build-index"

cd /app

echo "[pipeline] === Starting at $(date) ==="
echo "[pipeline] GCS_BUCKET=$GCS_BUCKET  INDEX_TARGET=$INDEX_TARGET"

if gsutil -q stat "gs://$GCS_BUCKET/faiss_contriever.index" 2>/dev/null; then
    echo "[pipeline] Existing FAISS index found in GCS — downloading..."
    mkdir -p /data
    gsutil cp "gs://$GCS_BUCKET/faiss_contriever.index" /data/
    gsutil cp "gs://$GCS_BUCKET/passages_meta.pkl" /data/
else
    echo "[pipeline] No index in GCS — building from scratch ($INDEX_TARGET)..."
    make download-data
    make "$INDEX_TARGET"
    echo "[pipeline] Uploading index to GCS for future reuse..."
    gsutil cp /data/faiss_contriever.index "gs://$GCS_BUCKET/"
    gsutil cp /data/passages_meta.pkl "gs://$GCS_BUCKET/"
fi

echo "[pipeline] Running TriviaQA eval..."
SAMPLE_FLAG=""
[ "${RUN_FULL_INDEX}" != "true" ] && SAMPLE_FLAG="--sample 500"

${PYTHON:-uv run python} -m rag.pipeline \
    --model mistralai/Mistral-7B-Instruct-v0.1 \
    --index-path "${DATA_DIR}/faiss_contriever.index" \
    --meta-path "${DATA_DIR}/passages_meta.pkl" \
    --results-path "${RESULTS_PATH}" \
    --device cpu \
    --batch-size 1 \
    ${SAMPLE_FLAG}

echo "[pipeline] Uploading results to gs://$GCS_BUCKET/results.json ..."
gsutil cp "$RESULTS_PATH" "gs://$GCS_BUCKET/results.json"

echo "[pipeline] === Done at $(date) ==="
