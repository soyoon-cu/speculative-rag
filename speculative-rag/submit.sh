#!/bin/bash
IMAGE_URI=$1
REGION=$2
EXPERIMENT=${3:-test}
PROJECT_ID=$4
INDEX_BUCKET=$5
OUTPUT_BUCKET=$6

: "${HF_TOKEN:?HF_TOKEN must be set in your environment or .env before running make submit}"
PROJECT_ID="${PROJECT_ID:?PROJECT_ID is required}"
INDEX_BUCKET="${INDEX_BUCKET:?INDEX_BUCKET is required}"
OUTPUT_BUCKET="${OUTPUT_BUCKET:?OUTPUT_BUCKET is required}"

echo "Generating Vertex AI config for run: ${EXPERIMENT}..."
cat <<EOF > vertex_config.yaml
workerPoolSpecs:
  - machineSpec:
      machineType: a2-ultragpu-1g
      acceleratorType: NVIDIA_A100_80GB
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 250
    containerSpec:
      imageUri: ${IMAGE_URI}
      env:
        - name: HF_TOKEN
          value: "${HF_TOKEN}"
        - name: HF_HUB_OFFLINE
          value: "0"
        - name: VERIFIER_MODEL_PATH
          value: "mistralai/Mistral-7B-Instruct-v0.1"
        - name: DRAFTER_MODEL_PATH
          value: "mistralai/Mistral-7B-Instruct-v0.1"
        - name: INDEX_PATH
          value: "/gcs/${INDEX_BUCKET}/faiss_contriever.index"
        - name: PASSAGES_META_PATH
          value: "/gcs/${INDEX_BUCKET}/passages_meta_arrow"
        - name: RESULTS_PATH
          value: "/gcs/${OUTPUT_BUCKET}/verifier_output"
        - name: N_SAMPLES
          value: "${N_SAMPLES:-100}"
EOF
if [ "$EXPERIMENT" = "run_vllm" ]; then
    echo "PROFILING DETECTED: Injecting Nsys global wrapper for ${EXPERIMENT}..."
    cat <<EOF >> vertex_config.yaml
      command: 
        - /opt/nvidia/nsight-systems/bin/nsys
        - profile
        - --trace=cuda,nvtx,cublas
        - --cuda-memory-usage=true
        - --force-overwrite=true
        - -o
        - /gcs/${OUTPUT_BUCKET}/verifier_output/${EXPERIMENT}_hardware_trace
        - python
        - e2e_eval.py
      args:
        - --run
        - ${EXPERIMENT}
EOF
elif [ "$EXPERIMENT" = "scout" ]; then
    echo "SCOUT DETECTED: Searching the container for Nsys..."
    cat <<EOF >> vertex_config.yaml
      command: 
        - bash
        - -c
        - "find / -type f -name nsys 2>/dev/null > /gcs/${OUTPUT_BUCKET}/nsys_location.txt"
EOF
elif [ "$EXPERIMENT" = "verify_no_opt" ] || [ "$EXPERIMENT" = "verify_saved" ]; then
    echo "VERIFIER-ONLY DETECTED: Reusing saved drafter outputs from GCS..."
    cat <<EOF >> vertex_config.yaml
      command: 
        - bash
        - -c
        - |
          export RESULTS_PATH="/gcs/${OUTPUT_BUCKET}/verifier_output"
          echo "Starting verifier-only pipeline..."
          python e2e_eval.py --run ${EXPERIMENT}
EOF
else
    echo "SWEEP DETECTED: Running pure Python for maximum throughput..."
    cat <<EOF >> vertex_config.yaml
      command: 
        - bash
        - -c
        - |
          echo "Bypassing GCS FUSE: Copying 64GB FAISS index to local SSD..."
          mkdir -p /tmp/index
          cp /gcs/${INDEX_BUCKET}/faiss_contriever.index /tmp/index/faiss_contriever.index
          export INDEX_PATH="/tmp/index/faiss_contriever.index"
          export PASSAGES_META_PATH="/gcs/${INDEX_BUCKET}/passages_meta_arrow"
          export RESULTS_PATH="/gcs/${OUTPUT_BUCKET}/verifier_output"
          echo "Starting Python Pipeline..."
          python e2e_eval.py --run ${EXPERIMENT}
EOF
fi
echo "Submitting ${EXPERIMENT} to Vertex AI..."
gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name=speculative-rag-${EXPERIMENT} \
    --config=vertex_config.yaml \
    --project=${PROJECT_ID}
