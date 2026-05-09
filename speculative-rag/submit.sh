#!/bin/bash
IMAGE_URI=$1
REGION=$2
EXPERIMENT=${3:-test}
N_SAMPLES=${4:-100}
GCS_ASSETS_BUCKET=${5:-standard-rag-results-2026}
GCS_RESULTS_BUCKET=${6:-speculative-rag-results-2026}
echo "Generating Vertex AI config for run: ${EXPERIMENT}..."
cat <<EOF > vertex_config.yaml
workerPoolSpecs:
  - machineSpec:
      machineType: a2-highgpu-1g
      acceleratorType: NVIDIA_TESLA_A100
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 250
    containerSpec:
      imageUri: ${IMAGE_URI}
      env:
        - name: HF_TOKEN
          value: "PUT TOKEN HERE"
        - name: HF_HUB_OFFLINE
          value: "0"
        - name: VERIFIER_MODEL_PATH
          value: "/gcs/${GCS_ASSETS_BUCKET}/models/mistral-7b-instruct-v0.1"
        - name: DRAFTER_MODEL_PATH
          value: "/gcs/${GCS_ASSETS_BUCKET}/models/mistral-7b-instruct-v0.1"
        - name: N_SAMPLES
          value: "${N_SAMPLES}"
EOF
if [ "$EXPERIMENT" = "run_vllm" ]; then
    echo "Injecting Nsys global wrapper for ${EXPERIMENT}..."
    cat <<EOF >> vertex_config.yaml
      command: 
        - /opt/nvidia/nsight-systems/bin/nsys
        - profile
        - --trace=cuda,nvtx,cublas
        - --cuda-memory-usage=true
        - --force-overwrite=true
        - -o
        - /gcs/${GCS_RESULTS_BUCKET}/verifier_output/${EXPERIMENT}_hardware_trace
        - python
        - e2e_eval.py
      args:
        - --run
        - ${EXPERIMENT}
EOF
else
    cat <<EOF >> vertex_config.yaml
      echo "Run: ${EXPERIMENT}..."
      command: 
        - bash
        - -c
        - |
          echo "Bypassing GCS FUSE: Copying assets to local SSD..."
          mkdir -p /tmp/index /tmp/model /tmp/meta
          cp    /gcs/${GCS_ASSETS_BUCKET}/faiss_contriever.index /tmp/index/faiss_contriever.index &
          cp -r /gcs/${GCS_ASSETS_BUCKET}/models/mistral-7b-instruct-v0.1 /tmp/model/ &
          cp -r /gcs/${GCS_ASSETS_BUCKET}/passages_meta_arrow /tmp/meta/ &
          wait
          echo "All assets copied to local SSD."
          export INDEX_PATH="/tmp/index/faiss_contriever.index"
          export VERIFIER_MODEL_PATH="/tmp/model/mistral-7b-instruct-v0.1"
          export DRAFTER_MODEL_PATH="/tmp/model/mistral-7b-instruct-v0.1"
          export PASSAGES_META_PATH="/tmp/meta/passages_meta_arrow"
          echo "Starting Python Pipeline..."
          python e2e_eval.py --run ${EXPERIMENT}
EOF
fi
echo "Submitting ${EXPERIMENT} to Vertex AI..."
gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name=speculative-rag-${EXPERIMENT} \
    --config=vertex_config.yaml