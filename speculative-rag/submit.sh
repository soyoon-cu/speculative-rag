#!/bin/bash
IMAGE_URI=$1
REGION=$2
EXPERIMENT=${3:-test}
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
          value: "1"
        - name: VERIFIER_MODEL_PATH
          value: "/gcs/standard-rag-results-2026/models/mistral-7b-instruct-v0.1"
        - name: DRAFTER_MODEL_PATH
          value: "/gcs/standard-rag-results-2026/models/mistral-7b-instruct-v0.1" 
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
        - /gcs/speculative-rag-results-2026/verifier_output/${EXPERIMENT}_hardware_trace
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
        - "find / -type f -name nsys 2>/dev/null > /gcs/speculative-rag-results-2026/nsys_location.txt"
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
          cp /gcs/standard-rag-results-2026/faiss_contriever.index /tmp/index/faiss_contriever.index
          export INDEX_PATH="/tmp/index/faiss_contriever.index"
          echo "Starting Python Pipeline..."
          python e2e_eval.py --run ${EXPERIMENT}
EOF
fi
echo "Submitting ${EXPERIMENT} to Vertex AI..."
gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name=speculative-rag-${EXPERIMENT} \
    --config=vertex_config.yaml