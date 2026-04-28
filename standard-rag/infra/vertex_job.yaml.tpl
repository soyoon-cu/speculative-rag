# Vertex AI CustomJob spec — filled in by `make vertex-submit` via envsubst.
# Variables: IMAGE_URI, GCS_BUCKET, HF_TOKEN, RUN_FULL_INDEX, VERTEX_SA
workerPoolSpecs:
  - machineSpec:
      machineType: a2-highgpu-1g
      acceleratorType: NVIDIA_TESLA_A100
      acceleratorCount: 1
    replicaCount: 1
    containerSpec:
      imageUri: $IMAGE_URI
      env:
        - name: GCS_BUCKET
          value: "$GCS_BUCKET"
        - name: HF_TOKEN
          value: "$HF_TOKEN"
        - name: RUN_FULL_INDEX
          value: "$RUN_FULL_INDEX"
        - name: DATA_DIR
          value: /data
        - name: INDEX_PATH
          value: /data/faiss_contriever.index
        - name: PASSAGES_META_PATH
          value: /data/passages_meta.pkl
        - name: RESULTS_PATH
          value: /app/output/results.json
serviceAccount: $VERTEX_SA
