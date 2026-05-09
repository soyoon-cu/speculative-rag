FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git wget curl gnupg apt-transport-https ca-certificates \
    && wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    # nsys
    && apt-get update \
    && apt-get install -y nsight-systems-cli \
    # Google Cloud SDK
    && curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y google-cloud-cli \
    # uv
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

ENV PYTHONPATH="/app/src"
WORKDIR /app/src

ENV INDEX_PATH="/gcs/standard-rag-results-2026/faiss_contriever.index"
ENV PASSAGES_META_PATH="/gcs/standard-rag-results-2026/passages_meta_arrow"
ENV RESULTS_PATH="/gcs/speculative-rag-results-2026/verifier_output"

ENV PATH="/opt/nvidia/nsight-systems/bin:${PATH}"

ENTRYPOINT ["python", "e2e_eval.py"]
CMD ["--run", "run_vllm"]