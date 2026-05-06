# Use the devel image for profiling support
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 1. Install System Dependencies + NVIDIA DevTools + GCloud SDK + uv
RUN apt-get update && apt-get install -y \
    git wget curl gnupg apt-transport-https ca-certificates \
    # 1a. Fetch the specific nvidia.pub key for the DevTools repo
    && wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    # 1b. Update and install Nsys
    && apt-get update \
    && apt-get install -y nsight-systems-cli \
    # 1c. Add Google Cloud SDK
    && curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y google-cloud-cli \
    # 1d. THE UPGRADE: Install uv directly into the system path
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
COPY requirements.txt .
# We use `--system` so uv installs alongside the PyTorch 2.3.1 installation 
# already baked into the base image, rather than isolating it in a venv.
RUN uv pip install --system --no-cache -r requirements.txt

# 3. Copy Codebase
COPY . .

# 4. Environment & Paths
ENV PYTHONPATH="/app/src"
WORKDIR /app/src

# GCS FUSE paths
ENV INDEX_PATH="/gcs/standard-rag-results-2026/faiss_contriever.index"
ENV PASSAGES_META_PATH="/gcs/standard-rag-results-2026/passages_meta_arrow"
ENV RESULTS_PATH="/gcs/speculative-rag-results-2026/verifier_output"

# Ensure nsys is in the path
ENV PATH="/opt/nvidia/nsight-systems/bin:${PATH}"

ENTRYPOINT ["python", "e2e_eval.py", "--run", "run_vllm"]