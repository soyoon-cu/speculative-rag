# vLLM base already has CUDA 12.1, PyTorch, and vLLM installed in system Python.
FROM vllm/vllm-openai:v0.4.3

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
       | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY Makefile ./

# Create venv with --system-site-packages so torch, vllm, and CUDA libs
# from the base image are visible without reinstalling them.
RUN uv venv --system-site-packages && uv sync --no-dev

ENV DATA_DIR=/data \
    RESULTS_PATH=/app/output/results.json \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data /app/output

ENTRYPOINT ["/app/scripts/run_pipeline.sh"]
