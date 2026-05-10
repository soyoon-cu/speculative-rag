# Speculative RAG Pipeline

This subproject runs the Speculative RAG implementation on Vertex AI. It combines dense retrieval, multi-perspective passage subset sampling, batched drafter generation, and verifier-based draft selection.

The default cloud workflow builds the Docker image with Google Cloud Build, submits a Vertex AI custom job, writes drafter/verifier outputs to GCS, and optionally records Nsight Systems or PyTorch profiler traces.

## Project Layout

```text
speculative-rag/
├── Dockerfile              # CUDA/PyTorch/vLLM container
├── Makefile                # build, submit, fetch-results
├── cloudbuild.yaml         # Cloud Build image build with cache
├── config.mk.example       # copy to config.mk and fill in local values
├── submit.sh               # generates vertex_config.yaml and submits the job
├── requirements.txt        # container Python dependencies
└── src/
    ├── data/               # TriviaQA loading and answer matching helpers
    ├── sampling/           # FAISS index, retriever, multi-perspective sampler
    ├── drafter/            # batched draft generation and experiment entrypoints
    ├── verifier/           # verifier scoring and metrics
    ├── e2e_eval.py         # drafter + verifier orchestration
    └── pipeline.py         # lightweight Typer pipeline entrypoint
```

## Prerequisites

- GCP project with billing enabled
- Vertex AI API and Cloud Build API enabled
- Artifact Registry repository for the Docker image
- GCS bucket containing the Standard RAG assets:
  - `faiss_contriever.index`
  - `passages_meta_arrow/`
  - `models/mistral-7b-instruct-v0.1/`
- GCS bucket for Speculative RAG outputs
- HuggingFace token with access to `mistralai/Mistral-7B-Instruct-v0.1`
- A100 quota in the configured region

## Configure

Create a local config file:

```bash
cp config.mk.example config.mk
```

Edit `config.mk`:

```make
PROJECT_ID = your-gcp-project-id
REGION     = us-central1
REPO_NAME  = hpml-repo
IMAGE_NAME = speculative-rag

INDEX_BUCKET  = standard-rag-results-2026
OUTPUT_BUCKET = speculative-rag-results-2026
```

`config.mk` is gitignored. Do not commit real project IDs, bucket names, or credentials if they are private.

Export your HuggingFace token before submitting a job:

```bash
export HF_TOKEN=hf_xxxx
```

## Build

Run from this directory:

```bash
make build
```

This runs:

```bash
gcloud builds submit --config=cloudbuild.yaml --substitutions=_IMAGE_URI=<image> .
```

## Submit Experiments

General form:

```bash
make submit ARGS="<experiment> <n_samples> <index_bucket> <output_bucket>"
```

Examples:

```bash
make submit ARGS="test 20"
make submit ARGS="run_vllm 100"
make submit ARGS="run_m 100"
make submit ARGS="run_k 100"
make submit ARGS="verify_saved 100"
make submit ARGS="verify_no_opt 100"
```

If you omit bucket arguments, `submit.sh` uses these defaults:

```text
GCS_ASSETS_BUCKET=standard-rag-results-2026
GCS_RESULTS_BUCKET=speculative-rag-results-2026
```

To override them directly:

```bash
make submit ARGS="run_vllm 100 my-index-bucket my-output-bucket"
```

## Experiment Names

`src/e2e_eval.py` supports these names:

| Experiment | Purpose |
|---|---|
| `test` | TinyLlama smoke test without profiling |
| `test_p` | TinyLlama smoke test with PyTorch profiler |
| `no_opt` | Mistral drafter without vLLM or quantization |
| `nf4` | NF4 quantized drafter run |
| `int8` | INT8 quantized drafter run |
| `run_vllm` | Main vLLM batched drafter + verifier run |
| `run_m` | Sweep number of drafts `m` |
| `run_k` | Sweep documents per subset `k` |
| `verify_saved` | Re-run verifier on saved vLLM drafter output |
| `verify_no_opt` | Re-run verifier on saved no-optimization drafter output |

## Fetch Results

```bash
make fetch-results
```

This downloads:

```text
gs://$(OUTPUT_BUCKET)/verifier_output
```

into local `./verifier_output`.

Typical output files include:

- drafter JSON records
- verifier JSON records
- `*_metrics.json` summaries
- Nsight Systems traces for `run_vllm`
- PyTorch profiler traces for profiled runs

## Local Checks

Before pushing code changes:

```bash
python3 -m compileall -q src
bash -n submit.sh
```

You can also dry-run the Makefile targets without launching cloud work:

```bash
make -n build
make -n submit ARGS="run_vllm 10"
make -n fetch-results
```

## Notes

- `config.mk` must exist before running any Makefile target.
- `make submit` depends on `make build`, so it rebuilds the image before submitting.
- `run_vllm` wraps `e2e_eval.py` with Nsight Systems through `submit.sh`.
- The Docker image sets `PYTHONPATH=/app/src` and runs from `/app/src`.
- Large data, generated outputs, model weights, local config, and profiler artifacts are intentionally gitignored.
