# Speculative RAG: Latency-Quality Trade-offs in Multi-Draft Retrieval

An end-to-end implementation of the [Speculative Retrieval-Augmented Generation (Speculative RAG)](doc/speculative-rag-iclr2025.pdf) pipeline, designed to quantify the trade-off between answer quality and inference latency on knowledge-intensive QA tasks.

## Project Objective

Standard RAG architectures suffer from a critical bottleneck: processing long retrieved contexts leads to prohibitive latency and reasoning errors.

This project addresses this by implementing a **Speculative RAG pipeline:**

1. **Multi-Perspective Selection:** Diversifying retrieved documents into distinct subsets.

2. **Batched Multi-Draft Prompting:** Using a smaller, specialist Drafter model to generate multiple `{answer, rationale}` drafts in parallel.

3. **Verifier-Based Selection:** Employing a larger Verifier model to select the best draft using log-probability-based confidence scoring.

### Performance Targets

- **Speedup:** $\ge 1.3\times$ p50 latency speedup at matched accuracy (within 1.0 EM of baseline).

- **GPU Utilization:** $\ge 60\%$ average SM utilization via continuous batching.

- **Stretch Goals:** Match or exceed the original paper's gains (e.g., $\ge +5.0$ EM or $\ge 1.8\times$ speedup at baseline EM).

## Architecture & Approach

- **Subset Selection (Diversity):** Retrieves top-n chunks and forms m subsets using embedding-based k-means clustering or an MMR diversification heuristic. Embeddings are precomputed offline.

- **Batched Parallel Drafting:** Implements parallel drafting as batched multi-prompt generation through a single drafter engine using vLLM continuous batching (generating m drafts in one scheduling window).

- **Verifier-Based Selection:** Selects the best draft based on a log-probability scoring function: `Score = log P_V(a_j | q, r_j) + λ log P_D(r_j | q, S_j)`.

## Hardware & Software Stack

Hardware: Single NVIDIA A100 (40GB) on GCP.

- **Inference Engine:** [vLLM](https://github.com/vllm-project/vllm) (leveraging continuous batching and PagedAttention).

- **Models**

  - **Drafter:** 7B class model (e.g., Mistral-7B).

  - **Verifier:** 7B-13B model (Stretch goal: Mixtral-8x7B).

- **Retrieval:** FAISS vector store with offline precomputed embeddings (InBedder-Roberta, E5, or BGE).

- **Optimizations:** INT8/NF4 quantization via `bitsandbytes`, KV-cache limits.

- **Profiling:** Nsight Systems (`nsys`) and PyTorch profiler.

## Baselines & Evaluation

Evaluated on knowledge-intensive datasets such as TriviaQA or PubHealth. Metrics include Exact Match (EM), p50/p95 latency, and GPU utilization.

### Baselines for Comparison:

- **Standard RAG:** Concatenate top-k retrieved chunks into one prompt and generate.

- **CRAG-Inspired (Filter-then-Generate):** Lightweight filtering/reranking of retrieved chunks followed by generation with shorter context.

## Repository Structure

```
speculative-rag/
├── doc/
│   └── speculative-rag-iclr2025.pdf
├── standard-rag/          ← Standard RAG baseline (implemented)
│   ├── README.md          ← full setup & run instructions
│   ├── Makefile
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── infra/             ← Terraform: GCS, Artifact Registry, service account
│   └── src/rag/
└── speculative-rag/       ← Speculative RAG implementation (in progress)
```

Each subdirectory is an independent project with its own environment, Docker image, and GCP infrastructure.

## Status

| Component             | Status        | Notes |
|-----------------------|---------------|-------|
| Standard RAG baseline | **Complete**  | Vertex AI pipeline; see `standard-rag/` |
| Speculative RAG       | In progress   | |

## Getting Started

See **[standard-rag/README.md](standard-rag/README.md)** for the full setup guide including:
- GCP project configuration and API enablement
- GPU quota increase instructions (required — new projects default to 0 GPU quota)
- Terraform infra provisioning
- Docker build and push
- Running smoke tests and full evaluation on Vertex AI

### Quick reference

```bash
cd standard-rag

# First-time setup
make gcp-enable-apis     # enable Vertex AI, Artifact Registry, GCS APIs
make infra-apply         # provision GCS bucket + Artifact Registry
make docker-push         # build and push the container image

# Run evaluation
make vertex-submit               # smoke test (100k passages, 500 examples)
make vertex-submit ENV=prod      # full eval (21M passages, 11k examples)
make fetch-results               # download results.json and print table
```

## Demo Dashboard

The project includes a live performance dashboard featuring:

- **Live Latency Panel:** p50/p95 latency, speedup ratios, tokens/sec, queries/sec.

- **Subset/Diversity View:** 2D projection/table showing evidence diversity across draft subsets.

- **Verifier Scoring Breakdown:** Visualization of the score components used for final selection.

- **Rationale Inspection:** Real-time viewing of document-grounded rationales.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the team, branch workflow, commit style, and code standards.

## Security

See [SECURITY.md](SECURITY.md) for guidance on secret management, GCP service account scoping, and what to do if a credential is accidentally exposed.

## Team

- [Alexandar Vassilev](https://github.com/alex-is-busy-coding)

- [Soyoon Park](https://github.com/soyoon-cu)

- [Hsuan-Ting Lin](https://github.com/Hsuan-Ting)

- [Rupeet Kaur](https://github.com/RupeetK)

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## References

- Wang et al., *Speculative RAG: Enhancing Retrieval Augmented Generation Through Drafting*, ICLR 2025. ([paper PDF](doc/speculative-rag-iclr2025.pdf))
