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

## Code Instruction

### Setup
Ensure `uv` is installed in your environment. <br>
After cloning the Git repo, run: <br>
`uv sync`

## Demo Dashboard

The project includes a live performance dashboard featuring:

- **Live Latency Panel:** p50/p95 latency, speedup ratios, tokens/sec, queries/sec.

- **Subset/Diversity View:** 2D projection/table showing evidence diversity across draft subsets.

- **Verifier Scoring Breakdown:** Visualization of the score components used for final selection.

- **Rationale Inspection:** Real-time viewing of document-grounded rationales.

## Team

- [Alexandar Vassilev](https://github.com/alex-is-busy-coding)

- [Soyoon Park](https://github.com/soyoon-cu)

- [Hsuan-Ting Lin](https://github.com/Hsuan-Ting)

- [Rupeet Kaur](https://github.com/RupeetK)

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## References

- Wang, Z., et al. "SPECULATIVE RAG: ENHANCING RETRIEVAL AUGMENTED GENERATION THROUGH DRAFTING." ICLR 2025. ([paper PDF](doc/speculative-rag-iclr2025.pdf))

- Asai, A., et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." ICLR 2024.

- Yan, S.-Q., et al. "Corrective Retrieval Augmented Generation." arXiv:2401.15884.

- Hsieh, C.-Y., et al. "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data." ACL 2023.

- Leviathan, Y., et al. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
