# Speculative RAG Drafter Pipeline

A GPU-optimised **Speculative RAG** (Retrieval-Augmented Generation) drafting pipeline.  
For each TriviaQA question the system retrieves top-k passages via a pre-built FAISS index, samples m multi-perspective document subsets using k-means clustering (MultiPerspectiveSampler), generates m parallel answer drafts in a **single batched forward pass** (BatchedDrafter / vLLM), and serialises a `VerifierInput` record per question for downstream verifier scoring.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Prerequisites](#prerequisites)
3. [Installation — A100 80 GB](#installation--a100-80-gb)
4. [FAISS Index & Passage Embeddings](#faiss-index--passage-embeddings)
5. [Default Configuration & What the Constants Mean](#default-configuration--what-the-constants-mean)
6. [Run Functions — When & How to Use Each](#run-functions--when--how-to-use-each)
7. [Profiling Strategy by Run](#profiling-strategy-by-run)
8. [Creating the nsys Traces Directory & Running Nsight Profiles](#creating-the-nsys-traces-directory--running-nsight-profiles)
9. [Viewing PyTorch Profiler Traces in TensorBoard (VS Code / SSH)](#viewing-pytorch-profiler-traces-in-tensorboard-vs-code--ssh)
10. [Viewing Nsight Systems Traces](#viewing-nsight-systems-traces)
11. [Output Data — Where It Is Saved & What It Contains](#output-data--where-it-is-saved--what-it-contains)

---

## Repository Structure

```
.
├── drafter/
│   ├── draft_output.py          # DraftOutput dataclass — prompt builder, parser, log-prob
│   ├── vllm_.py                 # VLLM wrapper — loads vLLM engine, continuous-batch generate
│   ├── batched_drafter.py       # BatchedDrafter — HuggingFace + quantisation + PyTorch Profiler
│   ├── drafter_pipeline.py      # Full pipeline — FAISS retrieval → sampling → drafting → JSON
│   ├── run_tests.py             # CLI entry-point with all named experiment functions
│   ├── profiler_traces/         # Auto-created — PyTorch TensorBoard events
│   ├── nsys_traces/             # Create manually before nsys runs (see §8)
│   └── drafter_output/          # Auto-created — per-experiment JSON results
│   ├── data/
│       ├── index.faiss          # ← precomputed FAISS index  
│       └── index_meta.pkl       # ← passage metadata / text  
├── sampling/
│   ├── index.py             # FAISSIndex — load precomputed passage index
│   ├── retriever.py         # ContrieverRetriever — dense retrieval over FAISS
│   └── multi_perspective.py # MultiPerspectiveSampler — k-means subset sampling
├── data/
│   ├── loader.py            # iter_samples(), TriviaQASample
│   ├── preprocess.py        # answer_in_response() — EM scoring helper
 

```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| CUDA | 12.1+ |
| GPU | NVIDIA A100 80 GB |
| Driver | ≥ 525 |
| NVIDIA Nsight Systems (`nsys`) | ≥ 2023.3 (for nsys profiling) |

---

## Installation — A100 80 GB

All steps assume a fresh environment on an A100 node. Run every command from the repository root.

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Upgrade pip and install core dependencies

```bash
pip install --upgrade pip

pip install torch==2.3.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install HuggingFace Transformers and Datasets

```bash
pip install transformers==4.41.0 \
            datasets \
            accelerate \
            sentencepiece \
            protobuf
```

### 4. Install FAISS (GPU build)

```bash
pip install faiss-gpu
```

> **Note:** If `faiss-gpu` is not available for your CUDA version, use `faiss-cpu` as a fallback — retrieval will still work, only slightly slower.

### 5. Install bitsandbytes (INT8 / NF4 quantisation)

```bash
pip install bitsandbytes==0.43.1
```

Verify the install targets the A100:

```bash
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

### 6. Install vLLM (PagedAttention + continuous batching)

```bash
pip install vllm==0.4.2
```

> vLLM requires CUDA 12.1+. On the A100 this gives you **PagedAttention**, **continuous batching**, and native `bfloat16` support.

### 7. Install Contriever retrieval dependencies

```bash
pip install sentence-transformers
```

### 8. Install profiling and utility packages

```bash
pip install tensorboard \
            torch-tb-profiler \
            nvidia-ml-py \
            tqdm \
            scipy
```

### 9. Verify the full GPU stack

```bash
python - <<'EOF'
import torch, vllm, bitsandbytes as bnb, faiss
print("torch      :", torch.__version__, "| CUDA:", torch.version.cuda)
print("vllm       :", vllm.__version__)
print("bnb        :", bnb.__version__)
print("faiss      : ok  (GPU res:", faiss.get_num_gpus(), ")")
print("A100 VRAM  :", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), "GB")
EOF
```

---

## FAISS Index & Passage Embeddings

The pipeline expects **two precomputed files** that must be placed (or symlinked) under `data/` before any run:

| Variable in code | Default path | Description |
|---|---|---|
| `INDEX_PATH` | `data/index.faiss` | FAISS flat/IVF index of passage embeddings produced by Contriever |
| `META_PATH` | `data/index_meta.pkl` | Pickled metadata: maps FAISS integer IDs → raw passage text strings |

These paths are set at the top of `drafter_pipeline.py`:

```python
INDEX_PATH = Path("data/index.faiss")
META_PATH  = Path("data/index_meta.pkl")
```

**To use a different location**, edit those two lines — or override them when calling `load_pipeline()` directly:

```python
retriever, sampler, drafter = load_pipeline(
    model_name   = MODEL_MISTRAL_INSTRUCT,
    use_vllm     = True,
    use_bnb_nf4  = False,
    use_int8     = False,
    index_path   = Path("/path/to/your/index.faiss"),
    meta_path    = Path("/path/to/your/index_meta.pkl"),
)
```

> **Why precomputed?** `ContrieverRetriever.retrieve_with_embeddings()` returns both the passage texts *and* their cached Contriever vectors. `MultiPerspectiveSampler` then runs k-means directly on these vectors, avoiding a second encoder forward pass per question.

---

## Default Configuration & What the Constants Mean

The following constants are defined at the top of `drafter_pipeline.py` and `run_tests.py`. Changing them changes the behaviour of every run that does not override them explicitly.

### `drafter_pipeline.py`

| Constant | Default | Meaning |
|---|---|---|
| `MAX_NEW_TOKENS` | `300` | Maximum tokens the drafter generates per draft |
| `MAX_INPUT_LEN` | `1024` | Tokeniser truncation limit for each prompt |
| `DO_SAMPLE` | `False` | `False` = greedy decoding; `True` = sampling |
| `TEMPERATURE` | `1.0` | Softmax temperature (only active when `DO_SAMPLE=True`) |
| `TOP_K` | `10` | Number of passages retrieved from FAISS per question |
| `INDEX_PATH` | `data/index.faiss` | Path to precomputed FAISS passage index |
| `META_PATH` | `data/index_meta.pkl` | Path to passage metadata pickle |

### `run_tests.py`

| Constant | Default | Meaning |
|---|---|---|
| `VLLM_AVAILABLE` | `False` | Master switch — set to `True` to enable vLLM for any run |
| `BNB_AVAILABLE` | `False` | Master switch — set to `True` to enable NF4 quantisation |
| `INT8_Q` | `False` | Master switch — set to `True` to enable INT8 quantisation |
| `PROFILE_RUN` | `False` | Set to `True` to activate PyTorch Profiler in smoke-test runs |
| `PROFILE_BASE_DIR` | `./profiler_traces` | Root directory for all TensorBoard profiler output |
| `DRAFTER_OUTPUT_PATH` | `./drafter_output` | Root directory for all JSON result files |
| `M` | `5` | Number of drafts generated per question (parallel subsets) |
| `K_DOCS` | `2` | Number of documents per subset fed to the drafter |
| `TOP_K` | `10` | Passages retrieved from FAISS (passed into `run()`) |

### `drafter_pipeline.run()` key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `n_samples` | `1000` | When `test=True`, the pipeline stops after this many questions |
| `log_every` | `100` | Draft coverage is logged to stdout every N questions |
| `test` | `True` | `True` → run on first `n_samples` questions; `False` → full TriviaQA validation split |

---

## Run Functions — When & How to Use Each

Run any experiment with:

```bash
python run_tests.py --run <name>
```

---

### `test` — Smoke test, no optimisation, no profiler

**When to run:** First check after installation. Verifies the end-to-end pipeline works with a tiny model and no GPU-specific dependencies.

```bash
python run_tests.py --run test
```

- **Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Optimisation:** None (`use_vllm=False`, `use_bnb_nf4=False`, `use_int8=False`)
- **Profiler:** Off (`PROFILE_RUN=False`)
- **Subset:** `test=True` → first **1 000 samples**, progress logged every **100**
- **Output:** `./drafter_output/test.json`

---

### `test_p` — Smoke test with PyTorch Profiler

**When to run:** Confirm that the PyTorch Profiler + TensorBoard trace handler writes correctly before running full profiling experiments.

```bash
python run_tests.py --run test_p
```

- **Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Optimisation:** None
- **Profiler:** On — writes TensorBoard events to `./profiler_traces/test/`
- **Subset:** `test=True` → first 1 000 samples
- **Output:** `./drafter_output/test_profiler.json`

---

### `no_opt` — Baseline (Mistral-7B-Instruct, no quantisation, full dataset)

**When to run:** Establish the baseline latency and coverage numbers against which quantisation and vLLM experiments are compared.

```bash
python run_tests.py --run no_opt
```

- **Model:** `mistralai/Mistral-7B-Instruct-v0.1`
- **Optimisation:** None — `use_vllm=False`, `use_bnb_nf4=False`, `use_int8=False`
- **Profiler:** **PyTorch Profiler** on — trace written to `./profiler_traces/no_opt/`
- **Subset:** `test=False` → **full TriviaQA validation split**
- **Output:** `./drafter_output/no_opt.json`

> **Note:** Profiler records CPU + CUDA activities, per-kernel FLOPS, and peak VRAM. Only the **first question** is profiled (`:profile_run = (i == 0)`).

---

### `nf4` — BitsAndBytes NF4 4-bit quantisation

**When to run:** Measure the VRAM reduction and throughput impact of 4-bit NF4 quantisation (QLoRA-style double quantisation enabled).

```bash
python run_tests.py --run nf4
```

- **Model:** `mistralai/Mistral-7B-Instruct-v0.1`
- **Optimisation:** `use_bnb_nf4=True` — loads model with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)`
- **Profiler:** **PyTorch Profiler** on — trace written to `./profiler_traces/bnb/`
- **Subset:** `test=False` → full dataset
- **Output:** `./drafter_output/bnb.json`

> Set `BNB_AVAILABLE = False` (default) to disable NF4; the run function overrides it explicitly with `use_bnb_nf4=True`, so the global flag does not block this run.

---

### `int8` — BitsAndBytes INT8 quantisation

**When to run:** Compare 8-bit integer quantisation against the NF4 and baseline runs for latency/quality trade-off.

```bash
python run_tests.py --run int8
```

- **Model:** `mistralai/Mistral-7B-Instruct-v0.1`
- **Optimisation:** `use_int8=True` — loaded with `load_in_8bit=True` via bitsandbytes
- **Profiler:** **PyTorch Profiler** on — trace written to `./profiler_traces/int8/`
- **Subset:** `test=False` → full dataset
- **Output:** `./drafter_output/int8.json`

---

### `run_vllm` — vLLM continuous batching (primary production run)

**When to run:** Full-speed production run with vLLM's PagedAttention engine. All m prompts are dispatched in a single scheduled batch — maximising GPU SM utilisation.

```bash
python run_tests.py --run run_vllm
```

- **Model:** `mistralai/Mistral-7B-Instruct-v0.1`
- **Optimisation:** `use_vllm=True` (`use_bnb_nf4=False`, `use_int8=False`)
- **Profiler:** **Nsight Systems (nsys)** — not PyTorch Profiler. NVTX ranges are annotated at `vllm.generate`, `vllm.logprob_extraction`, `pipeline.retrieve`, `pipeline.sample`, and `pipeline.draft` (see §8)
- **Subset:** `test=False` → full dataset
- **Output:** `./drafter_output/vllm_m5_k2.json`
- **Profile dir:** `./profiler_traces/vllm_m5_k2/` (written by NVTX; used with nsys output path separately)

> **Important:** When using vLLM, set `use_vllm=True` and ensure `use_bnb_nf4=False` and `use_int8=False` (both are `False` by default). vLLM manages its own memory via PagedAttention and is incompatible with bitsandbytes quantisation in the same process.

---

### `run_m` — Sweep over m (number of drafts per question) with vLLM

**When to run:** Ablation study on the effect of draft count. Runs m ∈ {5, 10, 15, 20} sequentially over the full dataset, saving one JSON file per value of m.

```bash
python run_tests.py --run run_m
```

- **Model:** `mistralai/Mistral-7B-Instruct-v0.1`
- **Optimisation:** `use_vllm=True`
- **Profiler:** Off by default (`profile_run=False`). To profile a specific m, set `profile_run=True` inside `run_m()` and re-run
- **Subset:** `test=False` → full dataset for each m
- **Outputs:** `./drafter_output/vllm_m5.json`, `vllm_m10.json`, `vllm_m15.json`, `vllm_m20.json`

---

### `run_k` — Sweep over k (documents per subset) with vLLM

**When to run:** Ablation study on evidence density. Runs k ∈ {2, 4, 6, 10} sequentially, with m fixed at 5.

```bash
python run_tests.py --run run_k
```

- **Model:** `mistralai/Mistral-7B-Instruct-v0.1`
- **Optimisation:** `use_vllm=True`
- **Profiler:** Off by default. Same override approach as `run_m` to enable
- **Subset:** `test=False` → full dataset for each k
- **Outputs:** `./drafter_output/vllm_k2.json`, `vllm_k4.json`, `vllm_k6.json`, `vllm_k10.json`

---

## Profiling Strategy by Run

| Run | Profiler Used | What Is Captured |
|---|---|---|
| `no_opt` (baseline) | **PyTorch Profiler** | CPU+CUDA kernel times, per-op FLOPS, peak VRAM, TensorBoard trace |
| `nf4` | **PyTorch Profiler** | Same as above — shows quantisation overhead vs baseline |
| `int8` | **PyTorch Profiler** | Same as above — INT8 matmul kernels visible in trace |
| `run_vllm` | **Nsight Systems (nsys)** | CUDA kernel timeline, NVTX ranges (retrieve / sample / draft / vllm.generate / logprob_extraction), HBM memory traffic |
| `run_m` / `run_k` | None by default | Set `profile_run=True` inside the loop for a specific m or k value |

The PyTorch Profiler writes a `trace_*.pt.trace.json` file (Chrome trace) **and** TensorBoard event files into `profile_dir` via `torch.profiler.tensorboard_trace_handler(profile_dir)`.

Only the **first question** in each run is profiled (`profile_run = (i == 0)` in `drafter_pipeline.py`), keeping trace file sizes manageable.

---

## Creating the nsys Traces Directory & Running Nsight Profiles

### Step 1 — Create the output directory

```bash
mkdir -p ./nsys_traces
```

### Step 2 — Smoke test without nsys first

```bash
python run_tests.py --run test_p
```

### Step 3 — Run under Nsight Systems

Use nsys to wrap the Python process. The `--output` flag sets the `.nsys-rep` file name.

```bash
nsys profile \
    --trace=cuda,nvtx,cublas \
    --cuda-memory-usage=true \
    --output=./nsys_traces/a100_vllm_m5_k2 \
    python run_tests.py --run run_vllm
```

**Flag breakdown:**

| Flag | Purpose |
|---|---|
| `--trace=cuda,nvtx,cublas` | Captures CUDA API calls, NVTX annotation ranges, and cuBLAS kernel events |
| `--cuda-memory-usage=true` | Tracks device memory alloc/free over time (HBM usage curve) |
| `--output=./nsys_traces/a100_vllm_m5_k2` | Writes `a100_vllm_m5_k2.nsys-rep` to `./nsys_traces/` |

This produces `./nsys_traces/a100_vllm_m5_k2.nsys-rep`.

### Step 4 — Additional nsys runs for m and k sweeps

```bash
# m=10 example
nsys profile \
    --trace=cuda,nvtx,cublas \
    --cuda-memory-usage=true \
    --output=./nsys_traces/a100_vllm_m10_k2 \
    python run_tests.py --run run_m

# k=4 example
nsys profile \
    --trace=cuda,nvtx,cublas \
    --cuda-memory-usage=true \
    --output=./nsys_traces/a100_vllm_m5_k4 \
    python run_tests.py --run run_k
```

---

## Viewing PyTorch Profiler Traces in TensorBoard (VS Code / SSH)

The PyTorch Profiler writes TensorBoard-compatible event files to `./profiler_traces/<experiment>/`.

### Option A — Terminal over SSH

```bash
# On the remote VM (in the repo root)
tensorboard --logdir=./profiler_traces --port=6006 --bind_all
```

Then on your **local machine**, open a new terminal and forward the port:

```bash
ssh -L 6006:localhost:6006 <user>@<vm-ip-or-hostname>
```

Open `http://localhost:6006` in your browser. Go to the **PyTorch Profiler** tab.

### Option B — VS Code Remote SSH (recommended)

1. Open VS Code and connect to the VM via **Remote – SSH** (`Ctrl+Shift+P` → `Remote-SSH: Connect to Host`).
2. Open the repository folder on the remote.
3. Open a terminal inside VS Code (`Ctrl+\``) and run:
   ```bash
   tensorboard --logdir=./profiler_traces --port=6006
   ```
4. VS Code detects the port automatically and shows a popup — click **Open in Browser**, or go to the **Ports** tab (bottom panel) and click the globe icon next to port 6006.

### What to look for in TensorBoard

- **Overview:** Wall-time breakdown across CPU, CUDA, and idle
- **Operator view:** Per-operator CUDA time — compare `no_opt` vs `int8` vs `nf4`
- **Trace view:** Chrome-style timeline — look for `batched_drafter.generate` spans (the `record_function` label used in `generate_with_profiler`)
- **Memory view:** Peak VRAM curve per step — verify NF4 reduces footprint vs baseline

---

## Viewing Nsight Systems Traces

### Option A — GUI on a local machine (recommended)

1. Copy the `.nsys-rep` file from the VM:
   ```bash
   scp <user>@<vm>:~/path/to/nsys_traces/a100_vllm_m5_k2.nsys-rep ./
   ```
2. Download and install [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) on your local machine.
3. Open Nsight Systems → **File → Open** → select `a100_vllm_m5_k2.nsys-rep`.

### Option B — CLI report on the VM (no GUI required)

```bash
nsys stats ./nsys_traces/a100_vllm_m5_k2.nsys-rep
```

This prints a text summary of kernel statistics, NVTX ranges, and CUDA API call counts directly in the terminal.

### What to look for in the Nsight Systems timeline

- **NVTX rows:** Look for `vllm.generate`, `vllm.logprob_extraction`, `pipeline.retrieve`, `pipeline.sample`, and `pipeline.draft` rows — each annotated with `nvtx.range_push/pop` in the code
- **CUDA kernel row:** Verify kernel overlap within `vllm.generate` — continuous batching should show near-zero gaps
- **Memory row:** HBM allocation spikes during model load vs flat utilisation during inference
- **cuBLAS row:** Tensor-core GEMM kernels fired during attention and MLP layers

---

## Output Data — Where It Is Saved & What It Contains

### Location

All results are written to `./drafter_output/` (auto-created on first run):

| Run | Output file |
|---|---|
| `test` | `./drafter_output/test.json` |
| `test_p` | `./drafter_output/test_profiler.json` |
| `no_opt` | `./drafter_output/no_opt.json` |
| `nf4` | `./drafter_output/bnb.json` |
| `int8` | `./drafter_output/int8.json` |
| `run_vllm` | `./drafter_output/vllm_m5_k2.json` |
| `run_m` | `./drafter_output/vllm_m{5,10,15,20}.json` |
| `run_k` | `./drafter_output/vllm_k{2,4,6,10}.json` |

### JSON schema — one record per question

```json
{
  "question_id":    "tc_2351",
  "question":       "Which country hosted the 1998 FIFA World Cup?",
  "gold_answers":   ["France"],
  "retrieval_time_s": 0.012,
  "sampling_time_s":  0.003,
  "drafting_time_s":  1.847,
  "drafts": [
    {
      "subset_index":  0,
      "answer_draft":  "France",
      "rationale":     "Evidence [1] states the 1998 World Cup was held in France.",
      "draft_logprob": -3.142
    }
  ],
"drafts_tokens_in"  : 4000,   
"drafts_tokens_out"  : 500  
}
```

### What each field records

| Field | Source | Description |
|---|---|---|
| `question_id` | TriviaQA loader | Unique question identifier |
| `question` | TriviaQA loader | Raw question string Q |
| `gold_answers` | TriviaQA loader | List of acceptable gold answers for EM scoring |
| `retrieval_time_s` | `ContrieverRetriever` | Wall-clock seconds for FAISS top-k retrieval |
| `sampling_time_s` | `MultiPerspectiveSampler` | Wall-clock seconds for k-means subset generation |
| `drafting_time_s` | `BatchedDrafter` / vLLM | Wall-clock seconds for all m drafts (single batched call) |
| `drafts[].subset_index` | `DraftOutput` | Index j of the document subset (0-based) |
| `drafts[].answer_draft` | `DraftOutput.parse_draft_output` | Extracted answer α_j from model output |
| `drafts[].rationale` | `DraftOutput.parse_draft_output` | Extracted rationale β_j from model output |
| `drafts[].draft_logprob` | `DraftOutput.compute_seq_logprob` | Log P(β_j, α_j \| Q, docs) — sum of per-token log-probs |
| `drafts_tokens_in` | `drafter_pipeline` | total prompt tokens across all m drafts |
| `drafts_tokens_out` |`drafter_pipeline` | total generated tokens across all m drafts |

### How the data feeds the verifier

Each JSON record is the serialised form of a `VerifierInput` dataclass. The downstream verifier reads these files, selects the best draft using the `draft_logprob` scores and its own re-ranking, and then evaluates exact-match (EM) accuracy against `gold_answers` using `answer_in_response()`.

### Progress logging

While the pipeline runs, draft coverage is printed to stdout every **100 questions** (`log_every=100`):

```
INFO | Processed 100 | draft coverage so far 72.00%
INFO | Processed 200 | draft coverage so far 70.50%
```

Coverage = fraction of questions where **at least one** of the m drafts contains a gold answer.

---

## Quick Reference — Run Map

```
python run_tests.py --run test        # TinyLlama, no opt, no profiler (sanity check)
python run_tests.py --run test_p      # TinyLlama, no opt, PyTorch Profiler
python run_tests.py --run no_opt      # Mistral-7B, baseline, PyTorch Profiler, full data
python run_tests.py --run nf4         # Mistral-7B, NF4 4-bit, PyTorch Profiler, full data
python run_tests.py --run int8        # Mistral-7B, INT8, PyTorch Profiler, full data
python run_tests.py --run run_vllm    # Mistral-7B, vLLM, NVTX (use with nsys), full data
python run_tests.py --run run_m       # Mistral-7B, vLLM, m ∈ {5,10,15,20}, full data
python run_tests.py --run run_k       # Mistral-7B, vLLM, k ∈ {2,4,6,10}, full data
```
