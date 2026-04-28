.DEFAULT_GOAL := help
SHELL         := /bin/bash

# Configuration ────────────────────────────────────────────────────────────────

DATA_DIR ?= /data
RESULTS  ?= output/results.json
N        ?= 500

# ENV=dev  → 100k-passage index, 500-example smoke test
# ENV=prod → full 21M-passage index, full TriviaQA validation set
ENV ?= dev
ifeq ($(ENV),prod)
  RUN_FULL_INDEX := true
  EVAL_SAMPLE    :=
else
  RUN_FULL_INDEX := false
  EVAL_SAMPLE    := --sample 500
endif

-include .env
export

# Derived variables ────────────────────────────────────────────────────────────

ARTIFACT_REGISTRY := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(PROJECT_NAME)
IMAGE             := $(ARTIFACT_REGISTRY)/triviaqa-rag
VERTEX_SA         := $(PROJECT_NAME)-vertex@$(GCP_PROJECT_ID).iam.gserviceaccount.com

# ── Development ────────────────────────────────────────────────────────────────

.PHONY: setup
setup:  ## Install all Python dependencies via uv
	uv sync --all-extras

.PHONY: fmt
fmt:  ## Auto-format with ruff
	uv run ruff format src/ tests/

.PHONY: lint
lint:  ## Lint with ruff
	uv run ruff check src/ tests/

.PHONY: test
test:  ## Run pytest
	uv run pytest -v

# ── Data & index ───────────────────────────────────────────────────────────────

.PHONY: download-data
download-data:  ## Download TriviaQA + DPR Wikipedia passage corpus
	DATA_DIR=$(DATA_DIR) bash scripts/download_data.sh

.PHONY: build-index
build-index:  ## Embed all 21M passages with Contriever and build FAISS index
	DATA_DIR=$(DATA_DIR) bash scripts/build_index.sh

.PHONY: build-index-subset
build-index-subset:  ## Build a 100k-passage index for smoke-testing
	DATA_DIR=$(DATA_DIR) PASSAGES_SUBSET=100000 bash scripts/build_index.sh

# ── Local evaluation ───────────────────────────────────────────────────────────

.PHONY: eval-mistral
eval-mistral:  ## Run TriviaQA eval with Mistral-7B (ENV=prod for full eval)
	uv run python -m rag.pipeline \
		--model mistralai/Mistral-7B-Instruct-v0.1 \
		--index-path $(DATA_DIR)/faiss_contriever.index \
		--meta-path $(DATA_DIR)/passages_meta.pkl \
		--results-path $(RESULTS) \
		$(EVAL_SAMPLE)

.PHONY: eval-mixtral
eval-mixtral:  ## Run TriviaQA eval with Mixtral-8x7B (4-GPU)
	uv run python -m rag.pipeline \
		--model mistralai/Mixtral-8x7B-Instruct-v0.1 \
		--index-path $(DATA_DIR)/faiss_contriever.index \
		--meta-path $(DATA_DIR)/passages_meta.pkl \
		--tensor-parallel-size 4 \
		--results-path mixtral_$(RESULTS)

.PHONY: eval-sample
eval-sample:  ## Quick smoke-test on N examples (default N=500)
	uv run python -m rag.pipeline \
		--model mistralai/Mistral-7B-Instruct-v0.1 \
		--index-path $(DATA_DIR)/faiss_contriever.index \
		--meta-path $(DATA_DIR)/passages_meta.pkl \
		--sample $(N) \
		--results-path sample_$(RESULTS)

.PHONY: results
results:  ## Print results table from output/results.json
	uv run python scripts/print_results.py $(RESULTS)

# ── Docker ─────────────────────────────────────────────────────────────────────

.PHONY: docker-build
docker-build:  ## Build linux/amd64 Docker image
	docker build --platform linux/amd64 -t $(IMAGE):latest .

.PHONY: docker-push
docker-push: docker-build  ## Build and push Docker image to Artifact Registry
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev --quiet
	docker push $(IMAGE):latest

# ── Vertex AI ──────────────────────────────────────────────────────────────────

.PHONY: vertex-submit
vertex-submit:  ## Submit pipeline as a Vertex AI custom job (ENV=dev|prod)
	@echo "Submitting Vertex AI job (ENV=$(ENV))..."
	IMAGE_URI=$(IMAGE):latest \
	GCS_BUCKET=$(GCS_BUCKET) \
	HF_TOKEN=$(HF_TOKEN) \
	RUN_FULL_INDEX=$(RUN_FULL_INDEX) \
	VERTEX_SA=$(VERTEX_SA) \
	envsubst < infra/vertex_job.yaml.tpl > /tmp/vertex_job.yaml
	gcloud ai custom-jobs create \
		--region=$(GCP_REGION) \
		--display-name=triviaqa-rag-$(shell date +%Y%m%d-%H%M%S) \
		--config=/tmp/vertex_job.yaml \
		--project=$(GCP_PROJECT_ID) \
	| tee /tmp/vertex_job_id.txt
	@echo "Job submitted. Run 'make vertex-logs' to stream logs."

.PHONY: vertex-logs
vertex-logs:  ## Stream logs from the most recent Vertex AI job
	$(eval JOB_NAME := $(shell grep -o 'projects/[^ ]*' /tmp/vertex_job_id.txt | head -1))
	@test -n "$(JOB_NAME)" || (echo "ERROR: no job name found — run make vertex-submit first" && exit 1)
	gcloud ai custom-jobs stream-logs $(JOB_NAME) \
		--region=$(GCP_REGION) \
		--project=$(GCP_PROJECT_ID)

.PHONY: vertex-status
vertex-status:  ## Show status of the last 5 Vertex AI jobs
	gcloud ai custom-jobs list \
		--region=$(GCP_REGION) \
		--project=$(GCP_PROJECT_ID) \
		--limit=5 \
		--format="table(displayName,state,createTime,endTime)"

.PHONY: fetch-results
fetch-results:  ## Download results.json from GCS and print the table
	gsutil cp gs://$(GCS_BUCKET)/results.json $(RESULTS)
	uv run python scripts/print_results.py $(RESULTS)

.PHONY: clear-index-cache
clear-index-cache:  ## Delete cached FAISS index from GCS (required before switching ENV=dev↔prod)
	gsutil rm gs://$(GCS_BUCKET)/faiss_contriever.index
	gsutil rm gs://$(GCS_BUCKET)/passages_meta.pkl
	@echo "Index cache cleared. Next vertex-submit will rebuild the index."

# ── GCP & Infrastructure ───────────────────────────────────────────────────────

.PHONY: gcp-enable-apis
gcp-enable-apis:  ## Enable required GCP APIs (run once per project)
	gcloud services enable \
		aiplatform.googleapis.com \
		artifactregistry.googleapis.com \
		secretmanager.googleapis.com \
		storage.googleapis.com \
		--project=$(GCP_PROJECT_ID)

.PHONY: infra-init
infra-init:  ## Initialize Terraform
	terraform -chdir=infra init

.PHONY: infra-fmt
infra-fmt:  ## Auto-format Terraform files
	terraform -chdir=infra fmt

.PHONY: infra-lint
infra-lint:  ## Validate Terraform configuration
	terraform -chdir=infra fmt -check
	terraform -chdir=infra validate

.PHONY: infra-plan
infra-plan:  ## Preview infrastructure changes
	terraform -chdir=infra plan

.PHONY: infra-apply
infra-apply:  ## Provision GCS bucket, Artifact Registry, and service account
	terraform -chdir=infra apply

.PHONY: infra-destroy
infra-destroy:  ## Tear down all GCP infrastructure
	terraform -chdir=infra destroy || (echo "ERROR: destroy failed. If the bucket is not empty, run 'make fetch-results' then 'make clear-index-cache' first." && exit 1)

# Help ─────────────────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help
	@printf "\n\033[1mStandard RAG\033[0m\n"
	@printf "\033[1mUsage:\033[0m  make \033[36m<target>\033[0m [ENV=dev|prod] [N=500]\n\n"
	@printf "\033[1mTargets:\033[0m\n"
	@awk 'BEGIN {FS = ":.*##"} \
	    /^# ── / { \
	        gsub(/^# ── /, ""); gsub(/ ─+$$/, ""); \
	        printf "\n  \033[1m%s\033[0m\n", $$0; next \
	    } \
	    /^[a-zA-Z_-]+:.*## / && !/^help:/ { \
	        printf "    \033[36m%-22s\033[0m %s\n", $$1, $$2 \
	    }' $(firstword $(MAKEFILE_LIST))
	@printf "\n\033[1mExamples:\033[0m\n"
	@printf "    make vertex-submit              # smoke test (100k passages, 500 examples)\n"
	@printf "    make vertex-submit ENV=prod     # full eval (21M passages, 11k examples)\n"
	@printf "    make eval-sample N=100          # local quick test on 100 examples\n"
	@printf "\n"
