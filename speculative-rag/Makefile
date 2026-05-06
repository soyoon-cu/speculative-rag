# GCP Configuration
PROJECT_ID = hpml-487915
REGION = us-central1
REPO_NAME = hpml-repo
IMAGE_NAME = speculative-rag
IMAGE_URI = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/$(IMAGE_NAME):latest

# The bucket where your 64GB index files are
INDEX_BUCKET = standard-rag-results-2026
# The bucket where you want Vertex to drop your final JSON outputs
OUTPUT_BUCKET = speculative-rag-results-2026

.PHONY: build submit fetch-results

# THE FIX: We replaced local docker build/push with Google Cloud Build.
# This pushes only your code (kilobytes) and compiles the 10GB image directly in the cloud.
build:
	@echo "Delegating Docker build to Google Cloud Build (with layer cache)..."
	gcloud builds submit --config=cloudbuild.yaml --substitutions=_IMAGE_URI=$(IMAGE_URI) .

# This triggers the cloud build automatically before running submit.sh
submit: build
	@echo "Submitting Vertex AI Job..."
	bash submit.sh $(IMAGE_URI) $(REGION) $(ARGS)

fetch-results:
	@echo "Fetching all evaluation results and Nsys traces from GCS..."
	gcloud storage cp -r gs://$(OUTPUT_BUCKET)/verifier_output ./
	@echo "Results downloaded to local ./verifier_output directory"