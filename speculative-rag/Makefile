# Local config — copy config.mk.example to config.mk and fill in your values.
include config.mk

IMAGE_URI = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/$(IMAGE_NAME):latest

.PHONY: build submit fetch-results

build:
	@echo "Delegating Docker build to Google Cloud Build (with layer cache)..."
	gcloud builds submit --config=cloudbuild.yaml --substitutions=_IMAGE_URI=$(IMAGE_URI) .

submit: build
	@echo "Submitting Vertex AI Job..."
	bash submit.sh $(IMAGE_URI) $(REGION) $(ARGS)

fetch-results:
	@echo "Fetching all evaluation results and Nsys traces from GCS..."
	gcloud storage cp -r gs://$(OUTPUT_BUCKET)/verifier_output ./
	@echo "Results downloaded to local ./verifier_output directory"