variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Vertex AI, GCS, and Artifact Registry"
  type        = string
  default     = "us-central1"
}

variable "project_name" {
  description = "Prefix for all resource names and labels"
  type        = string
  default     = "triviaqa-rag"
}

variable "gcs_bucket_name" {
  description = "Globally unique GCS bucket name for FAISS index, corpus, and results"
  type        = string
}
