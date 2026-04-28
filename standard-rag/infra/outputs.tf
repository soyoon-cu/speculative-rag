output "gcs_bucket" {
  description = "GCS bucket for index, corpus, and results"
  value       = google_storage_bucket.data.name
}

output "artifact_registry" {
  description = "Artifact Registry host for docker push/pull"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.project_name}"
}

output "docker_image" {
  description = "Full Docker image URI (append :tag)"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.project_name}/triviaqa-rag"
}

output "service_account" {
  description = "Service account email for Vertex AI jobs"
  value       = google_service_account.vertex.email
}
