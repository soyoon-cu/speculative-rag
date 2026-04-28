terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  labels = {
    project    = var.project_name
    managed_by = "terraform"
  }
}

resource "google_storage_bucket" "data" {
  name                        = var.gcs_bucket_name
  location                    = var.region
  force_destroy               = false
  uniform_bucket_level_access = true
  labels                      = local.labels

  versioning {
    enabled = true
  }
}

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = var.project_name
  description   = "Docker images for ${var.project_name}"
  format        = "DOCKER"
  labels        = local.labels
}

resource "google_service_account" "vertex" {
  account_id   = "${var.project_name}-vertex"
  display_name = "${var.project_name} Vertex AI job runner"
}

resource "google_project_iam_member" "vertex_gcs" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.vertex.email}"
}

resource "google_project_iam_member" "vertex_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.vertex.email}"
}

resource "google_project_iam_member" "vertex_secret" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.vertex.email}"
}

resource "google_project_iam_member" "vertex_registry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.vertex.email}"
}
