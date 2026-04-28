# Security Policy

This is an academic research project for **COMS E6998 — High Performance Machine Learning** at Columbia University. It is not a production system, but we take credential hygiene seriously.

## Sensitive files

The following are **never committed to the repository**:

- `.env` — GCP project ID, HuggingFace token, bucket name
- `*.tfvars` — Terraform variable overrides
- `*.tfstate` / `*.tfstate.backup` — Terraform state (may contain resource IDs)
- Any service account key files (`*.json` credential files)

These are covered by `.gitignore`. If you accidentally stage one, remove it immediately:

```bash
git rm --cached <file>
git commit -m "chore(git): remove accidentally staged secret"
```

If a secret has already been pushed, **rotate it immediately** — assume it is compromised.

## HuggingFace tokens

Use a **read-only** HuggingFace token scoped only to the models needed (`mistralai/Mistral-7B-Instruct-v0.1`). Do not use a write token or your primary account token.

## GCP service account

The Terraform-managed service account is granted only the minimum roles required:
- `storage.objectAdmin` — read/write to the project GCS bucket
- `logging.logWriter` — write Vertex AI job logs
- `artifactregistry.reader` — pull the Docker image

Do not grant `owner`, `editor`, or `project-level admin` roles.

## Reporting a vulnerability

This is a course project with no public-facing deployment. If you find a credential accidentally exposed in the repository, contact the team directly via GitHub.
