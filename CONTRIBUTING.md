# Contributing

This project is course work for **COMS E6998 — High Performance Machine Learning (HPML)** at **Columbia University**, taught by **Dr. Kaoutar El Maghraoui**, Adjunct Professor of Computer Science and Principal Research Scientist at IBM T.J. Watson Research Center, NY.

## Team

| Name | GitHub |
|------|--------|
| Alexandar Vassilev | [@alex-is-busy-coding](https://github.com/alex-is-busy-coding) |
| Soyoon Park | [@soyoon-cu](https://github.com/soyoon-cu) |
| Hsuan-Ting Lin | [@Hsuan-Ting](https://github.com/Hsuan-Ting) |
| Rupeet Kaur | [@RupeetK](https://github.com/RupeetK) |

## Workflow

- **Branch from `main`** for all changes. Use descriptive branch names: `feature/speculative-pipeline`, `fix/oom-a100`, etc.
- **One pull request per logical change.** Keep PRs focused — a bug fix and a new feature should be separate PRs.
- **Request a review** from at least one other team member before merging.
- **Squash or rebase** before merging to keep the history clean.

## Commit style

Use short, imperative commit messages with a scope:

```
feat(pipeline): add verifier log-probability scoring
fix(index): reduce passage batch size to prevent OOM
chore(infra): update Terraform outputs
docs(readme): add GPU quota instructions
test(retriever): add FAISS search unit tests
```

Format: `<type>(<scope>): <message>`

Common types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`.
Common scopes: `pipeline`, `index`, `retriever`, `vllm`, `infra`, `docker`, `readme`, `scripts`.

## Code style

**Python** (run from `standard-rag/`):

```bash
make fmt     # auto-format with ruff
make lint    # lint with ruff
make test    # run pytest
```

**Terraform** (run from `standard-rag/`):

```bash
make infra-fmt   # auto-format .tf files
make infra-lint  # fmt --check + validate
```

All checks should pass before opening a PR. `infra-lint` requires `terraform init` to have been run at least once (`make infra-init`).

## Project layout

Each subdirectory (`standard-rag/`, `speculative-rag/`) is an independent project with its own environment and Docker image. Changes to one should not break the other. See each subdirectory's `README.md` for setup instructions.

## Questions

Open a GitHub issue or reach out to the team directly.
