# Contributing to AIShield

## Branching strategy

We use **trunk-based development with short-lived feature branches**:

- `main` — always deployable, protected, only updated via reviewed PRs
- Feature branches: `phase-<n>/<short-description>`, e.g. `phase-1/schema-validation`
- Branch from `main`, PR back into `main`. No long-lived `develop` branch — with a
  3-person team, a second permanent branch just adds merge overhead.

## Workflow

1. Pick up an issue tagged with the current phase (see [README roadmap](README.md#build-roadmap))
2. `git checkout -b phase-1/your-task-name`
3. Commit in small, logical chunks (see commit convention below)
4. Push and open a PR against `main` using the PR template — it'll auto-fill
5. CI must pass (lint, security scan, tests) before merge is allowed
6. At least 1 teammate approval required (CODEOWNERS auto-assigns the right reviewer)
7. Squash-merge to keep `main` history clean

## Commit convention

We use [Conventional Commits](https://www.conventionalcommits.org/) — makes history
scannable and could later auto-generate a changelog:

```
feat(feature-engineering): add velocity features
fix(api): correct health check response schema
test(validation): add schema violation test cases
docs(readme): update phase 1 status
chore(ci): add pip-audit to security job
```

Types: `feat`, `fix`, `test`, `docs`, `chore`, `refactor`, `perf`

## Before opening a PR

```bash
pytest tests/ --cov=src --cov=api --cov-report=term-missing
bandit -r src api scripts -ll
pylint src api scripts
black src api tests
isort src api tests
```

Consider setting up `pre-commit` locally so these run automatically:
```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Code ownership

See [`.github/CODEOWNERS`](.github/CODEOWNERS) — PRs touching a given module
auto-request review from that module's owner. Update this file as the team's
areas of responsibility shift between phases.

## Data & secrets

- Never commit real datasets, trained model files, or `.env` — `.gitignore`
  already excludes these, don't fight it
- Sample/synthetic data for local dev goes in `data/raw/` but stays out of git
  (share via a team drive link in Slack/Discord instead)
