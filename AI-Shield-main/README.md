# AIShield — Real-Time Fraud Detection MLOps Platform

AIShield is an end-to-end MLOps system for real-time transaction fraud detection.
It covers the full lifecycle: data ingestion and validation, feature engineering,
model training and registry, low-latency serving, explainability, security scanning,
production monitoring, drift detection, and automated self-healing — with an
LLM-powered copilot that reads system telemetry to answer operational questions.

> **Status:** actively in development. See [Build Roadmap](#build-roadmap) below for
> what's implemented vs. planned — this is intentionally transparent rather than
> claiming a finished system.

## Why this project

Most portfolio ML projects stop at "train a model, wrap it in an API." AIShield is
built to demonstrate the *operational* half of ML engineering that production systems
actually require: CI/CD security gates, model registry and versioning, drift detection,
observability, and automated response to model degradation.

## Architecture

```
CSV Dataset → Ingestion → Validation → Feature Engineering → Feature Store
                                                                    │
                                                                    ▼
                                                          Model Training (MLflow)
                                                                    │
                                                                    ▼
                                                            Model Registry
                                                                    │
                                                                    ▼
                        Auth → Rate Limit → FastAPI /predict → SHAP Explainability
                                                                    │
                                                                    ▼
                                                    Prometheus → Grafana Dashboard
                                                                    │
                                                                    ▼
                                                            Drift Detection
                                                                    │
                                                                    ▼
                                                    Self-Healing Decision Engine
                                                          (retrain / rollback / alert)
```

CI/CD: GitHub Actions runs static analysis (Bandit, Pylint, Gitleaks, pip-audit),
unit tests with coverage, Docker build, and a Trivy container scan before deploying
to Kubernetes.

## Tech Stack

| Layer | Tools |
|---|---|
| API | FastAPI, Pydantic, JWT auth |
| ML | scikit-learn, XGBoost, SHAP |
| Experiment tracking | MLflow |
| Feature store | Flat parquet (dev) → Feast (planned) |
| CI/CD | GitHub Actions, Bandit, Gitleaks, pip-audit, Trivy |
| Deployment | Docker, Kubernetes (kind for local dev) |
| Monitoring | Prometheus, Grafana |
| Copilot | LLM reading Prometheus/Grafana/MLflow/logs |

## Getting Started

```bash
# 1. Clone and set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

# 2. Run tests
pytest tests/ --cov=src --cov=api

# 3. Run the API locally
uvicorn api.main:app --reload

# 4. Or run via Docker
docker compose -f docker/docker-compose.yml up --build
```

Visit `http://localhost:8000/docs` for interactive API docs, or `/health` for a
liveness check.

## Project Structure

```
AIShield/
├── api/                  # FastAPI application
├── configs/              # Central settings (pydantic-settings)
├── data/                 # raw / processed / validated / features / drift
├── feature_store/        # Feature storage layer
├── models/               # Registered model artifacts
├── src/
│   ├── ingestion/         # Data loading
│   ├── validation/        # Schema & quality checks
│   ├── feature_engineering/
│   ├── training/          # Model training
│   ├── evaluation/         # Metrics & reports
│   ├── explainability/    # SHAP
│   ├── deployment/        # Model loading for serving
│   ├── monitoring/        # Metrics instrumentation
│   ├── security/          # Auth, scanning helpers
│   ├── self_healing/      # Drift response, retrain triggers
│   └── llm_copilot/       # Ops-facing LLM assistant (planned)
├── tests/
├── docker/
├── kubernetes/
├── prometheus/ grafana/
├── mlflow/
└── .github/workflows/     # CI pipeline
```

## Build Roadmap

- [x] **Phase 0 — Foundation**: repo scaffold, config system, CI skeleton, Dockerfile
- [ ] **Phase 1 — Data pipeline**: ingestion, validation, feature engineering, flat-file feature store
- [ ] **Phase 2 — Model pipeline**: training, evaluation, MLflow registry
- [ ] **Phase 3 — Serving**: `/predict` endpoint wired to the registered model
- [ ] **Phase 4 — Explainability & Auth**: SHAP on predictions, JWT auth, rate limiting
- [ ] **Phase 5 — Security & Deployment**: full CI security gate, Docker + Trivy, Kubernetes (kind)
- [ ] **Phase 6 — Observability**: Prometheus instrumentation, Grafana dashboards
- [ ] **Phase 7 — Drift & Self-Healing**: statistical drift tests, automated retrain/rollback/alert
- [ ] **Phase 8 — LLM Copilot**: natural-language ops assistant over telemetry

## Contributing

This is a team project. See [CONTRIBUTING.md](CONTRIBUTING.md) for branching
strategy, commit conventions, and the PR workflow. Module ownership is defined
in [`.github/CODEOWNERS`](.github/CODEOWNERS).

## License

MIT
# AIShield
# AI-Shield
