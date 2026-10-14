"""
AIShield API.

Public:  GET /health, GET /, GET /metrics, POST /auth/token
Auth:    POST /predict, POST /feedback, POST /copilot/ask          (any authenticated role)
Admin:   GET /admin/model, GET /admin/drift, POST /admin/heal, POST /admin/rollback
"""

import asyncio
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.schemas import (
    CopilotRequest,
    FeedbackRequest,
    LoginRequest,
    PredictionResponse,
    TokenResponse,
    TransactionRequest,
)
from configs.settings import settings
from src.copilot.assistant import answer_question
from src.deployment import registry
from src.deployment.service import model_service
from src.feature_engineering.features import FEATURE_COLUMNS, engineer_features
from src.monitoring import feedback as feedback_store
from src.monitoring import metrics as prom
from src.monitoring.buffer import PredictionBuffer
from src.monitoring.drift import detect_drift, save_report
from src.security import auth
from src.security.rate_limit import rate_limit
from src.self_healing import engine
from src.utils.logger import get_logger

logger = get_logger(__name__)
buffer = PredictionBuffer(settings.monitoring_buffer_size)


def _sync_model_gauge() -> None:
    if model_service.version is not None:
        prom.MODEL_VERSION.set(model_service.version)


async def _monitor_loop() -> None:
    """Periodic drift check (and optional auto-heal) so Prometheus gauges stay fresh."""
    while True:
        await asyncio.sleep(settings.drift_check_interval_seconds)
        try:
            await asyncio.to_thread(monitor_tick)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Background monitor tick failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    auth.assert_secure_config()
    if model_service.load():
        _sync_model_gauge()
    task = (
        asyncio.create_task(_monitor_loop()) if settings.drift_check_interval_seconds > 0 else None
    )
    yield
    if task:
        task.cancel()


app = FastAPI(
    title=settings.app_name,
    description="Real-time fraud detection microservice",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    response = await call_next(request)
    route = request.scope.get("route")
    prom.HTTP_REQUESTS.labels(
        method=request.method,
        path=route.path if route else "unmatched",
        status=response.status_code,
    ).inc()
    return response


def _to_frame(req: TransactionRequest) -> pd.DataFrame:
    raw = {**req.features, "Time": req.time, "Amount": req.amount}
    return engineer_features(pd.DataFrame([raw]))


@app.get("/health", tags=["system"])
def health_check() -> dict:
    """Liveness/readiness probe target for Kubernetes."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
        "model_loaded": model_service.is_loaded,
        "model_version": model_service.version,
    }


@app.get("/", tags=["system"])
def root() -> dict:
    return {"message": f"{settings.app_name} API is running. See /docs for endpoints."}


@app.get("/metrics", tags=["system"], include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/auth/token", response_model=TokenResponse, tags=["auth"], dependencies=[Depends(rate_limit)]
)
def login(body: LoginRequest) -> TokenResponse:
    role = auth.authenticate(body.username, body.password)
    if role is None:
        raise HTTPException(401, "Invalid credentials")
    return TokenResponse(
        access_token=auth.create_access_token(body.username, role),
        expires_in=settings.jwt_expire_minutes * 60,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
    tags=["inference"],
    dependencies=[Depends(rate_limit), Depends(auth.get_current_user)],
)
def predict(req: TransactionRequest) -> dict:
    if not model_service.is_loaded:
        raise HTTPException(503, "No production model loaded")
    start = time.perf_counter()
    features = _to_frame(req)
    result = model_service.predict(features, explain=req.explain, top_k=settings.explanation_top_k)[
        0
    ]
    buffer.add(features.iloc[0].to_dict())
    prom.PREDICTIONS.labels(outcome="fraud" if result["is_fraud"] else "legit").inc()
    prom.FRAUD_SCORE.observe(result["fraud_probability"])
    prom.PREDICTION_LATENCY.observe(time.perf_counter() - start)
    return result


@app.post(
    "/feedback",
    tags=["inference"],
    dependencies=[Depends(rate_limit), Depends(auth.get_current_user)],
)
def feedback(req: FeedbackRequest) -> dict:
    """Submit the ground-truth label for a transaction (fuel for retraining)."""
    total = feedback_store.add_feedback(_to_frame(req).iloc[0].to_dict(), int(req.is_fraud))
    return {"stored": True, "feedback_rows": total}


@app.get("/admin/model", tags=["admin"], dependencies=[Depends(auth.require_role("admin"))])
def model_info() -> dict:
    return {"production": registry.get_production_version(), "versions": registry.list_versions()}


def _current_drift_report():
    reference = model_service.reference
    if reference is None:
        raise HTTPException(503, "No production model loaded")
    report = detect_drift(
        reference, buffer.to_frame() if len(buffer) else pd.DataFrame(columns=FEATURE_COLUMNS)
    )
    save_report(report)
    prom.DRIFT_FEATURE_SHARE.set(report.drift_share)
    prom.DRIFT_MAX_PSI.set(report.max_psi)
    return report


@app.get("/admin/drift", tags=["admin"], dependencies=[Depends(auth.require_role("admin"))])
def drift() -> dict:
    report = _current_drift_report()
    return {"action": engine.decide(report).value, "report": report.to_dict()}


def run_heal_cycle() -> dict:
    """Drift check -> decide -> (if warranted) retrain, compare, promote, hot-reload."""
    report = _current_drift_report()
    action = engine.decide(report)
    if action != engine.Action.RETRAIN:
        return {"action": action.value, "executed": False, "reason": "No retrain warranted"}
    outcome = engine.heal(feedback_store.load_feedback())
    if outcome.promoted:
        model_service.load()
        _sync_model_gauge()
        buffer.clear()
        feedback_store.clear_feedback()
    return outcome.to_dict()


def monitor_tick() -> dict:
    """One background-monitor iteration; only heals automatically when ``auto_heal`` is on."""
    if not model_service.is_loaded or len(buffer) == 0:
        return {"skipped": True}
    if settings.auto_heal:
        return run_heal_cycle()
    report = _current_drift_report()
    return {"action": engine.decide(report).value, "executed": False}


@app.post("/admin/heal", tags=["admin"], dependencies=[Depends(auth.require_role("admin"))])
def heal() -> dict:
    """Run one drift -> decide -> retrain -> promote cycle now."""
    return run_heal_cycle()


@app.post("/admin/rollback", tags=["admin"], dependencies=[Depends(auth.require_role("admin"))])
def rollback() -> dict:
    try:
        version = registry.rollback()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    model_service.load()
    _sync_model_gauge()
    return {"production": version}


@app.post(
    "/copilot/ask",
    tags=["copilot"],
    dependencies=[Depends(rate_limit), Depends(auth.get_current_user)],
)
def copilot(req: CopilotRequest) -> dict:
    return answer_question(req.question)


logger.info("AIShield API initialized (environment=%s)", settings.environment)
