"""
Ops copilot: answers questions about model health, drift, and self-healing actions.

It builds a compact telemetry context (registry, latest drift report, healing log) and
either sends it to the Anthropic API (when ANTHROPIC_API_KEY is set) or answers from
deterministic rules. Only aggregate metrics are sent to the LLM -- never transaction rows.
"""

import json
from typing import Any, Dict

import httpx

from configs.settings import settings
from src.deployment import registry
from src.monitoring.drift import load_latest_report
from src.self_healing.engine import read_healing_log
from src.utils.logger import get_logger

logger = get_logger(__name__)
_API_URL = "https://api.anthropic.com/v1/messages"
_SYSTEM = (
    "You are the AIShield ops copilot for a fraud-detection MLOps platform. Answer ONLY from the "
    "JSON telemetry provided. Be concise and concrete. If the telemetry does not contain the "
    "answer, say so. Never invent numbers."
)


def build_context() -> Dict[str, Any]:
    production = registry.get_production_version()
    versions = {v["version"]: v for v in registry.list_versions()}
    drift = load_latest_report() or {}
    drift.pop("feature_scores", None)
    top = None
    latest = load_latest_report()
    if latest and latest.get("feature_scores"):
        ranked = sorted(latest["feature_scores"].items(), key=lambda kv: kv[1]["psi"], reverse=True)
        top = [{"feature": k, "psi": round(v["psi"], 3)} for k, v in ranked[:5]]
    return {
        "production_version": production,
        "production_metrics": versions.get(production, {}).get("metrics") if production else None,
        "registered_versions": [
            {"version": v["version"], "status": v["status"], "pr_auc": v["metrics"].get("pr_auc")}
            for v in versions.values()
        ],
        "latest_drift": drift or None,
        "top_drifting_features": top,
        "recent_healing_actions": read_healing_log(5),
    }


def _rule_based(question: str, ctx: Dict[str, Any]) -> str:
    q = question.lower()
    if any(w in q for w in ("drift", "shift", "distribution")):
        drift = ctx["latest_drift"]
        if not drift:
            return "No drift report exists yet. Run a drift check (GET /admin/drift) first."
        if drift.get("insufficient_data"):
            return f"Not enough recent traffic ({drift['n_current']} rows) to assess drift."
        top = ", ".join(
            f"{t['feature']} (PSI {t['psi']})" for t in ctx["top_drifting_features"] or []
        )
        state = "DETECTED" if drift["dataset_drift"] else "not detected"
        return (
            f"Dataset drift {state}: {len(drift['drifted_features'])} features drifted "
            f"({drift['drift_share']:.0%}), max PSI {drift['max_psi']:.2f}. Top movers: {top}."
        )
    if any(w in q for w in ("heal", "retrain", "rollback", "action", "promot")):
        actions = ctx["recent_healing_actions"]
        if not actions:
            return "No self-healing actions have run yet."
        last = actions[-1]
        return (
            f"Last action: {last['action']} at {last['created_at']} - {last['reason']} "
            f"(champion v{last['champion_version']}, challenger v{last['challenger_version']})."
        )
    if any(w in q for w in ("model", "version", "performance", "metric", "accuracy", "recall")):
        metrics = ctx["production_metrics"]
        if ctx["production_version"] is None or not metrics:
            return "No production model is registered yet."
        return (
            f"Serving v{ctx['production_version']}: PR-AUC {metrics['pr_auc']:.3f}, precision "
            f"{metrics['precision']:.3f}, recall {metrics['recall']:.3f} at threshold "
            f"{metrics['threshold']:.2f} (held-out test set)."
        )
    return (
        "I can report on the serving model and its metrics, data drift, and recent self-healing "
        "actions. Try: 'is there drift?', 'which model is live?', or 'what did self-healing do?'"
    )


def _ask_llm(question: str, ctx: Dict[str, Any]) -> str:
    prompt = f"Telemetry:\n{json.dumps(ctx, default=str)}\n\nQuestion: {question}"
    response = httpx.post(
        _API_URL,
        headers={
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": settings.copilot_model,
            "max_tokens": 500,
            "system": _SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=20.0,
    )
    response.raise_for_status()
    return "".join(b.get("text", "") for b in response.json()["content"]).strip()


def answer_question(question: str) -> Dict[str, str]:
    ctx = build_context()
    if settings.anthropic_api_key:
        try:
            return {"answer": _ask_llm(question, ctx), "source": "llm"}
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning("LLM copilot failed (%s); falling back to rules", type(exc).__name__)
    return {"answer": _rule_based(question, ctx), "source": "rules"}
