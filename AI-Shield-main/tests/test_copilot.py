import httpx

from configs.settings import settings
from src.copilot import assistant
from src.monitoring.drift import DriftReport, save_report
from src.self_healing.engine import HealingOutcome, _record


def test_no_model_answers_gracefully():
    assert "No production model" in assistant.answer_question("which model version?")["answer"]


def test_model_question(trained_env):
    answer = assistant.answer_question("how is model performance?")["answer"]
    assert "PR-AUC" in answer and "v1" in answer


def test_drift_question_states(trained_env):
    assert "No drift report" in assistant.answer_question("any drift?")["answer"]
    save_report(DriftReport(1, 10, insufficient_data=True, created_at="t"))
    assert "Not enough" in assistant.answer_question("any drift?")["answer"]
    report = DriftReport(
        1000,
        900,
        drifted_features=["V1", "V2"],
        drift_share=0.06,
        max_psi=0.9,
        dataset_drift=True,
        created_at="t",
        feature_scores={"V1": {"psi": 0.9, "ks_p_value": 0.0}},
    )
    save_report(report)
    answer = assistant.answer_question("is there drift?")["answer"]
    assert "DETECTED" in answer and "V1" in answer


def test_healing_question(trained_env):
    assert "No self-healing" in assistant.answer_question("what did self-healing do?")["answer"]
    _record(HealingOutcome("retrain", True, True, "promoted it", 1, 2))
    assert "promoted it" in assistant.answer_question("last retrain action?")["answer"]


def test_unknown_question_lists_capabilities():
    assert "I can report" in assistant.answer_question("tell me a joke")["answer"]


def test_llm_path_and_fallback(monkeypatch, trained_env):
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"content": [{"type": "text", "text": "LLM says hi"}]}

    seen = {}

    def fake_post(url, headers, json, timeout):  # noqa: A002
        seen.update(url=url, headers=headers, body=json)
        return Resp()

    monkeypatch.setattr(assistant.httpx, "post", fake_post)
    out = assistant.answer_question("status?")
    assert out == {"answer": "LLM says hi", "source": "llm"}
    assert (
        seen["headers"]["x-api-key"] == "test-key"
        and "Telemetry" in seen["body"]["messages"][0]["content"]
    )

    def boom(*_a, **_k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(assistant.httpx, "post", boom)
    assert assistant.answer_question("which model?")["source"] == "rules"
