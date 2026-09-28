import math

from west_workshop.config import TARGET_MLFLOW, preflight
from west_workshop.runtime import _sanitize


def test_openai_route_needs_a_key(monkeypatch):
    report = preflight("openai")
    assert not report["ready"] and any("OPENAI_API_KEY" in reason for reason in report["reasons"])
    assert report["network_checked"] is False and report["credential_values_included"] is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-offline-0000000000000000")
    assert preflight("openai")["ready"]


def test_openai_route_rejects_redirected_endpoints_and_remote_tracking(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-offline-0000000000000000")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    assert not preflight("openai")["ready"]
    monkeypatch.delenv("OPENAI_BASE_URL")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    assert not preflight("openai")["ready"]


def test_databricks_route_lists_every_missing_setting():
    reasons = " ".join(preflight("databricks")["reasons"])
    for setting in ("Databricks SDK authentication", "WORKSHOP_DATABRICKS_MODEL", "MLFLOW_EXPERIMENT_NAME"):
        assert setting in reasons


def test_databricks_route_is_ready_with_complete_settings(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-test")
    monkeypatch.setenv("WORKSHOP_DATABRICKS_MODEL", "databricks-example-chat")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "/Users/someone@example.com/odsc-west-2026")
    report = preflight("databricks")
    if report["versions"]["databricks-sdk"] and report["versions"]["databricks-agents"]:
        assert report["ready"], report["reasons"]
    assert report["judge_model"] == "databricks:/databricks-example-chat"


def test_the_installed_mlflow_matches_the_workshop_pin():
    assert preflight("openai")["versions"]["mlflow"] == TARGET_MLFLOW


def test_evidence_is_sanitized(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-this-is-a-secret-value-1234")
    value = _sanitize({"a": "key sk-this-is-a-secret-value-1234 at https://api.example.com/v1?x=1",
                       "b": [math.nan, 1.5, True, None], "c": object()})
    assert "secret" not in value["a"] and "https://" not in value["a"]
    assert value["b"] == [None, 1.5, True, None]
    assert value["c"] == "object"
