"""Check the judge contract and authored data, never model behavior offline."""

import hashlib
import json
import socket

import pytest

from west_workshop import judge_v2
from west_workshop.data import (
    CURRENT_POLICY, calibration_dataset, dataset, judge_validation_dataset,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Judge contract tests must not call a provider or network.")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    yield
    assert not attempts, "A dependency attempted a network call during an offline contract test."


@pytest.mark.parametrize("provider,env_key,model,name", [
    ("openai", "WORKSHOP_OPENAI_JUDGE_MODEL", "test-openai-model", "policy_judge"),
    ("databricks", "WORKSHOP_DATABRICKS_JUDGE_MODEL", "test-endpoint", "review_judge"),
])
def test_factory_uses_the_configured_route_and_frozen_contract(
    monkeypatch, provider, env_key, model, name,
):
    import mlflow.genai.judges

    monkeypatch.setenv(env_key, model)
    calls = []
    sentinel = object()

    def capture(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(mlflow.genai.judges, "make_judge", capture)
    assert judge_v2.build_judge(provider, name=name) is sentinel
    assert calls == [{
        "name": name,
        "instructions": judge_v2.INSTRUCTIONS,
        "model": f"{provider}:/{model}",
        "feedback_value_type": bool,
        "generate_rationale_first": True,
        "inference_params": {"temperature": 0, "max_tokens": 500},
    }]


def test_real_mlflow_factory_serializes_and_restores_without_model_execution():
    from mlflow.genai.judges import Judge
    from mlflow.genai.scorers import Scorer

    judge = judge_v2.build_judge("openai")
    assert isinstance(judge, Judge)
    assert judge.name == "policy_judge"
    serialized = json.loads(json.dumps(judge.model_dump()))
    contract = serialized["instructions_judge_pydantic_data"]
    assert contract["instructions"] == judge_v2.INSTRUCTIONS
    assert contract["model"] == "openai:/gpt-4o-mini"
    assert contract["feedback_value_type"]["type"] == "boolean"
    assert contract["generate_rationale_first"] is True
    assert contract["inference_params"] == {"temperature": 0, "max_tokens": 500}
    restored = Scorer.model_validate(serialized)
    assert restored.model_dump() == serialized


def test_definition_fingerprint_is_frozen_and_does_not_hide_route_identity():
    # Any rubric change after independent controls are revealed needs a new version.
    assert judge_v2.JUDGE_VERSION == "policy_judge_v2"
    digest = hashlib.sha256(json.dumps(
        judge_v2.definition(), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    assert digest == judge_v2.JUDGE_DEFINITION_SHA256
    assert digest == "cbec7a786d1736e939bac148ec6ae2fd588a13844666fb2cf0cce5a571fd9da3"
    assert not {"model", "provider", "name", "mlflow_version"} & judge_v2.definition().keys()
    changed = judge_v2.definition()
    changed["inference_params"]["max_tokens"] = 1
    assert judge_v2.definition()["inference_params"]["max_tokens"] == 500


def test_v1_rubric_is_preserved_as_a_distinct_factory():
    from west_workshop.runtime import _policy_judge

    old_judge = _policy_judge("openai")
    assert hashlib.sha256(old_judge.instructions.encode("utf-8")).hexdigest() == (
        "f12f814f3d4e8c62e667d2b53aebd222d68eb5bf1266363a7dafc5a425aa0b59"
    )
    assert old_judge.instructions != judge_v2.INSTRUCTIONS


def test_development_labels_are_explicit_authored_expectations_and_not_prompt_inputs():
    rows = judge_v2.development_controls()
    assert len(rows) == 8
    assert [row["expectations"]["authored_human_label"] for row in rows] == [True, False] * 4
    assert "{{ expectations }}" not in judge_v2.INSTRUCTIONS
    assert judge_v2.INSTRUCTIONS.count("{{ inputs }}") == 1
    assert judge_v2.INSTRUCTIONS.count("{{ outputs }}") == 1
    assert CURRENT_POLICY in judge_v2.INSTRUCTIONS
    for row in rows:
        assert set(row) == {"inputs", "outputs", "expectations"}
        assert set(row["inputs"]) == {"case_id", "days_since_purchase", "defective", "question"}
        assert type(row["expectations"]["authored_human_label"]) is bool
        assert row["expectations"]["label_origin"] == "authored_development"
        assert row["expectations"]["policy"] == CURRENT_POLICY
        assert row["expectations"]["label_reason"]
        assert row["outputs"]


def test_development_controls_are_separate_from_existing_controls_and_application_cases():
    rows = judge_v2.development_controls()
    ids = [row["inputs"]["case_id"] for row in rows]
    assert len(set(ids)) == len(ids)
    previous_rows = judge_validation_dataset() + calibration_dataset() + dataset()
    assert set(ids).isdisjoint(row["inputs"]["case_id"] for row in previous_rows)
    assert len(judge_validation_dataset()) == 8
    for accepted, rejected in zip(rows[::2], rows[1::2]):
        assert accepted["expectations"]["development_issue"] == rejected["expectations"]["development_issue"]
        assert {k: v for k, v in accepted["inputs"].items() if k != "case_id"} == {
            k: v for k, v in rejected["inputs"].items() if k != "case_id"
        }
        assert accepted["outputs"] != rejected["outputs"]


def test_attribution_and_execution_pairs_place_the_claim_in_the_intended_source():
    attribution_ok, attribution_bad, next_step_ok, next_step_bad = judge_v2.development_controls()[:4]
    claim = "already approved and processed"
    assert claim in attribution_ok["inputs"]["question"]
    assert claim not in attribution_ok["outputs"]
    assert claim in attribution_bad["outputs"]
    assert "Please reach out to our support team to initiate the process!" in next_step_ok["outputs"]
    assert "I will issue your store credit now." in next_step_bad["outputs"]


@pytest.mark.parametrize("pair_offset,days,eligibility", [
    (4, 12, "full_refund"), (6, 91, "store_credit"),
])
def test_stale_window_pairs_change_only_the_explanation_not_the_correct_label(
    pair_offset, days, eligibility,
):
    accepted, rejected = judge_v2.development_controls()[pair_offset:pair_offset + 2]
    assert accepted["inputs"]["days_since_purchase"] == days
    assert accepted["outputs"].splitlines()[0] == f"Eligibility: {eligibility}"
    assert rejected["outputs"].splitlines()[0] == f"Eligibility: {eligibility}"
    assert accepted["outputs"].count("30-day") == 1
    assert rejected["outputs"] == accepted["outputs"].replace("30-day", "90-day")
    assert accepted["expectations"]["authored_human_label"] is True
    assert rejected["expectations"]["authored_human_label"] is False


def test_development_controls_return_fresh_nested_data_with_a_frozen_identity():
    rows = judge_v2.development_controls()
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == "322a25422a57cc07a688fb1f0117f341943f21a583a48183cd7ac11971e7f9ae"
    rows[0]["inputs"]["question"] = "mutated"
    rows[0]["expectations"]["authored_human_label"] = False
    fresh = judge_v2.development_controls()
    assert fresh[0]["inputs"]["question"] != "mutated"
    assert fresh[0]["expectations"]["authored_human_label"] is True
