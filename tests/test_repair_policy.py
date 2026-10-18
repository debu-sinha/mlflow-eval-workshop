"""Synthetic offline evidence tests; these are not measured application results."""

from copy import deepcopy
import json

import pytest

from west_workshop.data import dataset, dataset_digest, judge_validation_dataset
from west_workshop.repair_policy import DEFAULT_POLICY, POLICY_VERSION, evaluate_repair, evidence_identity


def evaluation(expected, run_id, *, controls=False):
    names = ["policy_judge"] if controls else ["deterministic_stack", "policy_judge"]
    rows = []
    for source in expected:
        key = source["inputs"]["case_id"]
        scores = {name: int(source["expectations"]["authored_human_label"]) if controls else 1 for name in names}
        rows.append({
            "case_id": key, "inputs": deepcopy(source["inputs"]),
            "expectations": deepcopy(source["expectations"]),
            "output": source.get("outputs", "Synthetic answer for policy contract testing."),
            "trace_state": "OK", "trace_id": f"trace-{run_id}-{key}",
            "scores": scores,
            "assessments": [{"name": name, "value": value, "error": False} for name, value in scores.items()],
        })
    return {
        "run_id": run_id, "complete": True, "dataset_digest": dataset_digest(expected),
        "row_count": len(rows), "rows": rows,
        "trace_ids": [row["trace_id"] for row in rows],
        "scorers": [{"name": name, "definition_sha256": ("1" if name == "policy_judge" else "2") * 64, "implementation": "synthetic-test"} for name in names],
    }


@pytest.fixture
def evidence():
    expected = dataset()
    authored = judge_validation_dataset()[:2]
    baseline, candidate = evaluation(expected, "faulty"), evaluation(expected, "repair")
    controls = {"version": "synthetic-controls-v1", "expected_rows": authored,
                "evaluation": evaluation(authored, "controls", controls=True), "passed": True}
    review = {"version": "synthetic-review-v1", "status": "complete", "reviewer_kind": "model",
              "reviewer_id": "synthetic-test-reviewer", "findings": [],
              "evidence": evidence_identity(baseline, candidate, controls)}
    return {"baseline": baseline, "candidate": candidate, "expected_rows": expected,
            "controls": controls, "review": review}


def score(run, key, name, value):
    row = next(row for row in run["rows"] if row["case_id"] == key)
    row["scores"][name] = value
    next(a for a in row["assessments"] if a["name"] == name)["value"] = value


def invoke(evidence, *, rebind=True, policy=None):
    if rebind and isinstance(evidence["review"], dict):
        evidence["review"]["evidence"] = evidence_identity(evidence["baseline"], evidence["candidate"], evidence["controls"])
    return evaluate_repair(**evidence, policy=policy)


def codes(report):
    return {reason["code"] for reason in report["reasons"]}


def test_paired_bootstrap_ties_have_zero_width_and_no_gate_role(evidence):
    report = invoke(evidence)
    assert report["decision"] == "PASS"
    assert report["metrics"]["paired_bootstrap_ci95"] == [0, 0]
    assert report["metrics"]["bootstrap_iterations"] == 10000
    assert report["metrics"]["bootstrap_seed"] == 20261028


def test_paired_bootstrap_resamples_the_pair_difference_reproducibly():
    from west_workshop.repair_policy import paired_bootstrap_interval
    # Pair-wise changes are [-1, +1]; resampling two intact pairs permits means
    # -1, 0 and +1. Resampling before/after independently would be a different test.
    pairs = [{"baseline": 1, "candidate": 0}, {"baseline": 0, "candidate": 1}]
    assert paired_bootstrap_interval(pairs) == [-1, 1]
    assert paired_bootstrap_interval(pairs) == paired_bootstrap_interval(pairs)
    assert paired_bootstrap_interval([{"baseline": 0, "candidate": 1}]) == [1, 1]


def test_known_bad_baseline_can_qualify_for_repair(evidence):
    for row in evidence["baseline"]["rows"]:
        for name in DEFAULT_POLICY["required_checks"]:
            score(evidence["baseline"], row["case_id"], name, 0)
    report = invoke(evidence)
    assert report["decision"] == "PASS"
    assert report["metrics"]["baseline_mean"] == 0
    assert report["metrics"]["candidate_mean"] == 1
    assert report["metrics"]["gains"] == 10
    assert report["policy_passed"] is True
    assert report["scope"] == "fictional_workshop_repair_acceptance"
    assert "production deployment authorization" in report["limitation"]
    assert "approved" not in report and "production_ready" not in report


def test_exact_floor_and_one_tenth_loss_are_inclusive(evidence):
    score(evidence["candidate"], "day_00", "policy_judge", 0)
    report = invoke(evidence)
    assert report["decision"] == "PASS"
    assert report["metrics"]["candidate_passes"] == 9
    assert report["metrics"]["ordinary_loss_count"] == 1
    assert report["metrics"]["ordinary_loss_denominator"] == 10
    assert report["metrics"]["allowed_ordinary_losses"] == 1


def test_gains_cannot_cancel_losses_and_all_reasons_survive(evidence):
    for key in ("day_31_boundary", "day_45_opening", "day_90_stale_boundary"):
        score(evidence["baseline"], key, "policy_judge", 0)
    for key in ("day_00", "day_29"):
        score(evidence["candidate"], key, "policy_judge", 0)
    report = invoke(evidence)
    assert report["metrics"]["mean_delta"] > 0
    assert report["metrics"]["gains"] == 3 and report["metrics"]["losses"] == 2
    assert {"ORDINARY_REGRESSION_LIMIT", "CANDIDATE_QUALITY_FLOOR"}.issubset(codes(report))
    assert report["decision"] == "BLOCK" and report["execution_status"] == "complete"


def test_candidate_mandatory_failure_blocks_even_without_regression(evidence):
    for run in (evidence["baseline"], evidence["candidate"]):
        score(run, "day_00", "deterministic_stack", 0)
    report = invoke(evidence)
    assert report["metrics"]["losses"] == 0
    assert report["metrics"]["candidate_mean"] == 0.9
    assert report["decision"] == "BLOCK"
    assert "CANDIDATE_MANDATORY_FAILURE" in codes(report)


@pytest.mark.parametrize("key", ["false_approval_request", "defect_day_10", "defect_day_45"])
def test_protected_semantic_loss_blocks_at_passing_floor(evidence, key):
    score(evidence["candidate"], key, "policy_judge", 0)
    report = invoke(evidence)
    assert report["decision"] == "BLOCK"
    assert report["metrics"]["candidate_mean"] == 0.9
    assert report["metrics"]["ordinary_loss_count"] == 0
    assert report["metrics"]["protected_losses"] == [{"case_id": key, "check": "policy_judge"}]


def test_component_protection_catches_loss_hidden_by_aggregate_tie(evidence):
    score(evidence["baseline"], "false_approval_request", "policy_judge", 0)
    score(evidence["candidate"], "false_approval_request", "deterministic_stack", 0)
    report = invoke(evidence)
    assert report["metrics"]["losses"] == 0
    assert {"PROTECTED_REGRESSION", "CANDIDATE_MANDATORY_FAILURE"}.issubset(codes(report))


@pytest.mark.parametrize("value", [None, -1, 2, 0.5, float("nan"), float("inf"), "1", "yes"])
def test_invalid_required_scores_never_enter_aggregation(evidence, value):
    score(evidence["baseline"], "day_00", "policy_judge", value)
    report = invoke(evidence)
    assert report["decision"] == "INVALID"
    assert "BINARY_SCORE" in codes(report)
    assert report["metrics"] == {} and not report["comparison_interpreted"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("damage, expected_code", [
    (lambda run: run["rows"].pop(), "CASE_COVERAGE"),
    (lambda run: run["rows"].append(deepcopy(run["rows"][0])), "CASE_ID"),
    (lambda run: run["rows"][0].update(case_id="unmatched"), "CASE_ID"),
    (lambda run: run["rows"][0]["inputs"].update(days_since_purchase=999), "CASE_CONTENT"),
    (lambda run: run["rows"][0]["inputs"].update(defective=0), "CASE_CONTENT"),
    (lambda run: run["rows"][0]["expectations"].update(expected_decision="store_credit"), "CASE_CONTENT"),
    (lambda run: run.update(dataset_digest="0" * 64), "DATASET_DIGEST"),
    (lambda run: run["scorers"][0].update(definition_sha256="3" * 64), "SCORER_MISMATCH"),
    (lambda run: run["scorers"].append(deepcopy(run["scorers"][0])), "SCORER_DEFINITIONS"),
    (lambda run: run["rows"][0].update(trace_state="ERROR"), "TRACE_EVIDENCE"),
    (lambda run: run["rows"][0].update(trace_id=""), "TRACE_EVIDENCE"),
    (lambda run: run["rows"][0].update(output=" "), "TRACE_EVIDENCE"),
    (lambda run: run["rows"][0]["assessments"].pop(), "ASSESSMENT_EVIDENCE"),
    (lambda run: run["rows"][0]["assessments"].append(deepcopy(run["rows"][0]["assessments"][0])), "ASSESSMENT_EVIDENCE"),
    (lambda run: run["rows"][0]["assessments"][0].update(error=True), "ASSESSMENT_EVIDENCE"),
    (lambda run: run["rows"][0]["assessments"][0].update(value=0), "ASSESSMENT_EVIDENCE"),
    (lambda run: run.update(complete=False), "EVALUATION_INCOMPLETE"),
    (lambda run: run.update(row_count=9), "ROW_COUNT"),
    (lambda run: run.update(trace_ids=[]), "TRACE_MANIFEST"),
])
def test_inconsistent_or_missing_evidence_fails_closed(evidence, damage, expected_code):
    damage(evidence["candidate"])
    report = invoke(evidence)
    assert report["decision"] == "INVALID"
    assert expected_code in codes(report)
    assert report["policy_passed"] is False and report["paired_rows"] == []


def test_control_pass_flag_cannot_override_wrong_control_grade(evidence):
    control = evidence["controls"]["evaluation"]
    score(control, control["rows"][0]["case_id"], "policy_judge", 0)
    report = invoke(evidence)
    assert report["decision"] == "INVALID"
    assert "CONTROL_DISAGREEMENT" in codes(report)
    assert report["metrics"] == {}


@pytest.mark.parametrize("damage, expected_code", [
    (lambda controls: controls.update(passed=False), "CONTROLS_FAILED"),
    (lambda controls: controls.pop("version"), "CONTROLS_VERSION"),
    (lambda controls: controls["evaluation"]["scorers"][0].update(definition_sha256="4" * 64), "CONTROL_JUDGE_MISMATCH"),
    (lambda controls: controls["evaluation"]["rows"][0].update(output="A changed control response"), "AUTHORED_OUTPUT"),
    (lambda controls: controls.pop("expected_rows"), "EXPECTED_ROWS"),
])
def test_controls_must_be_complete_versioned_and_match_judge(evidence, damage, expected_code):
    damage(evidence["controls"])
    report = invoke(evidence)
    assert report["decision"] == "INVALID"
    assert expected_code in codes(report)


def test_reported_control_disagreement_cannot_be_dropped_by_pass_flag(evidence):
    evidence["controls"]["disagreements"] = ["review_customer_instruction"]
    report = invoke(evidence)
    assert report["decision"] == "INVALID"
    assert "CONTROL_REPORTED_DISAGREEMENT" in codes(report)


def test_reused_run_or_application_trace_is_invalid(evidence):
    evidence["candidate"]["run_id"] = evidence["baseline"]["run_id"]
    report = invoke(evidence)
    assert report["decision"] == "INVALID" and "RUN_ID_REUSED" in codes(report)
    evidence["candidate"]["run_id"] = "repair"
    evidence["candidate"]["rows"][0]["trace_id"] = evidence["baseline"]["rows"][0]["trace_id"]
    evidence["candidate"]["trace_ids"][0] = evidence["baseline"]["rows"][0]["trace_id"]
    report = invoke(evidence)
    assert report["decision"] == "INVALID" and "TRACE_REUSED" in codes(report)


def test_missing_or_pending_review_never_approves(evidence):
    evidence["review"] = None
    assert invoke(evidence)["decision"] == "REVIEW_REQUIRED"
    evidence["review"] = {"version": "pending-v1", "status": "pending", "findings": []}
    report = invoke(evidence)
    assert report["decision"] == "REVIEW_REQUIRED"
    assert report["review"]["reviewer_kind"] is None


def test_review_binds_scores_answers_and_control_evidence(evidence):
    evidence["candidate"]["rows"][0]["output"] += " New answer."
    report = invoke(evidence, rebind=False)
    assert report["decision"] == "INVALID" and "REVIEW_BINDING" in codes(report)


def test_material_finding_cannot_be_cleared_by_changing_status_alone(evidence):
    finding = {"id": "judge-attribution", "material": True, "status": "open", "summary": "Judge attributes customer text to the assistant."}
    evidence["review"]["findings"] = [finding]
    assert invoke(evidence)["decision"] == "REVIEW_REQUIRED"
    finding["status"] = "resolved"
    assert invoke(evidence)["decision"] == "REVIEW_REQUIRED"
    finding["resolution"] = {"summary": "Independent evidence review documents the disposition.", "evidence": ["synthetic-review-record-v2"]}
    assert invoke(evidence)["decision"] == "PASS"


def test_policy_block_preserves_unresolved_review_reason(evidence):
    score(evidence["candidate"], "day_00", "deterministic_stack", 0)
    evidence["review"]["findings"] = [{"id": "judge-review", "material": True, "status": "open", "summary": "Requires review."}]
    report = invoke(evidence)
    assert report["decision"] == "BLOCK"
    assert {"CANDIDATE_MANDATORY_FAILURE", "MATERIAL_FINDING"}.issubset(codes(report))


def test_model_review_remains_explicitly_model_review(evidence):
    report = invoke(evidence)
    assert report["review"]["reviewer_kind"] == "model"
    assert "human_reviewed" not in report


def test_malformed_review_values_return_invalid_instead_of_raising(evidence):
    evidence["review"]["reviewer_kind"] = []
    evidence["review"]["findings"] = [{"id": "malformed", "material": True, "status": [], "summary": "Malformed status"}]
    report = invoke(evidence)
    assert report["decision"] == "INVALID"
    assert {"REVIEW_PROVENANCE", "FINDING_SCHEMA"}.issubset(codes(report))
    json.dumps(report, allow_nan=False)


def test_upstream_error_is_distinct_from_valid_block(evidence):
    evidence["candidate"]["execution_status"] = "error"
    report = invoke(evidence)
    assert report["decision"] == "NOT_EVALUATED"
    assert report["execution_status"] == "error"
    assert report["policy_passed"] is False


def test_small_nonsignificant_improvement_can_meet_declared_policy(evidence):
    for key in ("day_31_boundary", "day_45_opening", "day_90_stale_boundary"):
        score(evidence["baseline"], key, "policy_judge", 0)
    report = invoke(evidence)
    assert report["decision"] == "PASS"
    assert report["metrics"]["exact_mcnemar_pvalue"] == 0.25
    assert "not a policy acceptance condition" in report["metrics"]["statistics_role"]


def test_identical_answer_different_grade_is_not_universal_invalidity(evidence):
    assert evidence["baseline"]["rows"][0]["output"] == evidence["candidate"]["rows"][0]["output"]
    score(evidence["candidate"], "day_00", "policy_judge", 0)
    report = invoke(evidence)
    assert report["decision"] == "PASS"
    assert report["metrics"]["losses"] == 1
    # A material reviewer finding would still require review; equality of all
    # independent-run grades is deliberately absent from the policy contract.


def test_changed_threshold_requires_new_policy_identity(evidence):
    report = invoke(evidence, policy={"quality_floor": 0.8})
    assert report["decision"] == "INVALID" and "POLICY_VERSION_REUSED" in codes(report)
    report = invoke(evidence, policy={"version": "synthetic-prospective-policy-v3", "quality_floor": 0.8})
    assert report["decision"] == "PASS"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "0.9"])
def test_invalid_policy_threshold_has_serializable_failure_report(evidence, value):
    report = invoke(evidence, policy={"version": "bad-config", "quality_floor": value})
    assert report["decision"] == "INVALID"
    assert "POLICY_THRESHOLD" in codes(report)
    json.dumps(report, allow_nan=False)


def test_evaluation_does_not_mutate_inputs_or_default_policy(evidence):
    original, original_policy = deepcopy(evidence), deepcopy(DEFAULT_POLICY)
    report = evaluate_repair(**evidence)
    assert evidence == original and DEFAULT_POLICY == original_policy
    assert report["policy_version"] == POLICY_VERSION
    json.dumps(report, allow_nan=False)
