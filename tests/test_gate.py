import math
from types import SimpleNamespace

import pytest

from eval_gate import _mcnemar_exact_pvalue, _parse_score, _scores_from_traces, run_gate
from west_workshop.data import dataset
from west_workshop.runtime import QUALITY_FLOOR, REGRESSION_LIMIT, _complete, _gate

KEYS = [f"case_{i:02d}" for i in range(10)]


def scores(passing):
    return {key: 1.0 if key in passing else 0.0 for key in KEYS}


def test_identical_runs_pass():
    passed, reason = run_gate(scores(KEYS), scores(KEYS), min_overlap=10)
    assert passed and reason == "No significant regression detected"


def test_regressions_above_the_limit_block():
    passed, reason = run_gate(scores(KEYS), scores(KEYS[2:]), max_regression_rate=0.10, min_overlap=10)
    assert not passed and reason.startswith("Regression rate 20.0% exceeds threshold 10.0%")


def test_one_regression_in_ten_is_within_a_ten_percent_limit():
    passed, _ = run_gate(scores(KEYS), scores(KEYS[1:]), max_regression_rate=0.10, min_overlap=10)
    assert passed


def test_different_case_sets_block():
    candidate = scores(KEYS)
    candidate["extra"] = candidate.pop("case_00")
    passed, reason = run_gate(scores(KEYS), candidate, min_overlap=9)
    assert not passed and "Case coverage differs" in reason


def test_too_few_cases_block():
    three = {key: 1.0 for key in KEYS[:3]}
    passed, reason = run_gate(three, dict(three), min_overlap=10)
    assert not passed and "minimum 10 required" in reason


@pytest.mark.parametrize("bad", [math.nan, math.inf, None, "maybe"])
def test_unparseable_scores_block(bad):
    candidate = scores(KEYS)
    candidate["case_00"] = bad
    passed, reason = run_gate(scores(KEYS), candidate, min_overlap=10)
    assert not passed and "parseable and finite" in reason


def test_invalid_configuration_blocks_before_comparing():
    passed, reason = run_gate(scores(KEYS), scores(KEYS), max_regression_rate=1.5)
    assert not passed and reason.startswith("Invalid gate configuration")


@pytest.mark.parametrize(("regressions", "improvements", "expected"), [
    (0, 0, 1.0), (0, 8, 2 / 2**8), (1, 1, 1.0), (0, 5, 2 / 2**5), (2, 10, 2 * (1 + 12 + 66) / 2**12),
])
def test_exact_mcnemar_p_values(regressions, improvements, expected):
    assert _mcnemar_exact_pvalue(regressions, improvements) == pytest.approx(expected)


@pytest.mark.parametrize(("raw", "parsed"), [
    (True, 1.0), (False, 0.0), ("yes", 1.0), ("No", 0.0), ("factual", 1.0), ("hallucinated", 0.0),
    (0.25, 0.25), ("0.5", 0.5), (math.nan, None), ("unknown", None), (None, None),
])
def test_score_parsing(raw, parsed):
    assert _parse_score(raw) == parsed


def _evaluation(deterministic, judge, *, state="OK", output="Eligibility: store_credit\nAnswer."):
    rows = []
    for row in dataset():
        key = row["inputs"]["case_id"]
        rows.append({"case_id": key, "trace_state": state, "output": output,
                     "scores": {"deterministic_stack": deterministic.get(key, 1.0),
                                "policy_judge": judge.get(key, 1.0)}})
    return {"rows": rows}


def test_release_rules_are_the_documented_values():
    assert QUALITY_FLOOR == 0.90 and REGRESSION_LIMIT == 0.10


def test_a_clean_candidate_ships():
    gate = _gate(_evaluation({}, {}), _evaluation({}, {}), dataset())
    assert gate["passed"] and gate["decision"] == "ship" and gate["candidate_mean"] == 1.0


def test_one_judge_rejection_can_still_ship_at_the_floor():
    gate = _gate(_evaluation({}, {}), _evaluation({}, {"day_29": 0.0}), dataset())
    assert gate["passed"] and gate["candidate_mean"] == pytest.approx(0.9)


def test_a_mandatory_deterministic_failure_always_blocks():
    gate = _gate(_evaluation({}, {}), _evaluation({"day_29": 0.0}, {}), dataset())
    assert not gate["passed"] and gate["mandatory_failures"]["candidate"] == ["day_29"]


def test_below_the_floor_blocks():
    gate = _gate(_evaluation({}, {"day_29": 0.0}), _evaluation({}, {"day_29": 0.0, "day_00": 0.0}), dataset())
    assert not gate["passed"] and gate["decision"] == "block"


def test_missing_or_failed_evidence_fails_closed():
    incomplete = _evaluation({}, {})
    incomplete["rows"].pop()
    assert _gate(_evaluation({}, {}), incomplete, dataset())["complete"] is False
    errored = _evaluation({}, {}, state="ERROR")
    assert _gate(_evaluation({}, {}), errored, dataset())["decision"] == "block"
    unscored = _evaluation({}, {})
    unscored["rows"][0]["scores"]["policy_judge"] = None
    assert not _complete(unscored["rows"], dataset(), ["policy_judge"])


def _scored_trace(state="OK", assessment_error=None):
    return SimpleNamespace(info=SimpleNamespace(
        state=state, client_request_id="case_1",
        assessments=[SimpleNamespace(name="policy_decision", valid=True,
                                     feedback=SimpleNamespace(value=True), error=assessment_error)]))


@pytest.mark.parametrize("state", ["ERROR", "IN_PROGRESS", None])
def test_ci_rejects_failed_or_unfinished_traces_even_with_a_passing_score(state):
    with pytest.raises(ValueError, match="did not finish successfully"):
        _scores_from_traces([_scored_trace(state)], "policy_decision")


def test_ci_rejects_an_assessment_error_even_with_a_numeric_value():
    with pytest.raises(ValueError, match="missing, failed"):
        _scores_from_traces([_scored_trace(assessment_error=SimpleNamespace(error_code="BAD_REQUEST"))], "policy_decision")


def test_ci_accepts_a_successful_scored_trace():
    assert _scores_from_traces([_scored_trace()], "policy_decision") == {"case_1": 1.0}
