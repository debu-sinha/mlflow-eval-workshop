import copy

import pytest

from west_workshop.data import dataset, dataset_digest
from west_workshop.report import render_release_report, score_breakdown

SCORERS = [{"name": name, "definition_sha256": "0" * 64}
           for name in ("policy_decision", "deterministic_stack", "policy_judge")]


def _rows(version):
    rows = []
    for row in dataset():
        key = row["inputs"]["case_id"]
        stale = version == "candidate" and key in {"day_31_boundary", "day_45_opening", "day_90_stale_boundary"}
        label = "full_refund" if stale else row["expectations"]["expected_decision"]
        rows.append({
            "case_id": key, "inputs": row["inputs"], "expectations": row["expectations"],
            "output": f"Eligibility: {label}\nAnswer for {key}.", "trace_state": "OK",
            "trace_id": f"tr-{version}-{key}",
            "scores": {"policy_decision": 0.0 if stale else 1.0,
                       "deterministic_stack": 0.0 if stale else 1.0, "policy_judge": 0.0 if stale else 1.0},
            "assessments": [{"name": "policy_judge", "value": not stale, "rationale": "Checked.", "error": False}],
            "spans": [{"name": "retrieve_refund_policy", "type": "RETRIEVER", "inputs": {},
                       "outputs": [{"page_content": "policy text", "metadata": {}}]}],
        })
    return rows


def _summary():
    evaluations = []
    for version in ("baseline", "candidate", "repaired"):
        rows = _rows(version)
        evaluations.append({"run_id": version * 4, "rows": rows, "row_count": len(rows), "complete": True,
                            "scorers": SCORERS, "dataset_digest": dataset_digest()})
    ship = {"passed": True, "decision": "ship", "complete": True, "reason": "No significant regression detected",
            "quality_floor": 0.9, "max_regression_rate": 0.1}
    block = {"passed": False, "decision": "block", "complete": True, "reason": "A mandatory deterministic invariant failed."}
    return {"checkpoint": 4, "status": "passed", "expected_outcome_observed": True, "evaluations": evaluations,
            "gates": {"candidate": block, "repaired": ship},
            "repair_comparison": {"changed_component": "retrieved policy", "same_dataset": True, "same_scorers": True},
            "provider": "openai", "utc_timestamp": "2026-09-28T00:00:00+00:00"}


def test_a_complete_run_shows_both_decisions():
    html = render_release_report(_summary())
    assert ">BLOCK<" in html and ">SHIP<" in html and "INCOMPLETE" not in html
    assert "Recorded run" in html


def test_counts_separate_eligibility_from_combined_checks():
    counts = {item["version"]: item for item in score_breakdown(_summary())}
    assert counts["candidate"]["correct_eligibility"] == 7 and counts["candidate"]["combined_passes"] == 7
    assert counts["baseline"]["cases"] == 10


def test_missing_evidence_can_never_show_a_ship_card():
    summary = _summary()
    summary["evaluations"][2]["rows"].pop()
    html = render_release_report(summary)
    assert ">SHIP<" not in html and "INCOMPLETE" in html


def test_a_failed_trace_is_incomplete_evidence():
    summary = _summary()
    summary["evaluations"][2]["rows"][0]["trace_state"] = "ERROR"
    assert ">SHIP<" not in render_release_report(summary)


def test_different_datasets_or_scorers_are_not_comparable():
    summary = _summary()
    summary["evaluations"][2]["dataset_digest"] = "different"
    assert ">SHIP<" not in render_release_report(summary)
    summary = _summary()
    summary["evaluations"][2]["scorers"] = SCORERS[:2]
    assert ">SHIP<" not in render_release_report(summary)


def test_model_text_is_escaped():
    summary = _summary()
    for evaluation in summary["evaluations"]:
        for row in evaluation["rows"]:
            row["output"] = "Eligibility: store_credit\n<script>alert('x')</script>"
    html = render_release_report(summary)
    assert "<script>alert" not in html and "&lt;script&gt;" in html


def test_only_checkpoint_4_can_render():
    summary = copy.deepcopy(_summary())
    summary["checkpoint"] = 2
    with pytest.raises(ValueError):
        render_release_report(summary)
