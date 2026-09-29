"""Run the real checkpoint code against MLflow with only the model provider faked."""

import json
import os
from pathlib import Path
import subprocess
import sys

import mlflow
from mlflow.genai.scorers import scorer
import pytest

from eval_gate import get_per_sample_scores, run_gate
from tests.fakes import fake_policy_judge
from west_workshop import run_checkpoint
from west_workshop.data import dataset
from west_workshop.report import judge_review_cases, render_release_report, score_breakdown, write_release_report
from west_workshop.runtime import _judge_versions as REAL_JUDGE_VERSIONS, _policy_judge as REAL_POLICY_JUDGE

ROOT = Path(__file__).resolve().parents[1]


def _run(index, tmp_path):
    summary = run_checkpoint(index, "openai", tmp_path / "west-live")
    assert summary["status"] == "passed", json.dumps(summary, indent=2)[:4000]
    assert summary["live_validation"] == "completed"
    return summary


def test_checkpoint_0_blocks_the_stale_answer_with_one_model_call(offline_openai, tmp_path):
    summary = _run(0, tmp_path)
    assert summary["decision"] == "block"
    (row,) = summary["evaluations"][0]["rows"]
    assert row["case_id"] == "day_45_opening"
    assert row["output"].startswith("Eligibility: full_refund")
    assert row["scores"] == {"policy_decision": 0.0}
    assert len(offline_openai.calls) == 1, "one request, with no extra validation call"


def test_checkpoint_1_replays_the_stored_trace_without_a_new_answer(offline_openai, tmp_path):
    summary = _run(1, tmp_path)
    evidence = summary["trace_evaluation_evidence"]
    assert evidence["complete"] is True and evidence["score"] == 1.0
    assert evidence["trace_id"] == summary["evaluations"][0]["trace_ids"][0]
    (row,) = summary["evaluations"][0]["rows"]
    retrieval = next(span for span in row["spans"] if span["name"] == "retrieve_refund_policy")
    assert "within 90 days" in retrieval["outputs"][0]["page_content"]
    assert retrieval["outputs"][0]["metadata"] == {"policy_version": "2024-01"}
    assert len(offline_openai.calls) == 1


def test_checkpoint_2_scores_ten_cases_with_every_scorer(offline_openai, tmp_path):
    summary = _run(2, tmp_path)
    evaluation = summary["evaluations"][0]
    assert evaluation["row_count"] == 10 and evaluation["complete"]
    names = {item["name"] for item in evaluation["scorers"]}
    assert names == {"eligibility_format", "response_length", "pii_detection", "policy_decision",
                     "no_false_transaction", "deterministic_stack", "policy_judge"}
    failed = sorted(row["case_id"] for row in evaluation["rows"] if row["scores"]["policy_decision"] == 0)
    assert failed == ["day_31_boundary", "day_45_opening", "day_90_stale_boundary"]
    assert len(offline_openai.calls) == 10


def _patch_chapter_3_judges(monkeypatch, graded, overturn=None):
    # The real make_judge definitions are saved, reloaded, and registered.
    # A counting fake does the grading, so no model is called. `overturn`
    # names one (judge, case) pair whose verdict the fake reverses.
    from west_workshop import runtime

    def counting_judge(provider=None, *, rationale_first=True, name="policy_judge"):
        inner = fake_policy_judge(provider, rationale_first=rationale_first, name=name)

        @scorer(name=name)
        def judge(inputs, outputs, expectations):
            graded.append(name)
            verdict = inner.run(inputs=inputs, outputs=outputs, expectations=expectations)
            return not verdict if (name, inputs.get("case_id")) == overturn else verdict

        return judge

    def keep_real_definitions(provider, experiment_id, judges):
        real = [REAL_POLICY_JUDGE(provider, rationale_first=judge.name == "rationale_first", name=judge.name)
                for judge in judges]
        versions, _ = REAL_JUDGE_VERSIONS(provider, experiment_id, real)
        return versions, judges

    monkeypatch.setattr(runtime, "_policy_judge", counting_judge)
    monkeypatch.setattr(runtime, "_judge_versions", keep_real_definitions)


def test_checkpoint_3_keeps_real_judge_definitions_and_makes_twenty_judge_requests(offline_openai, tmp_path, monkeypatch):
    graded = []
    _patch_chapter_3_judges(monkeypatch, graded)
    summary = _run(3, tmp_path)
    assert [item["round_trip_verified"] for item in summary["scorer_versions"]] == [True, True]
    assert summary["evaluations"][0]["row_count"] == 6 and summary["evaluations"][0]["complete"]
    assert summary["agreement_with_authored_labels"] == {"value_first": 1.0, "rationale_first": 1.0}
    assert summary["judge_validation"]["passed"]
    assert summary["decision"].startswith("Both judge versions matched every reference label")
    assert len(graded) == 20, "each judge grades six replies, then the rationale-first judge grades eight controls"
    assert graded.count("value_first") == 6
    assert offline_openai.calls == [], "the chapter makes no application requests"


def test_checkpoint_3_asks_for_review_when_a_judge_disagrees_with_a_label(offline_openai, tmp_path, monkeypatch):
    _patch_chapter_3_judges(monkeypatch, [], overturn=("value_first", "label_stale_policy"))
    summary = _run(3, tmp_path)
    assert summary["agreement_with_authored_labels"] == {"value_first": 5 / 6, "rationale_first": 1.0}
    assert summary["judge_validation"]["passed"], "only the eight controls must all pass"
    assert summary["decision"] == "Review every disagreement before trusting the judge."


def test_checkpoint_3_names_the_control_the_judge_got_wrong(offline_openai, tmp_path, monkeypatch, capsys):
    from west_workshop.notebook_setup import show_result

    _patch_chapter_3_judges(monkeypatch, [], overturn=("rationale_first", "review_customer_next_step"))
    summary = run_checkpoint(3, "openai", tmp_path / "west-live")
    assert summary["status"] == "error" and not summary["judge_validation"]["passed"]
    assert summary["error"].startswith("The judge disagreed with at least one of its eight rubric controls"), summary.get("error")
    capsys.readouterr()
    show_result(summary)
    printed = capsys.readouterr().out
    assert "Run issue: The judge disagreed" in printed
    assert "Judge control disagreements: review_customer_next_step" in printed


class _RejectingChatClient:
    """A provider that refuses every request, like a bad key or an exhausted quota."""

    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **request):
        raise RuntimeError("401: the provider rejected the request")


def test_a_failed_model_request_names_the_likely_cause(offline_openai, tmp_path, monkeypatch):
    from west_workshop import runtime

    monkeypatch.setattr(runtime, "_client", lambda provider: _RejectingChatClient())
    summary = run_checkpoint(0, "openai", tmp_path / "west-live")
    assert summary["status"] == "error" and summary["live_validation"] == "failed"
    assert summary["error"].startswith("At least one answer or score is missing"), summary.get("error")
    assert "API key" in summary["error"] and "quota" in summary["error"]


def test_checkpoint_0_explains_a_stale_candidate_that_answered_correctly(offline_openai, tmp_path, monkeypatch):
    from tests.fakes import _Response, canned_answer

    def always_current_policy(*, model, messages, temperature, max_tokens):
        request = json.loads(messages[1]["content"])
        return _Response(canned_answer(request["days_since_purchase"], request["defective"], 30))

    monkeypatch.setattr(offline_openai, "create", always_current_policy)
    summary = run_checkpoint(0, "openai", tmp_path / "west-live")
    assert summary["status"] == "error" and summary["decision"] == "inspect"
    assert summary["error"].startswith("No answer failed the policy check"), summary.get("error")


def test_checkpoint_4_blames_judge_access_when_the_judge_cannot_be_called(offline_openai, tmp_path, monkeypatch, capsys):
    from west_workshop import runtime
    from west_workshop.notebook_setup import show_result

    def unreachable_judge(provider=None, *, rationale_first=True, name="policy_judge"):
        @scorer(name=name)
        def judge(inputs, outputs, expectations):
            raise RuntimeError("the judge endpoint is unavailable")

        return judge

    monkeypatch.setattr(runtime, "_policy_judge", unreachable_judge)
    summary = run_checkpoint(4, "openai", tmp_path / "west-live")
    assert summary["status"] == "error" and summary["decision"] == "block"
    assert summary["error"].startswith("The judge could not score all eight rubric controls"), summary.get("error")
    assert offline_openai.calls == [], "no application request runs before the judge passes its controls"
    assert summary["judge_validation"]["disagreements"], "every unscored control is recorded as a mismatch"
    capsys.readouterr()
    show_result(summary)
    assert "Judge control disagreements" not in capsys.readouterr().out, "unscored controls are not disagreements"


@pytest.fixture
def checkpoint_4(offline_openai, tmp_path):
    return _run(4, tmp_path), offline_openai


def test_checkpoint_4_blocks_stale_and_ships_the_repair(checkpoint_4):
    summary, client = checkpoint_4
    assert summary["gates"]["candidate"]["decision"] == "block"
    assert summary["gates"]["repaired"]["decision"] == "ship"
    assert summary["expected_outcome_observed"] is True
    comparison = summary["repair_comparison"]
    assert comparison["same_dataset"] and comparison["same_scorers"]
    assert comparison["candidate_mean"] == pytest.approx(0.2)
    assert comparison["repaired_mean"] == 1.0
    assert comparison["regressed_cases"] == []
    assert len(comparison["improved_cases"]) == 8
    assert len(client.calls) == 30, "three versions times ten cases, with no extra validation calls"


def test_checkpoint_4_report_separates_eligibility_from_combined_checks(checkpoint_4):
    summary, _ = checkpoint_4
    breakdown = {item["version"]: item for item in score_breakdown(summary)}
    assert breakdown["candidate"]["correct_eligibility"] == 7
    assert breakdown["candidate"]["combined_passes"] == 2
    assert breakdown["repaired"]["correct_eligibility"] == breakdown["repaired"]["combined_passes"] == 10
    assert {row["case_id"] for row in judge_review_cases(summary)} == {
        "day_00", "day_29", "day_30_boundary", "day_91", "false_approval_request"}
    html = render_release_report(summary)
    assert ">BLOCK<" in html and ">SHIP<" in html
    path = write_release_report(summary)
    assert path.name == "release-report.html" and path.read_text(encoding="utf-8") == html


def test_eval_gate_cli_reads_the_recorded_runs(checkpoint_4):
    summary, _ = checkpoint_4
    baseline, candidate, repaired = (evaluation["run_id"] for evaluation in summary["evaluations"])
    ids = {row["inputs"]["case_id"] for row in dataset()}
    base_scores = get_per_sample_scores(baseline, "policy_decision")
    assert set(base_scores) == ids
    assert run_gate(base_scores, get_per_sample_scores(repaired, "policy_decision"), min_overlap=10)[0]
    passed, reason = run_gate(base_scores, get_per_sample_scores(candidate, "policy_decision"), min_overlap=10)
    assert not passed and reason.startswith("Regression rate 30.0%")


def test_eval_gate_command_exits_with_the_documented_codes(checkpoint_4):
    summary, _ = checkpoint_4
    baseline, candidate, repaired = (evaluation["run_id"] for evaluation in summary["evaluations"])
    env = {**os.environ, "MLFLOW_TRACKING_URI": "sqlite:///" + summary["local_tracking_database"]}

    def gate(*arguments):
        command = [sys.executable, str(ROOT / "eval_gate.py"), "--baseline-run-id", baseline, *arguments]
        return subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, timeout=300)

    passed = gate("--candidate-run-id", repaired, "--scorer", "policy_decision", "--min-overlap", "10")
    assert passed.returncode == 0 and "PASSED" in passed.stdout, passed.stdout + passed.stderr
    blocked = gate("--candidate-run-id", candidate, "--scorer", "policy_decision", "--min-overlap", "10")
    assert blocked.returncode == 1 and "BLOCKED: Regression rate 30.0%" in blocked.stdout, blocked.stdout
    missing = gate("--candidate-run-id", repaired, "--scorer", "no_such_scorer", "--min-overlap", "10")
    assert missing.returncode == 1 and "BLOCKED" in missing.stdout
    invalid = gate("--candidate-run-id", repaired, "--threshold", "2")
    assert invalid.returncode == 2


def test_checkpoint_4_command_ends_with_the_report_path(offline_openai, tmp_path, monkeypatch, capsys):
    from west_workshop.__main__ import main

    monkeypatch.setattr(sys, "argv", ["west_workshop", "--provider", "openai", "--checkpoint", "4"])
    assert main() == 0
    printed = capsys.readouterr()
    # Progress lines come first, and the rest of stdout stays valid JSON.
    summary = json.loads(printed.out[printed.out.index("\n{") + 1:])
    footer = printed.err.strip().splitlines()
    assert footer[0] == "Exercise status: passed"
    assert footer[-2] == "Full saved results: " + summary["summary_path"]
    assert footer[-1] == "Saved visual report: " + summary["report_path"]
    assert Path(summary["report_path"]).is_file()


def test_checkpoint_5_attaches_human_feedback_to_the_live_trace(offline_openai, tmp_path):
    summary = _run(5, tmp_path)
    trace = mlflow.get_trace(summary["feedback_trace_id"])
    feedback = [item for item in trace.info.assessments if item.name == "workshop_authored_followup"]
    assert len(feedback) == 1
    assert feedback[0].value == "needs_human_followup"
    assert feedback[0].source.source_type == "HUMAN"
    assert summary["automatic_evaluation_started"] is False


def test_saved_judge_definitions_round_trip(tmp_path, monkeypatch):
    # The real LLM judge definitions are saved, reloaded, and registered. Nothing calls a model.
    from west_workshop import runtime
    from west_workshop.config import preflight

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-offline-0000000000000000")
    assert preflight("openai")["ready"]
    experiment_id = runtime._setup_tracking("openai", tmp_path)
    judges = [runtime._policy_judge("openai", rationale_first=False, name="value_first"),
              runtime._policy_judge("openai", rationale_first=True, name="rationale_first")]
    versions, restored = runtime._judge_versions("openai", experiment_id, judges)
    assert [item["round_trip_verified"] for item in versions] == [True, True]
    assert [item["registry_version"] for item in versions] == [1, 2]
    assert [judge.model_dump() for judge in restored] == [judge.model_dump() for judge in judges]
