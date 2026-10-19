"""Offline evidence/CLI tests. Test doubles never become released saved evidence."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from west_workshop.data import CURRENT_POLICY, dataset, dataset_digest
from west_workshop import repair_workshop as workshop
from west_workshop.repair_policy import evidence_identity
from west_workshop.runtime import _score, _complete, eligibility_declaration, policy_decision_value

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", [-1, 2, 0.5, 10 ** 1000, float("nan"), float("inf"), "1", "true", "pass", "factual", {}, []])
def test_release_score_contract_rejects_nonbinary_values(value):
    assert _score(value) is None


@pytest.mark.parametrize("value, expected", [(False, 0), (True, 1), (0, 0), (1.0, 1), ("yes", 1), ("No", 0)])
def test_release_score_contract_keeps_native_binary_feedback(value, expected):
    assert _score(value) == expected


def test_native_dataframe_boolean_and_numeric_scalars_keep_strict_binary_contract():
    import numpy as np
    assert _score(np.bool_(True)) == 1
    assert _score(np.bool_(False)) == 0
    assert _score(np.int64(1)) == 1
    assert _score(np.float64(0.5)) is None


def test_primary_cli_help_exposes_the_full_policy_entrypoint():
    result = subprocess.run([sys.executable, "-m", "west_workshop", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert "evaluate" in result.stdout and "gate" in result.stdout and "rescore-legacy" in result.stdout


@pytest.mark.parametrize("answer", [
    "Eligibility: store_credit\nEligibility: full_refund",
    "Eligibility: store_credit\nOther option: Eligibility: full_refund.",
    "Eligibility: store_credit\nELIGIBILITY : full_refund",
    "Eligibility: store_credit\nEligibility: unknown",
    "Eligibility: store_credit_plus",
    "Eligibility: store_credit/full_refund",
    "Eligibility: store_credit OR full_refund",
    "Eligibility: store_credit full_refund",
    "Here is my answer: Eligibility: store_credit",
])
def test_one_declaration_applies_to_shared_format_and_business_checks(answer):
    assert eligibility_declaration(answer) is None
    assert not policy_decision_value(answer, {"expected_decision": "store_credit"})


@pytest.mark.parametrize("separator", ["\n", ". "])
def test_the_two_actual_model_layouts_remain_supported(separator):
    assert eligibility_declaration("Eligibility: store_credit" + separator + "Contact support to request it.") == "store_credit"


def _record(cases, run_id, scores=None, judge_only=False):
    names = ["policy_judge"] if judge_only else ["deterministic_stack", "policy_judge"]
    rows = []
    for case in cases:
        key = case["inputs"]["case_id"]
        values = {name: (scores or {}).get((key, name), 1) for name in names}
        if judge_only:
            values["policy_judge"] = int(case["expectations"]["authored_human_label"])
        rows.append({"case_id": key, "inputs": copy.deepcopy(case["inputs"]), "expectations": copy.deepcopy(case["expectations"]),
                     "output": case.get("outputs", "Eligibility: store_credit\nTest-only answer."), "trace_state": "OK",
                     "trace_id": run_id + "-" + key, "scores": values,
                     "assessments": [{"name": name, "value": value, "error": False} for name, value in values.items()]})
    return {"run_id": run_id, "complete": True, "rows": rows, "row_count": len(rows), "trace_ids": [row["trace_id"] for row in rows],
            "scorers": [{"name": name, "definition_sha256": hashlib.sha256(name.encode()).hexdigest(), "implementation": "test-only"} for name in names],
            "dataset_digest": dataset_digest(cases)}


@pytest.fixture
def saved_evidence(tmp_path):
    """Explicitly fabricated unit-test fixture, never used by production saved mode."""
    cases = dataset()
    baseline = _record(cases, "before")
    candidate = _record(cases, "after", {("day_00", "deterministic_stack"): 0})
    control_cases = []
    for number, label in enumerate((False, True)):
        row = copy.deepcopy(cases[number])
        row["inputs"]["case_id"] = "control-" + str(number)
        row["outputs"] = "Test-only authored answer " + str(number)
        row["expectations"]["authored_human_label"] = label
        control_cases.append(row)
    controls = {"version": "test-controls", "expected_rows": control_cases, "evaluation": _record(control_cases, "controls", judge_only=True), "passed": True, "disagreements": []}
    review = {"version": "test-review", "status": "complete", "reviewer_kind": "human", "reviewer_id": "unit-test-only", "findings": [],
              "evidence": evidence_identity(baseline, candidate, controls)}
    directory = tmp_path / "fixture-evidence"
    directory.mkdir()
    manifest = {"schema_version": 2, "evidence_id": "unit-test-only", "created_at": "2026-10-18T00:00:00+00:00", "provider": "openai", "files": {}, "records": {}, "decisions": []}
    for key, value in {"cases": cases, "deployed": baseline, "repair": candidate, "controls": controls}.items():
        name = key + ".json"
        encoded = workshop._json_bytes(value)
        (directory / name).write_bytes(encoded)
        manifest["files"][name] = hashlib.sha256(encoded).hexdigest()
        manifest["records"][key] = name
    (directory / "manifest.json").write_bytes(workshop._json_bytes(manifest))
    return directory, review


def test_saved_mode_is_read_only_offline_and_full_policy_not_a_single_scorer(saved_evidence, monkeypatch):
    directory, review = saved_evidence
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("saved mode attempted network"))
    session = workshop.notebook_session("saved", evidence_dir=directory)
    assert not session.needs("deployed") and not session.needs("repair")
    result = session.finish(review)
    assert result["execution_status"] == "complete"
    assert result["policy_result"]["decision"] == "BLOCK"
    assert result["policy_result"]["evidence_checks_passed"]
    assert original == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_saved_import_and_read_need_no_optional_or_mlflow_dependencies(saved_evidence):
    directory, _ = saved_evidence
    script = "import sys; from west_workshop.repair_workshop import read_evidence; read_evidence(sys.argv[1]); assert 'mlflow' not in sys.modules; assert 'openai' not in sys.modules; assert 'pandas' not in sys.modules"
    result = subprocess.run([sys.executable, "-S", "-c", script, str(directory)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_hash_verification_prevents_modified_saved_scores(saved_evidence):
    directory, _ = saved_evidence
    (directory / "repair.json").write_text("{}", encoding="utf-8")
    with pytest.raises(workshop.EvidenceError, match="integrity"):
        workshop.read_evidence(directory)


def test_manifest_cannot_point_outside_evidence_directory(saved_evidence, tmp_path):
    directory, _ = saved_evidence
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../outside.json"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    path.write_bytes(workshop._json_bytes(manifest))
    with pytest.raises(workshop.EvidenceError, match="escapes"):
        workshop.read_evidence(directory)


def test_saved_mode_has_no_synthetic_fallback(tmp_path):
    result = workshop.workshop_preflight("saved", evidence_dir=tmp_path)
    assert not result["ready"] and "never invents" in result["reasons"][0]
    with pytest.raises(workshop.EvidenceError):
        workshop.notebook_session("saved", evidence_dir=tmp_path)


@pytest.mark.parametrize("route, decision", [("local", "REVIEW_REQUIRED"), ("databricks", "PASS")])
def test_released_review_metadata_is_anonymous_and_preserves_bound_policy(route, decision):
    directory = ROOT / "evidence" / "workshop" / route / "final"
    session = workshop.WorkshopSession("saved", evidence_dir=directory)
    report = session.finish()
    from west_workshop.repair_policy import evaluate_repair
    assert evaluate_repair(**session.policy_inputs()) == report["policy_result"]
    assert report["policy_result"]["decision"] == decision
    assert report["review"]["reviewer_kind"] == "model"
    assert report["review"]["reviewer_id"] == "automated-review"
    assert report["review"]["evidence"] == report["policy_result"]["evidence"]
    assert report["production_authorization"] is False
    if route == "local":
        assert any(item["material"] and item["status"] == "open" for item in report["review"]["findings"])
    for name, saved in zip(session.manifest["decisions"], session.bundle["decisions"]):
        assert Path(name).stem == workshop._digest(saved)[:20]


def test_review_template_is_bound_but_never_automatically_complete(saved_evidence):
    directory, _ = saved_evidence
    form = workshop.review_skeleton(directory)
    assert form["status"] == "pending" and form["reviewer_id"] == ""
    records = workshop.read_evidence(directory)["records"]
    assert form["evidence"] == evidence_identity(records["deployed"], records["repair"], records["controls"])


def test_saved_notebook_review_panel_shows_exact_bound_evidence_before_vote_without_mutation(saved_evidence, monkeypatch):
    from html import escape
    from IPython.display import HTML
    directory, _ = saved_evidence
    record_path = directory / "repair.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    row = next(item for item in record["rows"] if item["case_id"] == "false_approval_request")
    rationale = 'Test-only rationale: customer text <script>alert("unsafe")</script> was attributed to the assistant.'
    next(item for item in row["assessments"] if item["name"] == "policy_judge")["rationale"] = rationale
    record_path.write_bytes(workshop._json_bytes(record))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["repair.json"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(workshop._json_bytes(manifest))
    original_files = {path.name: path.read_bytes() for path in directory.iterdir()}
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: pytest.fail("saved review panel attempted network"))
    session = workshop.WorkshopSession("saved", evidence_dir=directory)
    review = workshop.review_skeleton(directory)
    review.update(version="test-only-model-review", status="complete", reviewer_kind="model", reviewer_id="named-test-reviewer",
                  findings=[{"id": "attribution-1", "summary": "Customer <text> attributed to the assistant",
                             "status": "open", "material": True, "variant": "repair",
                             "case_id": "false_approval_request", "check": "policy_judge"}])
    report = session.finish(review)
    comparison = session.policy_inputs(review)
    before = copy.deepcopy((report, comparison, session.bundle))
    document = json.loads((ROOT / "notebooks/workshop/02_compare_and_decide.ipynb").read_text(encoding="utf-8"))
    cell_index = next(index for index, cell in enumerate(document["cells"]) if cell["id"] == "02-review-evidence")
    assert "paired_table(report)" in document["cells"][cell_index - 1]["source"]
    assert "Vote before revealing" in document["cells"][cell_index + 1]["source"]
    rendered = []
    exec(compile(document["cells"][cell_index]["source"], "02-review-evidence", "exec"),
         {"report": report, "comparison": comparison, "json": json, "HTML": HTML, "display": rendered.append})
    content = rendered[0].data
    assert "complete" in content
    assert "Reviewer:" not in content and "named-test-reviewer" not in content
    assert "model /" not in content
    assert "Status: open" in content and "Material: True" in content
    assert "Customer &lt;text&gt; attributed to the assistant" in content
    assert escape(json.dumps(row["inputs"], ensure_ascii=False, indent=2)) in content
    assert escape(row["output"]) in content and escape(rationale) in content
    assert "policy_judge rationale" in content and "<script>" not in content
    assert before == (report, comparison, session.bundle)
    assert original_files == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_pairs_can_be_inspected_before_decision_and_reasons_name_the_case(saved_evidence):
    directory, review = saved_evidence
    report = workshop.WorkshopSession("saved", evidence_dir=directory).finish(review)
    pairs = workshop.paired_table(report)
    assert "day_00" in pairs and "loss" in pairs
    assert "BLOCK" not in pairs and "Workshop repair policy" not in pairs
    rendered = workshop.render_report(report)
    assert "case id: day_00" in rendered and "check: deterministic_stack" in rendered
    assert "Paired bootstrap 95% interval" in rendered and "10,000 resamples" in rendered


def test_invalid_measurement_withholds_pairs_and_names_missing_evidence(saved_evidence):
    directory, review = saved_evidence
    review["evidence"]["baseline_sha256"] = "stale-binding"
    report = workshop.WorkshopSession("saved", evidence_dir=directory).finish(review)
    table = workshop.paired_table(report)
    assert "Paired comparison withheld" in table
    assert "stale" in table
    assert "INVALID" not in table


def test_measurement_block_exits_zero_and_gate_blocks_after_writing_report(saved_evidence, tmp_path):
    directory, review = saved_evidence
    review_path = tmp_path / "review.json"
    review_path.write_bytes(workshop._json_bytes(review))
    for action, expected in (("evaluate", 0), ("gate", 2)):
        destination = tmp_path / action
        assert workshop.main([action, "--evidence", str(directory), "--review-json", str(review_path), "--output", str(destination)]) == expected
        report_path = destination / "report" / "result.json"
        assert report_path.is_file()
        assert json.loads(report_path.read_text())["policy_result"]["decision"] == "BLOCK"
        assert (destination / "report" / "report.html").is_file()


def test_invalid_evidence_is_distinct_from_execution_failure(saved_evidence, tmp_path):
    directory, review = saved_evidence
    review["evidence"]["baseline_sha256"] = "wrong"
    review_path = tmp_path / "review.json"
    review_path.write_bytes(workshop._json_bytes(review))
    assert workshop.main(["gate", "--evidence", str(directory), "--review-json", str(review_path), "--output", str(tmp_path / "invalid")]) == 4
    assert workshop.main(["gate", "--evidence", str(tmp_path / "missing"), "--output", str(tmp_path / "error")]) == 1
    assert (tmp_path / "error" / "execution-error.json").is_file()


def test_malformed_review_is_invalid_evidence_in_the_session_not_a_wrapper_exception(saved_evidence):
    directory, _ = saved_evidence
    report = workshop.WorkshopSession("saved", evidence_dir=directory).finish(["not a review object"])
    assert report["execution_status"] == "complete" and report["policy_result"]["decision"] == "INVALID"
    assert report["review_status"] == "invalid"


def test_automatic_feedback_cannot_be_promoted_to_ground_truth():
    case = copy.deepcopy(dataset()[0])
    case["inputs"]["case_id"] = "new-case-v1"
    approval = {"status": "approved", "reviewer_kind": "human", "review_id": "review-1", "reviewer_id": "test-reviewer",
                "reviewed_at": "2026-10-18", "label_reason": "Reviewed current-policy label", "source_evidence": "test-trace", "case": case}
    assert workshop.promote_reviewed_case(approval)["compare_every_configuration_again"]
    approval["reviewer_kind"] = "model"
    with pytest.raises(workshop.EvidenceError, match="Automatic feedback"):
        workshop.promote_reviewed_case(approval)


def test_primary_notebooks_are_portable_and_expose_real_mlflow_calls():
    for path in sorted((ROOT / "notebooks" / "workshop").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
        tree = ast.parse(source)
        assert not any(isinstance(node, ast.Name) and node.id == "__file__" for node in ast.walk(tree))
        if not path.name.startswith("00"):
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "evaluate"]
            assert any({keyword.arg for keyword in node.keywords} >= {"data", "predict_fn", "scorers"} for node in calls)
        assert all(cell.get("outputs", []) == [] for cell in notebook["cells"]), "released source notebooks must not conceal test-double outputs"


def test_notebook_saves_and_setup_edits_preserve_live_handoff_but_retain_hashes(offline_openai, tmp_path, monkeypatch):
    (tmp_path / "west_workshop").mkdir()
    (tmp_path / "west_workshop" / "data.py").write_text("# root marker", encoding="utf-8")
    notebooks = tmp_path / "notebooks" / "workshop"
    notebooks.mkdir(parents=True)
    path = notebooks / "01_example.ipynb"
    document = {"cells": [{"cell_type": "code", "source": "MODE = 'saved'\n", "execution_count": None, "outputs": [], "metadata": {}}]}
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setenv("WORKSHOP_PROJECT_ROOT", str(tmp_path))
    before = workshop._workshop_source_manifest()
    original = workshop._notebook_provenance()
    first = workshop.WorkshopSession("live", output_dir=tmp_path / "cross-kernel")
    assert first.manifest["notebook_provenance"]["at_creation"] == original
    document["cells"][0].update(execution_count=1, outputs=[{"output_type": "stream", "name": "stdout", "text": "Recorded real output"}], metadata={"scrolled": True})
    document["cells"][0]["source"] = ["MODE = 'saved'\n"]  # Both valid notebook source encodings are equivalent.
    path.write_text(json.dumps(document), encoding="utf-8")
    assert workshop._workshop_source_manifest() == before
    saved = workshop._notebook_provenance()
    key = "notebooks/workshop/01_example.ipynb"
    assert saved[key]["raw_sha256"] != original[key]["raw_sha256"]
    assert saved[key]["cell_source_sha256"] == original[key]["cell_source_sha256"]
    document["cells"][0]["source"] = "MODE = 'live'\n"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert workshop._workshop_source_manifest() == before
    assert workshop._notebook_provenance()[key]["cell_source_sha256"] != saved[key]["cell_source_sha256"]
    second = workshop.WorkshopSession("live", output_dir=first.directory)
    assert second.manifest == first.manifest and offline_openai.calls == []
    monkeypatch.setattr(workshop, "_workshop_source_manifest", lambda: {**before, "runtime_changed.py": "different"})
    with pytest.raises(workshop.EvidenceError, match="another source"):
        workshop.WorkshopSession("live", output_dir=first.directory)


def test_real_mlflow_workflow_reuses_batches_across_fresh_sessions(offline_openai, tmp_path, monkeypatch):
    from tests.fakes import fake_policy_judge
    from west_workshop import judge_v2
    monkeypatch.setattr(judge_v2, "build_judge", fake_policy_judge)
    directory = tmp_path / "live-test-only"
    first = workshop.WorkshopSession("live", output_dir=directory)
    import mlflow
    with first.evaluation("deployed") as run:
        result = mlflow.genai.evaluate(data=run.data, predict_fn=run.predict_fn, scorers=run.scorers)
        first.record_evaluation("deployed", result, run)
    assert len(offline_openai.calls) == 10
    assert {span["name"] for span in first.recorded("deployed")["rows"][0]["spans"]} == {"refund_assistant", "retrieve_refund_policy", "generate_support_answer"}
    second = workshop.WorkshopSession("live", output_dir=directory)
    report = workshop.run_evaluation(second)
    assert len(offline_openai.calls) == 20, "notebook 02 must not regenerate notebook 01's deployed answers"
    assert report["evaluation_status"] == "complete"
    assert report["control_status"] == "passed"
    assert mlflow.get_run(second.recorded("controls")["evaluation"]["run_id"]).data.params["application_policy"] == "authored_control_outputs"
    assert report["policy_result"]["decision"] == "REVIEW_REQUIRED", report["policy_result"]
    assert report["policy_result"]["metrics"]["candidate_passes"] == 10
    original = second.recorded("repair")
    third = workshop.WorkshopSession("live", output_dir=directory)
    assert workshop.run_evaluation(third) == report
    assert len(offline_openai.calls) == 20
    assert third.recorded("repair") == original
    saved = workshop.WorkshopSession("saved", evidence_dir=directory)
    assert saved.finish() == report

    rescored = workshop.rescore_frozen(directory, tmp_path / "frozen-rescore-test-only", provider="openai")
    assert rescored["policy_result"]["decision"] == "REVIEW_REQUIRED", rescored["policy_result"]
    rescore_records = workshop.read_evidence(tmp_path / "frozen-rescore-test-only")["records"]
    assert mlflow.get_run(rescore_records["repair"]["run_id"]).data.params["application_policy"] == "frozen_source_outputs"
    assert len(offline_openai.calls) == 20, "rescoring must not regenerate application answers"


def test_legacy_rescore_preserves_all_three_versions_and_never_relabels_history(offline_openai, tmp_path, monkeypatch):
    from tests.fakes import fake_policy_judge
    from west_workshop import judge_v2
    monkeypatch.setattr(judge_v2, "build_judge", fake_policy_judge)
    cases = dataset()
    evaluations = []
    for variant in ("baseline", "candidate", "repaired"):
        evaluation = _record(cases, variant)
        evaluation["rows_path"] = "/original/" + variant + "-rows.json"
        evaluations.append(evaluation)
    source = {"provider": "openai", "utc_timestamp": "2026-10-18", "git_commit": "original-test-only",
              "evaluations": evaluations, "gates": {"candidate": {"decision": "block"}, "repaired": {"decision": "block"}}}
    path = tmp_path / "original-v1.json"
    raw = workshop._json_bytes(source)
    path.write_bytes(raw)
    output = tmp_path / "legacy-test-only"
    report = workshop.rescore_legacy(path, output, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert report["execution_status"] == "complete" and report["control_status"] == "passed"
    assert report["new_policy_decision"] is None and not report["comparison_interpreted"]
    assert report["original_policy_decisions"] == source["gates"]
    assert set(report["updated_evaluations"]) == {"baseline", "candidate", "repaired"}
    assert path.read_bytes() == raw
    assert len(offline_openai.calls) == 0
    for variant, evaluation in report["updated_evaluations"].items():
        assert {row["case_id"] for row in evaluation["rows"]} == {row["inputs"]["case_id"] for row in cases}
        assert all(row["output"] == "Eligibility: store_credit\nTest-only answer." for row in evaluation["rows"])
        assert len(report["differences"][variant]) == len(cases)
    with pytest.raises(workshop.EvidenceError, match="already exists"):
        workshop.rescore_legacy(path, output)


def test_legacy_rescore_rejects_a_wrong_source_hash_before_any_client(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(workshop.EvidenceError, match="SHA-256"):
        workshop.rescore_legacy(path, tmp_path / "output", expected_sha256="0" * 64)


def test_failed_application_execution_is_not_reported_as_a_completed_block(offline_openai, tmp_path, monkeypatch):
    from tests.fakes import fake_policy_judge
    from west_workshop import judge_v2
    monkeypatch.setattr(judge_v2, "build_judge", fake_policy_judge)

    def rejected(**request):
        raise RuntimeError("test-only provider failure")

    monkeypatch.setattr(offline_openai, "create", rejected)
    output = tmp_path / "failed-execution"
    code = workshop.main(["evaluate", "--mode", "live", "--provider", "openai", "--output", str(output)])
    assert code == 1
    report = json.loads((output / "report" / "result.json").read_text())
    assert report["execution_status"] == "error"
    assert report["policy_result"]["decision"] == "NOT_EVALUATED"
    records = workshop.read_evidence(output)["records"]
    assert "repair" not in records and "controls" not in records, "do not make more model requests after an application execution failure"


def test_take_home_adapter_and_reviewed_eleventh_case_use_the_actual_mlflow_path(offline_openai, tmp_path, monkeypatch):
    """An explicit alternative application, not the default fake chat client.

    The reviewed-case record is synthetic test input. No real human approval,
    model response, or release evidence is claimed by this offline test.
    """
    import mlflow
    from tests.fakes import fake_policy_judge
    from west_workshop import data, judge_v2, runtime

    proposed = {"inputs": {"case_id": "day_32_reviewed_v1", "days_since_purchase": 32, "defective": False,
                           "question": "I bought this 32 days ago. What return option is available?"},
                "expectations": {"expected_decision": "store_credit", "policy": CURRENT_POLICY}}
    approved = workshop.promote_reviewed_case({"status": "approved", "reviewer_kind": "human",
                                               "review_id": "synthetic-review-v1", "reviewer_id": "unit-test-only",
                                               "reviewed_at": "2026-10-18", "label_reason": "Day 32 is after the inclusive 30-day window.",
                                               "source_evidence": "synthetic-test-finding", "case": proposed})
    expanded = dataset() + [approved["case"]]
    monkeypatch.setattr(data, "dataset", lambda: copy.deepcopy(expanded))
    monkeypatch.setattr(workshop, "dataset", lambda: copy.deepcopy(expanded))
    monkeypatch.setattr(judge_v2, "build_judge", fake_policy_judge)
    received = []

    def application_adapter(provider, variant="repaired"):
        # Replace this business service with your own application. The argument
        # names match each case's inputs, just as the workshop adapter requires.
        window = 90 if variant == "candidate" else 30

        @mlflow.trace(name="refund_assistant", span_type="CHAIN")
        def predict(case_id: str, days_since_purchase: int, defective: bool, question: str):
            mlflow.update_current_trace(client_request_id=case_id, metadata={"client_request_id": case_id, "adapter": "offline-business-service"})
            received.append({"variant": variant, "inputs": {"case_id": case_id, "days_since_purchase": days_since_purchase,
                                                              "defective": defective, "question": question}})
            if defective:
                label, explanation = "support_review", "Contact support to review the defective product."
            else:
                label = "full_refund" if days_since_purchase <= window else "store_credit"
                explanation = f"The full-refund window is {window}-day. Contact support to request your eligible option."
            return f"Eligibility: {label}\n{explanation}"

        return predict

    monkeypatch.setattr(runtime, "make_predictor", application_adapter)
    session = workshop.WorkshopSession("live", output_dir=tmp_path / "adaptation-test-only")
    report = workshop.run_evaluation(session)
    assert len(received) == 22 and offline_openai.calls == [], "the replacement application, not either provider, must answer each case once"
    assert report["policy_result"]["evidence_checks_passed"]
    assert report["policy_result"]["comparison_interpreted"]
    assert report["policy_result"]["decision"] == "REVIEW_REQUIRED"
    assert report["policy_result"]["metrics"]["case_count"] == 11
    assert report["policy_result"]["metrics"]["candidate_passes"] == 11
    for variant, expected_label in (("deployed", "full_refund"), ("repair", "store_credit")):
        row = next(row for row in session.recorded(variant)["rows"] if row["case_id"] == proposed["inputs"]["case_id"])
        assert row["inputs"] == proposed["inputs"] and row["expectations"] == proposed["expectations"]
        assert row["output"].startswith("Eligibility: " + expected_label)
        assert row["trace_id"] and row["trace_state"] == "OK"
        assert row["spans"][0]["inputs"] == proposed["inputs"]
    pair = next(row for row in report["policy_result"]["paired_rows"] if row["case_id"] == proposed["inputs"]["case_id"])
    assert pair["baseline"] == 0 and pair["candidate"] == 1 and pair["change"] == "gain"


def test_changed_rubric_cannot_reuse_independent_control_provenance(offline_openai, tmp_path, monkeypatch):
    import mlflow
    from west_workshop import judge_v2
    evaluations = []
    monkeypatch.setattr(judge_v2, "JUDGE_DEFINITION_SHA256", "0" * 64)
    monkeypatch.setattr(mlflow.genai, "evaluate", lambda **kwargs: evaluations.append(kwargs))
    session = workshop.WorkshopSession("live", output_dir=tmp_path / "changed-rubric-test-only")
    with pytest.raises(workshop.EvidenceError, match="former held-out controls are now development evidence"):
        session.qualify_judge()
    assert evaluations == [] and offline_openai.calls == []
    assert not session._tracking_ready


def test_nonbinary_required_score_makes_evaluation_incomplete():
    cases = dataset()
    record = _record(cases, "before")
    record["rows"][0]["scores"]["policy_judge"] = 0.8
    assert not _complete(record["rows"], cases, ["deterministic_stack", "policy_judge"])


def test_failed_judge_request_retains_case_and_check_and_exits_execution_failure(offline_openai, tmp_path, monkeypatch, capsys):
    from mlflow.genai.scorers import scorer
    from west_workshop import judge_v2

    @scorer(name="policy_judge")
    def rejected_judge(inputs, outputs, expectations):
        if inputs["case_id"] == "day_31_boundary":
            raise RuntimeError("test-secret-service-detail-omit-from-output")
        return True

    monkeypatch.setattr(judge_v2, "build_judge", lambda provider: rejected_judge)
    output = tmp_path / "judge-failure-test-only"
    assert workshop.main(["evaluate", "--mode", "live", "--output", str(output)]) == 1
    evidence = workshop.read_evidence(output)
    record = evidence["records"]["deployed"]
    report = json.loads((output / "report" / "result.json").read_text())
    assert record["execution_status"] == "error" and not record["complete"]
    issue = next(item for item in record["execution_issues"] if item["code"] == "SCORER_EXECUTION_ERROR")
    assert issue["case_id"] == "day_31_boundary" and issue["check"] == "policy_judge"
    assert report["policy_result"]["decision"] == "NOT_EVALUATED" and report["execution_status"] == "error"
    assert any(item.get("case_id") == "day_31_boundary" and item.get("check") == "policy_judge" for item in report["policy_result"]["reasons"])
    assert "day_31_boundary" in workshop.render_report(report)
    assert "repair" not in evidence["records"] and "controls" not in evidence["records"]
    assert len(offline_openai.calls) == 10
    captured = capsys.readouterr()
    assert "test-secret-service-detail" not in captured.out + captured.err + json.dumps(record)


def test_completed_nonbinary_judge_feedback_is_invalid_evidence_not_execution_failure(offline_openai, tmp_path, monkeypatch):
    from mlflow.genai.scorers import scorer
    from west_workshop import judge_v2

    @scorer(name="policy_judge")
    def malformed_judge(inputs, outputs, expectations):
        if "authored_human_label" in expectations:
            return bool(expectations["authored_human_label"])
        return 0.5 if inputs["case_id"] == "day_31_boundary" else 1.0

    monkeypatch.setattr(judge_v2, "build_judge", lambda provider: malformed_judge)
    session = workshop.WorkshopSession("live", output_dir=tmp_path / "nonbinary-test-only")
    report = workshop.run_evaluation(session)
    assert report["execution_status"] == "complete" and report["policy_result"]["decision"] == "INVALID"
    for variant in workshop.VARIANTS:
        record = session.recorded(variant)
        assert record["execution_status"] == "complete" and record["execution_issues"] == [] and not record["complete"]
    assert len(offline_openai.calls) == 20


def test_failed_controls_preserve_first_outcome_and_skip_repair(offline_openai, tmp_path, monkeypatch):
    from mlflow.genai.scorers import scorer
    from west_workshop import data, judge_v2
    failed_id = data.judge_validation_dataset()[0]["inputs"]["case_id"]

    @scorer(name="policy_judge")
    def disagreeing_judge(inputs, outputs, expectations):
        if "authored_human_label" in expectations:
            label = bool(expectations["authored_human_label"])
            return not label if inputs["case_id"] == failed_id else label
        return True

    monkeypatch.setattr(judge_v2, "build_judge", lambda provider: disagreeing_judge)
    directory = tmp_path / "failed-control-test-only"
    session = workshop.WorkshopSession("live", output_dir=directory)
    report = workshop.run_evaluation(session)
    controls = session.recorded("controls")
    assert controls["evaluation"]["complete"] and controls["disagreements"] == [failed_id]
    assert not controls["passed"] and "repair" not in session.bundle["records"]
    assert report["execution_status"] == "complete" and report["control_status"] == "failed"
    assert report["policy_result"]["decision"] == "INVALID" and not report["policy_result"]["comparison_interpreted"]
    assert len(offline_openai.calls) == 10
    assert workshop.run_evaluation(workshop.WorkshopSession("live", output_dir=directory)) == report
    assert len(offline_openai.calls) == 10


@pytest.mark.parametrize("stage", ["deployed", "repair", "control-evaluation"])
def test_session_failure_record_produces_not_evaluated_without_more_calls(offline_openai, tmp_path, stage):
    from west_workshop.repair_policy import evaluate_repair
    session = workshop.WorkshopSession("live", output_dir=tmp_path / ("failure-" + stage))
    session._record("failure-" + stage, {"execution_status": "error", "stage": stage,
                                        "message": "Synthetic test-only interrupted evaluation."})
    report = session.finish()
    assert report["execution_status"] == "error" and report["policy_result"]["decision"] == "NOT_EVALUATED"
    assert evaluate_repair(**session.policy_inputs()) == report["policy_result"]
    assert workshop.run_evaluation(session) == report and offline_openai.calls == []


@pytest.mark.parametrize("exception_name", ["WorkshopExecutionError", "MlflowException"])
def test_setup_failures_are_safe_and_recorded_before_cli_exit(offline_openai, tmp_path, monkeypatch, capsys, exception_name):
    from mlflow.exceptions import MlflowException
    from west_workshop import runtime
    exception = runtime.WorkshopExecutionError if exception_name == "WorkshopExecutionError" else MlflowException

    def rejected_setup(*args):
        raise exception("https://private-service.example test-secret-service-detail")

    monkeypatch.setattr(runtime, "_setup_tracking", rejected_setup)
    output = tmp_path / "setup-failure-test-only"
    assert workshop.main(["evaluate", "--mode", "live", "--output", str(output)]) == 1
    record = json.loads((output / "execution-error.json").read_text())
    assert record["execution_status"] == "error"
    assert workshop.read_evidence(output)["records"]["failure-setup"]["category"] == "execution_or_setup"
    report = json.loads((output / "report" / "result.json").read_text())
    assert report["policy_result"]["decision"] == "NOT_EVALUATED"
    captured = capsys.readouterr()
    assert "test-secret-service-detail" not in captured.out + captured.err + json.dumps(record)
    assert offline_openai.calls == []


def test_cli_and_notebook_default_live_directory_reuse_the_same_bundle(offline_openai, tmp_path, monkeypatch):
    from tests.fakes import fake_policy_judge
    from west_workshop import judge_v2
    monkeypatch.setattr(judge_v2, "build_judge", fake_policy_judge)
    monkeypatch.setenv("WORKSHOP_OUTPUT_DIR", str(tmp_path / "shared-live-default"))
    session = workshop.notebook_session("live")
    workshop.run_evaluation(session)
    assert workshop.main(["evaluate", "--mode", "live"]) == 0
    assert (session.directory / "report" / "result.json").is_file()
    assert len(offline_openai.calls) == 20, "CLI review or gate must reuse the notebook's twenty answers"
    monkeypatch.delenv("WORKSHOP_OUTPUT_DIR")
    assert workshop.default_output_dir("openai") == Path("artifacts/workshop/openai/live")


def test_missing_trace_identifies_the_missing_case_as_execution_failure():
    cases = dataset()
    record = _record(cases, "missing-trace-test-only")
    missing = record["rows"].pop()["case_id"]
    issues = workshop._execution_issues(record["rows"], cases, ["deterministic_stack", "policy_judge"])
    assert issues == [{"code": "MISSING_CASE_RESULT", "case_id": missing,
                       "message": "The evaluated case has no retrievable result trace."}]


@pytest.mark.parametrize("starting_state", ["fresh", "partial-deployed", "failure-deployed", "failure-setup"])
def test_literal_compare_notebook_stops_provider_work_after_execution_failure(offline_openai, tmp_path, monkeypatch, starting_state):
    """Execute the actual notebook cells, including its fresh-kernel handoff.

    Only the provider and semantic judge are offline test doubles. MLflow runs,
    trace readback, notebook branching and the final policy/report are real.
    """
    import mlflow
    from tests.fakes import fake_policy_judge
    from west_workshop import judge_v2

    monkeypatch.setattr(judge_v2, "build_judge", fake_policy_judge)
    monkeypatch.setenv("WORKSHOP_MODE", "live")
    monkeypatch.delenv("WORKSHOP_REVIEW_JSON", raising=False)
    monkeypatch.setenv("WORKSHOP_OUTPUT_DIR", str(tmp_path / "literal-notebook-failure-test-only"))
    original_create = offline_openai.create
    attempts = []

    def sometimes_unavailable(**request):
        inputs = json.loads(request["messages"][1]["content"])
        attempts.append(inputs)
        if inputs["days_since_purchase"] == 31:
            raise RuntimeError("test-only application unavailable on day 31")
        return original_create(**request)

    monkeypatch.setattr(offline_openai, "create", sometimes_unavailable)
    original_evaluate = mlflow.genai.evaluate
    evaluations = []

    def observed_evaluate(**kwargs):
        evaluations.append({"authored": kwargs.get("predict_fn") is None, "cases": len(kwargs["data"])})
        return original_evaluate(**kwargs)

    monkeypatch.setattr(mlflow.genai, "evaluate", observed_evaluate)
    previous = workshop.WorkshopSession("live")
    if starting_state == "partial-deployed":
        with previous.evaluation("deployed") as run:
            result = mlflow.genai.evaluate(data=run.data, predict_fn=run.predict_fn, scorers=run.scorers)
            previous.record_evaluation("deployed", result, run)
        assert previous.recorded("deployed")["execution_status"] == "error"
    elif starting_state.startswith("failure-"):
        previous._record(starting_state, {"execution_status": "error", "stage": starting_state.removeprefix("failure-"),
                                          "message": "Synthetic test-only interrupted stage."})
    before_attempts, before_evaluations = len(attempts), len(evaluations)
    namespace = {"__name__": "literal_notebook_test"}
    document = json.loads((ROOT / "notebooks/workshop/02_compare_and_decide.ipynb").read_text(encoding="utf-8"))
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] == "code":
            exec(compile(cell["source"], f"02_compare_and_decide.ipynb:cell-{index}", "exec"), namespace)
    records = namespace["session"].bundle["records"]
    assert "repair" not in records
    assert namespace["report"]["execution_status"] == "error"
    assert namespace["policy_result"]["decision"] == "NOT_EVALUATED"
    if starting_state == "fresh":
        assert len(attempts) == 10
        assert evaluations == [{"authored": True, "cases": 20}, {"authored": False, "cases": 10}]
        assert {issue["case_id"] for issue in records["deployed"]["execution_issues"]} == {"day_31_boundary"}
    else:
        assert len(attempts) == before_attempts and len(evaluations) == before_evaluations
        assert "controls" not in records, "a previously failed deployed stage must not trigger new judge calls"
