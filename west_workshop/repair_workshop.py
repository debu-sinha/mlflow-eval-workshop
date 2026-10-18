"""Two-configuration workshop: immutable evidence, real MLflow, explicit policy.

Saved mode reads verified dated evidence with the standard library only. Live
mode is an explicit opt-in and calls the configured application and judge. The
same session directory connects independent notebook kernels; an existing stage
is read back, never silently regenerated to obtain a more attractive result.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import sys
import uuid

from .data import CURRENT_POLICY, STALE_POLICY, dataset, dataset_digest

SCHEMA_VERSION = 2
VARIANTS = {"deployed": "candidate", "repair": "repaired"}
VARIANT_LABELS = {
    "deployed": "Simulated deployed assistant / obsolete policy",
    "repair": "Proposed repair / current policy",
}


class EvidenceError(RuntimeError):
    """An evidence problem with a safe, actionable message."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def project_root(start=None):
    """Find a checkout from cwd, including a local or Databricks notebook cwd."""
    explicit = os.environ.get("WORKSHOP_PROJECT_ROOT")
    origin = Path(explicit or start or Path.cwd()).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / "west_workshop" / "data.py").is_file():
            return candidate
    raise EvidenceError("Open the notebook inside the workshop checkout, or set WORKSHOP_PROJECT_ROOT to that checkout.")


def _notebook_defaults():
    # Presence-only setup. Do not resolve credentials or call a workspace API.
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        os.environ.setdefault("WORKSHOP_PROVIDER", "databricks")
        os.environ.setdefault("WORKSHOP_DATABRICKS_MODEL", "databricks-qwen3-next-80b-a3b-instruct")


def default_evidence_dir(root=None):
    return Path(os.environ.get("WORKSHOP_EVIDENCE_DIR", "")) if os.environ.get("WORKSHOP_EVIDENCE_DIR") else (root or project_root()) / "evidence" / "workshop" / "local" / "final"


def default_output_dir(provider=None):
    """One persisted live directory for the CLI and independent notebook kernels."""
    if os.environ.get("WORKSHOP_OUTPUT_DIR"):
        return Path(os.environ["WORKSHOP_OUTPUT_DIR"])
    from .config import selected_provider
    return Path("artifacts/workshop") / selected_provider(provider) / "live"


def _read_manifest(directory):
    directory = Path(directory).resolve()
    path = directory / "manifest.json"
    if not path.is_file():
        raise EvidenceError("No saved evidence manifest was found. Use the released evidence folder or explicitly run live mode; saved mode never invents answers or scores.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != SCHEMA_VERSION or not isinstance(manifest.get("files"), dict)
                or not isinstance(manifest.get("records"), dict) or not isinstance(manifest.get("decisions"), list)
                or not isinstance(manifest.get("evidence_id"), str) or not manifest["evidence_id"]
                or not isinstance(manifest.get("created_at"), str) or manifest.get("provider") not in {"openai", "databricks"}):
            raise ValueError("schema")
        for name, expected in manifest["files"].items():
            member = (directory / name).resolve()
            if not member.is_relative_to(directory) or not member.is_file():
                raise EvidenceError("A saved evidence file is missing or escapes its evidence folder.")
            if hashlib.sha256(member.read_bytes()).hexdigest() != expected:
                raise EvidenceError("Saved evidence integrity check failed. Restore the released files; do not treat modified scores as the original result.")
        for record in manifest.get("records", {}).values():
            if record not in manifest["files"]:
                raise ValueError("unverified record")
        for decision in manifest.get("decisions", []):
            if decision not in manifest["files"]:
                raise ValueError("unverified decision")
    except (ValueError, TypeError, KeyError, OSError) as error:
        if isinstance(error, EvidenceError):
            raise
        raise EvidenceError("The saved evidence manifest is invalid or unreadable.") from None
    return manifest


def read_evidence(directory):
    """Verify every manifest member before exposing any saved result."""
    directory = Path(directory).resolve()
    manifest = _read_manifest(directory)
    records = {name: json.loads((directory / path).read_text(encoding="utf-8")) for name, path in manifest.get("records", {}).items()}
    decisions = [json.loads((directory / path).read_text(encoding="utf-8")) for path in manifest.get("decisions", [])]
    return {"manifest": manifest, "records": records, "decisions": decisions}


def workshop_preflight(mode="saved", *, evidence_dir=None, provider=None):
    """Presence, dependency and file checks only; never contacts a model/store."""
    _notebook_defaults()
    if mode == "saved":
        try:
            evidence = read_evidence(evidence_dir or default_evidence_dir())
            manifest = evidence["manifest"]
            return {"ready": True, "mode": "saved", "network_checked": False,
                    "credential_values_included": False, "evidence_id": manifest["evidence_id"],
                    "recorded_at": manifest["created_at"], "provider": manifest["provider"],
                    "verified_files": len(manifest["files"]), "reasons": []}
        except EvidenceError as error:
            return {"ready": False, "mode": "saved", "network_checked": False,
                    "credential_values_included": False, "reasons": [str(error)]}
    if mode != "live":
        raise ValueError("Mode must be saved or live.")
    from .config import preflight
    return {**preflight(provider), "mode": "live", "python": platform.python_version()}


def _pending_review(baseline, candidate, controls):
    from .repair_policy import evidence_identity
    return {"version": "review-pending-v1", "status": "pending", "findings": [],
            "evidence": evidence_identity(baseline, candidate, controls)}


def _workshop_source_manifest():
    from .runtime import _source_manifest
    values = _source_manifest()
    root = project_root()
    files = [root / name for name in ("pyproject.toml", "uv.lock", "requirements-workshop.txt")]
    values.update({path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path.is_file()})
    return values


def _notebook_provenance():
    """Record notebook state without blocking saves or attendee setup-cell edits.

    The blocking identity covers executed runtime, data, scorer, policy, model
    and dependency definitions. Notebooks remain editable callers of that API.
    Both raw-file and canonical cell-source hashes are retained for inspection.
    """
    root = project_root()
    values = {}
    for path in (root / "notebooks" / "workshop").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = [{"cell_type": cell["cell_type"],
                   "source": "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]}
                  for cell in notebook["cells"]]
        values[path.relative_to(root).as_posix()] = {"raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                    "cell_source_sha256": _digest(source)}
    return values


def _execution_issues(rows, expected_rows, scorer_names):
    """Separate failed/missing execution from completed but nonbinary feedback."""
    expected_ids = {row["inputs"]["case_id"] for row in expected_rows}
    found_ids = {row["case_id"] for row in rows}
    issues = [{"code": "MISSING_CASE_RESULT", "case_id": key,
               "message": "The evaluated case has no retrievable result trace."}
              for key in sorted(expected_ids - found_ids)]
    for row in rows:
        key = row["case_id"]
        if row.get("trace_state") != "OK":
            issues.append({"code": "TRACE_EXECUTION_ERROR", "case_id": key,
                           "message": "The application trace did not finish successfully."})
        if not isinstance(row.get("output"), str) or not row["output"].strip():
            issues.append({"code": "MISSING_APPLICATION_OUTPUT", "case_id": key,
                           "message": "The application returned no usable answer text."})
        assessments = row.get("assessments", [])
        for name in scorer_names:
            matching = [item for item in assessments if item.get("name") == name]
            if not matching:
                issues.append({"code": "MISSING_ASSESSMENT", "case_id": key, "check": name,
                               "message": "The evaluation did not return this scorer's assessment."})
            elif any(item.get("error") for item in matching):
                issues.append({"code": "SCORER_EXECUTION_ERROR", "case_id": key, "check": name,
                               "message": "The scorer failed. Its raw provider exception is omitted."})
    return issues


def _safe_command_error(error):
    """Allowlisted diagnostics only; provider exceptions can contain credentials."""
    if isinstance(error, EvidenceError):
        return "evidence", str(error)
    if isinstance(error, (OSError, ValueError)):
        return "input_or_storage", "The command could not read or write its declared inputs and evidence."
    if isinstance(error, ImportError):
        return "environment", "The locked workshop environment could not be loaded."
    return "execution_or_setup", "Execution or MLflow setup failed. Check model access, tracking permissions and the locked environment. Raw service details were omitted."


def review_skeleton(directory):
    """A bound review form, deliberately pending until a reviewer completes it."""
    records = read_evidence(directory)["records"]
    return {**_pending_review(records.get("deployed", {}), records.get("repair", {}), records.get("controls", {})),
            "reviewer_kind": "", "reviewer_id": "",
            "instructions": "Inspect the answers, rationales and controls. Record material findings and their evidence. Set complete only after review; an empty form is not approval."}


class WorkshopSession:
    """Small persistence boundary around visible native MLflow evaluation calls."""

    def __init__(self, mode="saved", *, evidence_dir=None, output_dir=None, provider=None):
        _notebook_defaults()
        if mode not in {"saved", "live"}:
            raise ValueError("Mode must be saved or live.")
        self.mode = mode
        self._scorers = None
        self._tracking_ready = False
        self._active = None
        if mode == "saved":
            self.directory = Path(evidence_dir or default_evidence_dir()).resolve()
            self.bundle = read_evidence(self.directory)
            self.provider = self.bundle["manifest"]["provider"]
            return
        from .config import selected_provider
        from .runtime import _commit
        from .repair_policy import POLICY_VERSION
        from .judge_v2 import JUDGE_VERSION, JUDGE_DEFINITION_SHA256
        self.provider = selected_provider(provider)
        readiness = workshop_preflight("live", provider=self.provider)
        if not readiness["ready"]:
            raise EvidenceError("Live setup is incomplete. " + " ".join(readiness["reasons"]))
        self.directory = Path(output_dir or default_output_dir(self.provider)).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        identity = {"provider": self.provider, "application_model": readiness["application_model"],
                    "judge_model": readiness["judge_model"], "judge_version": JUDGE_VERSION,
                    "judge_definition_sha256": JUDGE_DEFINITION_SHA256, "policy_version": POLICY_VERSION,
                    "dataset_digest": dataset_digest(), "source_sha256": _workshop_source_manifest(),
                    "versions": readiness["versions"]}
        if (self.directory / "manifest.json").exists():
            self.bundle = read_evidence(self.directory)
            if self.bundle["manifest"].get("measurement_identity") != identity:
                raise EvidenceError("This directory belongs to another source, environment, dataset, model or policy version. Use a new output directory and record the reason; existing evidence is preserved.")
        else:
            self.bundle = {"manifest": {"schema_version": SCHEMA_VERSION, "evidence_id": "repair-" + uuid.uuid4().hex,
                                       "created_at": _now(), "provider": self.provider, "git_commit": _commit(),
                                       "measurement_identity": identity, "files": {}, "records": {}, "decisions": [],
                                       "notebook_provenance": {"informational_only": True, "at_creation": _notebook_provenance()},
                                       "production_authorization": False}, "records": {}, "decisions": []}
            self._save_manifest()
            self._record("cases", dataset())
            self._record("environment", {**readiness, "python": platform.python_version()})

    @property
    def manifest(self):
        return self.bundle["manifest"]

    def _save_manifest(self):
        path = self.directory / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(_json_bytes(self.manifest))
        temporary.replace(path)

    def _write_member(self, name, value):
        from .runtime import _sanitize
        encoded = _json_bytes(_sanitize(value))
        path = self.directory / name
        if path.exists() and path.read_bytes() != encoded:
            raise EvidenceError("An immutable evidence record already exists. Keep it and use a new version or evidence directory.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
        self.manifest["files"][name] = hashlib.sha256(encoded).hexdigest()
        return json.loads(encoded)

    def _record(self, key, value):
        stored = self._write_member(key + ".json", value)
        self.manifest["records"][key] = key + ".json"
        self.bundle["records"][key] = stored
        self._save_manifest()
        return stored

    def needs(self, variant):
        if variant not in VARIANTS:
            raise ValueError("Use deployed or repair.")
        return self.mode == "live" and variant not in self.bundle["records"] and "failure-" + variant not in self.bundle["records"]

    def has_execution_failure(self):
        """Stop further calls after any persisted failed request or stage setup."""
        return any(key.startswith("failure-") or
                   isinstance(record, dict) and (record.get("execution_status") == "error" or
                                                isinstance(record.get("evaluation"), dict) and record["evaluation"].get("execution_status") == "error")
                   for key, record in self.bundle["records"].items())

    def recorded(self, variant):
        if variant not in self.bundle["records"]:
            raise EvidenceError("This evidence set has no recorded " + variant + " evaluation.")
        return copy.deepcopy(self.bundle["records"][variant])

    def _setup(self):
        if self.mode != "live":
            raise EvidenceError("Saved mode never initializes a model client or tracking server.")
        if not self._tracking_ready:
            from .runtime import _configure_timeouts, _setup_tracking, _quiet_dependencies
            try:
                with _quiet_dependencies():
                    _configure_timeouts(self.provider)
                    self.experiment_id = _setup_tracking(self.provider, self.directory)
            except Exception as error:
                category, message = _safe_command_error(error)
                self._record("failure-setup", {"stage": "setup", "at": _now(), "error_type": type(error).__name__,
                                               "execution_status": "error", "category": category, "message": message})
                raise EvidenceError(message) from None
            self._tracking_ready = True

    def scorers(self):
        if self._scorers is None:
            from .runtime import build_scorers
            from .judge_v2 import build_judge
            self._scorers = build_scorers(self.provider, include_judge=False) + [build_judge(self.provider)]
        return self._scorers

    @contextmanager
    def evaluation(self, variant, *, data=None, scorers=None, authored=False):
        """Expose data, predict_fn and scorers while recording native run provenance."""
        if variant in self.bundle["records"]:
            raise EvidenceError("That evaluation is already frozen; read it instead of regenerating it.")
        if "failure-" + variant in self.bundle["records"]:
            raise EvidenceError("This stage already recorded an execution failure. Preserve it and record a retry reason in a new evidence directory.")
        if self._active is not None:
            raise EvidenceError("Finish the current evaluation before starting another.")
        self._setup()
        import mlflow
        import pandas as pd
        from types import SimpleNamespace
        from .runtime import _scorer_manifest, make_predictor, _quiet_dependencies
        rows = copy.deepcopy(data if data is not None else dataset())
        checks = scorers if scorers is not None else self.scorers()
        definition = _scorer_manifest(checks)
        predict = None if authored else make_predictor(self.provider, VARIANTS[variant])
        with _quiet_dependencies(), mlflow.start_run(run_name="workshop-v2-" + variant, nested=mlflow.active_run() is not None) as run:
            run_info = SimpleNamespace(data=rows, predict_fn=predict, scorers=checks, manifest=definition, run_id=run.info.run_id, variant=variant)
            self._active = run_info
            try:
                application_policy = ("authored_control_outputs" if variant == "control-evaluation" else
                                      "frozen_source_outputs" if authored else
                                      "2024-01" if variant == "deployed" else "2026-09")
                mlflow.log_params({"workshop_variant": variant, "dataset_sha256": dataset_digest(rows),
                                   "evidence_id": self.manifest["evidence_id"], "authored_outputs": authored,
                                   "application_policy": application_policy})
                mlflow.set_tags({"workshop": "odsc-west-repair-v2", "git_commit": self.manifest["git_commit"]})
                mlflow.log_dict(self.manifest["measurement_identity"], "measurement_identity.json")
                mlflow.log_dict(_notebook_provenance(), "notebook_provenance_informational.json")
                mlflow.log_dict(rows, "evaluation_input.json")
                mlflow.log_dict(definition, "scorer_definitions.json")
                mlflow.log_dict({check.name: check.model_dump() for check in checks}, "scorer_definitions_full.json")
                mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(rows), name="repair-cases-" + dataset_digest(rows)[:12]), context="evaluation")
                yield run_info
            except Exception as error:
                self._record("failure-" + variant, {"stage": variant, "at": _now(), "error_type": type(error).__name__,
                                                 "execution_status": "error", "run_id": run.info.run_id,
                                                 "message": "Evaluation did not finish. Inspect provider access and the recorded run. No answer or score was substituted."})
                raise EvidenceError("Evaluation failed; partial evidence and its run ID were preserved. Inspect the failure before retrying into a new evidence directory.") from None
            finally:
                self._active = None

    def record_evaluation(self, variant, result, evaluation):
        """Read actual traced assessments, then persist before the next stage."""
        import mlflow
        from .runtime import _complete, _trace_rows, _sanitize
        if evaluation is not self._active or evaluation.variant != variant:
            raise EvidenceError("Evaluation result is not bound to the active stage.")
        run_id = result.run_id or evaluation.run_id
        if run_id != evaluation.run_id:
            raise EvidenceError("MLflow returned a different evaluation run identity.")
        rows = _trace_rows(run_id, evaluation.data)
        issues = _execution_issues(rows, evaluation.data, [item.name for item in evaluation.scorers])
        record = {"run_id": run_id, "rows": rows, "row_count": len(rows),
                  "execution_status": "error" if issues else "complete", "execution_issues": issues,
                  "complete": _complete(rows, evaluation.data, [item.name for item in evaluation.scorers]),
                  "metrics": _sanitize(dict(result.metrics)), "scorers": evaluation.manifest,
                  "trace_ids": [row["trace_id"] for row in rows],
                  "dataset_digest": dataset_digest([{key: row[key] for key in ("inputs", "expectations")} for row in evaluation.data]) if variant in VARIANTS else dataset_digest(evaluation.data),
                  "evaluation_data_digest": dataset_digest(evaluation.data),
                  "recorded_at": _now(), "variant": variant,
                  "notebook_provenance": {"informational_only": True, "at_evaluation": _notebook_provenance()}}
        mlflow.log_dict(_sanitize(record), "evaluation_evidence.json")
        return self._record(variant, record)

    def qualify_judge(self):
        """Run each frozen development and independent control once, with no app calls."""
        if "controls" in self.bundle["records"]:
            return self.recorded("controls")
        if self.mode == "saved":
            raise EvidenceError("This saved evidence set has no evaluator controls.")
        import mlflow
        from .judge_v2 import development_controls, JUDGE_VERSION, JUDGE_DEFINITION_SHA256
        from .qualification_controls import qualification_controls, CONTROL_VERSION, FROZEN_JUDGE_DEFINITION_SHA256
        from .data import judge_validation_dataset
        if JUDGE_DEFINITION_SHA256 != FROZEN_JUDGE_DEFINITION_SHA256:
            raise EvidenceError("The judge rubric differs from the version frozen before the independent controls were authored. Declare a new judge version and new control provenance; the former held-out controls are now development evidence. No evaluation was started.")
        legacy, development, independent = judge_validation_dataset(), development_controls(), qualification_controls()
        cases = legacy + development + independent
        judge = next(check for check in self.scorers() if check.name == "policy_judge")
        with self.evaluation("control-evaluation", data=cases, scorers=[judge], authored=True) as run:
            result = mlflow.genai.evaluate(data=run.data, predict_fn=None, scorers=run.scorers)
            evaluation = self.record_evaluation("control-evaluation", result, run)
        disagreements = [row["case_id"] for row in evaluation["rows"]
                         if row["scores"].get("policy_judge") != float(row["expectations"]["authored_human_label"])]
        return self._record("controls", {"version": CONTROL_VERSION, "judge_version": JUDGE_VERSION,
                                         "frozen_judge_definition_sha256": FROZEN_JUDGE_DEFINITION_SHA256,
                                         "expected_rows": cases, "evaluation": evaluation,
                                         "passed": evaluation["complete"] and not disagreements,
                                         "disagreements": disagreements,
                                         "partitions": {"legacy_authored": [row["inputs"]["case_id"] for row in legacy],
                                                        "development": [row["inputs"]["case_id"] for row in development],
                                                        "independent": [row["inputs"]["case_id"] for row in independent]},
                                         "provenance": "preserved original authored controls, versioned development controls, and separately authored independent controls",
                                         "representative_accuracy_claim": False})

    def finish(self, review=None):
        """Calculate the full policy without changing any answer, score or old decision."""
        from .repair_policy import evaluate_repair
        if review is None and self.bundle["decisions"]:
            return copy.deepcopy(self.bundle["decisions"][-1])
        inputs = self.policy_inputs(review)
        baseline, candidate, controls, review = (inputs[key] for key in ("baseline", "candidate", "controls", "review"))
        policy = evaluate_repair(**inputs)
        report = {"schema_version": SCHEMA_VERSION, "evidence_id": self.manifest["evidence_id"],
                  "evidence_recorded_at": self.manifest["created_at"], "provider": self.provider,
                  "mode": "recorded_live_evidence", "execution_status": policy["execution_status"],
                  "evaluation_status": "complete" if baseline.get("complete") and candidate.get("complete") else "incomplete",
                  "control_status": "passed" if controls.get("passed") else "failed" if controls else "not_run",
                  "review_status": review.get("status", "invalid") if isinstance(review, dict) else "invalid", "review": review,
                  "execution_issues": [{"scope": name, **issue} for name, record in
                                       (("deployed", baseline), ("repair", candidate), ("controls", controls.get("evaluation", {})))
                                       for issue in record.get("execution_issues", [])],
                  "policy_result": policy, "production_authorization": False,
                  "limitation": "Ten teaching cases and a small evaluator smoke test do not authorize a production deployment."}
        # Read-only saved mode may calculate a review, but never modifies a release bundle.
        if self.mode == "live":
            name = "decisions/" + _digest(report)[:20] + ".json"
            report = self._write_member(name, report)
            if name not in self.manifest["decisions"]:
                self.manifest["decisions"].append(name)
                self.bundle["decisions"].append(report)
            self._save_manifest()
        return report

    def policy_inputs(self, review=None):
        """Inputs for the visible pure-policy call, with the original reviewed binding."""
        records = self.bundle["records"]
        if review is None and self.bundle["decisions"]:
            review = copy.deepcopy(self.bundle["decisions"][-1]["review"])
        baseline, candidate, controls = records.get("deployed", {}), records.get("repair", {}), records.get("controls", {})
        if not baseline and ("failure-deployed" in records or "failure-setup" in records):
            baseline = records.get("failure-deployed", records.get("failure-setup"))
        if not candidate and "failure-repair" in records:
            candidate = records["failure-repair"]
        if not controls and "failure-control-evaluation" in records:
            controls = {"passed": False, "evaluation": records["failure-control-evaluation"]}
        return {"baseline": copy.deepcopy(baseline), "candidate": copy.deepcopy(candidate),
                "expected_rows": copy.deepcopy(records.get("cases", dataset())), "controls": copy.deepcopy(controls),
                "review": copy.deepcopy(review) if review is not None else _pending_review(baseline, candidate, controls)}

    def banner(self):
        return {"mode": self.mode.upper(), "recorded_at": self.manifest["created_at"],
                "evidence_id": self.manifest["evidence_id"], "provider": self.provider,
                "app_configuration": "simulated deployed obsolete policy → proposed current-policy repair",
                "production_authorization": False}


def notebook_session(mode=None, **kwargs):
    return WorkshopSession(mode or os.environ.get("WORKSHOP_MODE", "saved"), **kwargs)


def run_evaluation(session):
    """CLI equivalent of the two notebooks' visible native MLflow calls."""
    if session.mode == "saved":
        return session.finish()
    if session.has_execution_failure():
        return session.finish()
    import mlflow
    if session.needs("deployed"):
        with session.evaluation("deployed") as run:
            result = mlflow.genai.evaluate(data=run.data, predict_fn=run.predict_fn, scorers=run.scorers)
            session.record_evaluation("deployed", result, run)
    if session.has_execution_failure():
        return session.finish()
    controls = session.qualify_judge()
    if controls["passed"] and session.needs("repair"):
        with session.evaluation("repair") as run:
            result = mlflow.genai.evaluate(data=run.data, predict_fn=run.predict_fn, scorers=run.scorers)
            session.record_evaluation("repair", result, run)
    return session.finish()


def rescore_frozen(source_dir, output_dir, *, provider=None):
    """Score every frozen compared answer under v2; never generate new answers.

    Source must be a verified workshop bundle. Both versions use one new scorer
    definition. Original scores and decisions remain in the source directory.
    """
    source = read_evidence(source_dir)
    if not all(name in source["records"] for name in VARIANTS):
        raise EvidenceError("Rescoring requires both frozen compared answer sets.")
    if source["records"].get("cases") != dataset():
        raise EvidenceError("The source case version differs from this checkout. Select the matching case version before rescoring.")
    session = WorkshopSession("live", output_dir=output_dir, provider=provider)
    if session.manifest["provider"] != source["manifest"]["provider"]:
        raise EvidenceError("Keep local and Databricks routes separate when rescoring frozen answers.")
    if any(name in session.bundle["records"] for name in VARIANTS):
        raise EvidenceError("Use a new output directory for a versioned rescoring experiment.")
    session._record("rescore-source", {"evidence_id": source["manifest"]["evidence_id"],
                                      "manifest_sha256": hashlib.sha256((Path(source_dir) / "manifest.json").read_bytes()).hexdigest(),
                                      "reason": "Versioned evaluator correction; fixed answers, inputs and expected labels."})
    import mlflow
    for variant in VARIANTS:
        original = source["records"][variant]
        by_id = {row["case_id"]: row for row in original["rows"]}
        frozen = [{"inputs": case["inputs"], "expectations": case["expectations"], "outputs": by_id[case["inputs"]["case_id"]]["output"]} for case in source["records"]["cases"]]
        with session.evaluation(variant, data=frozen, authored=True) as run:
            result = mlflow.genai.evaluate(data=run.data, predict_fn=None, scorers=run.scorers)
            updated = session.record_evaluation(variant, result, run)
        originals = {row["case_id"]: row for row in original["rows"]}
        changes = []
        for row in updated["rows"]:
            before = originals[row["case_id"]]
            if row["inputs"] != before["inputs"] or row["output"] != before["output"] or row["expectations"] != before["expectations"]:
                raise EvidenceError("Rescoring changed frozen input, output or expectations; evidence is invalid.")
            changes.append({"case_id": row["case_id"], "answer_sha256": hashlib.sha256(row["output"].encode()).hexdigest(),
                            "original_trace_id": before["trace_id"], "new_trace_id": row["trace_id"],
                            "before_scores": before["scores"], "after_scores": row["scores"]})
        session._record("rescore-diff-" + variant, changes)
    session.qualify_judge()
    return session.finish()


def rescore_legacy(summary_path, output_dir, *, provider=None, expected_sha256=None):
    """Rescore all three archived v1 answer sets without assigning them a v2 verdict.

    The original file stays untouched. The new bundle retains a hash of its exact
    bytes, selected original provenance, every frozen answer/assessment, the old
    decisions, and new scores. No private source path is exported.
    """
    raw = Path(summary_path).read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and source_hash != expected_sha256:
        raise EvidenceError("The original summary does not match the declared source SHA-256.")
    original = json.loads(raw)
    source_provider = original.get("provider")
    if source_provider not in {"openai", "databricks"} or (provider is not None and provider != source_provider):
        raise EvidenceError("Keep the original local and Databricks routes separate for legacy rescoring.")
    evaluations = original.get("evaluations", [])
    if len(evaluations) != 3:
        raise EvidenceError("Historical rescoring requires the original three-variant checkpoint 4 summary.")
    by_variant = {}
    for evaluation in evaluations:
        basename = str(evaluation.get("rows_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
        variant = basename.removesuffix("-rows.json")
        if variant not in {"baseline", "candidate", "repaired"} or variant in by_variant:
            raise EvidenceError("Original variant identities are missing or ambiguous; no role is inferred from score or list order.")
        if not evaluation.get("complete") or not isinstance(evaluation.get("rows"), list):
            raise EvidenceError("Freeze complete original answer evidence before versioned rescoring.")
        by_variant[variant] = evaluation
    session = WorkshopSession("live", output_dir=output_dir, provider=source_provider)
    if "legacy-source" in session.bundle["records"]:
        raise EvidenceError("This historical rescoring experiment already exists. Preserve its first outcome.")
    selected = {key: copy.deepcopy(original.get(key)) for key in ("provider", "utc_timestamp", "git_commit", "source_sha256", "gates", "repair_comparison")}
    selected.update(source_summary_sha256=source_hash, measurement_type="historical_v1_frozen_answers",
                    original_policy_unchanged=True, original_summary_preserved_at_source=True,
                    evaluations={variant: {key: copy.deepcopy(evaluation.get(key)) for key in ("run_id", "rows", "row_count", "complete", "scorers", "dataset_digest", "metrics", "trace_ids")} for variant, evaluation in by_variant.items()})
    session._record("legacy-source", selected)
    controls = session.qualify_judge()
    updated_records, differences = {}, {}
    if controls["passed"]:
        import mlflow
        for variant in ("baseline", "candidate", "repaired"):
            before = by_variant[variant]
            frozen = [{"inputs": row["inputs"], "expectations": row["expectations"], "outputs": row["output"]} for row in before["rows"]]
            key = "legacy-" + variant
            with session.evaluation(key, data=frozen, authored=True) as run:
                result = mlflow.genai.evaluate(data=run.data, predict_fn=None, scorers=run.scorers)
                updated = session.record_evaluation(key, result, run)
            original_rows = {row["case_id"]: row for row in before["rows"]}
            changed = []
            for row in updated["rows"]:
                previous = original_rows[row["case_id"]]
                if row["inputs"] != previous["inputs"] or row["expectations"] != previous["expectations"] or row["output"] != previous["output"]:
                    raise EvidenceError("Historical rescoring changed a frozen answer or its case identity.")
                changed.append({"case_id": row["case_id"], "answer_sha256": hashlib.sha256(row["output"].encode()).hexdigest(),
                                "original_trace_id": previous["trace_id"], "new_trace_id": row["trace_id"],
                                "before_scores": previous["scores"], "after_scores": row["scores"]})
            updated_records[variant] = updated
            differences[variant] = session._record("legacy-diff-" + variant, changed)
    report = {"measurement_type": "historical_v1_frozen_answer_rescore", "evidence_id": session.manifest["evidence_id"],
              "source_summary_sha256": source_hash, "provider": source_provider,
              "execution_status": "error" if any(item.get("execution_status") == "error" for item in [controls["evaluation"], *updated_records.values()]) else "complete",
              "evaluation_status": "complete" if controls["evaluation"]["complete"] and all(item["complete"] for item in updated_records.values()) else "incomplete",
              "control_status": "passed" if controls["passed"] else "failed", "comparison_interpreted": False,
              "original_policy_decisions": copy.deepcopy(original.get("gates")), "new_policy_decision": None,
              "updated_evaluations": updated_records, "differences": differences,
              "note": "These are new evaluator measurements of all three frozen historical answer sets. The old verdicts remain unchanged. No repair-policy v2 verdict is assigned to this history."}
    return session._record("legacy-rescore-result", report)


def promote_reviewed_case(review_record, *, existing_cases=None):
    """Promote an explicitly approved human-labelled case; feedback alone is not truth."""
    required = ("review_id", "reviewer_id", "reviewed_at", "label_reason", "source_evidence")
    if review_record.get("status") != "approved" or review_record.get("reviewer_kind") != "human" or any(not review_record.get(key) for key in required):
        raise EvidenceError("Promotion needs a named human approval, review date, label rationale and source evidence. Automatic feedback is not a ground-truth label.")
    case = review_record.get("case")
    if not isinstance(case, dict) or not isinstance(case.get("inputs"), dict) or not isinstance(case.get("expectations"), dict):
        raise EvidenceError("The approved case must contain inputs and expectations.")
    inputs, expectations = case["inputs"], case["expectations"]
    if not isinstance(inputs.get("case_id"), str) or not inputs["case_id"].strip() or expectations.get("expected_decision") not in {"full_refund", "store_credit", "support_review"}:
        raise EvidenceError("Use a stable case ID and a reviewed expected eligibility label.")
    if inputs["case_id"] in {row["inputs"]["case_id"] for row in (existing_cases or dataset())}:
        raise EvidenceError("The approved case ID already exists; use a new case version.")
    if set(inputs) != {"case_id", "days_since_purchase", "defective", "question"} or type(inputs["days_since_purchase"]) is not int or inputs["days_since_purchase"] < 0 or type(inputs["defective"]) is not bool or not isinstance(inputs["question"], str) or not inputs["question"].strip():
        raise EvidenceError("The approved case does not satisfy the application's input contract.")
    if expectations.get("policy") != CURRENT_POLICY:
        raise EvidenceError("Bind the reviewed label to the current evaluation policy.")
    return {"case": copy.deepcopy(case), "review_provenance": {key: copy.deepcopy(review_record[key]) for key in (*required, "reviewer_kind")},
            "dataset_version_required": True, "compare_every_configuration_again": True}


def _reason_text(reason):
    message = reason.get("message", "")
    details = [str(key).replace("_", " ") + ": " + str(reason[key])
               for key in ("case_id", "check", "case_ids", "finding_id", "scope") if reason.get(key) is not None]
    return message + (" [" + "; ".join(details) + "]" if details else "")


def paired_table(report):
    """Display the case evidence before revealing the policy decision."""
    policy = report["policy_result"]
    if not policy.get("comparison_interpreted"):
        reasons = "".join("<li>" + html.escape(_reason_text(reason)) + "</li>" for reason in policy.get("reasons", [])
                          if reason.get("category") in {"invalid", "execution"})
        return "<div style='font:18px system-ui'><p><strong>Paired comparison withheld.</strong> Resolve the evidence issues before interpreting gains or losses.</p><ul>" + reasons + "</ul></div>"
    headings = ("Case", "Starting answer", "Proposed repair", "Change", "Protected loss")
    header = "<tr>" + "".join("<th style='text-align:left;padding:8px'>" + text + "</th>" for text in headings) + "</tr>"
    body = ""
    for row in policy.get("paired_rows", []):
        values = [row["case_id"], row["baseline"], row["candidate"], row["change"], ", ".join(row.get("protected_regressions", [])) or "none"]
        body += "<tr>" + "".join("<td style='padding:8px;border-top:1px solid #ccd'>" + html.escape(str(value)) + "</td>" for value in values) + "</tr>"
    return "<table style='font:18px system-ui;border-collapse:collapse'><caption style='text-align:left;padding:8px'>1 = every required check passed; 0 = at least one required check failed.</caption>" + header + body + "</table>"


def render_report(report):
    """A compact offline report: evidence and policy remain visibly separate."""
    policy = report["policy_result"]
    escape = lambda value: html.escape(str(value))
    reasons = "".join("<li>" + escape(_reason_text(reason)) + "</li>" for reason in policy.get("reasons", []))
    rows = policy.get("paired_rows", [])
    columns = list(dict.fromkeys(key for row in rows for key in row))
    table = "<tr>" + "".join("<th>" + escape(key) + "</th>" for key in columns) + "</tr>"
    table += "".join("<tr>" + "".join("<td>" + escape(row.get(key, "")) + "</td>" for key in columns) + "</tr>" for row in rows)
    metrics = policy.get("metrics", {})
    statistics = ""
    if metrics.get("paired_bootstrap_ci95") is not None:
        low, high = metrics["paired_bootstrap_ci95"]
        statistics = f'<p>Mean paired change: {metrics["mean_delta"]:+.2f} · Paired bootstrap 95% interval: [{low:+.2f}, {high:+.2f}] · Exact McNemar p: {metrics["exact_mcnemar_pvalue"]:.4f}</p><p>Descriptive uncertainty across these teaching cases; not an acceptance condition or evidence of safety. Bootstrap: {metrics["bootstrap_iterations"]:,} resamples, seed {metrics["bootstrap_seed"]}.</p>'
    style = ".repair-report{max-width:1100px;margin:1em auto;padding:0 1em;font:18px system-ui;color:#152638}.repair-report h1{font-size:40px}.repair-report td,.repair-report th{padding:.6em;border-bottom:1px solid #ccc;text-align:left}.repair-report table{border-collapse:collapse;width:100%}.repair-report code{overflow-wrap:anywhere}"
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>Workshop repair decision</title><style>' + style + '</style><main class="repair-report">' + f'<h1>{escape(policy["decision"])}</h1><p>Workshop repair policy · {escape(policy.get("policy_version"))}</p><p>Execution: {escape(report["execution_status"])} · Controls: {escape(report["control_status"])} · Review: {escape(report["review_status"])}</p><p>Recorded {escape(report["evidence_recorded_at"])} · {escape(report["provider"])}</p><p><code>{escape(report["evidence_id"])}</code></p><ul>{reasons}</ul><table>{table}</table>{statistics}<p>{escape(report["limitation"])}</p></main></html>'


def write_report(report, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_bytes(_json_bytes(report))
    (directory / "report.html").write_text(render_report(report), encoding="utf-8")
    return directory / "report.html"


def show_case(session, variant="deployed", case_id="day_45_opening"):
    row = next(row for row in session.recorded(variant)["rows"] if row["case_id"] == case_id)
    print(VARIANT_LABELS[variant], "|", row["case_id"])
    print("Customer:", row["inputs"]["question"])
    print("Answer:\n" + row["output"])
    print("Expected:", row["expectations"].get("expected_decision"))
    print("Scores:", json.dumps(row["scores"], sort_keys=True))
    print("Trace:", row["trace_id"])
    return row


def show_decision(report):
    policy = report["policy_result"]
    print("Execution:", report["execution_status"], "| Controls:", report["control_status"], "| Review:", report["review_status"])
    print("Workshop policy:", policy["decision"])
    for reason in policy.get("reasons", []):
        print("•", reason.get("message", reason))
    print(report["limitation"])


def evaluation_table(evaluation):
    """Small HTML table with literal recorded values, safe for offline notebook views."""
    headings = ("Case ID", "Expected eligibility", "Label check", "All deterministic checks", "Semantic judge")
    columns = "".join("<th style='text-align:left;padding:8px'>" + heading + "</th>" for heading in headings)
    body = ""
    for row in evaluation["rows"]:
        values = [row["case_id"], row["expectations"].get("expected_decision", ""),
                  *[row["scores"].get(check, "missing") for check in ("policy_decision", "deterministic_stack", "policy_judge")]]
        body += "<tr>" + "".join("<td style='padding:8px;border-top:1px solid #ccd'>" + html.escape(str(value)) + "</td>" for value in values) + "</tr>"
    return "<table style='font-size:18px;border-collapse:collapse'><tr>" + columns + "</tr>" + body + "</table>"


def main(argv=None):
    """Full-policy CLI. Evaluation and enforcement deliberately have distinct exits."""
    import argparse
    parser = argparse.ArgumentParser(description="Full repair policy: evaluate evidence, then optionally enforce the decision.")
    parser.add_argument("action", choices=("evaluate", "gate", "review-template", "rescore", "rescore-legacy", "promote-case"))
    parser.add_argument("--mode", choices=("saved", "live"), default="saved")
    parser.add_argument("--provider", choices=("openai", "databricks"))
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to the notebooks' live evidence directory in live mode, or artifacts/workshop-report in saved mode")
    parser.add_argument("--review-json", type=Path)
    parser.add_argument("--source-sha256", help="Expected SHA-256 of an original summary used by rescore-legacy")
    args = parser.parse_args(argv)
    session = None
    output = args.output or Path("artifacts/workshop-report")
    try:
        _notebook_defaults()
        args.output = output = args.output or (default_output_dir(args.provider) if args.mode == "live" else output)
        review = json.loads(args.review_json.read_text(encoding="utf-8")) if args.review_json else None
        if args.action == "review-template":
            form = review_skeleton(args.evidence or default_evidence_dir())
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "review.json").write_bytes(_json_bytes(form))
            print("Pending bound review template:", args.output / "review.json")
            return 0
        if args.action == "promote-case":
            if review is None:
                raise EvidenceError("Supply --review-json with the explicit approved case review.")
            promoted = promote_reviewed_case(review)
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "promoted-case.json").write_bytes(_json_bytes(promoted))
            print("Approved case proposal saved:", args.output / "promoted-case.json")
            return 0
        if args.action == "rescore-legacy":
            if args.mode != "live" or args.evidence is None:
                raise EvidenceError("Historical rescoring requires --mode live and --evidence pointing to the original summary JSON.")
            report = rescore_legacy(args.evidence, args.output, provider=args.provider, expected_sha256=args.source_sha256)
            print(json.dumps({"execution_status": report["execution_status"], "control_status": report["control_status"], "new_policy_decision": None, "result": str(args.output / "legacy-rescore-result.json")}, indent=2))
            return 0 if report["execution_status"] == "complete" else 1
        if args.action == "rescore":
            if args.mode != "live" or args.evidence is None:
                raise EvidenceError("Rescoring requires --mode live and --evidence identifying frozen answers.")
            report = rescore_frozen(args.evidence, args.output, provider=args.provider)
        else:
            session = WorkshopSession(args.mode, evidence_dir=args.evidence, output_dir=args.output if args.mode == "live" else None, provider=args.provider)
            report = run_evaluation(session)
            if review is not None:
                report = session.finish(review)
        path = write_report(report, args.output / "report")
        print(json.dumps({"execution_status": report["execution_status"], "decision": report["policy_result"]["decision"], "report": str(path)}, indent=2))
        # The report exists before the caller receives the enforcement result.
        if report["execution_status"] != "complete" or report["policy_result"]["decision"] == "NOT_EVALUATED":
            return 1
        if args.action != "gate":
            return 0
        return {"PASS": 0, "BLOCK": 2, "REVIEW_REQUIRED": 3, "INVALID": 4}.get(report["policy_result"]["decision"], 4)
    except Exception as error:
        # Do not print raw provider exceptions or private connection details.
        category, message = _safe_command_error(error)
        try:
            output.mkdir(parents=True, exist_ok=True)
            (output / "execution-error.json").write_bytes(_json_bytes({"execution_status": "error", "category": category,
                                                                       "error": message, "error_type": type(error).__name__, "at": _now()}))
            if session is not None and any(key.startswith("failure-") for key in session.bundle["records"]):
                write_report(session.finish(), output / "report")
        except (OSError, EvidenceError):
            message += " The output directory could not preserve the failure report."
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
