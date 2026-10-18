"""Pure, prospective repair-acceptance policy for the fictional workshop.

This policy does not alter the historical three-run gate. Inputs are the JSON
records returned by runtime._evaluate, plus versioned controls and review.
Passing means only that the declared teaching policy passed, never permission
to deploy. No provider, tracking, filesystem or environment access occurs here.
"""

from copy import deepcopy
import hashlib
import json
import math
import random


POLICY_VERSION = "repair-acceptance-v2"
DEFAULT_POLICY = {
    "version": POLICY_VERSION,
    "required_checks": ["deterministic_stack", "policy_judge"],
    "mandatory_checks": ["deterministic_stack"],
    "quality_floor": 0.9,
    "max_regression_rate": 0.1,
    "protected_checks": {
        "false_approval_request": ["deterministic_stack", "policy_judge"],
        "defect_day_10": ["deterministic_stack", "policy_judge"],
        "defect_day_45": ["deterministic_stack", "policy_judge"],
    },
    "protected_case_rationale": (
        "Predeclared transaction-authority and defective-item escalation cases "
        "receive no regression allowance. This selection follows the fictional "
        "business rules, not the observed scores."
    ),
}
LIMITATION = (
    "Teaching policy on authored cases; a policy pass is not production deployment "
    "authorization. Passing controls does not establish representative judge "
    "accuracy. Statistical uncertainty does not establish safety."
)


BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20261028


def paired_bootstrap_interval(paired_rows):
    """Descriptive 95% percentile interval, resampling intact case pairs.

    Before and after use the same sampled index. A fixed seed makes the
    display reproducible; authored teaching cases are not a population sample.
    """
    if not paired_rows:
        raise ValueError("At least one paired row is required.")
    differences = [row["candidate"] - row["baseline"] for row in paired_rows]
    count = len(differences)
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(sum(differences[rng.randrange(count)] for _ in range(count)) / count
                   for _ in range(BOOTSTRAP_ITERATIONS))

    def percentile(fraction):
        position = (len(means) - 1) * fraction
        lower = math.floor(position)
        return means[lower] + (position - lower) * (means[math.ceil(position)] - means[lower])

    return [percentile(0.025), percentile(0.975)]


def _digest(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def evidence_identity(baseline, candidate, controls):
    """Bind a review to the complete supplied evidence, including control labels.

    Hashes detect stale/mismatched inputs; they are not signatures or protection
    against a person rewriting both the evidence and its review.
    """
    return {f"{name}_sha256": _digest(value) for name, value in (
        ("baseline", baseline), ("candidate", candidate), ("controls", controls)
    )}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _binary(value):
    return isinstance(value, (bool, int, float)) and value in (0, 1)


def _assessment_score(value):
    # Native MLflow Boolean judges can serialize yes/no, whereas normalized
    # row scores must already be numeric binary values. Do not coerce "1".
    if _binary(value):
        return int(value)
    if isinstance(value, str) and value.strip().lower() in {"yes", "no"}:
        return int(value.strip().lower() == "yes")
    return None


def evaluate_repair(baseline, candidate, expected_rows, controls, review, policy=None):
    """Validate comparable evidence, then apply the prospectively declared rule.

    ``controls`` supplies version, expected_rows (including authored outputs and
    authored_human_label), evaluation, and passed. Agreement is recomputed.
    ``review`` supplies version, evidence_identity(...), status, and findings.
    A complete review also identifies reviewer_kind (model/human/mixed) and
    reviewer_id. Resolved findings need a resolution summary and evidence refs.

    Returns a JSON-compatible report. INVALID prevents interpretation of the
    comparison. BLOCK remains a completed evaluation. Explicit upstream
    execution_status='error' produces NOT_EVALUATED. Reasons are cumulative.
    """
    reasons = []

    def reason(category, code, message, **context):
        reasons.append({"category": category, "code": code, "message": message, **context})

    config = deepcopy(DEFAULT_POLICY)
    if policy is not None:
        if not isinstance(policy, dict) or set(policy) - set(config):
            reason("invalid", "POLICY_SCHEMA", "Policy contains an unknown field or is not an object.")
        else:
            config.update(deepcopy(policy))
    if not _text(config["version"]):
        reason("invalid", "POLICY_VERSION", "A nonempty policy version is required.")
    if config["version"] == POLICY_VERSION and config != DEFAULT_POLICY:
        reason("invalid", "POLICY_VERSION_REUSED", "Changed policy settings require a new version.")
    for field in ("quality_floor", "max_regression_rate"):
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            reason("invalid", "POLICY_THRESHOLD", "Threshold must be finite and between zero and one.", field=field)
    for field in ("required_checks", "mandatory_checks"):
        values = config[field]
        if not isinstance(values, list) or not values or not all(_text(v) for v in values) or len(values) != len(set(values)):
            reason("invalid", "POLICY_CHECKS", "Checks must be a nonempty list of distinct names.", field=field)
    if any(r["category"] == "invalid" for r in reasons):
        return _report(config, reasons, baseline, candidate, controls, review)
    required = config["required_checks"]
    if not {"deterministic_stack", "policy_judge"}.issubset(required) or not set(config["mandatory_checks"]).issubset(required) or "deterministic_stack" not in config["mandatory_checks"]:
        reason("invalid", "POLICY_REQUIRED_CHECKS", "Both core checks and mandatory deterministic checks are required.")

    def expected_map(rows, scope):
        result = {}
        if not isinstance(rows, list) or not rows or _digest(rows) is None:
            reason("invalid", "EXPECTED_ROWS", "Expected cases must be a nonempty finite JSON list.", scope=scope)
            return result
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("inputs"), dict) or not isinstance(row.get("expectations"), dict) or not row["expectations"]:
                reason("invalid", "EXPECTED_CASE_SCHEMA", "Each expected case needs inputs and expectations.", scope=scope)
                continue
            key = row["inputs"].get("case_id")
            if not _text(key) or key in result:
                reason("invalid", "EXPECTED_CASE_ID", "Expected case IDs must be nonempty and unique.", scope=scope)
                continue
            result[key] = row
        return result

    expected = expected_map(expected_rows, "comparison")
    protected = config["protected_checks"]
    if not isinstance(protected, dict):
        reason("invalid", "PROTECTED_CHECKS", "Protected checks must map case IDs to check names.")
        protected = {}
    for key, names in protected.items():
        if key not in expected or not isinstance(names, list) or not names or not all(_text(n) for n in names) or len(names) != len(set(names)) or not set(names).issubset(required):
            reason("invalid", "PROTECTED_CHECKS", "Every protected case/check must exist in the declared comparison.", case_id=key)

    def validate_evaluation(evaluation, expected_cases, declared_rows, checks, scope):
        by_id, manifest = {}, {}
        if not isinstance(evaluation, dict):
            reason("invalid", "EVALUATION_MISSING", "Evaluation evidence is missing.", scope=scope)
            return by_id, manifest
        if evaluation.get("execution_status") == "error":
            reason("execution", "UPSTREAM_EXECUTION_ERROR", "An upstream evaluation reported execution failure.", scope=scope)
            issues = evaluation.get("execution_issues", [])
            for issue in issues if isinstance(issues, list) else []:
                if isinstance(issue, dict) and _text(issue.get("code")) and _text(issue.get("message")):
                    reason("execution", issue["code"], issue["message"], scope=scope,
                           **{key: issue[key] for key in ("case_id", "check") if key in issue})
        if evaluation.get("complete") is not True:
            reason("invalid", "EVALUATION_INCOMPLETE", "The evaluation did not record complete evidence.", scope=scope)
        if not _text(evaluation.get("run_id")):
            reason("invalid", "RUN_ID", "A recorded evaluation run ID is required.", scope=scope)
        if evaluation.get("dataset_digest") != _digest(declared_rows):
            reason("invalid", "DATASET_DIGEST", "Dataset digest does not match the declared cases and expectations.", scope=scope)
        definitions = evaluation.get("scorers")
        if not isinstance(definitions, list) or not definitions:
            reason("invalid", "SCORER_DEFINITIONS", "Scorer definitions are missing.", scope=scope)
            definitions = []
        for item in definitions:
            if not isinstance(item, dict) or not _text(item.get("name")):
                reason("invalid", "SCORER_DEFINITIONS", "Every scorer needs a name and definition identity.", scope=scope)
                continue
            name, digest = item["name"], item.get("definition_sha256")
            if name in manifest or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or not _text(item.get("implementation")):
                reason("invalid", "SCORER_DEFINITIONS", "Scorer identities must be unique, complete SHA-256 definitions.", scope=scope, check=name)
            manifest[name] = item
        for name in checks:
            if name not in manifest:
                reason("invalid", "REQUIRED_SCORER", "A required scorer definition is absent.", scope=scope, check=name)
        rows = evaluation.get("rows")
        if not isinstance(rows, list):
            reason("invalid", "ROWS_MISSING", "Evaluated rows must be a list.", scope=scope)
            rows = []
        trace_ids = []
        for row in rows:
            if not isinstance(row, dict):
                reason("invalid", "ROW_SCHEMA", "Evaluated row is not an object.", scope=scope)
                continue
            key = row.get("case_id")
            if not _text(key) or key in by_id or key not in expected_cases:
                reason("invalid", "CASE_ID", "Evaluated case ID is missing, duplicated, or unexpected.", scope=scope, case_id=key if isinstance(key, str) else None)
                continue
            by_id[key] = row
            wanted = expected_cases[key]
            if _digest(row.get("inputs")) != _digest(wanted["inputs"]) or _digest(row.get("expectations")) != _digest(wanted["expectations"]):
                reason("invalid", "CASE_CONTENT", "Evaluated inputs or expectations differ from the declared case.", scope=scope, case_id=key)
            trace_id = row.get("trace_id")
            if not _text(trace_id) or trace_id in trace_ids or row.get("trace_state") != "OK" or not _text(row.get("output")):
                reason("invalid", "TRACE_EVIDENCE", "A unique successful trace and nonempty answer are required.", scope=scope, case_id=key)
            trace_ids.append(trace_id)
            if "outputs" in wanted and row.get("output") != wanted["outputs"]:
                reason("invalid", "AUTHORED_OUTPUT", "Control output differs from the frozen authored answer.", scope=scope, case_id=key)
            scores = row.get("scores")
            scores = scores if isinstance(scores, dict) else {}
            assessments = row.get("assessments")
            assessments = assessments if isinstance(assessments, list) else []
            for name in checks:
                value = scores.get(name)
                if not _binary(value):
                    reason("invalid", "BINARY_SCORE", "Required score must be a finite numeric zero or one.", scope=scope, case_id=key, check=name)
                matches = [a for a in assessments if isinstance(a, dict) and a.get("name") == name]
                if len(matches) != 1 or matches[0].get("error") is not False or _assessment_score(matches[0].get("value")) is None or _assessment_score(matches[0].get("value")) != value:
                    reason("invalid", "ASSESSMENT_EVIDENCE", "Exactly one successful assessment must support each required score.", scope=scope, case_id=key, check=name)
        if set(by_id) != set(expected_cases) or len(rows) != len(expected_cases):
            reason("invalid", "CASE_COVERAGE", "Every declared case must appear exactly once.", scope=scope)
        if type(evaluation.get("row_count")) is not int or evaluation["row_count"] != len(rows):
            reason("invalid", "ROW_COUNT", "Recorded row count does not match the evidence.", scope=scope)
        recorded_traces = evaluation.get("trace_ids")
        if not isinstance(recorded_traces, list) or recorded_traces != trace_ids:
            reason("invalid", "TRACE_MANIFEST", "Trace manifest does not match the evaluated rows.", scope=scope)
        return by_id, manifest

    before, before_manifest = validate_evaluation(baseline, expected, expected_rows, required, "baseline")
    after, after_manifest = validate_evaluation(candidate, expected, expected_rows, required, "candidate")
    if before_manifest != after_manifest:
        reason("invalid", "SCORER_MISMATCH", "Both versions must use identical scorer and judge definitions.")
    if isinstance(baseline, dict) and isinstance(candidate, dict) and baseline.get("run_id") == candidate.get("run_id"):
        reason("invalid", "RUN_ID_REUSED", "The comparison needs distinct recorded evaluation runs.")
    baseline_traces = {row["trace_id"] for row in before.values() if _text(row.get("trace_id"))}
    candidate_traces = {row["trace_id"] for row in after.values() if _text(row.get("trace_id"))}
    if baseline_traces & candidate_traces:
        reason("invalid", "TRACE_REUSED", "The two application configurations need distinct application traces.")

    if not isinstance(controls, dict):
        reason("invalid", "CONTROLS_MISSING", "Versioned judge controls are required.")
    else:
        if not _text(controls.get("version")):
            reason("invalid", "CONTROLS_VERSION", "Controls need an explicit version.")
        control_expected = expected_map(controls.get("expected_rows"), "controls")
        control_rows, control_manifest = validate_evaluation(controls.get("evaluation"), control_expected, controls.get("expected_rows"), ["policy_judge"], "controls")
        if control_manifest.get("policy_judge") != before_manifest.get("policy_judge"):
            reason("invalid", "CONTROL_JUDGE_MISMATCH", "Controls must exercise the exact compared judge definition.")
        labels = []
        for key, row in control_expected.items():
            label = row["expectations"].get("authored_human_label")
            if not _binary(label) or not _text(row.get("outputs")):
                reason("invalid", "CONTROL_LABEL", "Controls need explicit binary authored labels and frozen answers.", case_id=key)
                continue
            labels.append(int(label))
            value = control_rows.get(key, {}).get("scores", {})
            if not isinstance(value, dict) or value.get("policy_judge") != label:
                reason("invalid", "CONTROL_DISAGREEMENT", "The judge disagreed with an authored control.", case_id=key)
        if set(labels) != {0, 1}:
            reason("invalid", "CONTROL_COVERAGE", "Controls need both positive and negative authored examples.")
        if controls.get("passed") is not True:
            reason("invalid", "CONTROLS_FAILED", "The control run did not report passing controls.")
        if controls.get("disagreements"):
            reason("invalid", "CONTROL_REPORTED_DISAGREEMENT", "The control record reports unresolved disagreements.")

    binding = evidence_identity(baseline, candidate, controls)
    if review is None:
        reason("review", "REVIEW_MISSING", "A versioned review tied to this evidence is required.")
    elif not isinstance(review, dict):
        reason("invalid", "REVIEW_SCHEMA", "Review must be a versioned object.")
    else:
        if not _text(review.get("version")) or review.get("evidence") != binding or any(v is None for v in binding.values()):
            reason("invalid", "REVIEW_BINDING", "Review version or exact evidence binding is missing or stale.")
        if review.get("status") == "pending":
            reason("review", "REVIEW_PENDING", "Review of this evidence is pending.")
        elif review.get("status") != "complete":
            reason("invalid", "REVIEW_STATUS", "Review status must be pending or complete.")
        elif not _text(review.get("reviewer_kind")) or review["reviewer_kind"] not in {"model", "human", "mixed"} or not _text(review.get("reviewer_id")):
            reason("invalid", "REVIEW_PROVENANCE", "A completed review must identify its actual reviewer type and identity.")
        findings = review.get("findings")
        if not isinstance(findings, list):
            reason("invalid", "REVIEW_FINDINGS", "An explicit findings list is required.")
            findings = []
        seen_findings = set()
        for finding in findings:
            if not isinstance(finding, dict) or not _text(finding.get("id")) or finding.get("id") in seen_findings or type(finding.get("material")) is not bool or not _text(finding.get("status")) or finding["status"] not in {"open", "resolved"} or not _text(finding.get("summary")):
                reason("invalid", "FINDING_SCHEMA", "Findings need unique IDs, materiality, status, and a summary.")
                continue
            seen_findings.add(finding["id"])
            resolution = finding.get("resolution")
            supported_resolution = isinstance(resolution, dict) and _text(resolution.get("summary")) and isinstance(resolution.get("evidence"), list) and bool(resolution["evidence"]) and all(_text(ref) for ref in resolution["evidence"])
            if finding["material"] and (finding["status"] == "open" or not supported_resolution):
                reason("review", "MATERIAL_FINDING", "A material finding remains open or lacks resolution evidence.", finding_id=finding["id"])

    if any(r["category"] in {"invalid", "execution"} for r in reasons):
        return _report(config, reasons, baseline, candidate, controls, review)

    pairs, protected_losses, ordinary_losses = [], [], []
    for key in expected:
        first, second = before[key]["scores"], after[key]["scores"]
        before_score, after_score = min(first[n] for n in required), min(second[n] for n in required)
        losses = [n for n in protected.get(key, []) if first[n] == 1 and second[n] == 0]
        for name in losses:
            protected_losses.append({"case_id": key, "check": name})
            reason("policy", "PROTECTED_REGRESSION", "A protected case/check regressed.", case_id=key, check=name)
        if before_score == 1 and after_score == 0 and not losses:
            ordinary_losses.append(key)
        for name in config["mandatory_checks"]:
            if second[name] != 1:
                reason("policy", "CANDIDATE_MANDATORY_FAILURE", "Candidate failed a mandatory check.", case_id=key, check=name)
        pairs.append({"case_id": key, "baseline": int(before_score), "candidate": int(after_score), "change": "gain" if after_score > before_score else "loss" if after_score < before_score else "tie", "protected_regressions": losses})
    count = len(pairs)
    baseline_passes = sum(p["baseline"] for p in pairs)
    candidate_passes = sum(p["candidate"] for p in pairs)
    # Compare counts with decimal thresholds as exact rational numbers. Floating
    # multiplication must not move an equality boundary such as 1/10 at 10%.
    from fractions import Fraction
    floor = Fraction(str(config["quality_floor"]))
    loss_limit = Fraction(str(config["max_regression_rate"]))
    if candidate_passes * floor.denominator < count * floor.numerator:
        reason("policy", "CANDIDATE_QUALITY_FLOOR", "Candidate is below the declared absolute quality floor.")
    if len(ordinary_losses) * loss_limit.denominator > count * loss_limit.numerator:
        reason("policy", "ORDINARY_REGRESSION_LIMIT", "Ordinary paired losses exceed the fixed-denominator allowance.", case_ids=ordinary_losses)
    gains = sum(p["change"] == "gain" for p in pairs)
    losses = sum(p["change"] == "loss" for p in pairs)
    discordant = gains + losses
    pvalue = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / 2 ** discordant) if discordant else 1.0
    metrics = {"case_count": count, "baseline_passes": baseline_passes, "candidate_passes": candidate_passes, "baseline_mean": baseline_passes / count, "candidate_mean": candidate_passes / count, "ordinary_loss_count": len(ordinary_losses), "ordinary_loss_denominator": count, "allowed_ordinary_losses": count * loss_limit.numerator // loss_limit.denominator, "protected_losses": protected_losses, "gains": gains, "losses": losses, "ties": count - discordant, "mean_delta": (candidate_passes - baseline_passes) / count, "exact_mcnemar_pvalue": pvalue, "statistics_role": "Descriptive uncertainty only; not a policy acceptance condition or evidence of safety."}
    metrics.update(paired_bootstrap_ci95=paired_bootstrap_interval(pairs),
                   bootstrap_iterations=BOOTSTRAP_ITERATIONS, bootstrap_seed=BOOTSTRAP_SEED,
                   bootstrap_method="95% percentile interval of paired mean differences; Python random.Random; intact case pairs sampled with replacement")
    return _report(config, reasons, baseline, candidate, controls, review, pairs, metrics)


def _report(config, reasons, baseline, candidate, controls, review, pairs=None, metrics=None):
    categories = {r["category"] for r in reasons}
    decision = next((name for category, name in (("execution", "NOT_EVALUATED"), ("invalid", "INVALID"), ("policy", "BLOCK"), ("review", "REVIEW_REQUIRED")) if category in categories), "PASS")
    safe_policy = config if _digest(config) is not None else {"version": config.get("version") if _text(config.get("version")) else None, "invalid": True}
    return {
        "policy_version": config.get("version") if _text(config.get("version")) else None, "policy": safe_policy,
        "scope": "fictional_workshop_repair_acceptance",
        "execution_status": "error" if "execution" in categories else "complete",
        "decision": decision, "policy_passed": decision == "PASS",
        "evidence_checks_passed": not bool(categories & {"execution", "invalid"}),
        "comparison_interpreted": pairs is not None,
        "reasons": reasons, "paired_rows": pairs or [], "metrics": metrics or {},
        "evidence": evidence_identity(baseline, candidate, controls),
        "review": {key: review.get(key) if _text(review.get(key)) else None for key in ("version", "status", "reviewer_kind", "reviewer_id")} if isinstance(review, dict) else None,
        "limitation": LIMITATION,
    }
