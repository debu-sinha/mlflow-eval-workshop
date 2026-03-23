#!/usr/bin/env python3
"""Evaluation gate for CI/CD pipelines.

Compares two MLflow evaluation runs by aligning samples on a stable key
and exits with code 1 if the candidate regresses beyond the configured
threshold.

Sample alignment prefers MLflow-native identifiers (client_request_id or
dataset_record_id from trace metadata) when available, and falls back to
hashing the request body when they are not set.

Wire this into GitHub Actions, GitLab CI, or any CI system that checks
exit codes.

Usage:
    python eval_gate.py --baseline-run-id <RUN_ID> --candidate-run-id <RUN_ID>
    python eval_gate.py --baseline-run-id <RUN_ID> --candidate-run-id <RUN_ID> --threshold 0.05

Environment:
    MLFLOW_TRACKING_URI: MLflow tracking server URL (default: http://localhost:5000)

Exit codes:
    0: Candidate passes (no significant regression)
    1: Candidate fails (regression exceeds threshold)
"""

import argparse
import hashlib
import sys

import mlflow
import numpy as np

# Minimum number of aligned samples required before the gate will pass.
# Below this count, the gate fails closed to prevent misconfigured pipelines
# from silently promoting bad models.
_MIN_OVERLAP = 2


def _stable_key(trace) -> str | None:
    """Derive a stable sample key from a trace.

    Prefers MLflow-native identifiers when available:
      1. client_request_id (set by caller)
      2. dataset_record_id (set by evaluation datasets)
    Falls back to hashing the request body when neither is present.
    Returns None if no key can be derived.
    """
    metadata = getattr(trace.info, "metadata", None) or {}

    for id_field in ("client_request_id", "dataset_record_id"):
        native_id = metadata.get(id_field)
        if native_id:
            return native_id

    request = trace.data.request if trace.data else None
    if request:
        return hashlib.sha256(request.encode("utf-8")).hexdigest()[:16]
    return None


def _parse_score(value) -> float | None:
    """Convert a scorer feedback value to a numeric score.

    Handles the common value formats returned by MLflow scorers:
      - Binary strings: "yes"/"no", "pass"/"fail"
      - Phoenix-style labels: "factual"/"hallucinated"
      - Booleans: True/False
      - Numeric: int or float pass through directly
    Returns None if the value cannot be parsed.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, bool):
        return 1.0 if value else 0.0

    if isinstance(value, str):
        lower = value.lower().strip()
        _positive = {"yes", "pass", "true", "factual", "correct", "grounded", "safe"}
        _negative = {
            "no",
            "fail",
            "false",
            "hallucinated",
            "incorrect",
            "ungrounded",
            "unsafe",
        }
        if lower in _positive:
            return 1.0
        if lower in _negative:
            return 0.0
        try:
            return float(lower)
        except ValueError:
            return None

    return None


def get_per_sample_scores(
    run_id: str, scorer_name: str = "correctness"
) -> dict[str, float]:
    """Extract per-sample scores from an MLflow evaluation run.

    Returns a dict mapping a stable sample key to a numeric score.
    Samples without a parseable key or an unparseable value are skipped
    with a warning.
    """
    client = mlflow.tracking.MlflowClient()

    experiment_id = client.get_run(run_id).info.experiment_id
    traces = client.search_traces(
        experiment_ids=[experiment_id],
        filter_string=f"metadata.`mlflow.sourceRun` = '{run_id}'",
    )

    scores: dict[str, float] = {}
    skipped = 0
    for trace in traces:
        key = _stable_key(trace)
        if key is None:
            skipped += 1
            continue
        for assessment in trace.info.assessments:
            if assessment.name == scorer_name and assessment.feedback:
                raw = assessment.feedback.get("value")
                parsed = _parse_score(raw)
                if parsed is not None:
                    scores[key] = parsed
                else:
                    skipped += 1
                    print(
                        f"  Warning: unparseable score value {raw!r} for trace {trace.info.trace_id}"
                    )
    if skipped:
        print(f"  Skipped {skipped} samples (no key or unparseable value)")
    return scores


def run_gate(
    baseline_scores: dict[str, float],
    candidate_scores: dict[str, float],
    max_regression_rate: float = 0.10,
    min_overlap: int = _MIN_OVERLAP,
) -> tuple[bool, str]:
    """Compare baseline and candidate scores aligned by sample key.

    Performs an inner join on the shared keys so that only samples
    present in both runs are compared. Fails closed when overlap is
    below min_overlap.

    Returns (passed, reason).
    """
    shared_keys = sorted(set(baseline_scores) & set(candidate_scores))
    bl_only = len(baseline_scores) - len(shared_keys)
    cd_only = len(candidate_scores) - len(shared_keys)
    n = len(shared_keys)

    print(f"Aligned samples:    {n}")
    if bl_only > 0 or cd_only > 0:
        print(f"  Baseline-only:    {bl_only} (skipped)")
        print(f"  Candidate-only:   {cd_only} (skipped)")

    if n < min_overlap:
        return (
            False,
            f"Only {n} overlapping samples (minimum {min_overlap} required). "
            f"Check that both runs evaluated the same dataset.",
        )

    bl_values = np.array([baseline_scores[k] for k in shared_keys])
    cd_values = np.array([candidate_scores[k] for k in shared_keys])

    bl_acc = float(np.mean(bl_values))
    cd_acc = float(np.mean(cd_values))
    delta = cd_acc - bl_acc

    regressions = int(np.sum((bl_values > cd_values)))
    improvements = int(np.sum((cd_values > bl_values)))
    regression_rate = regressions / n

    print(f"Baseline mean:      {bl_acc:.1%}")
    print(f"Candidate mean:     {cd_acc:.1%}")
    print(f"Delta:              {delta:+.1%}")
    print(f"Regressions:        {regressions}/{n} ({regression_rate:.1%})")
    print(f"Improvements:       {improvements}/{n}")
    print(f"Threshold:          {max_regression_rate:.1%}")

    if regression_rate > max_regression_rate:
        return (
            False,
            f"Regression rate {regression_rate:.1%} exceeds threshold {max_regression_rate:.1%}",
        )
    if delta < -max_regression_rate:
        return False, f"Score drop {delta:+.1%} exceeds threshold"
    return True, "No significant regression detected"


def main():
    parser = argparse.ArgumentParser(description="MLflow evaluation gate for CI/CD")
    parser.add_argument(
        "--baseline-run-id", required=True, help="MLflow run ID for baseline"
    )
    parser.add_argument(
        "--candidate-run-id", required=True, help="MLflow run ID for candidate"
    )
    parser.add_argument(
        "--scorer",
        default="correctness",
        help="Scorer name to compare (default: correctness)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Max regression rate (default: 0.10)",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=_MIN_OVERLAP,
        help=f"Minimum overlapping samples required to pass (default: {_MIN_OVERLAP})",
    )
    args = parser.parse_args()

    print("Evaluation Gate")
    print(f"Baseline:  {args.baseline_run_id}")
    print(f"Candidate: {args.candidate_run_id}")
    print(f"Scorer:    {args.scorer}")
    print()

    baseline_scores = get_per_sample_scores(args.baseline_run_id, args.scorer)
    candidate_scores = get_per_sample_scores(args.candidate_run_id, args.scorer)

    print(f"Baseline samples:  {len(baseline_scores)}")
    print(f"Candidate samples: {len(candidate_scores)}")
    print()

    passed, reason = run_gate(
        baseline_scores, candidate_scores, args.threshold, args.min_overlap
    )

    print()
    if passed:
        print(f"PASSED: {reason}")
        sys.exit(0)
    else:
        print(f"BLOCKED: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
