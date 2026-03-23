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

Gate policy (same as notebook Module 4):
    1. Regression rate > threshold -> FAIL
    2. McNemar p-value < significance AND candidate accuracy lower -> FAIL
    3. Otherwise -> PASS

Usage:
    python eval_gate.py --baseline-run-id <RUN_ID> --candidate-run-id <RUN_ID>
    python eval_gate.py --baseline-run-id <RUN_ID> --candidate-run-id <RUN_ID> --threshold 0.05

Environment:
    MLFLOW_TRACKING_URI: MLflow tracking server URL (default: http://localhost:5000)

Exit codes:
    0: Candidate passes (no significant regression)
    1: Candidate fails (regression exceeds threshold or significant p-value)
"""

import argparse
import hashlib
import sys
from math import erfc, sqrt

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
        locations=[experiment_id],
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
                fb = assessment.feedback
                raw = (
                    fb.value
                    if hasattr(fb, "value")
                    else fb.get("value")
                    if hasattr(fb, "get")
                    else None
                )
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
    significance_threshold: float = 0.05,
    min_overlap: int = _MIN_OVERLAP,
) -> tuple[bool, str]:
    """Compare baseline and candidate scores aligned by sample key.

    Uses the same three-check policy as the notebook gate in Module 4:
      1. Regression rate exceeds threshold -> FAIL
      2. McNemar p-value significant AND candidate worse -> FAIL
      3. Otherwise -> PASS

    Performs an inner join on shared keys. Fails closed when overlap is
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

    # Detect whether scores are binary (0/1) or continuous
    unique_values = set(np.unique(bl_values)) | set(np.unique(cd_values))
    is_binary = unique_values <= {0.0, 1.0}

    if is_binary:
        regressions = int(np.sum((bl_values > cd_values)))
        improvements = int(np.sum((cd_values > bl_values)))
    else:
        # For continuous scores, only count meaningful changes.
        # A decrease must exceed 5% of the observed score range to qualify
        # as a regression, filtering out noise-level fluctuations.
        score_range = max(float(np.max(bl_values) - np.min(bl_values)), 0.01)
        min_delta = 0.05 * score_range
        regressions = int(np.sum((bl_values - cd_values) > min_delta))
        improvements = int(np.sum((cd_values - bl_values) > min_delta))
    regression_rate = regressions / n

    if is_binary:
        # McNemar's test for binary scorers (matches notebook Module 4)
        discordant = regressions + improvements
        if discordant > 0:
            chi2 = (abs(regressions - improvements) - 1) ** 2 / discordant
            p_value = erfc(sqrt(chi2 / 2))
        else:
            p_value = 1.0
        test_name = "McNemar"
    else:
        # Paired permutation test for continuous/graded scorers.
        # Under the null hypothesis, the sign of each paired difference
        # is equally likely to be positive or negative.
        diffs = cd_values - bl_values
        observed_delta = float(np.mean(diffs))
        rng = np.random.default_rng(42)
        n_perm = 10_000
        # Random sign flips simulate the null distribution
        signs = rng.choice([-1, 1], size=(n_perm, n))
        perm_deltas = np.mean(signs * diffs, axis=1)
        p_value = float(np.mean(np.abs(perm_deltas) >= abs(observed_delta)))
        test_name = "Permutation"

    # Cohen's d effect size
    diffs = cd_values - bl_values
    sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    effect_size = float(np.mean(diffs)) / sd if sd > 0 else 0.0

    print(f"Score type:         {'binary' if is_binary else 'continuous'}")
    print(f"Baseline mean:      {bl_acc:.1%}")
    print(f"Candidate mean:     {cd_acc:.1%}")
    print(f"Delta:              {delta:+.1%}")
    print(f"Regressions:        {regressions}/{n} ({regression_rate:.1%})")
    print(f"Improvements:       {improvements}/{n}")
    print(f"{test_name} p-value: {p_value:.4f}")
    print(f"Cohen's d:          {effect_size:.3f}")
    print(f"Threshold:          {max_regression_rate:.1%}")

    if regression_rate > max_regression_rate:
        return (
            False,
            f"Regression rate {regression_rate:.1%} exceeds threshold {max_regression_rate:.1%}",
        )
    if p_value < significance_threshold and cd_acc < bl_acc:
        return (
            False,
            f"Significant regression detected ({test_name} p={p_value:.4f}, d={effect_size:.3f})",
        )
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
        "--significance",
        type=float,
        default=0.05,
        help="Significance threshold for McNemar (binary) or permutation test (continuous). Default: 0.05",
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
        baseline_scores,
        candidate_scores,
        args.threshold,
        args.significance,
        args.min_overlap,
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
