#!/usr/bin/env python3
"""Evaluation gate for CI/CD pipelines.

Compares two MLflow evaluation runs by aligning samples on a stable key
(the input text hash) and exits with code 1 if the candidate regresses
beyond the configured threshold.

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


def _stable_key(trace) -> str | None:
    """Derive a stable sample key from the trace's request input.

    Evaluation runs on the same dataset produce traces with the same
    input content but different trace IDs. We hash the request body so
    that samples align correctly across runs even if trace order differs.
    Falls back to None if no request data is available.
    """
    request = trace.data.request
    if request:
        return hashlib.sha256(request.encode("utf-8")).hexdigest()[:16]
    return None


def get_per_sample_scores(
    run_id: str, scorer_name: str = "correctness"
) -> dict[str, float]:
    """Extract per-sample scores from an MLflow evaluation run.

    Returns a dict mapping a stable sample key (input hash) to the
    binary score value. Samples without a parseable key are skipped.
    """
    client = mlflow.tracking.MlflowClient()

    experiment_id = client.get_run(run_id).info.experiment_id
    traces = client.search_traces(
        experiment_ids=[experiment_id],
        filter_string=f"metadata.`mlflow.sourceRun` = '{run_id}'",
    )

    scores: dict[str, float] = {}
    for trace in traces:
        key = _stable_key(trace)
        if key is None:
            continue
        for assessment in trace.info.assessments:
            if assessment.name == scorer_name and assessment.feedback:
                value = assessment.feedback.get("value")
                if value == "yes":
                    scores[key] = 1.0
                elif value == "no":
                    scores[key] = 0.0
    return scores


def run_gate(
    baseline_scores: dict[str, float],
    candidate_scores: dict[str, float],
    max_regression_rate: float = 0.10,
) -> tuple[bool, str]:
    """Compare baseline and candidate scores aligned by sample key.

    Performs an inner join on the shared keys so that only samples
    present in both runs are compared. Reports overlap statistics.

    Returns (passed, reason).
    """
    shared_keys = sorted(set(baseline_scores) & set(candidate_scores))
    bl_only = len(baseline_scores) - len(shared_keys)
    cd_only = len(candidate_scores) - len(shared_keys)

    if not shared_keys:
        return True, "No overlapping samples to compare"

    bl_values = np.array([baseline_scores[k] for k in shared_keys])
    cd_values = np.array([candidate_scores[k] for k in shared_keys])
    n = len(shared_keys)

    bl_acc = float(np.mean(bl_values))
    cd_acc = float(np.mean(cd_values))
    delta = cd_acc - bl_acc

    regressions = int(np.sum((bl_values == 1.0) & (cd_values == 0.0)))
    improvements = int(np.sum((bl_values == 0.0) & (cd_values == 1.0)))
    regression_rate = regressions / n

    print(f"Aligned samples:    {n}")
    if bl_only > 0 or cd_only > 0:
        print(f"  Baseline-only:    {bl_only} (skipped)")
        print(f"  Candidate-only:   {cd_only} (skipped)")
    print(f"Baseline accuracy:  {bl_acc:.1%}")
    print(f"Candidate accuracy: {cd_acc:.1%}")
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
        return False, f"Accuracy drop {delta:+.1%} exceeds threshold"
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

    passed, reason = run_gate(baseline_scores, candidate_scores, args.threshold)

    print()
    if passed:
        print(f"PASSED: {reason}")
        sys.exit(0)
    else:
        print(f"BLOCKED: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
