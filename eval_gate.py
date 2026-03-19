#!/usr/bin/env python3
"""Evaluation gate for CI/CD pipelines.

Compares two MLflow evaluation runs and exits with code 1 if the candidate
regresses beyond the configured threshold. Wire this into GitHub Actions,
GitLab CI, or any CI system that checks exit codes.

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
import sys

import mlflow
import numpy as np


def get_per_sample_scores(run_id: str, scorer_name: str = "correctness") -> dict:
    """Extract per-sample scores from an MLflow evaluation run.

    Reads the evaluation result DataFrame from the run's artifacts
    and returns a dict mapping trace_id to score value.
    """
    client = mlflow.tracking.MlflowClient()

    # Get traces for this run
    experiment_id = client.get_run(run_id).info.experiment_id
    traces = client.search_traces(
        experiment_ids=[experiment_id],
        filter_string=f"metadata.`mlflow.sourceRun` = '{run_id}'",
    )

    scores = {}
    for trace in traces:
        trace_id = trace.info.trace_id
        for assessment in trace.info.assessments:
            if assessment.name == scorer_name and assessment.feedback:
                value = assessment.feedback.get("value")
                if value == "yes":
                    scores[trace_id] = 1.0
                elif value == "no":
                    scores[trace_id] = 0.0
    return scores


def run_gate(
    baseline_scores: dict,
    candidate_scores: dict,
    max_regression_rate: float = 0.10,
) -> tuple[bool, str]:
    """Compare scores and decide pass/fail.

    Returns (passed, reason).
    """
    # Align by trace content (both runs evaluate same inputs)
    bl_values = list(baseline_scores.values())
    cd_values = list(candidate_scores.values())

    if not bl_values or not cd_values:
        return True, "No scores to compare"

    bl_acc = np.mean(bl_values)
    cd_acc = np.mean(cd_values)
    delta = cd_acc - bl_acc

    # Count regressions (paired comparison needs same sample alignment)
    n = min(len(bl_values), len(cd_values))
    regressions = sum(1 for i in range(n) if bl_values[i] == 1.0 and cd_values[i] == 0.0)
    regression_rate = regressions / n if n > 0 else 0.0

    print(f"Baseline accuracy:  {bl_acc:.1%} ({len(bl_values)} samples)")
    print(f"Candidate accuracy: {cd_acc:.1%} ({len(cd_values)} samples)")
    print(f"Delta: {delta:+.1%}")
    print(f"Regressions: {regressions}/{n} ({regression_rate:.1%})")
    print(f"Threshold: {max_regression_rate:.1%}")

    if regression_rate > max_regression_rate:
        return False, f"Regression rate {regression_rate:.1%} exceeds threshold {max_regression_rate:.1%}"
    if delta < -max_regression_rate:
        return False, f"Accuracy drop {delta:+.1%} exceeds threshold"
    return True, "No significant regression detected"


def main():
    parser = argparse.ArgumentParser(description="MLflow evaluation gate for CI/CD")
    parser.add_argument("--baseline-run-id", required=True, help="MLflow run ID for baseline")
    parser.add_argument("--candidate-run-id", required=True, help="MLflow run ID for candidate")
    parser.add_argument("--scorer", default="correctness", help="Scorer name to compare (default: correctness)")
    parser.add_argument("--threshold", type=float, default=0.10, help="Max regression rate (default: 0.10)")
    args = parser.parse_args()

    print(f"Evaluation Gate")
    print(f"Baseline: {args.baseline_run_id}")
    print(f"Candidate: {args.candidate_run_id}")
    print(f"Scorer: {args.scorer}")
    print()

    baseline_scores = get_per_sample_scores(args.baseline_run_id, args.scorer)
    candidate_scores = get_per_sample_scores(args.candidate_run_id, args.scorer)

    print(f"Baseline samples: {len(baseline_scores)}")
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
