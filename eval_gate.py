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
    2. Paired significance test detects a candidate loss -> FAIL
       (McNemar with exact binomial fallback for small samples on binary
        scorers, paired sign-flip permutation test on continuous scorers)
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
import json
import re
import sys
from math import comb, erfc, sqrt

import mlflow
import numpy as np

# Minimum number of aligned samples required before the gate will pass.
# Below this count, the gate fails closed to prevent misconfigured pipelines
# from silently promoting bad models.
#
# Workshop demos use 2 so the small sample sets in Modules 3 and 4 can
# exercise the full path. Production gates should set --min-overlap to
# 30 or higher; see README.md "Production usage" for guidance.
_MIN_OVERLAP = 2

# MLflow run IDs are 32-character lowercase hex strings. Validate before
# interpolating into a search filter to prevent injection.
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# Threshold below which the asymptotic chi-square approximation for
# McNemar's test is unreliable. Below this we use the exact binomial
# test instead (two-sided mid-p on discordant pairs).
_MCNEMAR_EXACT_THRESHOLD = 25

# Page size for search_traces pagination. MLflow defaults to 100 per page
# and returns a PagedList whose .token continues the next page.
_TRACE_PAGE_SIZE = 500

_POSITIVE_LABELS = frozenset(
    {"yes", "pass", "true", "factual", "correct", "grounded", "safe"}
)
_NEGATIVE_LABELS = frozenset(
    {"no", "fail", "false", "hallucinated", "incorrect", "ungrounded", "unsafe"}
)


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
    if request is None:
        return None

    if isinstance(request, str):
        payload = request
    else:
        # Structured inputs (dicts, lists) come back from some trace stores.
        # Serialize with sorted keys so equal inputs hash to the same key.
        payload = json.dumps(request, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _parse_score(value) -> float | None:
    """Convert a scorer feedback value to a numeric score.

    Handles the common value formats returned by MLflow scorers:
      - Booleans: True/False
      - Numeric: int or float pass through directly
      - Binary strings: "yes"/"no", "pass"/"fail"
      - Phoenix-style labels: "factual"/"hallucinated"
    Returns None if the value cannot be parsed.
    """
    if value is None:
        return None

    # Check bool before int/float because bool is a subclass of int.
    if isinstance(value, bool):
        return 1.0 if value else 0.0

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        lower = value.lower().strip()
        if lower in _POSITIVE_LABELS:
            return 1.0
        if lower in _NEGATIVE_LABELS:
            return 0.0
        try:
            return float(lower)
        except ValueError:
            return None

    return None


def _search_all_traces(client, experiment_id: str, run_id: str) -> list:
    """Paginate through every trace for a given evaluation run.

    MlflowClient.search_traces defaults to 100 results per page and returns
    a PagedList with a .token continuation cursor. Without pagination, a
    large evaluation run is silently truncated.
    """
    all_traces = []
    page_token = None
    while True:
        page = client.search_traces(
            locations=[experiment_id],
            filter_string=f"metadata.`mlflow.sourceRun` = '{run_id}'",
            max_results=_TRACE_PAGE_SIZE,
            page_token=page_token,
        )
        all_traces.extend(page)
        page_token = getattr(page, "token", None)
        if not page_token:
            break
    return all_traces


def get_per_sample_scores(
    run_id: str, scorer_name: str = "correctness"
) -> dict[str, float]:
    """Extract per-sample scores from an MLflow evaluation run.

    Returns a dict mapping a stable sample key to a numeric score.
    Samples without a parseable key or an unparseable value are skipped
    with a warning.
    """
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(
            f"Invalid run ID {run_id!r}. Expected 32-character lowercase hex."
        )

    client = mlflow.tracking.MlflowClient()

    experiment_id = client.get_run(run_id).info.experiment_id
    traces = _search_all_traces(client, experiment_id, run_id)

    scores: dict[str, float] = {}
    skipped = 0
    for trace in traces:
        key = _stable_key(trace)
        if key is None:
            skipped += 1
            continue
        for assessment in trace.info.assessments:
            if assessment.name != scorer_name or not assessment.feedback:
                continue
            fb = assessment.feedback
            if hasattr(fb, "value"):
                raw = fb.value
            elif hasattr(fb, "get"):
                raw = fb.get("value")
            else:
                raw = None
            parsed = _parse_score(raw)
            if parsed is not None:
                scores[key] = parsed
            else:
                skipped += 1
                print(
                    f"  Warning: unparseable score value {raw!r} for trace {trace.info.trace_id}"
                )
            break
    if skipped:
        print(f"  Skipped {skipped} samples (no key or unparseable value)")
    return scores


def _mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Two-sided exact binomial p-value for McNemar's test.

    Under H0, each discordant pair is equally likely to favor baseline or
    candidate (prob 0.5). The test statistic is min(b, c). The two-sided
    p-value is the probability of observing a split at least as extreme as
    the one seen, doubled.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


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
      2. Paired significance test detects candidate loss -> FAIL
      3. Otherwise -> PASS

    Binary scorers (all values in {0.0, 1.0}) use McNemar's test, falling
    back to the exact binomial test when the discordant pair count is too
    small for the chi-square approximation. Continuous scorers use a
    sign-flip permutation test on the paired differences.

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

    # Detect whether scores are binary (0/1) or continuous. Inferred from
    # the observed sample values; a graded scorer that happens to produce
    # only 0.0/1.0 in this run will be treated as binary.
    unique_values = set(np.unique(bl_values)) | set(np.unique(cd_values))
    is_binary = unique_values <= {0.0, 1.0}

    if is_binary:
        regressions = int(np.sum(bl_values > cd_values))
        improvements = int(np.sum(cd_values > bl_values))
    else:
        # For continuous scores, only count meaningful changes. A decrease
        # must exceed 5% of the observed score range (combined across both
        # runs) to qualify as a regression, filtering out noise.
        combined_range = float(
            max(np.max(bl_values), np.max(cd_values))
            - min(np.min(bl_values), np.min(cd_values))
        )
        score_range = max(combined_range, 0.01)
        min_delta = 0.05 * score_range
        regressions = int(np.sum((bl_values - cd_values) > min_delta))
        improvements = int(np.sum((cd_values - bl_values) > min_delta))
    regression_rate = regressions / n

    diffs = cd_values - bl_values

    if is_binary:
        discordant = regressions + improvements
        if discordant == 0:
            p_value = 1.0
            test_name = "McNemar"
        elif discordant < _MCNEMAR_EXACT_THRESHOLD:
            p_value = _mcnemar_exact_pvalue(regressions, improvements)
            test_name = "McNemar-exact"
        else:
            chi2 = (abs(regressions - improvements) - 1) ** 2 / discordant
            p_value = erfc(sqrt(chi2 / 2))
            test_name = "McNemar"
    else:
        # Paired sign-flip permutation test. Under H0 (no systematic
        # direction), the sign of each paired difference is equally likely
        # to be positive or negative, so flipping signs generates the null
        # distribution of the mean difference.
        observed_delta = float(np.mean(diffs))
        rng = np.random.default_rng(42)
        n_perm = 10_000
        signs = rng.choice([-1, 1], size=(n_perm, n))
        perm_deltas = np.mean(signs * diffs, axis=1)
        p_value = float(np.mean(np.abs(perm_deltas) >= abs(observed_delta)))
        test_name = "Permutation"

    # Cohen's d_z for paired differences.
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
        help=(
            f"Minimum overlapping samples required to pass (default: {_MIN_OVERLAP}). "
            "Production gates should use 30 or higher."
        ),
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
