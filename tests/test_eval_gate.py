"""Unit tests for eval_gate.py.

Covers score parsing, stable key derivation, the gate decision on binary
and continuous scorers, the small-sample McNemar fallback, and the
fail-closed behavior when sample overlap is below the minimum.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from eval_gate import (
    _MCNEMAR_EXACT_THRESHOLD,
    _mcnemar_exact_pvalue,
    _parse_score,
    _stable_key,
    run_gate,
)


# ---------- _parse_score ----------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, 1.0),
        (False, 0.0),
        (1, 1.0),
        (0, 0.0),
        (0.75, 0.75),
        ("yes", 1.0),
        ("YES", 1.0),
        (" pass ", 1.0),
        ("factual", 1.0),
        ("grounded", 1.0),
        ("safe", 1.0),
        ("no", 0.0),
        ("fail", 0.0),
        ("hallucinated", 0.0),
        ("ungrounded", 0.0),
        ("unsafe", 0.0),
        ("0.42", 0.42),
        ("-1", -1.0),
        ("garbage", None),
        ([], None),
    ],
)
def test_parse_score(value, expected):
    assert _parse_score(value) == expected


# ---------- _stable_key ----------


def _fake_trace(metadata=None, request=None):
    info = SimpleNamespace(metadata=metadata)
    data = SimpleNamespace(request=request) if request is not None else None
    return SimpleNamespace(info=info, data=data)


def test_stable_key_prefers_client_request_id():
    trace = _fake_trace(
        metadata={"client_request_id": "req-123", "dataset_record_id": "ds-456"},
        request="fallback",
    )
    assert _stable_key(trace) == "req-123"


def test_stable_key_uses_dataset_record_id_when_no_client_id():
    trace = _fake_trace(
        metadata={"dataset_record_id": "ds-456"},
        request="fallback",
    )
    assert _stable_key(trace) == "ds-456"


def test_stable_key_hashes_string_request():
    trace = _fake_trace(metadata={}, request="What is MLflow?")
    expected = hashlib.sha256("What is MLflow?".encode("utf-8")).hexdigest()[:16]
    assert _stable_key(trace) == expected


def test_stable_key_hashes_dict_request_deterministically():
    payload = {"question": "What is MLflow?", "user_id": 42}
    trace_a = _fake_trace(metadata={}, request=payload)
    trace_b = _fake_trace(
        metadata={}, request={"user_id": 42, "question": "What is MLflow?"}
    )
    # Same content, different key order -> same hash.
    assert _stable_key(trace_a) == _stable_key(trace_b)
    # Hash matches sorted-key JSON.
    expected_payload = json.dumps(payload, sort_keys=True, default=str)
    assert (
        _stable_key(trace_a)
        == hashlib.sha256(expected_payload.encode()).hexdigest()[:16]
    )


def test_stable_key_returns_none_when_nothing_available():
    trace = _fake_trace(metadata=None, request=None)
    assert _stable_key(trace) is None


# ---------- _mcnemar_exact_pvalue ----------


@pytest.mark.parametrize(
    ("b", "c", "expected_bounds"),
    [
        (0, 0, (1.0, 1.0)),
        # All discordant pairs favor candidate -> should be far from symmetric.
        (0, 5, (0.0, 0.1)),
        # Balanced 50/50 split -> should not be significant.
        (5, 5, (0.5, 1.0)),
        # Heavy skew toward regressions.
        (8, 2, (0.0, 0.15)),
    ],
)
def test_mcnemar_exact_pvalue_bounds(b, c, expected_bounds):
    p = _mcnemar_exact_pvalue(b, c)
    low, high = expected_bounds
    assert low <= p <= high, f"p={p} not in [{low},{high}] for b={b} c={c}"


# ---------- run_gate: overlap and fail-closed ----------


def test_run_gate_fails_closed_on_no_overlap():
    passed, reason = run_gate({"a": 1.0}, {"b": 1.0})
    assert passed is False
    assert "overlapping samples" in reason


def test_run_gate_fails_closed_below_min_overlap():
    baseline = {f"k{i}": 1.0 for i in range(3)}
    candidate = {"k0": 1.0, "k_other": 1.0}
    passed, reason = run_gate(baseline, candidate, min_overlap=5)
    assert passed is False
    assert "minimum 5" in reason


# ---------- run_gate: binary scorers ----------


def test_run_gate_binary_passes_on_improvement():
    baseline = {f"k{i}": 1.0 if i < 8 else 0.0 for i in range(10)}
    candidate = {f"k{i}": 1.0 for i in range(10)}  # everyone correct now
    passed, reason = run_gate(baseline, candidate)
    assert passed is True
    assert "No significant regression" in reason


def test_run_gate_binary_blocks_on_regression_rate():
    # 15 out of 100 regress, well above default 10% threshold.
    baseline = {f"k{i}": 1.0 for i in range(100)}
    candidate = dict(baseline)
    for i in range(15):
        candidate[f"k{i}"] = 0.0
    passed, reason = run_gate(baseline, candidate)
    assert passed is False
    assert "Regression rate" in reason
    assert "15.0%" in reason


def test_run_gate_binary_uses_exact_test_on_small_samples():
    # 6 regressions and 0 improvements out of 10 samples. Below the
    # chi-square threshold, so the exact binomial McNemar path runs.
    # Exact two-sided p = 2 * (1/2)^6 = 0.03125 < 0.05.
    baseline = {f"k{i}": 1.0 for i in range(10)}
    candidate = dict(baseline)
    for i in range(6):
        candidate[f"k{i}"] = 0.0
    # Regression rate is 60%, which will trip the rate check first. Allow
    # a high threshold so the test exercises the significance branch.
    passed, reason = run_gate(baseline, candidate, max_regression_rate=0.9)
    assert passed is False
    assert "Significant regression" in reason
    assert "McNemar-exact" in reason


def test_run_gate_binary_exact_boundary_five_zero():
    # 5 regressions, 0 improvements: exact two-sided p = 0.0625, just above
    # the default significance level. The gate should NOT fire the significance
    # check. Set a permissive regression rate so the rate check also passes.
    baseline = {f"k{i}": 1.0 for i in range(10)}
    candidate = dict(baseline)
    for i in range(5):
        candidate[f"k{i}"] = 0.0
    passed, _reason = run_gate(baseline, candidate, max_regression_rate=0.9)
    assert passed is True


def test_run_gate_binary_chi_square_path_for_large_samples():
    # Force >= _MCNEMAR_EXACT_THRESHOLD discordant pairs.
    n = 200
    baseline = {f"k{i}": 1.0 for i in range(n)}
    candidate = dict(baseline)
    # Tiny regression rate (2%), a few improvements, no threshold tripping.
    for i in range(2):
        candidate[f"k{i}"] = 0.0
    for i in range(100, 100 + _MCNEMAR_EXACT_THRESHOLD + 5):
        # First mark these as 0 in baseline then 1 in candidate so they
        # show as improvements.
        baseline[f"k{i}"] = 0.0
        candidate[f"k{i}"] = 1.0
    passed, reason = run_gate(baseline, candidate)
    # Candidate is better, so gate should pass (no rate breach + positive delta).
    assert passed is True, reason


# ---------- run_gate: continuous scorers ----------


def test_run_gate_continuous_passes_on_noise():
    rng = np.random.default_rng(0)
    baseline = {f"k{i}": float(v) for i, v in enumerate(rng.uniform(0.4, 0.8, 50))}
    # Candidate is baseline plus small zero-mean noise well below the
    # gate's 5% of range "meaningful change" threshold.
    candidate = {
        k: max(0.0, min(1.0, v + rng.normal(0, 0.002))) for k, v in baseline.items()
    }
    passed, reason = run_gate(baseline, candidate)
    assert passed is True, reason


def test_run_gate_continuous_blocks_on_large_shift():
    rng = np.random.default_rng(1)
    baseline = {f"k{i}": float(v) for i, v in enumerate(rng.uniform(0.6, 0.9, 50))}
    # Candidate loses 0.3 on every sample: regression rate = 100% > threshold.
    candidate = {k: max(0.0, v - 0.3) for k, v in baseline.items()}
    passed, reason = run_gate(baseline, candidate)
    assert passed is False
    assert "Regression rate" in reason or "Significant regression" in reason
