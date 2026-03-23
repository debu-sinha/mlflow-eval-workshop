# Databricks notebook source
# MAGIC %md
# MAGIC # Module 3: Comparing Runs and Detecting Regressions
# MAGIC
# MAGIC **Objective:** Align samples across two evaluation runs, detect regressions,
# MAGIC and determine whether score differences are statistically significant.
# MAGIC
# MAGIC **Time:** 12 minutes

# COMMAND ----------

# MAGIC %pip install --upgrade mlflow[genai] numpy -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os

ON_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------

import mlflow

if ON_DATABRICKS:
    import json

    _ctx = json.loads(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()  # noqa: F821
    )
    mlflow.set_experiment(
        _ctx.get("extraContext", {}).get("notebook_path", "/tmp/odsc-workshop")
    )
else:
    mlflow.set_tracking_uri("sqlite:///mlflow_workshop.db")
    mlflow.set_experiment("odsc-eval-workshop")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 The problem (2 min)
# MAGIC
# MAGIC You upgraded your model. Accuracy went from 86% to 80%. Three questions:
# MAGIC
# MAGIC 1. **Which samples regressed?** You need to fix what broke.
# MAGIC 2. **Is the 6% drop real or noise?** Small datasets produce noisy metrics.
# MAGIC 3. **How big is the effect?** A statistically significant but tiny drop
# MAGIC    might not matter in practice.
# MAGIC
# MAGIC MLflow tracks individual evaluation runs. We need to compare across runs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Sample alignment and comparison (5 min)
# MAGIC
# MAGIC First, simulate two evaluation runs and align samples by ID.

# COMMAND ----------

import numpy as np

np.random.seed(42)
n_samples = 50

# Baseline: 86% accuracy
baseline_scores = np.random.choice([0.0, 1.0], size=n_samples, p=[0.14, 0.86])

# Candidate: introduce 5 regressions and 2 improvements
candidate_scores = baseline_scores.copy()

regression_idx = np.random.choice(
    np.where(baseline_scores == 1.0)[0], size=5, replace=False
)
candidate_scores[regression_idx] = 0.0

improvement_idx = np.random.choice(
    np.where(baseline_scores == 0.0)[0], size=2, replace=False
)
candidate_scores[improvement_idx] = 1.0

print(f"Baseline accuracy:  {baseline_scores.mean():.1%}")
print(f"Candidate accuracy: {candidate_scores.mean():.1%}")
print(f"Net change: {candidate_scores.mean() - baseline_scores.mean():+.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Align samples by ID
# MAGIC
# MAGIC In practice, evaluation datasets change between runs. Alignment handles
# MAGIC added, removed, and reordered samples.

# COMMAND ----------

from dataclasses import dataclass


@dataclass
class AlignedSample:
    sample_id: int
    baseline_score: float | None
    candidate_score: float | None


def align_samples(
    baseline: dict[int, float],
    candidate: dict[int, float],
) -> list[AlignedSample]:
    all_ids = sorted(set(baseline.keys()) | set(candidate.keys()))
    return [
        AlignedSample(
            sample_id=sid,
            baseline_score=baseline.get(sid),
            candidate_score=candidate.get(sid),
        )
        for sid in all_ids
    ]


baseline_dict = {i: float(s) for i, s in enumerate(baseline_scores)}
candidate_dict = {i: float(s) for i, s in enumerate(candidate_scores)}

aligned = align_samples(baseline_dict, candidate_dict)
print(f"Aligned {len(aligned)} samples")

# Show what changed
regressions = [
    s for s in aligned if s.baseline_score == 1.0 and s.candidate_score == 0.0
]
improvements = [
    s for s in aligned if s.baseline_score == 0.0 and s.candidate_score == 1.0
]
unchanged = [s for s in aligned if s.baseline_score == s.candidate_score]

print(f"Regressions:  {len(regressions)} samples")
print(f"Improvements: {len(improvements)} samples")
print(f"Unchanged:    {len(unchanged)} samples")

for s in regressions:
    print(f"  Sample {s.sample_id}: 1.0 -> 0.0 (REGRESSED)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Statistical analysis utilities
# MAGIC
# MAGIC The functions below implement comparison statistics without requiring scipy.
# MAGIC They are defined here for reference. The focus of this module is on
# MAGIC **using** them and **interpreting** the results.

# COMMAND ----------

from math import erfc, sqrt


def mcnemars_test(
    baseline_correct: list[bool],
    candidate_correct: list[bool],
    significance: float = 0.05,
) -> dict:
    """McNemar's test for paired binary outcomes.

    Tests whether the rate of discordant pairs (regressions vs improvements)
    differs significantly. Uses chi-square approximation with continuity
    correction. No scipy required.
    """
    b = sum(1 for bl, cd in zip(baseline_correct, candidate_correct) if bl and not cd)
    c = sum(1 for bl, cd in zip(baseline_correct, candidate_correct) if not bl and cd)

    discordant = b + c
    if discordant == 0:
        return {
            "p_value": 1.0,
            "significant": False,
            "regressions": b,
            "improvements": c,
        }

    chi2 = (abs(b - c) - 1) ** 2 / discordant
    p_value = erfc(sqrt(chi2 / 2))

    return {
        "p_value": p_value,
        "significant": p_value < significance,
        "regressions": b,
        "improvements": c,
        "discordant_pairs": discordant,
    }


def cohens_d(baseline: list[float], candidate: list[float]) -> float | None:
    """Cohen's d for paired samples. 0.2 = small, 0.5 = medium, 0.8 = large."""
    diffs = np.array(candidate) - np.array(baseline)
    if len(diffs) < 2:
        return None
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return 0.0
    return float(np.mean(diffs)) / sd


def bootstrap_ci(
    baseline: list[float],
    candidate: list[float],
    n_resamples: int = 10_000,
    significance: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for the difference in means."""
    rng = np.random.default_rng(seed)
    diffs = np.array(candidate) - np.array(baseline)
    n = len(diffs)
    observed = float(np.mean(diffs))

    boot_means = np.empty(n_resamples)
    batch_size = 1000
    for start in range(0, n_resamples, batch_size):
        end = min(start + batch_size, n_resamples)
        count = end - start
        indices = rng.integers(0, n, size=(count, n))
        boot_means[start:end] = np.mean(diffs[indices], axis=1)

    alpha = significance / 2
    ci_lower = float(np.percentile(boot_means, 100 * alpha))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha)))

    return {
        "observed_diff": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "excludes_zero": ci_lower > 0 or ci_upper < 0,
    }


def win_rate(baseline: list[float], candidate: list[float]) -> dict:
    """Win/loss/tie breakdown."""
    wins = sum(1 for b, c in zip(baseline, candidate) if c > b)
    losses = sum(1 for b, c in zip(baseline, candidate) if c < b)
    ties = sum(1 for b, c in zip(baseline, candidate) if c == b)
    total = len(baseline)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_pct": wins / total if total > 0 else 0.0,
        "loss_pct": losses / total if total > 0 else 0.0,
    }


print("Comparison utilities loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 Interpreting statistical results (5 min)
# MAGIC
# MAGIC Now we run all four analyses and interpret each result.

# COMMAND ----------

# MAGIC %md
# MAGIC ### McNemar's test: Is the difference real?
# MAGIC
# MAGIC For binary scores (correct/incorrect), McNemar's test checks whether
# MAGIC the number of regressions vs improvements is more than you would expect
# MAGIC by chance.

# COMMAND ----------

bl_scores = [
    s.baseline_score
    for s in aligned
    if s.baseline_score is not None and s.candidate_score is not None
]
cd_scores = [
    s.candidate_score
    for s in aligned
    if s.baseline_score is not None and s.candidate_score is not None
]

mcnemar = mcnemars_test(
    [s == 1.0 for s in bl_scores],
    [s == 1.0 for s in cd_scores],
)

print("McNemar's test:")
print(f"  Regressions:  {mcnemar['regressions']}")
print(f"  Improvements: {mcnemar['improvements']}")
print(f"  p-value:      {mcnemar['p_value']:.4f}")
print(f"  Significant:  {mcnemar['significant']}")
print()
if mcnemar["significant"]:
    print("  Interpretation: The imbalance between regressions and improvements")
    print("  is unlikely to be random. The candidate model is meaningfully different.")
else:
    print("  Interpretation: The imbalance could be due to chance. We cannot")
    print("  conclude the candidate is worse based on this sample size.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cohen's d: Does the difference matter?
# MAGIC
# MAGIC A p-value tells you the difference is real. Cohen's d tells you if it
# MAGIC matters in practice.
# MAGIC
# MAGIC - |d| < 0.2: negligible
# MAGIC - |d| 0.2-0.5: small
# MAGIC - |d| 0.5-0.8: medium
# MAGIC - |d| > 0.8: large

# COMMAND ----------

d = cohens_d(bl_scores, cd_scores)

if d is not None:
    if abs(d) < 0.2:
        magnitude = "negligible"
    elif abs(d) < 0.5:
        magnitude = "small"
    elif abs(d) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    print(f"Cohen's d: {d:+.3f} ({magnitude} effect)")
    print()
    if magnitude in ("negligible", "small"):
        print("  Interpretation: Even if statistically significant, the practical")
        print("  impact is small. Consider whether this matters for your use case.")
    else:
        print("  Interpretation: The effect is large enough to be practically")
        print("  meaningful. Investigate the regressed samples.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bootstrap CI: Where does the true difference lie?
# MAGIC
# MAGIC The confidence interval gives a range for the true difference in means.
# MAGIC If the interval excludes zero, the difference is significant at that
# MAGIC confidence level.

# COMMAND ----------

ci = bootstrap_ci(bl_scores, cd_scores)

print(f"Observed difference: {ci['observed_diff']:+.4f}")
print(f"95% CI: [{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]")
print(f"Excludes zero: {ci['excludes_zero']}")
print()
if ci["excludes_zero"]:
    if ci["ci_upper"] < 0:
        print("  Interpretation: The entire CI is below zero. The candidate is")
        print("  worse than the baseline with 95% confidence.")
    else:
        print("  Interpretation: The entire CI is above zero. The candidate is")
        print("  better than the baseline with 95% confidence.")
else:
    print("  Interpretation: The CI includes zero. We cannot be 95% confident")
    print("  that there is a real difference between baseline and candidate.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Win rate: The metric everyone understands
# MAGIC
# MAGIC For each sample, did the candidate win, lose, or tie?

# COMMAND ----------

wr = win_rate(bl_scores, cd_scores)

print(f"Wins:   {wr['wins']}/{len(bl_scores)} ({wr['win_pct']:.1%})")
print(f"Losses: {wr['losses']}/{len(bl_scores)} ({wr['loss_pct']:.1%})")
print(f"Ties:   {wr['ties']}/{len(bl_scores)}")
print()
print("  Win rate is the metric you show to stakeholders who do not care")
print("  about p-values. It answers: 'On how many questions did the new")
print("  model do better?'")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Full comparison summary

# COMMAND ----------

print("=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"Baseline:  model-v1 (accuracy: {np.mean(bl_scores):.1%})")
print(f"Candidate: model-v2 (accuracy: {np.mean(cd_scores):.1%})")
print(f"Samples:   {len(bl_scores)} aligned")
print()
print(f"Delta:     {np.mean(cd_scores) - np.mean(bl_scores):+.1%}")
print(f"Effect:    Cohen's d = {d:+.3f} ({magnitude})")
print(
    f"McNemar:   p = {mcnemar['p_value']:.4f} ({'significant' if mcnemar['significant'] else 'not significant'})"
)
print(f"Bootstrap: CI [{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]")
print(f"Win rate:  {wr['wins']}/{len(bl_scores)} ({wr['win_pct']:.1%})")
print()
if mcnemar["significant"] and np.mean(cd_scores) < np.mean(bl_scores):
    print("VERDICT: The candidate model is significantly worse. Do not deploy.")
elif not mcnemar["significant"]:
    print("VERDICT: The difference is not statistically significant at p=0.05.")
else:
    print("VERDICT: The candidate model is significantly better. Safe to deploy.")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key takeaways
# MAGIC
# MAGIC 1. **Align samples by ID** before comparing. Datasets change between runs.
# MAGIC 2. **McNemar's test** tells you if the difference is real (for binary scores).
# MAGIC 3. **Cohen's d** tells you if the difference matters in practice.
# MAGIC 4. **Bootstrap CI** gives you a range for the true difference.
# MAGIC 5. **Win rate** is the metric non-statisticians understand fastest.
# MAGIC
# MAGIC In Module 4, we wrap this into an automated evaluation gate.
# MAGIC
# MAGIC **Where this is going:** The comparison and statistical testing patterns shown
# MAGIC here are being contributed upstream to the
# MAGIC [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) framework as a
# MAGIC built-in comparison module
# MAGIC ([PR #3511](https://github.com/UKGovernmentBEIS/inspect_ai/pull/3511)). The
# MAGIC goal is to make regression detection a first-class feature of AI safety
# MAGIC evaluation, not something teams build ad hoc.
