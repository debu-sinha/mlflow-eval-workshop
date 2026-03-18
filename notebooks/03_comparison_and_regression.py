# Databricks notebook source
# MAGIC %md
# MAGIC # Module 3: Comparing Evaluation Runs and Detecting Regressions
# MAGIC
# MAGIC **Objective:** Align samples across two evaluation runs, compute score deltas,
# MAGIC and determine whether differences are statistically significant.
# MAGIC
# MAGIC **Tools:** MLflow, NumPy
# MAGIC
# MAGIC **Time:** 20 minutes

# COMMAND ----------

# Install dependencies (uncomment if running locally)
# !pip install mlflow[genai] scikit-learn numpy -q

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 The problem: "Did the model get better?"
# MAGIC
# MAGIC You run evals before and after a model change. Accuracy went from 90% to 85%.
# MAGIC Questions you need to answer:
# MAGIC - Which specific samples regressed?
# MAGIC - Is the 5% drop real or just noise?
# MAGIC - How big is the effect in practical terms?
# MAGIC
# MAGIC MLflow doesn't have built-in comparison. We'll build one.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Create two evaluation runs to compare

# COMMAND ----------

import mlflow
import numpy as np

if not ON_DATABRICKS:
    mlflow.set_experiment("odsc-eval-workshop-module-3-comparison")

# Simulate baseline evaluation (model v1)
np.random.seed(42)
n_samples = 50
baseline_scores = np.random.choice([0.0, 1.0], size=n_samples, p=[0.15, 0.85])

# Simulate candidate evaluation (model v2, slightly worse)
candidate_scores = baseline_scores.copy()
# Introduce 5 regressions (samples that were correct, now incorrect)
regression_indices = np.random.choice(
    np.where(baseline_scores == 1.0)[0], size=5, replace=False
)
candidate_scores[regression_indices] = 0.0
# Introduce 2 improvements (samples that were incorrect, now correct)
improvement_indices = np.random.choice(
    np.where(baseline_scores == 0.0)[0], size=2, replace=False
)
candidate_scores[improvement_indices] = 1.0

print(f"Baseline accuracy: {baseline_scores.mean():.1%}")
print(f"Candidate accuracy: {candidate_scores.mean():.1%}")
print(f"Regressions: {sum(1 for b, c in zip(baseline_scores, candidate_scores) if b == 1 and c == 0)}")
print(f"Improvements: {sum(1 for b, c in zip(baseline_scores, candidate_scores) if b == 0 and c == 1)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Sample alignment
# MAGIC
# MAGIC Before comparing, we need to match samples by ID. If the evaluation dataset
# MAGIC changed between runs (added/removed samples), alignment handles that.

# COMMAND ----------

from dataclasses import dataclass


@dataclass
class AlignedSample:
    id: int
    baseline_score: float | None
    candidate_score: float | None


def align_samples(
    baseline: dict[int, float],
    candidate: dict[int, float],
) -> list[AlignedSample]:
    """Align samples by ID, handling missing/new samples."""
    all_ids = sorted(set(baseline.keys()) | set(candidate.keys()))
    aligned = []
    for sample_id in all_ids:
        aligned.append(
            AlignedSample(
                id=sample_id,
                baseline_score=baseline.get(sample_id),
                candidate_score=candidate.get(sample_id),
            )
        )
    return aligned


# Create sample dicts (ID -> score)
baseline_dict = {i: float(s) for i, s in enumerate(baseline_scores)}
candidate_dict = {i: float(s) for i, s in enumerate(candidate_scores)}

aligned = align_samples(baseline_dict, candidate_dict)
print(f"Aligned {len(aligned)} samples")

# Show regressions
for s in aligned:
    if s.baseline_score == 1.0 and s.candidate_score == 0.0:
        print(f"  Sample {s.id}: 1.0 -> 0.0 (REGRESSED)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 Statistical significance: Is the difference real?
# MAGIC
# MAGIC With binary scores (correct/incorrect), McNemar's test tells us whether
# MAGIC the number of regressions vs improvements is statistically significant.

# COMMAND ----------

from math import erfc, sqrt


def mcnemars_test(
    baseline_correct: list[bool],
    candidate_correct: list[bool],
    significance: float = 0.05,
) -> dict:
    """McNemar's test for paired binary outcomes.

    Tests whether the rate of discordant pairs differs significantly.
    Uses chi-square approximation with continuity correction.
    No scipy needed.
    """
    b = sum(1 for bl, cd in zip(baseline_correct, candidate_correct) if bl and not cd)
    c = sum(1 for bl, cd in zip(baseline_correct, candidate_correct) if not bl and cd)

    discordant = b + c
    if discordant == 0:
        return {"p_value": 1.0, "significant": False, "regressions": b, "improvements": c}

    # Chi-square with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / discordant

    # p-value from chi-square (df=1): erfc(sqrt(x/2))
    p_value = erfc(sqrt(chi2 / 2))

    return {
        "p_value": p_value,
        "significant": p_value < significance,
        "regressions": b,
        "improvements": c,
        "discordant_pairs": discordant,
    }


result = mcnemars_test(
    [s.baseline_score == 1.0 for s in aligned],
    [s.candidate_score == 1.0 for s in aligned],
)

print(f"McNemar's test results:")
print(f"  Regressions: {result['regressions']}")
print(f"  Improvements: {result['improvements']}")
print(f"  p-value: {result['p_value']:.4f}")
print(f"  Significant at 0.05: {result['significant']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 Effect size: Does the difference matter?
# MAGIC
# MAGIC Statistical significance tells you the difference is real.
# MAGIC Effect size (Cohen's d) tells you if it matters in practice.

# COMMAND ----------


def cohens_d(baseline: list[float], candidate: list[float]) -> float | None:
    """Cohen's d for paired samples. 0.2 = small, 0.5 = medium, 0.8 = large."""
    diffs = np.array(candidate) - np.array(baseline)
    if len(diffs) < 2:
        return None
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return 0.0
    return float(np.mean(diffs)) / sd


bl_scores = [s.baseline_score for s in aligned if s.baseline_score is not None and s.candidate_score is not None]
cd_scores = [s.candidate_score for s in aligned if s.baseline_score is not None and s.candidate_score is not None]

d = cohens_d(bl_scores, cd_scores)
magnitude = "small" if abs(d) < 0.5 else ("medium" if abs(d) < 0.8 else "large")
print(f"Cohen's d: {d:+.3f} ({magnitude} effect)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6 Bootstrap confidence intervals
# MAGIC
# MAGIC For continuous scores, bootstrap the difference in means to get a
# MAGIC confidence interval. If the CI excludes zero, the difference is significant.

# COMMAND ----------


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


ci = bootstrap_ci(bl_scores, cd_scores)
print(f"Mean difference: {ci['observed_diff']:+.4f}")
print(f"95% CI: [{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]")
print(f"CI excludes zero: {ci['excludes_zero']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.7 Win rate
# MAGIC
# MAGIC The simplest summary: what fraction of samples did the candidate win on?

# COMMAND ----------

wins = sum(1 for b, c in zip(bl_scores, cd_scores) if c > b)
losses = sum(1 for b, c in zip(bl_scores, cd_scores) if c < b)
ties = sum(1 for b, c in zip(bl_scores, cd_scores) if c == b)

print(f"Win rate: {wins}/{len(bl_scores)} ({wins/len(bl_scores):.1%})")
print(f"Loss rate: {losses}/{len(bl_scores)} ({losses/len(bl_scores):.1%})")
print(f"Tie rate: {ties}/{len(bl_scores)} ({ties/len(bl_scores):.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.8 Putting it all together: comparison summary

# COMMAND ----------

print(f"Baseline:  model-v1 (accuracy: {np.mean(bl_scores):.1%})")
print(f"Candidate: model-v2 (accuracy: {np.mean(cd_scores):.1%})")
print(f"Samples:   {len(bl_scores)} aligned")
print()
print(f"Delta:     {np.mean(cd_scores) - np.mean(bl_scores):+.1%}")
print(f"Effect:    Cohen's d = {d:+.3f} ({magnitude})")
print(f"McNemar:   p = {result['p_value']:.4f} ({'significant' if result['significant'] else 'not significant'})")
print(f"Bootstrap: CI [{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]")
print(f"Win rate:  {wins}/{len(bl_scores)} ({wins/len(bl_scores):.1%})")
print()
if result["significant"]:
    print("VERDICT: The candidate model is significantly worse. Do not deploy.")
else:
    print("VERDICT: The difference is not statistically significant at p=0.05.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key takeaways
# MAGIC
# MAGIC 1. **Align samples by ID** before comparing. Datasets change between runs.
# MAGIC 2. **McNemar's test** for binary scores (correct/incorrect). No scipy needed.
# MAGIC 3. **Bootstrap CI** for continuous scores. Vectorized for speed.
# MAGIC 4. **Cohen's d** separates "is it real?" from "does it matter?"
# MAGIC 5. **Win rate** is the metric non-statisticians understand fastest.
