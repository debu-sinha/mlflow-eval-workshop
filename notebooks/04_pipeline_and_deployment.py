# Databricks notebook source
# MAGIC %md
# MAGIC # Module 4: End-to-End Pipeline and Deployment Patterns
# MAGIC
# MAGIC **Objective:** Chain all three modules into a single evaluation pipeline using
# MAGIC Databricks Workflows and understand CI/CD integration patterns.
# MAGIC
# MAGIC **Tools:** MLflow, NumPy
# MAGIC
# MAGIC **Time:** 10 minutes

# COMMAND ----------

# Install dependencies (uncomment if running locally)
# !pip install mlflow[genai] arize-phoenix guardrails-ai scikit-learn numpy -q

# COMMAND ----------

import os
ON_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 The evaluation gate pattern
# MAGIC
# MAGIC Before promoting a model from staging to production, run an evaluation
# MAGIC pipeline that:
# MAGIC 1. Scores the candidate model with multiple scorers
# MAGIC 2. Compares against the baseline (current production model)
# MAGIC 3. Blocks promotion if regressions exceed a threshold
# MAGIC
# MAGIC This is the "eval gate" pattern used by teams running LLMs in production.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 The evaluation pipeline script
# MAGIC
# MAGIC In production, this would be a Databricks Workflow task triggered by
# MAGIC a model registration event.

# COMMAND ----------

import mlflow
import numpy as np
from dataclasses import dataclass



@dataclass
class EvalGateResult:
    passed: bool
    baseline_accuracy: float
    candidate_accuracy: float
    regressions: int
    improvements: int
    p_value: float
    effect_size: float
    reason: str


def run_eval_gate(
    baseline_scores: list[float],
    candidate_scores: list[float],
    max_regression_rate: float = 0.10,
    significance_threshold: float = 0.05,
) -> EvalGateResult:
    """Run the evaluation gate. Returns pass/fail with details."""
    from math import erfc, sqrt

    n = len(baseline_scores)
    bl = np.array(baseline_scores)
    cd = np.array(candidate_scores)

    bl_acc = float(np.mean(bl))
    cd_acc = float(np.mean(cd))

    # Count regressions and improvements
    regressions = int(np.sum((bl == 1.0) & (cd == 0.0)))
    improvements = int(np.sum((bl == 0.0) & (cd == 1.0)))
    regression_rate = regressions / n if n > 0 else 0.0

    # McNemar's test
    discordant = regressions + improvements
    if discordant > 0:
        chi2 = (abs(regressions - improvements) - 1) ** 2 / discordant
        p_value = erfc(sqrt(chi2 / 2))
    else:
        p_value = 1.0

    # Cohen's d
    diffs = cd - bl
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    effect_size = float(np.mean(diffs)) / sd if sd > 0 else 0.0

    # Gate logic
    if regression_rate > max_regression_rate:
        passed = False
        reason = f"Regression rate {regression_rate:.1%} exceeds threshold {max_regression_rate:.1%}"
    elif p_value < significance_threshold and cd_acc < bl_acc:
        passed = False
        reason = f"Significant regression detected (p={p_value:.4f})"
    else:
        passed = True
        reason = "No significant regression detected"

    return EvalGateResult(
        passed=passed,
        baseline_accuracy=bl_acc,
        candidate_accuracy=cd_acc,
        regressions=regressions,
        improvements=improvements,
        p_value=p_value,
        effect_size=effect_size,
        reason=reason,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the gate with a passing candidate

# COMMAND ----------

np.random.seed(42)
n = 100
baseline = np.random.choice([0.0, 1.0], size=n, p=[0.15, 0.85])
candidate_good = baseline.copy()
# 3 regressions, 5 improvements -> net positive
flip_to_bad = np.random.choice(np.where(baseline == 1.0)[0], size=3, replace=False)
flip_to_good = np.random.choice(np.where(baseline == 0.0)[0], size=5, replace=False)
candidate_good[flip_to_bad] = 0.0
candidate_good[flip_to_good] = 1.0

result = run_eval_gate(baseline.tolist(), candidate_good.tolist())

print(f"Baseline accuracy:  {result.baseline_accuracy:.1%}")
print(f"Candidate accuracy: {result.candidate_accuracy:.1%}")
print(f"Regressions: {result.regressions}, Improvements: {result.improvements}")
print(f"p-value: {result.p_value:.4f}")
print(f"Effect size: {result.effect_size:+.3f}")
print(f"Gate: {'PASSED' if result.passed else 'BLOCKED'}")
print(f"Reason: {result.reason}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the gate with a failing candidate

# COMMAND ----------

candidate_bad = baseline.copy()
# 15 regressions, 2 improvements -> net negative, exceeds 10% threshold
flip_to_bad = np.random.choice(np.where(baseline == 1.0)[0], size=15, replace=False)
flip_to_good = np.random.choice(np.where(baseline == 0.0)[0], size=2, replace=False)
candidate_bad[flip_to_bad] = 0.0
candidate_bad[flip_to_good] = 1.0

result_bad = run_eval_gate(baseline.tolist(), candidate_bad.tolist())

print(f"Baseline accuracy:  {result_bad.baseline_accuracy:.1%}")
print(f"Candidate accuracy: {result_bad.candidate_accuracy:.1%}")
print(f"Regressions: {result_bad.regressions}, Improvements: {result_bad.improvements}")
print(f"p-value: {result_bad.p_value:.4f}")
print(f"Effect size: {result_bad.effect_size:+.3f}")
print(f"Gate: {'PASSED' if result_bad.passed else 'BLOCKED'}")
print(f"Reason: {result_bad.reason}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Integrating with Databricks Workflows
# MAGIC
# MAGIC In production, this pipeline runs as a Databricks Workflow:
# MAGIC
# MAGIC 1. **Trigger:** Model registration event (new model version registered)
# MAGIC 2. **Task 1:** Run evaluation with scorers (Module 1)
# MAGIC 3. **Task 2:** Compare against baseline (Module 3)
# MAGIC 4. **Task 3:** Gate decision (this module)
# MAGIC 5. **On pass:** Promote model to production alias
# MAGIC 6. **On fail:** Log failure reason, notify team, keep current model
# MAGIC
# MAGIC ```yaml
# MAGIC # Example Databricks Workflow configuration
# MAGIC tasks:
# MAGIC   - task_key: evaluate_candidate
# MAGIC     notebook_task:
# MAGIC       notebook_path: /Repos/.../01_third_party_scorers
# MAGIC     parameters:
# MAGIC       model_uri: "{{model_uri}}"
# MAGIC
# MAGIC   - task_key: compare_and_gate
# MAGIC     depends_on:
# MAGIC       - task_key: evaluate_candidate
# MAGIC     notebook_task:
# MAGIC       notebook_path: /Repos/.../04_pipeline_and_deployment
# MAGIC     parameters:
# MAGIC       baseline_run_id: "{{baseline_run_id}}"
# MAGIC       candidate_run_id: "{{candidate_run_id}}"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4 Logging gate results to MLflow

# COMMAND ----------

with mlflow.start_run(run_name="eval-gate-demo") as run:
    mlflow.log_metric("baseline_accuracy", result.baseline_accuracy)
    mlflow.log_metric("candidate_accuracy", result.candidate_accuracy)
    mlflow.log_metric("regressions", result.regressions)
    mlflow.log_metric("improvements", result.improvements)
    mlflow.log_metric("p_value", result.p_value)
    mlflow.log_metric("effect_size", result.effect_size)
    mlflow.log_param("gate_passed", result.passed)
    mlflow.log_param("gate_reason", result.reason)

    print(f"Gate results logged to run: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.5 Workshop summary
# MAGIC
# MAGIC What we built in 60 minutes:
# MAGIC
# MAGIC | Module | What | Why |
# MAGIC |--------|------|-----|
# MAGIC | 1 | Connected Phoenix, TruLens, Guardrails AI to MLflow | Run safety and quality evals from one place |
# MAGIC | 2 | Configured judge params, concurrency, and uv deps | Reproducible, rate-limit-safe evaluations |
# MAGIC | 3 | Built comparison with statistical testing | Know if model changes are real or noise |
# MAGIC | 4 | Assembled an eval gate for CI/CD | Block bad models before they reach production |
# MAGIC
# MAGIC **Next steps:**
# MAGIC - Add your own scorers to the pipeline
# MAGIC - Set up a Databricks Workflow to run this on model registration events
# MAGIC - Adjust the regression threshold for your risk tolerance
# MAGIC
# MAGIC **Code:** github.com/debu-sinha/mlflow-eval-workshop

# COMMAND ----------

# MAGIC %md
# MAGIC ## Questions?
# MAGIC
# MAGIC **Debu Sinha** | Lead Applied AI/ML Engineer @ Databricks
# MAGIC
# MAGIC LinkedIn: linkedin.com/in/debusinha
# MAGIC
# MAGIC GitHub: github.com/debu-sinha
