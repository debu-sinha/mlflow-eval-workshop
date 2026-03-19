# Databricks notebook source
# MAGIC %md
# MAGIC # Module 4: The Evaluation Gate
# MAGIC
# MAGIC **Objective:** Build an automated gate that blocks model deployments when
# MAGIC regressions exceed a threshold, then show how to integrate it into CI/CD.
# MAGIC
# MAGIC **Time:** 15 minutes

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
import numpy as np

if ON_DATABRICKS:
    import json

    _ctx = json.loads(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()  # noqa: F821
    )
    mlflow.set_experiment(
        _ctx.get("extraContext", {}).get("notebook_path", "/tmp/odsc-workshop")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Start with the punchline (2 min)
# MAGIC
# MAGIC Here is the output of an evaluation gate blocking a bad model:

# COMMAND ----------

print("=" * 60)
print("EVALUATION GATE RESULT")
print("=" * 60)
print("Status:     BLOCKED")
print("Reason:     Regression rate 15.0% exceeds threshold 10.0%")
print()
print("Baseline accuracy:  85.0%")
print("Candidate accuracy: 72.0%")
print("Regressions: 15 | Improvements: 2 | p-value: 0.0026")
print("=" * 60)
print()
print("This gate prevented a bad model from reaching production.")
print("Let's build it.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Build the gate step by step (8 min)
# MAGIC
# MAGIC The gate takes baseline and candidate scores, computes regression rate,
# MAGIC runs a significance test, and returns a pass/fail decision.

# COMMAND ----------

from dataclasses import dataclass
from math import erfc, sqrt


@dataclass
class EvalGateResult:
    passed: bool
    baseline_accuracy: float
    candidate_accuracy: float
    regressions: int
    improvements: int
    regression_rate: float
    p_value: float
    effect_size: float
    reason: str


def run_eval_gate(
    baseline_scores: list[float],
    candidate_scores: list[float],
    max_regression_rate: float = 0.10,
    significance_threshold: float = 0.05,
) -> EvalGateResult:
    """Run the evaluation gate. Returns pass/fail with supporting evidence."""
    bl = np.array(baseline_scores)
    cd = np.array(candidate_scores)
    n = len(bl)

    bl_acc = float(np.mean(bl))
    cd_acc = float(np.mean(cd))

    # Count regressions (was correct, now wrong) and improvements (was wrong, now correct)
    regressions = int(np.sum((bl == 1.0) & (cd == 0.0)))
    improvements = int(np.sum((bl == 0.0) & (cd == 1.0)))
    regression_rate = regressions / n if n > 0 else 0.0

    # McNemar's test for significance
    discordant = regressions + improvements
    if discordant > 0:
        chi2 = (abs(regressions - improvements) - 1) ** 2 / discordant
        p_value = erfc(sqrt(chi2 / 2))
    else:
        p_value = 1.0

    # Cohen's d for effect size
    diffs = cd - bl
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    effect_size = float(np.mean(diffs)) / sd if sd > 0 else 0.0

    # Gate decision
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
        regression_rate=regression_rate,
        p_value=p_value,
        effect_size=effect_size,
        reason=reason,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test with a passing candidate
# MAGIC
# MAGIC The candidate has 3 regressions and 5 improvements. Net positive.

# COMMAND ----------

np.random.seed(42)
n = 100
baseline = np.random.choice([0.0, 1.0], size=n, p=[0.15, 0.85])

# Good candidate: 3 regressions, 5 improvements
candidate_good = baseline.copy()
flip_bad = np.random.choice(np.where(baseline == 1.0)[0], size=3, replace=False)
flip_good = np.random.choice(np.where(baseline == 0.0)[0], size=5, replace=False)
candidate_good[flip_bad] = 0.0
candidate_good[flip_good] = 1.0

result_pass = run_eval_gate(baseline.tolist(), candidate_good.tolist())

print("=" * 50)
print(f"Gate:       {'PASSED' if result_pass.passed else 'BLOCKED'}")
print(f"Reason:     {result_pass.reason}")
print(f"Baseline:   {result_pass.baseline_accuracy:.1%}")
print(f"Candidate:  {result_pass.candidate_accuracy:.1%}")
print(f"Regressions: {result_pass.regressions} | Improvements: {result_pass.improvements}")
print(f"Regression rate: {result_pass.regression_rate:.1%}")
print(f"p-value:    {result_pass.p_value:.4f}")
print(f"Effect:     {result_pass.effect_size:+.3f}")
print("=" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test with a failing candidate
# MAGIC
# MAGIC The candidate has 15 regressions and 2 improvements. The regression rate
# MAGIC (15%) exceeds the 10% threshold.

# COMMAND ----------

# Bad candidate: 15 regressions, 2 improvements
candidate_bad = baseline.copy()
flip_bad = np.random.choice(np.where(baseline == 1.0)[0], size=15, replace=False)
flip_good = np.random.choice(np.where(baseline == 0.0)[0], size=2, replace=False)
candidate_bad[flip_bad] = 0.0
candidate_bad[flip_good] = 1.0

result_fail = run_eval_gate(baseline.tolist(), candidate_bad.tolist())

print("=" * 50)
print(f"Gate:       {'PASSED' if result_fail.passed else 'BLOCKED'}")
print(f"Reason:     {result_fail.reason}")
print(f"Baseline:   {result_fail.baseline_accuracy:.1%}")
print(f"Candidate:  {result_fail.candidate_accuracy:.1%}")
print(f"Regressions: {result_fail.regressions} | Improvements: {result_fail.improvements}")
print(f"Regression rate: {result_fail.regression_rate:.1%}")
print(f"p-value:    {result_fail.p_value:.4f}")
print(f"Effect:     {result_fail.effect_size:+.3f}")
print("=" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC The gate caught the regression. The 15% regression rate exceeds the 10%
# MAGIC threshold, so the candidate is blocked.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 CI/CD integration patterns (5 min)
# MAGIC
# MAGIC The eval gate runs as part of your deployment pipeline. Below are two
# MAGIC patterns: a generic Python script for any CI system, and a Databricks
# MAGIC Workflows configuration.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pattern 1: Generic Python script (GitHub Actions, GitLab CI, etc.)
# MAGIC
# MAGIC ```python
# MAGIC # eval_gate.py - runs in any CI environment
# MAGIC import sys
# MAGIC import mlflow
# MAGIC import numpy as np
# MAGIC
# MAGIC BASELINE_RUN_ID = sys.argv[1]
# MAGIC CANDIDATE_RUN_ID = sys.argv[2]
# MAGIC MAX_REGRESSION_RATE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
# MAGIC
# MAGIC # Load scores from MLflow runs
# MAGIC baseline_run = mlflow.get_run(BASELINE_RUN_ID)
# MAGIC candidate_run = mlflow.get_run(CANDIDATE_RUN_ID)
# MAGIC
# MAGIC # ... extract per-sample scores from evaluation artifacts ...
# MAGIC
# MAGIC result = run_eval_gate(baseline_scores, candidate_scores, MAX_REGRESSION_RATE)
# MAGIC
# MAGIC # Log the gate result
# MAGIC with mlflow.start_run(run_name="eval-gate"):
# MAGIC     mlflow.log_metric("baseline_accuracy", result.baseline_accuracy)
# MAGIC     mlflow.log_metric("candidate_accuracy", result.candidate_accuracy)
# MAGIC     mlflow.log_metric("regressions", result.regressions)
# MAGIC     mlflow.log_metric("p_value", result.p_value)
# MAGIC     mlflow.log_param("gate_passed", result.passed)
# MAGIC     mlflow.log_param("gate_reason", result.reason)
# MAGIC
# MAGIC if not result.passed:
# MAGIC     print(f"BLOCKED: {result.reason}")
# MAGIC     sys.exit(1)
# MAGIC
# MAGIC print("PASSED: Safe to deploy.")
# MAGIC ```
# MAGIC
# MAGIC In GitHub Actions, call it like:
# MAGIC ```yaml
# MAGIC - name: Run eval gate
# MAGIC   run: python eval_gate.py $BASELINE_RUN_ID $CANDIDATE_RUN_ID 0.10
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pattern 2: Databricks Workflows
# MAGIC
# MAGIC ```yaml
# MAGIC # databricks.yml (Databricks Asset Bundles)
# MAGIC resources:
# MAGIC   jobs:
# MAGIC     eval_gate_pipeline:
# MAGIC       name: "eval-gate-pipeline"
# MAGIC       tasks:
# MAGIC         - task_key: evaluate_candidate
# MAGIC           notebook_task:
# MAGIC             notebook_path: /Repos/.../01_mlflow_evaluation_ecosystem
# MAGIC           parameters:
# MAGIC             model_uri: "{{model_uri}}"
# MAGIC
# MAGIC         - task_key: compare_and_gate
# MAGIC           depends_on:
# MAGIC             - task_key: evaluate_candidate
# MAGIC           notebook_task:
# MAGIC             notebook_path: /Repos/.../04_evaluation_gate
# MAGIC           parameters:
# MAGIC             baseline_run_id: "{{baseline_run_id}}"
# MAGIC             candidate_run_id: "{{candidate_run_id}}"
# MAGIC             max_regression_rate: "0.10"
# MAGIC
# MAGIC         - task_key: promote_model
# MAGIC           depends_on:
# MAGIC             - task_key: compare_and_gate
# MAGIC           notebook_task:
# MAGIC             notebook_path: /Repos/.../promote_to_production
# MAGIC           # Only runs if compare_and_gate succeeds
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log gate results to MLflow

# COMMAND ----------

with mlflow.start_run(run_name="eval-gate-passing") as run:
    mlflow.log_metric("baseline_accuracy", result_pass.baseline_accuracy)
    mlflow.log_metric("candidate_accuracy", result_pass.candidate_accuracy)
    mlflow.log_metric("regressions", result_pass.regressions)
    mlflow.log_metric("improvements", result_pass.improvements)
    mlflow.log_metric("regression_rate", result_pass.regression_rate)
    mlflow.log_metric("p_value", result_pass.p_value)
    mlflow.log_metric("effect_size", result_pass.effect_size)
    mlflow.log_param("gate_passed", result_pass.passed)
    mlflow.log_param("gate_reason", result_pass.reason)
    print(f"Passing gate logged to run: {run.info.run_id}")

with mlflow.start_run(run_name="eval-gate-blocking") as run:
    mlflow.log_metric("baseline_accuracy", result_fail.baseline_accuracy)
    mlflow.log_metric("candidate_accuracy", result_fail.candidate_accuracy)
    mlflow.log_metric("regressions", result_fail.regressions)
    mlflow.log_metric("improvements", result_fail.improvements)
    mlflow.log_metric("regression_rate", result_fail.regression_rate)
    mlflow.log_metric("p_value", result_fail.p_value)
    mlflow.log_metric("effect_size", result_fail.effect_size)
    mlflow.log_param("gate_passed", result_fail.passed)
    mlflow.log_param("gate_reason", result_fail.reason)
    print(f"Blocking gate logged to run: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4 Workshop summary and next steps
# MAGIC
# MAGIC Here is what we built in 55 minutes:
# MAGIC
# MAGIC | Module | What we built | Why it matters |
# MAGIC |--------|--------------|----------------|
# MAGIC | 1 | Built-in, third-party, and custom scorers in one `evaluate()` call | Single interface for all evaluation needs |
# MAGIC | 2 | Deterministic judge config and concurrency control | Reproducible, rate-limit-safe evaluations |
# MAGIC | 3 | Sample alignment with statistical comparison | Know if model changes are real or noise |
# MAGIC | 4 | Automated evaluation gate with CI/CD patterns | Block bad models before they reach production |
# MAGIC
# MAGIC ### What to do next
# MAGIC
# MAGIC - **Try your own scorers.** Write a `@scorer` function for your domain-specific
# MAGIC   quality criteria and plug it into the same `evaluate()` call.
# MAGIC - **Set up the gate in your pipeline.** Use the generic Python script pattern
# MAGIC   for GitHub Actions, or the Databricks Workflows pattern if you are on
# MAGIC   Databricks.
# MAGIC - **Tune the thresholds.** The 10% regression rate and p=0.05 significance
# MAGIC   level are starting points. Adjust them based on your risk tolerance.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Beyond MLflow scorers
# MAGIC
# MAGIC MLflow's evaluation ecosystem extends beyond what we covered today:
# MAGIC - **More third-party scorers**: DeepEval and RAGAS integrations are also available
# MAGIC   for RAG-specific metrics like faithfulness and answer relevance
# MAGIC - **Inspect AI integration**: MLflow tracking hooks for the UK AI Safety Institute's
# MAGIC   evaluation framework, enabling experiment tracking across safety evaluation suites
# MAGIC - **Trace-based evaluation**: MLflow scorers can evaluate production traces directly,
# MAGIC   connecting observability with evaluation
# MAGIC - **Scheduled scorers**: Run evaluation scorers on a schedule against production traffic

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resources
# MAGIC
# MAGIC - Workshop code: [github.com/debu-sinha/mlflow-eval-workshop](https://github.com/debu-sinha/mlflow-eval-workshop)
# MAGIC - MLflow GenAI evaluation docs: [mlflow.org/docs/latest/genai](https://mlflow.org/docs/latest/genai)
# MAGIC - MLflow scorers API: [mlflow.org/docs/latest/python_api/mlflow.genai.scorers.html](https://mlflow.org/docs/latest/python_api/mlflow.genai.scorers.html)
# MAGIC
# MAGIC ### Speaker
# MAGIC
# MAGIC **Debu Sinha** | Lead Applied AI/ML Engineer @ Databricks
# MAGIC
# MAGIC - LinkedIn: [linkedin.com/in/debusinha](https://linkedin.com/in/debusinha)
# MAGIC - GitHub: [github.com/debu-sinha](https://github.com/debu-sinha)
