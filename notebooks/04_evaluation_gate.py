# Databricks notebook source
# MAGIC %md
# MAGIC # Module 4: The Evaluation Gate
# MAGIC
# MAGIC **Objective:** Build an automated gate that blocks model deployments when
# MAGIC regressions exceed a threshold, then show how to integrate it into CI/CD.
# MAGIC
# MAGIC **Time:** 15 minutes

# COMMAND ----------

import os

if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    _nb_path = (
        dbutils.notebook.entry_point.getDbutils()  # noqa: F821
        .notebook()
        .getContext()
        .notebookPath()
        .get()
    )
    _repo_root = "/Workspace" + "/".join(_nb_path.split("/")[:-2])
    REQ_PATH = f"{_repo_root}/requirements-workshop.txt"
    print(f"Installing workshop requirements from: {REQ_PATH}")
else:
    REQ_PATH = None
    print(
        "Local run detected. Skip the next two cells (Databricks "
        "%pip install + %run helper) and run `pip install '.[all]'` "
        "from the repo root before continuing."
    )

# COMMAND ----------

# MAGIC %pip install -q -r $REQ_PATH
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_verify_environment

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
    _user = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .userName()
        .get()
    )  # noqa: F821
    mlflow.set_experiment(f"/Users/{_user}/odsc-workshop-m4")
else:
    mlflow.set_tracking_uri("sqlite:///mlflow_workshop.db")
    mlflow.set_experiment("odsc-eval-workshop")

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
print("Baseline accuracy:  81.0%")
print("Candidate accuracy: 68.0%")
print("Regressions: 15 | Improvements: 2 | McNemar p-value: 0.0036")
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
print(
    f"Regressions: {result_pass.regressions} | Improvements: {result_pass.improvements}"
)
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
print(
    f"Regressions: {result_fail.regressions} | Improvements: {result_fail.improvements}"
)
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
# MAGIC
# MAGIC ### When does the gate fire?
# MAGIC
# MAGIC Three common trigger patterns. Pick one based on your release cadence:
# MAGIC
# MAGIC - **PR-triggered**: every pull request that touches model config runs
# MAGIC   the gate against the latest production run. Blocks merge on regression.
# MAGIC - **Scheduled**: nightly job evaluates the latest staging model against
# MAGIC   production. Pages oncall if the gate blocks.
# MAGIC - **Manual (`workflow_dispatch`)**: for human review of a candidate
# MAGIC   before promotion. The shipped `eval-gate.yml` in this repo uses this
# MAGIC   trigger.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pattern 1: CLI script (GitHub Actions, GitLab CI, etc.)
# MAGIC
# MAGIC The repo includes `eval_gate.py` which handles sample alignment, score
# MAGIC parsing, and statistical testing. Call it with named flags:
# MAGIC
# MAGIC ```bash
# MAGIC python eval_gate.py \
# MAGIC     --baseline-run-id $BASELINE_RUN_ID \
# MAGIC     --candidate-run-id $CANDIDATE_RUN_ID \
# MAGIC     --scorer correctness \
# MAGIC     --threshold 0.10
# MAGIC # Exit code 0 = pass, 1 = blocked
# MAGIC ```
# MAGIC
# MAGIC The script logs results to stdout and exits with code 1 if the candidate
# MAGIC regresses. Any CI system that checks exit codes can use it directly.
# MAGIC
# MAGIC In GitHub Actions:
# MAGIC ```yaml
# MAGIC - name: Run eval gate
# MAGIC   run: |
# MAGIC     python eval_gate.py \
# MAGIC       --baseline-run-id ${{ env.BASELINE_RUN_ID }} \
# MAGIC       --candidate-run-id ${{ env.CANDIDATE_RUN_ID }} \
# MAGIC       --scorer correctness \
# MAGIC       --threshold 0.10
# MAGIC ```
# MAGIC
# MAGIC The complete workflow file is at `.github/workflows/eval-gate.yml`
# MAGIC in this repo. It adds the checkout, Python setup, secret-based
# MAGIC tracking URI, and a `min_overlap` input for production use. Copy it
# MAGIC into your own repo and customize the trigger.

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
# MAGIC             notebook_path: /Workspace/Users/<your-email>/mlflow-eval-workshop/notebooks/01_mlflow_evaluation_ecosystem
# MAGIC           parameters:
# MAGIC             model_uri: "{{model_uri}}"
# MAGIC
# MAGIC         - task_key: compare_and_gate
# MAGIC           depends_on:
# MAGIC             - task_key: evaluate_candidate
# MAGIC           notebook_task:
# MAGIC             notebook_path: /Workspace/Users/<your-email>/mlflow-eval-workshop/notebooks/04_evaluation_gate
# MAGIC           parameters:
# MAGIC             baseline_run_id: "{{baseline_run_id}}"
# MAGIC             candidate_run_id: "{{candidate_run_id}}"
# MAGIC             max_regression_rate: "0.10"
# MAGIC
# MAGIC         - task_key: promote_model
# MAGIC           depends_on:
# MAGIC             - task_key: compare_and_gate
# MAGIC           notebook_task:
# MAGIC             notebook_path: /Workspace/Users/<your-email>/mlflow-eval-workshop/notebooks/promote_to_production
# MAGIC           # Only runs if compare_and_gate succeeds
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log gate results to MLflow
# MAGIC
# MAGIC Logging each gate decision as its own MLflow run gives you queryable
# MAGIC history of every gate firing. Useful for incident review and quarterly
# MAGIC model-quality reports: "show me every blocked model in Q1 with their
# MAGIC regression rate" becomes one MLflow search query.

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
# MAGIC ### Run the real eval_gate.py on Module 3's evaluation runs
# MAGIC
# MAGIC Module 3 produced two real evaluation runs (baseline and candidate) using
# MAGIC `Correctness`. We can feed those run IDs into `eval_gate.py` to see the
# MAGIC gate work end-to-end with real MLflow data.

# COMMAND ----------

import json as _json_m4
import subprocess
from pathlib import Path

# Locate eval_gate.py by searching up from the script or cwd
_search_dirs = []
if "__file__" in dir():
    _search_dirs.append(Path(__file__).resolve().parent.parent)
    _search_dirs.append(Path(__file__).resolve().parent)
_search_dirs.append(Path.cwd())
_search_dirs.append(Path.cwd().parent)

_gate_script = None
for _d in _search_dirs:
    _candidate = _d / "eval_gate.py"
    if _candidate.exists():
        _gate_script = _candidate
        break

# First try: read run IDs from Module 3's handoff file (deterministic)
baseline_id = None
candidate_id = None
for _d in _search_dirs:
    _handoff = _d / "notebooks" / "m3_run_ids.json"
    if not _handoff.exists():
        _handoff = _d / "m3_run_ids.json"
    if _handoff.exists():
        _ids = _json_m4.loads(_handoff.read_text())
        baseline_id = _ids.get("baseline_run_id")
        candidate_id = _ids.get("candidate_run_id")
        print(f"Loaded run IDs from {_handoff.name}")
        break

# Fall back: search experiment for evaluation runs
if not baseline_id or not candidate_id:
    client = mlflow.tracking.MlflowClient()
    _m3_candidates = [os.environ.get("MLFLOW_EXPERIMENT_NAME", "")]
    if ON_DATABRICKS:
        _m3_candidates.append(f"/Users/{_user}/odsc-workshop-m3")
    _m3_candidates.extend(["odsc-eval-workshop", "workshop-module3"])

    exp = None
    for _name in _m3_candidates:
        if not _name:
            continue
        exp = client.get_experiment_by_name(_name)
        if exp:
            break

    if exp:
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string="metrics.`correctness/mean` > 0",
            order_by=["start_time DESC"],
            max_results=2,
        )
        if len(runs) >= 2:
            candidate_id = runs[0].info.run_id
            baseline_id = runs[1].info.run_id
            print("Discovered run IDs from experiment search")

if _gate_script is None:
    print("eval_gate.py not found. Check repo structure.")
elif not baseline_id or not candidate_id:
    print("No evaluation runs found. Run Module 3 first.")
else:
    print(f"Baseline run:  {baseline_id}")
    print(f"Candidate run: {candidate_id}")
    print()

    # Print the manual command in case auto-discovery fails on your system
    print(
        f"Manual command:\n  python eval_gate.py --baseline-run-id {baseline_id} --candidate-run-id {candidate_id} --scorer correctness\n"
    )

    import sys

    # The parent notebook may have set the tracking URI via Python API
    # (e.g. mlflow.set_tracking_uri("sqlite:///...")) and on Databricks it
    # holds the host + token in the driver process. Neither propagates to a
    # subprocess, so we thread both through explicitly.
    _gate_env = {**os.environ, "MLFLOW_TRACKING_URI": mlflow.get_tracking_uri()}

    if ON_DATABRICKS:
        try:
            from mlflow.utils.databricks_utils import get_databricks_host_creds

            _creds = get_databricks_host_creds()
            if getattr(_creds, "host", None):
                _gate_env["DATABRICKS_HOST"] = _creds.host
            if getattr(_creds, "token", None):
                _gate_env["DATABRICKS_TOKEN"] = _creds.token
        except Exception as _auth_err:
            print(
                f"Could not extract Databricks credentials for subprocess: {_auth_err}"
            )

    result = subprocess.run(
        [
            sys.executable,
            str(_gate_script),
            "--baseline-run-id",
            baseline_id,
            "--candidate-run-id",
            candidate_id,
            "--scorer",
            "correctness",
        ],
        capture_output=True,
        text=True,
        cwd=str(_gate_script.parent),
        env=_gate_env,
    )

    # Strip known-noise lines from the subprocess output so attendees do
    # not see the Databricks Connect Spark-context warning that fires when
    # MLflow initializes on Serverless. The gate itself uses the MLflow
    # REST client and does not need Spark.
    _noise_patterns = (
        "Failed to initialize spark connection",
        "CONTEXT_UNAVAILABLE_FOR_REMOTE_CLIENT",
        "Remote client cannot create a SparkContext",
    )

    def _filter_noise(stream: str) -> str:
        return "\n".join(
            line
            for line in (stream or "").splitlines()
            if not any(p in line for p in _noise_patterns)
        )

    print(_filter_noise(result.stdout))
    if result.returncode != 0:
        print(f"Gate exit code: {result.returncode}")
        _err = _filter_noise(result.stderr)
        if _err.strip():
            print(_err)

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
# MAGIC - **Agent evaluation**: `ToolCallCorrectness` and `ToolCallEfficiency` scorers
# MAGIC   for evaluating agent tool usage, multi-turn conversations, and trajectory quality
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
