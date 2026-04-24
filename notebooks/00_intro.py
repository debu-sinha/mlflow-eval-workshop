# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluating LLM Applications with MLflow
# MAGIC
# MAGIC **ODSC AI East 2026** | April 28-30, Boston | 60 minutes
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Workshop Goal
# MAGIC
# MAGIC Build a production evaluation pipeline that goes from scorer selection to
# MAGIC deployment gating in one continuous story: evaluate a real LLM with built-in
# MAGIC and third-party scorers, compare two evaluation runs with statistical rigor,
# MAGIC and block bad models from production with an automated gate.
# MAGIC
# MAGIC By the end, you will have a working evaluation gate that integrates into CI/CD.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Workshop Flow
# MAGIC
# MAGIC ![Workshop Flow](images/workshop-flow.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Module Details
# MAGIC
# MAGIC ### Module 1: MLflow Evaluation Ecosystem (20 min)
# MAGIC **Notebook:** `01_mlflow_evaluation_ecosystem`
# MAGIC
# MAGIC | What you'll do | Tools used |
# MAGIC |----------------|-----------|
# MAGIC | Run built-in scorers (`Correctness`, `Safety`) | `mlflow.genai.evaluate()` |
# MAGIC | Add third-party scorers (Phoenix `Hallucination`, TruLens `Groundedness`) | `mlflow.genai.scorers.phoenix`, `.trulens` |
# MAGIC | Add optional safety scorer (Guardrails `DetectPII`) | `mlflow.genai.scorers.guardrails` |
# MAGIC | Write a custom scorer with `@scorer` decorator | `mlflow.genai.scorers.scorer` |
# MAGIC | Call a real LLM and evaluate its response | `predict_fn` with traces |
# MAGIC | Combine all scorers in one `evaluate()` call | Single result DataFrame |
# MAGIC
# MAGIC **Key takeaway:** One API surface for all evaluation needs.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Module 2: Production Evaluation Infrastructure (8 min)
# MAGIC **Notebook:** `02_production_infrastructure`
# MAGIC
# MAGIC | What you'll do | Tools used |
# MAGIC |----------------|-----------|
# MAGIC | Set `temperature=0.0` for deterministic judge scoring | `inference_params` |
# MAGIC | Control scorer concurrency for rate-limited APIs | `MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS` |
# MAGIC | Compare metrics across evaluation runs in the UI | MLflow Experiment view |
# MAGIC
# MAGIC **Key takeaway:** Reproducible, rate-limit-safe evaluations.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Module 3: Comparing Runs and Detecting Regressions (12 min)
# MAGIC **Notebook:** `03_comparison_and_regression`
# MAGIC
# MAGIC | What you'll do | Tools used |
# MAGIC |----------------|-----------|
# MAGIC | Align samples across two evaluation runs by ID | `align_samples()` |
# MAGIC | Detect per-sample regressions | Score delta analysis |
# MAGIC | Test significance with McNemar's test | `mcnemars_test()` |
# MAGIC | Measure effect size with Cohen's d | `cohens_d()` |
# MAGIC | Compute bootstrap confidence intervals | `bootstrap_ci()` |
# MAGIC | Compare real `mlflow.genai.evaluate()` runs | Key-aligned result DataFrames |
# MAGIC
# MAGIC **Key takeaway:** Same accuracy can hide different broken samples. Per-sample alignment catches what aggregates miss.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Module 4: The Evaluation Gate (15 min)
# MAGIC **Notebook:** `04_evaluation_gate`
# MAGIC
# MAGIC | What you'll do | Tools used |
# MAGIC |----------------|-----------|
# MAGIC | Build a gate that blocks deployment on regression | `run_eval_gate()` |
# MAGIC | Log gate decisions to MLflow | `mlflow.log_metric()`, `mlflow.log_param()` |
# MAGIC | Run the real `eval_gate.py` on Module 3's runs | CLI with `--baseline-run-id` |
# MAGIC | Wire the gate into GitHub Actions and Databricks Workflows | YAML configs |
# MAGIC
# MAGIC **Key takeaway:** A bad model should never reach production. Automate the gate.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Bonus: uv Dependencies (5 min)
# MAGIC **Notebook:** `bonus_uv_dependencies`
# MAGIC
# MAGIC | What you'll do | Tools used |
# MAGIC |----------------|-----------|
# MAGIC | Understand uv lockfiles for reproducible model serving | `uv.lock` |
# MAGIC | See how MLflow auto-detects uv projects | `mlflow.sklearn.log_model()` |
# MAGIC | Compare pip vs uv dependency resolution | Side-by-side |
# MAGIC
# MAGIC **Key takeaway:** Pin every transitive dependency for reproducible deployments.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC
# MAGIC **On Databricks:** Each module has three preamble cells: compute an
# MAGIC absolute path to `requirements-workshop.txt`, install it with `%pip`,
# MAGIC and `%run ./_verify_environment` to confirm the pinned versions
# MAGIC loaded. The verify helper is not meant to be run on its own.
# MAGIC
# MAGIC **Locally:** `pip install .` from the repo root. Set `OPENAI_API_KEY`.
# MAGIC Start MLflow: `mlflow server --backend-store-uri sqlite:///mlflow_workshop.db --port 5000`
# MAGIC
# MAGIC See the [README](https://github.com/debu-sinha/mlflow-eval-workshop) for full setup instructions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time Budget
# MAGIC
# MAGIC | Module | Duration | Cumulative |
# MAGIC |--------|----------|-----------|
# MAGIC | 1. Evaluation Ecosystem | 20 min | 20 min |
# MAGIC | 2. Production Infrastructure | 8 min | 28 min |
# MAGIC | 3. Comparing Runs | 12 min | 40 min |
# MAGIC | 4. Evaluation Gate | 15 min | 55 min |
# MAGIC | Bonus: uv Dependencies | 5 min | 60 min |
# MAGIC
# MAGIC If running behind, skip Module 2 (merge key points into Module 1 verbally).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's go!
# MAGIC
# MAGIC Open **`01_mlflow_evaluation_ecosystem`** to start the workshop.
