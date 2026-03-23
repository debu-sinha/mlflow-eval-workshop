# Databricks notebook source
# MAGIC %md
# MAGIC # Module 2: Production Evaluation Infrastructure
# MAGIC
# MAGIC **Objective:** Control LLM judge parameters for deterministic scoring,
# MAGIC manage scorer concurrency for rate-limited APIs, and understand what the
# MAGIC MLflow UI shows for evaluation runs.
# MAGIC
# MAGIC **Time:** 8 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC > **Prerequisites:** Run `00_setup` first (or `pip install -e .` locally).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os

ON_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

if ON_DATABRICKS:
    # Free Edition: "databricks:/databricks-gpt-oss-120b"
    # Enterprise:   "databricks:/databricks-gpt-5-4" or other premium models
    _default_model = "databricks:/databricks-gpt-oss-120b"
    JUDGE_MODEL = os.environ.get("WORKSHOP_JUDGE_MODEL", _default_model)
    print(f"Running on Databricks. Judge model: {JUDGE_MODEL}")
else:
    JUDGE_MODEL = "openai:/gpt-4o-mini"
    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY to run locally"
    print(f"Running locally. Judge model: {JUDGE_MODEL}")

# COMMAND ----------

import mlflow

if ON_DATABRICKS:
    _user = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .userName()
        .get()
    )  # noqa: F821
    mlflow.set_experiment(f"/Users/{_user}/odsc-workshop-m2")
else:
    mlflow.set_tracking_uri("sqlite:///mlflow_workshop.db")
    mlflow.set_experiment("odsc-eval-workshop")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Controlling judge parameters (3 min)
# MAGIC
# MAGIC LLM-based scorers use a model as a judge. The judge's temperature affects
# MAGIC scoring consistency. Lower temperature means more deterministic evaluations.
# MAGIC
# MAGIC `mlflow.genai.make_judge` lets you create a custom judge with full control
# MAGIC over inference parameters.
# MAGIC

# COMMAND ----------

eval_data = [
    {
        "inputs": {"question": "What is gradient descent?"},
        "outputs": (
            "Gradient descent is an optimization algorithm that iteratively adjusts "
            "parameters by moving in the direction of steepest decrease of the loss function."
        ),
        "expectations": {
            "expected_response": (
                "Gradient descent minimizes a loss function by iteratively updating "
                "parameters in the direction of the negative gradient."
            )
        },
    },
    {
        "inputs": {"question": "Explain batch normalization."},
        "outputs": (
            "Batch normalization normalizes layer inputs to have zero mean and unit "
            "variance within each mini-batch, stabilizing training and allowing "
            "higher learning rates."
        ),
        "expectations": {
            "expected_response": (
                "Batch normalization normalizes activations within a mini-batch "
                "to stabilize and accelerate training."
            )
        },
    },
    {
        "inputs": {"question": "What is a transformer model?"},
        "outputs": (
            "A transformer is a neural network architecture based on self-attention "
            "mechanisms. It processes all tokens in parallel rather than sequentially, "
            "which makes it much faster to train than RNNs."
        ),
        "expectations": {
            "expected_response": (
                "Transformers use self-attention to process sequences in parallel, "
                "replacing recurrent architectures for most NLP tasks."
            )
        },
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Default judge (model defaults, typically temperature=1.0)

# COMMAND ----------

from mlflow.genai.scorers import Correctness

results_default = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[Correctness()],
)

print("Default temperature results:")
for name, value in results_default.metrics.items():
    print(f"  {name}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Custom judge with temperature=0.0

# COMMAND ----------

deterministic_judge = mlflow.genai.make_judge(
    name="correctness_deterministic",
    model=JUDGE_MODEL,
    instructions=(
        "Evaluate whether the response is factually correct compared to "
        "the expected answer. Return 'yes' if correct, 'no' if incorrect.\n\n"
        "Response: {{ outputs }}\n"
        "Expected: {{ expectations }}"
    ),
    inference_params={"temperature": 0.0, "max_tokens": 50},
)

results_deterministic = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[deterministic_judge],
)

print("Deterministic judge results (temperature=0.0):")
for name, value in results_deterministic.metrics.items():
    print(f"  {name}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC Run the deterministic judge twice on the same data. With `temperature=0.0`,
# MAGIC the results should be identical across runs. This matters when you need
# MAGIC reproducible evaluation baselines.

# COMMAND ----------

results_deterministic_v2 = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[deterministic_judge],
)

print("Second run (same data, same judge):")
for name, value in results_deterministic_v2.metrics.items():
    print(f"  {name}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Managing scorer concurrency (2 min)
# MAGIC
# MAGIC When running multiple scorers that call external LLM APIs, you can hit
# MAGIC rate limits. The `MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS` environment variable
# MAGIC controls how many scorers run in parallel.

# COMMAND ----------

from mlflow.genai.scorers.phoenix import Hallucination

# Limit to 2 concurrent scorer workers (default is 10)
os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = "2"

results_limited = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[
        Correctness(),
        Hallucination(model=JUDGE_MODEL),
    ],
)

print(f"Ran {len(results_limited.metrics)} metrics with 2 concurrent workers:")
for name, value in results_limited.metrics.items():
    print(f"  {name}: {value}")

# Reset to default
del os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"]

# COMMAND ----------

# MAGIC %md
# MAGIC Setting `MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS=2` means at most 2 scorer
# MAGIC invocations run at the same time. Use this when your LLM API has a low
# MAGIC rate limit or when running on a shared endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 MLflow UI walkthrough (3 min)
# MAGIC
# MAGIC Open the MLflow Experiment page for this notebook. Here is what to look for.
# MAGIC
# MAGIC ### Evaluation runs
# MAGIC
# MAGIC Each `mlflow.genai.evaluate()` call creates a run. The runs from this module
# MAGIC show up in the experiment's run list. Click any run to see details.
# MAGIC
# MAGIC ### Traces tab
# MAGIC
# MAGIC The Traces tab shows every scorer invocation as a trace. For LLM-based
# MAGIC scorers (Correctness, Hallucination), you can see:
# MAGIC - The prompt sent to the judge model
# MAGIC - The judge's raw response
# MAGIC - How the response was parsed into a score
# MAGIC
# MAGIC This is critical for debugging when a scorer gives unexpected results. You
# MAGIC can see exactly what the judge was asked and what it said.
# MAGIC
# MAGIC ### Assessments pane
# MAGIC
# MAGIC Click any sample row in the evaluation results. The Assessments pane shows:
# MAGIC - The scorer name and its output value (yes/no for binary scorers)
# MAGIC - The rationale the judge provided
# MAGIC - Metadata like the model used and the framework (Phoenix, Guardrails, etc.)
# MAGIC
# MAGIC ### Per-sample scores
# MAGIC
# MAGIC The evaluation results table shows one row per sample, with columns for each
# MAGIC scorer. This is where you spot which specific samples failed. In Module 1,
# MAGIC the "Berlin" sample should show a correctness failure. In this module, all
# MAGIC samples should pass since the outputs are factually correct.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Appendix: Design context for production evaluation features
# MAGIC
# MAGIC The production features in this module address problems that come up when
# MAGIC evaluation moves from notebooks to pipelines:
# MAGIC
# MAGIC - **`inference_params`** solves evaluation reproducibility. Before this existed,
# MAGIC   judge temperature defaulted to the provider's default (usually 1.0), so scores
# MAGIC   varied across runs on identical data.
# MAGIC   [PR #19152](https://github.com/mlflow/mlflow/pull/19152)
# MAGIC
# MAGIC - **`MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS`** solves rate limiting under concurrent
# MAGIC   scoring. Four LLM-based scorers on 100 samples is 400 API calls. Without a
# MAGIC   concurrency cap, most provider rate limits break.
# MAGIC   [PR #19248](https://github.com/mlflow/mlflow/pull/19248)
