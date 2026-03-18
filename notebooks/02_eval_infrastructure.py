# Databricks notebook source
# MAGIC %md
# MAGIC # Module 2: Evaluation Infrastructure for Production
# MAGIC
# MAGIC **Objective:** Configure LLM judge parameters, manage scorer concurrency,
# MAGIC and lock model dependencies with uv for reproducible evaluations.
# MAGIC
# MAGIC **Tools:** MLflow, OpenAI API, uv package manager
# MAGIC
# MAGIC **Time:** 15 minutes

# COMMAND ----------

# MAGIC %pip install --upgrade mlflow[genai] arize-phoenix scikit-learn -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os

ON_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

if ON_DATABRICKS:
    JUDGE_MODEL = "databricks-claude-sonnet-4"
    print(f"Running on Databricks. Using model: {JUDGE_MODEL}")
else:
    JUDGE_MODEL = "openai:/gpt-4o-mini"
    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY to run locally"
    print(f"Running locally. Using model: {JUDGE_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Controlling LLM judge parameters
# MAGIC
# MAGIC LLM-based scorers use a model as a judge. The judge's temperature and
# MAGIC other parameters affect scoring consistency. Lower temperature produces
# MAGIC more deterministic evaluations.

# COMMAND ----------

import mlflow

if not ON_DATABRICKS:
    mlflow.set_experiment("odsc-eval-workshop-module-2-infrastructure")

eval_data = [
    {
        "inputs": {"question": "What is gradient descent?"},
        "outputs": "Gradient descent is an optimization algorithm that iteratively adjusts parameters by moving in the direction of steepest decrease of the loss function.",
        "expectations": {"expected_response": "Gradient descent minimizes a loss function by iteratively updating parameters in the direction of the negative gradient."},
    },
    {
        "inputs": {"question": "Explain batch normalization."},
        "outputs": "Batch normalization normalizes layer inputs to have zero mean and unit variance within each mini-batch, stabilizing training and allowing higher learning rates.",
        "expectations": {"expected_response": "Batch normalization normalizes activations within a mini-batch to stabilize and accelerate training."},
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run with default temperature (non-deterministic)

# COMMAND ----------

from mlflow.genai.scorers.phoenix import Hallucination

# Default temperature (model default, usually 1.0)
results_default = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[Hallucination(model=JUDGE_MODEL)],
)

print("Default temperature results:")
for metric_name, value in results_default.metrics.items():
    print(f"  {metric_name}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run with low temperature (more deterministic)
# MAGIC
# MAGIC Use `inference_params` on the judge to control temperature, top_p, etc.

# COMMAND ----------

judge = mlflow.genai.make_judge(
    name="correctness",
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
    scorers=[judge],
)

print("Low temperature results:")
for metric_name, value in results_deterministic.metrics.items():
    print(f"  {metric_name}: {value}")
if not results_deterministic.metrics:
    print("  (Check the Traces tab in MLflow UI for per-sample judge outputs)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Managing scorer concurrency
# MAGIC
# MAGIC When running multiple scorers that call external LLM APIs, you can hit
# MAGIC rate limits. Control concurrency with `MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS`.

# COMMAND ----------

import os

# Limit to 2 concurrent scorer workers (default is unbounded)
os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = "2"

from mlflow.genai.scorers.phoenix import Hallucination, Relevance

results_limited = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[
        Hallucination(model=JUDGE_MODEL),
        Relevance(model=JUDGE_MODEL),
    ],
)

print(f"Ran {len(results_limited.metrics)} metrics with 2 concurrent workers")
for metric_name, value in results_limited.metrics.items():
    print(f"  {metric_name}: {value}")

# Reset to default
del os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Locking dependencies with uv
# MAGIC
# MAGIC When you log a model from a uv-managed project, MLflow reads your
# MAGIC `uv.lock` file and pins every dependency (including transitive ones)
# MAGIC to the exact version you tested with.
# MAGIC
# MAGIC This works with all model flavors (sklearn, pytorch, transformers, pyfunc).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Log a sklearn model with uv auto-detection

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Train a simple model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=200).fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2%}")

# COMMAND ----------

# Log the model. If uv.lock exists in the working directory,
# MLflow auto-detects it and exports pinned requirements.
with mlflow.start_run(run_name="sklearn-with-uv") as run:
    model_info = mlflow.sklearn.log_model(model, name="iris-model")
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model logged: {model_info.model_uri}")

# COMMAND ----------

# Check what requirements were captured
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run.info.run_id}/iris-model",
        dst_path=tmp,
    )
    reqs_path = Path(local_path) / "requirements.txt"
    if reqs_path.exists():
        reqs = reqs_path.read_text()
        print("Captured requirements:")
        for line in reqs.strip().split("\n")[:15]:
            print(f"  {line}")
        total = len([l for l in reqs.strip().split("\n") if l.strip()])
        if total > 15:
            print(f"  ... and {total - 15} more")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key takeaway
# MAGIC
# MAGIC With uv, your model's `requirements.txt` contains exact pinned versions
# MAGIC from the lockfile, not versions inferred from the Python runtime. This means
# MAGIC the environment used for serving will match the environment used for training.
