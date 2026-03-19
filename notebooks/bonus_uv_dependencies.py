# Databricks notebook source
# MAGIC %md
# MAGIC # Bonus: Locking Dependencies with uv
# MAGIC
# MAGIC **Objective:** Understand how MLflow uses uv lockfiles to pin every
# MAGIC dependency (including transitive ones) when logging a model.
# MAGIC
# MAGIC **Time:** 5 minutes
# MAGIC
# MAGIC This is a standalone bonus module. It does not depend on Modules 1-4.

# COMMAND ----------

# MAGIC %pip install --upgrade mlflow[genai] scikit-learn -q
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## How uv lockfiles work with MLflow
# MAGIC
# MAGIC When you log a model from a uv-managed project, MLflow reads your
# MAGIC `uv.lock` file and pins every dependency (including transitive ones)
# MAGIC to the exact version you tested with.
# MAGIC
# MAGIC Without uv, MLflow infers dependencies from the Python runtime. With uv,
# MAGIC MLflow gets exact pins from the lockfile. The result is a `requirements.txt`
# MAGIC inside the model artifact that precisely matches your development environment.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: Log a sklearn model

# COMMAND ----------

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

# MAGIC %md
# MAGIC ## Inspect captured requirements

# COMMAND ----------

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
        lines = [line for line in reqs.strip().split("\n") if line.strip()]
        print(f"Captured {len(lines)} requirements:")
        for line in lines[:15]:
            print(f"  {line}")
        if len(lines) > 15:
            print(f"  ... and {len(lines) - 15} more")
    else:
        print("No requirements.txt found in model artifact")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why this matters
# MAGIC
# MAGIC Without lockfile support, the model's `requirements.txt` contains versions
# MAGIC inferred from the Python runtime at log time. These are often loose pins
# MAGIC (e.g., `scikit-learn==1.4.0`) that do not capture transitive dependencies.
# MAGIC
# MAGIC With uv, every dependency in the lockfile is included with its exact version
# MAGIC and hash. When you serve the model in a different environment, `pip install`
# MAGIC recreates the exact same dependency tree.
# MAGIC
# MAGIC This closes the gap between "works on my machine" and "works in production."
