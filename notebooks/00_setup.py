# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Setup
# MAGIC
# MAGIC **Optional pre-workshop step.** Each notebook has its own `%pip install`
# MAGIC cell that runs automatically on Databricks. This notebook is useful if you
# MAGIC want to install everything once before the session starts, or if you want
# MAGIC to verify that all imports work before going live.
# MAGIC
# MAGIC **Time:** 2-3 minutes
# MAGIC
# MAGIC **Local users:** Use `pip install -e .` or `uv sync` from the repo root instead.

# COMMAND ----------

# MAGIC %pip install mlflow[genai] arize-phoenix-evals trulens trulens-providers-litellm numpy scikit-learn databricks-agents -q

# COMMAND ----------

# MAGIC %md
# MAGIC Restart the Python environment so all packages are available.

# COMMAND ----------

# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify installation

# COMMAND ----------

import mlflow
from mlflow.genai import make_judge  # noqa: F401
from mlflow.genai.scorers import (  # noqa: F401
    Correctness,
    RelevanceToQuery,
    Safety,
    scorer,
)
from mlflow.genai.scorers.phoenix import Hallucination  # noqa: F401
from mlflow.genai.scorers.trulens import Groundedness  # noqa: F401

print(f"MLflow version: {mlflow.__version__}")
print("Core scorers: OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: Guardrails AI setup
# MAGIC
# MAGIC DetectPII needs outbound internet to install Hub validators.
# MAGIC Databricks Free Edition restricts this. Skip if it fails.
# MAGIC The must-succeed path is built-ins + Phoenix + TruLens.

# COMMAND ----------

import os
import subprocess

_rc_path = os.path.expanduser("~/.guardrailsrc")
if not os.path.exists(_rc_path):
    ON_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ
    if ON_DATABRICKS:
        try:
            _token = dbutils.secrets.get(scope="guardrails-hub", key="api-token")  # noqa: F821
        except Exception:
            _token = os.environ.get("GUARDRAILS_API_KEY", "")
    else:
        _token = os.environ.get("GUARDRAILS_API_KEY", "")

    if _token:
        with open(_rc_path, "w") as f:
            f.write(f"token={_token}\n")
        print("Guardrails Hub token configured")
    else:
        print("No Guardrails Hub token. DetectPII will be skipped in Module 1.")

try:
    import nltk

    nltk.download("punkt_tab", quiet=True)
except ImportError:
    pass

try:
    _result = subprocess.run(
        [
            "guardrails",
            "hub",
            "install",
            "hub://guardrails/detect_pii",
            "--quiet",
            "--no-install-local-models",
        ],
        capture_output=True,
        text=True,
    )
    if _result.returncode == 0:
        print("Guardrails DetectPII: installed")
    else:
        print(f"Guardrails optional, skipping: {_result.stderr.strip()[:120]}")
except FileNotFoundError:
    print("Guardrails CLI not installed. DetectPII will be skipped.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup complete
# MAGIC
# MAGIC You can now run Modules 1-4 without any per-notebook install steps.
# MAGIC
# MAGIC **Next:** Open `01_mlflow_evaluation_ecosystem` and start the workshop.
