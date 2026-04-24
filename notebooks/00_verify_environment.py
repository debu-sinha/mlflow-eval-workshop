# Databricks notebook source
# MAGIC %md
# MAGIC # Verify Workshop Environment
# MAGIC
# MAGIC **This notebook does not install dependencies.**
# MAGIC
# MAGIC Use it after either:
# MAGIC
# MAGIC 1. configuring the Databricks Serverless Environment side panel to
# MAGIC    point at `requirements-workshop.txt`, or
# MAGIC 2. running the first install cell in the notebook you plan to use.
# MAGIC
# MAGIC On Databricks Serverless, notebook-scoped installs are scoped to the
# MAGIC current notebook/session, so this verification notebook does not
# MAGIC prepare the other modules. Each module has its own install cell.
# MAGIC
# MAGIC **Time:** 1 minute.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pinned package versions
# MAGIC
# MAGIC If a package shows `NOT INSTALLED`, go back to the Serverless Environment
# MAGIC panel or the install cell in the module you want to run.

# COMMAND ----------

import importlib.metadata as md

_packages = [
    "mlflow",
    "arize-phoenix-evals",
    "trulens",
    "trulens-providers-litellm",
    "litellm",
    "databricks-agents",
    "openai",
    "numpy",
    "scikit-learn",
    "nltk",
]

for package in _packages:
    try:
        print(f"{package}: {md.version(package)}")
    except md.PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Core scorer imports

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
# MAGIC ## Optional: Guardrails AI setup
# MAGIC
# MAGIC DetectPII needs outbound internet to install Hub validators.
# MAGIC Databricks Free Edition restricts this. Skip if it fails.
# MAGIC The must-succeed path is built-ins + Phoenix + TruLens.

# COMMAND ----------

import os
import shutil
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

if shutil.which("guardrails") is None:
    print("Guardrails CLI not installed. Skipping optional DetectPII validator setup.")
else:
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
# MAGIC ### Verification complete
# MAGIC
# MAGIC If every package showed a version and the scorer imports succeeded, the
# MAGIC current notebook environment is ready. Remember that this environment
# MAGIC does not carry over to the other modules on Serverless: each module has
# MAGIC its own install cell at the top.
# MAGIC
# MAGIC **Next:** Open `01_mlflow_evaluation_ecosystem` and start the workshop.
