# Databricks notebook source
# MAGIC %md
# MAGIC # Shared verification helper
# MAGIC
# MAGIC This notebook is a helper. Each module runs it via `%run` after the
# MAGIC module's pinned `%pip install` cell and `dbutils.library.restartPython()`.
# MAGIC It does not install any dependencies.
# MAGIC
# MAGIC If you land here directly and the checks fail, that is expected: on
# MAGIC Databricks Serverless, notebook-scoped libraries are scoped to the
# MAGIC notebook/session that installed them. Run a module's first two cells
# MAGIC or configure the Serverless Environment panel with
# MAGIC `requirements-workshop.txt`, then retry.

# COMMAND ----------

import importlib.metadata as md

_REQUIRED_PACKAGES = [
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

_missing = []

print("Workshop dependency versions:")
for _package in _REQUIRED_PACKAGES:
    try:
        print(f"  {_package}: {md.version(_package)}")
    except md.PackageNotFoundError:
        print(f"  {_package}: NOT INSTALLED")
        _missing.append(_package)

if _missing:
    raise RuntimeError(
        "Workshop dependencies are missing: "
        + ", ".join(_missing)
        + ". Run the module's first %pip install cell, or configure the "
        "Databricks Serverless Environment panel with requirements-workshop.txt, "
        "then retry."
    )

# COMMAND ----------

print("Verifying MLflow GenAI scorer imports...")

from mlflow.genai import make_judge  # noqa: F401
from mlflow.genai.scorers import (  # noqa: F401
    Correctness,
    RelevanceToQuery,
    Safety,
    scorer,
)
from mlflow.genai.scorers.phoenix import Hallucination  # noqa: F401
from mlflow.genai.scorers.trulens import Groundedness  # noqa: F401

print("Core scorers: OK")
print("Phoenix scorer: OK")
print("TruLens scorer: OK")
print()
print("READY. Continue to the next cell.")
