# Databricks notebook source
# MAGIC %md
# MAGIC # Can another evaluator add useful evidence?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC Phoenix and TruLens evaluate a fresh support reply through their real MLflow integrations. Inspect what each metric measures before adding it to a release gate. Their scores are additional evidence; they do not replace the policy checks.
# MAGIC
# MAGIC **Setup:** Locally, install the `ecosystem` extra. In Databricks Free Edition, select Standard environment 5 and use `-r /Workspace/Users/<your-user>/mlflow-eval-workshop/requirements-ecosystem.txt` in the Environment panel. Click Apply and wait for installation. The same notebook-native authentication is used; no OpenAI key is needed for the Databricks route.

# COMMAND ----------

from pathlib import Path
import sys

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_root = next((p for p in (_start, *_start.parents) if (p / "west_workshop").is_dir()), None)
if _root is None:
    raise RuntimeError("Open this notebook inside the workshop Git folder.")
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from west_workshop.notebook_setup import configure_notebook
configure_notebook()

# COMMAND ----------

import os
from scripts.verify_west_environment import verify_environment

verify_environment(ecosystem=True, databricks=os.environ.get("WORKSHOP_PROVIDER") == "databricks")

# COMMAND ----------

import json
from west_workshop import run_integrations

summary = run_integrations()
print(json.dumps(summary, indent=2, sort_keys=True))
if summary.get("status") != "passed":
    raise RuntimeError("Integration incomplete. Inspect package compatibility and provider access.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the decision
# MAGIC
# MAGIC Inspect the response, evaluator rationales, and individual scores in MLflow. A complete run means both evaluators returned usable evidence. It does not mean that their different quality dimensions are interchangeable or that this one reply establishes general accuracy.
