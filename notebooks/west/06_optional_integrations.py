# Databricks notebook source
# MAGIC %md
# MAGIC # Can another evaluator add evidence?
# MAGIC
# MAGIC ODSC AI West 2026 · Debu Sinha · Optional chapter
# MAGIC
# MAGIC This notebook sends a fresh 45-day question to the repaired assistant and scores the answer with two third-party evaluators that run inside MLflow:
# MAGIC
# MAGIC | Evaluator | The question it investigates |
# MAGIC |---|---|
# MAGIC | Phoenix Hallucination | Is the answer supported by the supplied policy? |
# MAGIC | TruLens Coherence | Does the answer read clearly and hang together? |
# MAGIC
# MAGIC These measure different things. A coherent answer can still contradict the policy, as the stale candidate showed. Read each evaluator's value and explanation before deciding how to use it.
# MAGIC
# MAGIC ## Install the optional dependencies
# MAGIC
# MAGIC **Locally:** run `uv sync --locked --python 3.12 --extra ecosystem`, then `uv run --locked --extra ecosystem python -m west_workshop --provider openai --integrations`.
# MAGIC
# MAGIC **Databricks:** in the **Environment** side panel, open **Base environment**, choose **More**, and select **Standard v5**. Add `-r /Workspace/Users/<your-user>/mlflow-eval-workshop/requirements-ecosystem.txt` with your Git folder's path, in place of the core requirements file. Click **Apply** and wait for Python to restart. The notebook uses your workspace identity and an accessible model endpoint, so no OpenAI key is needed.

# COMMAND ----------

# Locate the cloned repository from a local script or a Databricks Git folder.
from pathlib import Path
import importlib
import sys

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_root = next((p for p in (_start, *_start.parents) if (p / "west_workshop").is_dir()), None)
if _root is None:
    raise RuntimeError("Open this notebook inside the workshop Git folder.")
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Refresh cached paths after a Git folder update.
importlib.invalidate_caches()
from west_workshop.notebook_setup import configure_notebook, show_result
configure_notebook()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the environment
# MAGIC
# MAGIC This checks installed package versions and builds both evaluators before any model call. The integrations need the combined pins. Installing the newest Phoenix evaluation package over this environment breaks MLflow's adapter.

# COMMAND ----------

import os
from scripts.verify_west_environment import verify_environment

verify_environment(ecosystem=True, databricks=os.environ.get("WORKSHOP_PROVIDER") == "databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run both evaluators
# MAGIC
# MAGIC The application and both evaluators make real provider requests. A passed exercise means both returned usable evidence. It does not make their scores interchangeable, and one answer establishes nothing general.

# COMMAND ----------

from west_workshop import run_integrations

summary = run_integrations()
show_result(summary)
if summary.get("status") != "passed":
    raise RuntimeError("The integrations did not complete. Inspect the package check and saved summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the answer and each explanation
# MAGIC
# MAGIC Compare what the two evaluators actually assessed. Phoenix receives the current policy as context. Read each original value and explanation instead of assuming every evaluator uses the same scale.

# COMMAND ----------

row = summary["evaluations"][0]["rows"][0]
print("Customer:", row["inputs"]["question"])
print("Assistant:", row["output"])
print("Recorded numeric scores:", row["scores"])
for assessment in row["assessments"]:
    print("\nEvaluator:", assessment["name"])
    print("Feedback value:", assessment["value"])
    print("Explanation:", assessment["rationale"])
print("\nTrace ID:", row["trace_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decide what belongs in your evaluation
# MAGIC
# MAGIC Which evaluator adds evidence that your policy checks do not? What examples would prove it? Count the extra calls, the latency, and the disagreements someone must review before adding another judge to every request.
# MAGIC
# MAGIC This chapter leaves the release gate from chapter 4 unchanged.
