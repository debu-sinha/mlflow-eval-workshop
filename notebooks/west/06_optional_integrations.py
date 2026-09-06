# Databricks notebook source
# MAGIC %md
# MAGIC # Can another evaluator add useful evidence?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC This optional notebook sends a fresh 45-day request to the repaired assistant. It evaluates the answer with two MLflow integrations:
# MAGIC
# MAGIC | Evaluator | Question to investigate |
# MAGIC |---|---|
# MAGIC | Phoenix Hallucination | Is the response supported by the supplied policy context? |
# MAGIC | TruLens Coherence | Does the response make sense and read coherently? |
# MAGIC
# MAGIC These are different dimensions. A coherent answer can still conflict with policy. Inspect each evaluator's label, score, and explanation before deciding how to use it.
# MAGIC
# MAGIC ## Install the optional dependencies
# MAGIC
# MAGIC Locally, use `uv sync --locked --python 3.12 --extra ecosystem`. In Databricks Free Edition, select **Standard environment 5** and apply `-r /Workspace/Users/<your-user>/mlflow-eval-workshop/requirements-ecosystem.txt`, replacing the path with your Git folder's actual path. Use this combined file in place of the core requirements file.
# MAGIC
# MAGIC Wait for installation and the Python restart. In Databricks this notebook uses your workspace identity and an accessible model endpoint. No OpenAI key is needed for that route.

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
# MAGIC This checks installed package versions before making model calls. The optional integrations require the combined pins; installing the newest Phoenix version over this environment can break MLflow's adapter.

# COMMAND ----------

import os
from scripts.verify_west_environment import verify_environment

verify_environment(ecosystem=True, databricks=os.environ.get("WORKSHOP_PROVIDER") == "databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run both evaluators
# MAGIC
# MAGIC The application and both evaluators make real provider requests. A passed exercise means both returned usable evidence. It does not mean their scores are interchangeable or that one answer establishes general accuracy.

# COMMAND ----------

from west_workshop import run_integrations

summary = run_integrations()
show_result(summary)
if summary.get("status") != "passed":
    raise RuntimeError("The integrations did not complete. Inspect the package check and saved summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the response and each explanation
# MAGIC
# MAGIC Compare what the two evaluators actually assessed. For the Phoenix check, the current policy is supplied as context. Read the original feedback value and explanation rather than assuming every evaluator uses the same scale.

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
# MAGIC ## Choose what belongs in your evaluation
# MAGIC
# MAGIC Which evaluator adds evidence your policy checks do not already provide? What examples would you use to test that claim? Consider the extra calls, latency, and disagreement review before adding another judge to every request.
# MAGIC
# MAGIC The core release gate remains the one you inspected in checkpoint 4. This exercise does not change it.
