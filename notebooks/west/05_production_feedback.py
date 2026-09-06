# Databricks notebook source
# MAGIC %md
# MAGIC # What do we do with feedback after release?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC A release decision covers the examples we tested. This notebook makes a fresh model request for a defective item bought 45 days ago, records its trace, and attaches an authored follow-up note.
# MAGIC
# MAGIC The correct eligibility is **support_review**. Defective items follow that route regardless of purchase age. The note demonstrates a review workflow; it does not claim that a real customer reported a failure or that this new model answer necessarily failed.
# MAGIC
# MAGIC The code attaches feedback to one trace. It does not start continuous monitoring or schedule evaluation jobs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up this notebook
# MAGIC Use the setup for your platform in the [README](https://github.com/debu-sinha/mlflow-eval-workshop#readme). In Databricks Free Edition, select **Standard environment 5**, add `-r /Workspace/Users/<your-user>/mlflow-eval-workshop/requirements-workshop.txt` using your Git folder's actual path, and click **Apply**. Wait for the Python restart before running the cells.
# MAGIC
# MAGIC The next cell finds the repository and configures this notebook's model and experiment. In Databricks it uses your workspace identity. Locally it keeps your terminal settings. Each notebook runs independently.

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
# MAGIC ## Generate the trace and attach feedback
# MAGIC
# MAGIC This cell calls the application and scorers, then uses `mlflow.log_feedback` to attach a note to the generated trace. The note is explicitly marked as a workshop example.

# COMMAND ----------

from west_workshop import run_checkpoint

summary = run_checkpoint(5)
show_result(summary)
if summary.get("status") != "passed":
    raise RuntimeError("This exercise did not complete. Read the setup or run issue above and the saved summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect the answer and the review note
# MAGIC
# MAGIC Open the printed trace in MLflow and find the assessment named `workshop_authored_followup`. Its value is `needs_human_followup`. Read its rationale and provenance alongside the actual model response.

# COMMAND ----------

row = summary["evaluations"][0]["rows"][0]
print("Customer:", row["inputs"]["question"])
print("Assistant:", row["output"])
print("Expected eligibility:", row["expectations"]["expected_decision"])
print("Scores:", row["scores"])
print("Feedback trace ID:", summary["feedback_trace_id"])
print("Feedback assessment ID:", summary["feedback_assessment_id"])
print("Feedback provenance:", summary["feedback_provenance"])
print("Continuous evaluation started:", summary["automatic_evaluation_started"])
print("Scheduled evaluation started:", summary["scheduled_evaluation_started"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Turn a review into a test
# MAGIC
# MAGIC Write down the behavior to investigate, who should review it, the expected answer under the policy, and a stable case name. Add an approved example to a new dataset version, then evaluate both baseline and candidate on that same version.
# MAGIC
# MAGIC Avoid adding every flagged response automatically. A flag is a request for review, and the reviewer may find that the application was correct.
# MAGIC
# MAGIC Finish by naming one risk, one example, and one release rule you can use in your own application. For the optional evaluator comparison, continue to **06_optional_integrations**.
