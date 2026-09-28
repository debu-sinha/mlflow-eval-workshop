# Databricks notebook source
# MAGIC %md
# MAGIC # What happens after release?
# MAGIC
# MAGIC ODSC AI West 2026 · Debu Sinha · Prologue · Trace · Score · Trust · Decide · **Learn**
# MAGIC
# MAGIC A release decision covers the cases we tested. Production brings the ones we did not. The evaluation keeps improving only if reviewed problems become the next tests.
# MAGIC
# MAGIC This notebook makes one fresh request, for a defective item bought 45 days ago, records its trace, and attaches a reviewer's note to that exact trace. The correct eligibility is **support_review**, because defective items go to review at any age.
# MAGIC
# MAGIC The note is written for the workshop. It is not a report from a real customer, and it does not claim that this new answer failed. The code attaches feedback to one trace. It does not start monitoring or schedule any job.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up
# MAGIC
# MAGIC **Databricks:** in the **Environment** side panel, open **Base environment**, choose **More**, and select **Standard v5**. Add `-r /Workspace/Users/<your-user>/mlflow-eval-workshop/requirements-workshop.txt` with your Git folder's path, click **Apply**, and wait for Python to restart.
# MAGIC
# MAGIC **Locally:** run this file from the repository with `uv run --locked python notebooks/west/05_production_feedback.py`, in the terminal where you loaded your API key.
# MAGIC
# MAGIC The next cell finds the repository and configures this notebook's model and experiment. Every notebook sets itself up, so each one runs on its own.

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
# MAGIC ## Answer, score, and attach the review
# MAGIC
# MAGIC This cell calls the repaired assistant once, scores the answer with every check and the judge, and then uses `mlflow.log_feedback` to attach the reviewer's note to the trace. The note's source is marked as a human reviewer, and its metadata marks it as a workshop example.

# COMMAND ----------

from west_workshop import run_checkpoint

summary = run_checkpoint(5)
show_result(summary)
if summary.get("status") != "passed":
    raise RuntimeError("This exercise did not complete. Read the setup or run issue above and the saved summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the answer, then the note
# MAGIC
# MAGIC Understand the behavior before you interpret a flag attached to it. Then open the printed trace in MLflow and find the assessment named `workshop_authored_followup`, with the value `needs_human_followup`. Its rationale and provenance sit beside the model's actual answer.

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
# MAGIC A flag is a request for review, not a label. The reviewer may find that the assistant was right. Before a flagged trace becomes a test case, someone who owns it writes down:
# MAGIC
# MAGIC 1. The behavior to investigate and the relevant policy clause
# MAGIC 2. The expected answer under that policy
# MAGIC 3. A stable case name and the reason it adds coverage
# MAGIC
# MAGIC Add the approved case to a new version of the dataset, then evaluate the baseline and the candidate again on that same version. Adding every flagged answer automatically does not make an evaluation better.
# MAGIC
# MAGIC ## What MLflow adds after release
# MAGIC
# MAGIC - **Review queues and label schemas** (MLflow 3.14) route traces to reviewers and write their answers back onto the trace, ready for evaluation. See [review queues](https://mlflow.org/docs/latest/genai/assessments/review-queues/).
# MAGIC - **Automatic evaluation** runs registered LLM judges on traces as they arrive, at a sampling rate you choose. It supports LLM judges only, so rule checks like `policy_decision` stay in CI and in the application. In open source MLflow, the judges run through an AI Gateway endpoint. See [automatic evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/).
# MAGIC - **Agents** add a trajectory to evaluate. Tool-call scorers such as `ToolCallCorrectness`, session-level scorers such as `UserFrustration`, and `ConversationSimulator` extend this same loop to tools and multi-turn conversations.
# MAGIC
# MAGIC ## Close the loop
# MAGIC
# MAGIC We started with one fluent, wrong answer. We traced it to a stale document, built checks that catch it on every answer, tested the judge that grades it, and made a release decision that answers three questions. Now reviewed feedback becomes the next test, and the loop starts again.
# MAGIC
# MAGIC Before you leave, write down one risk in your own application, one case that would catch it, and one release rule you would enforce. For an optional look at other evaluators, open **06_optional_integrations**.
