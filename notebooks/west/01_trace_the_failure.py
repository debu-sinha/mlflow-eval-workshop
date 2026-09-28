# Databricks notebook source
# MAGIC %md
# MAGIC # Where did the refund promise begin?
# MAGIC
# MAGIC ODSC AI West 2026 · Debu Sinha · Prologue · **Trace** · Score · Trust · Decide · Learn · Extend
# MAGIC
# MAGIC In the prologue, the candidate told a day-45 customer they qualify for a full refund. There are three suspects: the policy it retrieved, the prompt, or the model's handling of that evidence. A *trace* lets us check instead of guess. It records one request, and each *span* inside it records one step:
# MAGIC
# MAGIC | Span | What it records |
# MAGIC |---|---|
# MAGIC | `refund_assistant` | The whole request and the final answer |
# MAGIC | `retrieve_refund_policy` | The policy document the assistant received |
# MAGIC | `generate_support_answer` | The model call, its inputs, and its answer |
# MAGIC
# MAGIC The retriever here is deliberately tiny. It returns one of two policy documents, which keeps the cause easy to see. A production retriever is bigger, and the question stays the same: which source reached the model for this request?
# MAGIC
# MAGIC **Pick a suspect before you run the cell.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up
# MAGIC
# MAGIC **Databricks:** in the **Environment** side panel, open **Base environment**, choose **More**, and select **Standard v5**. Add `-r /Workspace/Users/<your-user>/mlflow-eval-workshop/requirements-workshop.txt` with your Git folder's path, click **Apply**, and wait for Python to restart.
# MAGIC
# MAGIC **Locally:** run this file from the repository with `uv run --locked python notebooks/west/01_trace_the_failure.py`, in the terminal where you loaded your API key.
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
# MAGIC ## Generate a trace, then score it again
# MAGIC
# MAGIC This cell asks the candidate the 45-day question once and checks its eligibility. Then it loads the stored trace and scores the saved answer again, this time with a format check. That replay reuses the recorded answer and makes no new application request.
# MAGIC
# MAGIC Scoring a stored trace is how you evaluate production traffic after the fact. Notice that a format check can pass while the refund decision is wrong, because the two checks answer different questions.

# COMMAND ----------

from west_workshop import run_checkpoint

summary = run_checkpoint(1)
show_result(summary)
if summary.get("status") != "passed":
    raise RuntimeError("This exercise did not complete. Read the setup or run issue above, and the saved summary if one was printed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find the source of the answer
# MAGIC
# MAGIC Read the retrieved text below. Then open the experiment in MLflow, select **Traces**, open the printed trace ID, and expand the retrieval and generation spans. In Databricks the experiment is `/Users/<your-user>/odsc-west-2026`, under **Experiments** in the sidebar. Locally it is `odsc-west-2026`.

# COMMAND ----------

row = summary["evaluations"][0]["rows"][0]
print("Trace ID:", row["trace_id"])
for span in row["spans"]:
    print("\nStep:", span["name"])
    if span["name"] == "retrieve_refund_policy":
        for document in span["outputs"]:
            print(document["page_content"])
    elif span["name"] == "generate_support_answer":
        print("Assistant:", span["outputs"])
print("\nStored trace replay:", summary["trace_evaluation_evidence"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the tracing code
# MAGIC
# MAGIC Three decorators create the trace and its spans. The predictor passes the retrieved document into the model call. Printing the source below makes no model calls.

# COMMAND ----------

import inspect
from west_workshop.runtime import make_predictor

print(inspect.getsource(make_predictor))

# COMMAND ----------

# MAGIC %md
# MAGIC ## The diagnosis
# MAGIC
# MAGIC The retrieval span says 90 days. The current policy says 30. The model explained the document it was given, and that document was stale. A bigger model would still receive the same stale document. The trace hands us one specific change to test: retrieve the current policy.
# MAGIC
# MAGIC One corrected answer proves little, though. Which other cases did the stale window break, and would we notice the next time? That needs checks that run on every answer.
# MAGIC
# MAGIC **Take it further.** Suppose a retriever returned two conflicting policies. What would you want the trace to record so a reviewer could tell which one the model used?
# MAGIC
# MAGIC Next, open **02_build_the_scorer_stack**: which checks should run on every answer?
