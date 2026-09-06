# Databricks notebook source
# MAGIC %md
# MAGIC # Where did the refund promise begin?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC A *trace* records one application request. A *span* records one step within that request. Our trace has three named steps:
# MAGIC
# MAGIC | Span | What it records |
# MAGIC |---|---|
# MAGIC | `refund_assistant` | The full request and response |
# MAGIC | `retrieve_refund_policy` | The policy supplied to the assistant |
# MAGIC | `generate_support_answer` | The model inputs and answer |
# MAGIC
# MAGIC This example uses a small retriever that returns a policy document selected by the application variant. It does not use a vector database. Keeping retrieval simple makes the failure easy to inspect.
# MAGIC
# MAGIC Choose a likely cause before running: the retrieved policy, the prompt, or the model's handling of the evidence.

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
# MAGIC ## Generate and inspect a trace
# MAGIC
# MAGIC This cell creates a fresh answer to the 45-day request, checks eligibility, then fetches the stored trace and evaluates its answer format. The replay uses the saved response; it does not generate a second application answer.
# MAGIC
# MAGIC A format check can pass even when the refund decision is wrong. The two checks answer different questions.

# COMMAND ----------

from west_workshop import run_checkpoint

summary = run_checkpoint(1)
show_result(summary)
if summary.get("status") != "passed":
    raise RuntimeError("This exercise did not complete. Read the setup or run issue above and the saved summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find the source of the answer
# MAGIC
# MAGIC Read the actual retrieved text below. Then open your experiment in **MLflow > Traces**, select the printed trace ID, and expand the retrieval and generation spans. In Databricks the experiment is normally `/Users/<your-user>/odsc-west-2026`; locally it is `odsc-west-2026`.

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
# MAGIC The decorators on the three functions create the trace and its spans. The predictor passes the retrieved document into the model call. Inspect the implementation below; printing it makes no model calls.

# COMMAND ----------

import inspect
from west_workshop.runtime import make_predictor

print(inspect.getsource(make_predictor))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the diagnosis
# MAGIC
# MAGIC Point to the 90-day window in the retrieval span. Compare it with the current 30-day policy. The stored trace gives us a specific change to test: retrieve the current document.
# MAGIC
# MAGIC If a future retriever returns two conflicting policies, what additional evidence would you want in the trace?
# MAGIC
# MAGIC Next, open **02_build_the_scorer_stack**.
