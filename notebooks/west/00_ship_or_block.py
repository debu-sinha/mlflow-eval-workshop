# Databricks notebook source
# MAGIC %md
# MAGIC # Would you ship this answer?
# MAGIC
# MAGIC ODSC AI West 2026 · Debu Sinha · **Prologue** · Trace · Score · Trust · Decide · Learn
# MAGIC
# MAGIC A customer bought a tote 45 days ago and asks for a full refund. Our fictional Northstar Shop has a short policy:
# MAGIC
# MAGIC | Situation | Correct guidance |
# MAGIC |---|---|
# MAGIC | Non-defective item, day 0 through day 30 | Full refund |
# MAGIC | Non-defective item, after day 30 | Store credit |
# MAGIC | Defective item, any age | Support review |
# MAGIC
# MAGIC The assistant explains eligibility. It can never approve or process a transaction.
# MAGIC
# MAGIC The rule is deliberately simple, and ordinary code could compute it. What we are testing is the generated advice. Does it follow the current policy, explain it correctly, and avoid claiming an action it cannot take?
# MAGIC
# MAGIC The candidate in this notebook retrieves an outdated policy with a 90-day window. **Before you run anything, decide what the right answer is on day 45.**
# MAGIC
# MAGIC ![A recorded release report](https://raw.githubusercontent.com/debu-sinha/mlflow-eval-workshop/main/notebooks/images/west/release-report.png)
# MAGIC
# MAGIC This is where the story ends: a release report that blocks one version and ships another. Chapter 4 builds it from your own run. The chapters in between explain every number on it.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up
# MAGIC
# MAGIC **Databricks:** in the **Environment** side panel, open **Base environment**, choose **More**, and select **Standard v5**. Add `-r /Workspace/Users/<your-user>/mlflow-eval-workshop/requirements-workshop.txt` with your Git folder's path, click **Apply**, and wait for Python to restart.
# MAGIC
# MAGIC **Locally:** run this file from the repository with `uv run --locked python notebooks/west/00_ship_or_block.py`, in the terminal where you loaded your API key.
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
# MAGIC ## Ask the candidate
# MAGIC
# MAGIC This cell sends the 45-day question to the stale-policy candidate once, then checks the declared eligibility against the reference label. There is no judge yet, only one rule you can read.
# MAGIC
# MAGIC Two lines matter in the output. **Exercise status: passed** means the exercise ran and caught what it was built to catch. **Decision: block** is the verdict on the candidate. Keep them apart. A passing exercise can block a bad release.

# COMMAND ----------

from west_workshop import run_checkpoint

summary = run_checkpoint(0)
show_result(summary)
if summary.get("status") != "passed":
    raise RuntimeError("This exercise did not complete. Read the setup or run issue above and the saved summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the answer before the score
# MAGIC
# MAGIC A *reference label* is the decision a reviewer wrote into the test case. `policy_decision` scores 1 when the answer's declared eligibility matches that label and 0 when it does not.
# MAGIC
# MAGIC Read the assistant's words first. Which sentence makes a promise the current policy cannot keep?

# COMMAND ----------

row = summary["evaluations"][0]["rows"][0]
print("Customer:", row["inputs"]["question"])
print("Assistant:", row["output"])
print("Expected eligibility:", row["expectations"]["expected_decision"])
print("Policy decision score:", row["scores"]["policy_decision"])
print("Trace ID:", row["trace_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## What we know, and what we don't
# MAGIC
# MAGIC We know the candidate offered a day-45 customer a full refund. We do not know why. The cause could be the model, the prompt, or the information the model received. Guessing is expensive, so the next chapter looks at what actually happened inside the request.
# MAGIC
# MAGIC **Take it further.** Day 30 is still a full refund and day 31 is store credit. Find both cases in `west_workshop/data.py` and explain why a test set needs both.
# MAGIC
# MAGIC Next, open **01_trace_the_failure**: where did the refund promise begin?
