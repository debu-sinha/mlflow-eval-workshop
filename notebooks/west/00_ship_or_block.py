# Databricks notebook source
# MAGIC %md
# MAGIC # Would you ship this answer?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC ![A recorded release report](https://raw.githubusercontent.com/debu-sinha/mlflow-eval-workshop/main/notebooks/images/west/release-report.png)
# MAGIC
# MAGIC Start with the finished report in **04_compare_and_gate**. It compares the customer's answers, the scores, and the release decisions. This image is one saved local run; your own run may differ.
# MAGIC
# MAGIC ## The customer's request
# MAGIC
# MAGIC A customer asks for a full refund 45 days after buying an item. Our fictional Northstar Shop policy says:
# MAGIC
# MAGIC | Situation | Correct guidance |
# MAGIC |---|---|
# MAGIC | Non-defective item, day 0 through day 30 | Full refund eligibility |
# MAGIC | Non-defective item, after day 30 | Store credit eligibility |
# MAGIC | Defective item, any purchase age | Support review |
# MAGIC
# MAGIC The assistant explains eligibility. It cannot approve or process a transaction.
# MAGIC
# MAGIC The candidate has an outdated 90-day refund policy. Before reading its score, decide whether its answer follows the current policy.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up this notebook
# MAGIC Use the setup for your platform in the [README](https://github.com/debu-sinha/mlflow-eval-workshop#readme). In Databricks Free Edition, select **Standard environment 5**, add `-r /Workspace/Users/<your-user>/mlflow-eval-workshop/requirements-workshop.txt` using your Git folder's actual path, and click **Apply**. Wait for the Python restart before running the cells.
# MAGIC
# MAGIC The next cell finds the repository and configures this notebook's model and experiment. In Databricks it uses your workspace identity. Locally it keeps your terminal settings. Each notebook runs independently.

# COMMAND ----------

# Locate the cloned repository from a local script or a Databricks Git folder.
from pathlib import Path
import sys

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_root = next((p for p in (_start, *_start.parents) if (p / "west_workshop").is_dir()), None)
if _root is None:
    raise RuntimeError("Open this notebook inside the workshop Git folder.")
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from west_workshop.notebook_setup import configure_notebook, show_result
configure_notebook()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the opening case
# MAGIC
# MAGIC This cell makes one application request and checks the declared eligibility against the reference label. It does not call an LLM judge.
# MAGIC
# MAGIC **Exercise status: passed** means the exercise completed and caught the intended policy failure. **Decision: block** means the candidate answer should be stopped. These are different outcomes.

# COMMAND ----------

from west_workshop import run_checkpoint

summary = run_checkpoint(0)
show_result(summary)
if summary.get("status") != "passed":
    raise RuntimeError("This exercise did not complete. Read the setup or run issue above and the saved summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the answer and the check
# MAGIC
# MAGIC A *reference label* is the expected decision written into the test case. `policy_decision` checks the answer's declared eligibility against that label: 1 means it matches; 0 means it does not.
# MAGIC
# MAGIC Read the actual response below. Which sentence creates a promise the current policy cannot support?

# COMMAND ----------

row = summary["evaluations"][0]["rows"][0]
print("Customer:", row["inputs"]["question"])
print("Assistant:", row["output"])
print("Expected eligibility:", row["expectations"]["expected_decision"])
print("Policy decision score:", row["scores"]["policy_decision"])
print("Trace ID:", row["trace_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Take it further
# MAGIC
# MAGIC On day 30, a non-defective purchase is still eligible for a full refund. On day 31, it is eligible for store credit. Find those named cases in `west_workshop/data.py` and explain why both belong in a test set.
# MAGIC
# MAGIC Next, open **01_trace_the_failure**. We'll inspect the policy the assistant actually received.
