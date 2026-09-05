# Databricks notebook source
# MAGIC %md
# MAGIC # Would you ship this answer?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC A customer bought an item 45 days ago. The new support assistant retrieves an old policy that offers cash refunds within 90 days. Our current policy allows a full refund within 30 days and store credit after 30 days. A defective item goes to support. The assistant cannot approve or process a transaction.
# MAGIC
# MAGIC **Watch:** Vote on the answer before inspecting its scores. Fluency and speed are not evidence of a correct refund.
# MAGIC
# MAGIC **Build:** Run the prepared checkpoint below and inspect the first response.
# MAGIC
# MAGIC **Extend:** Change the purchase age to 30 days. Predict whether the decision should change.
# MAGIC

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

# COMMAND ----------

# Run this cell after completing the setup in README.md.
# It calls the configured provider. Failures stop the checkpoint.
import json
from west_workshop import run_checkpoint

summary = run_checkpoint(0)
print(json.dumps(summary, indent=2, sort_keys=True))
if summary.get("status") != "passed":
    raise RuntimeError("Checkpoint incomplete. Resolve the readiness or validation issue in the summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the decision
# MAGIC
# MAGIC ![Local MLflow example](https://raw.githubusercontent.com/debu-sinha/mlflow-eval-workshop/west-2026/notebooks/images/west/00-answer.png)
# MAGIC
# MAGIC The recorded 45-day request received a full-refund answer under the stale policy.
# MAGIC
# MAGIC Can a polished response still create a refund promise we cannot honor?
# MAGIC
# MAGIC Next, open 01_trace_the_failure and find the evidence that reached the application.
