# Databricks notebook source
# MAGIC %md
# MAGIC # Where did the refund promise begin?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC A trace records what the application actually did. Find the customer question, the policy that was retrieved, and the model response. The same model can give different answers when the evidence changes. A trace supports that diagnosis without asking anyone to guess from the final answer.
# MAGIC
# MAGIC **Watch:** Choose the likely cause before opening the trace: retrieval, instructions, or the model response.
# MAGIC
# MAGIC **Build:** Run the checkpoint and open its trace in MLflow.
# MAGIC
# MAGIC **Extend:** Explain which span would change if the retriever found two conflicting policies.
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

summary = run_checkpoint(1)
print(json.dumps(summary, indent=2, sort_keys=True))
if summary.get("status") != "passed":
    raise RuntimeError("Checkpoint incomplete. Resolve the readiness or validation issue in the summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the decision
# MAGIC
# MAGIC ![Local MLflow example](https://raw.githubusercontent.com/debu-sinha/mlflow-eval-workshop/west-2026/notebooks/images/west/01-retrieval.png)
# MAGIC
# MAGIC The retrieval span shows the actual stale policy supplied to the application.
# MAGIC
# MAGIC Which component should we change first, and what observation supports that choice?
# MAGIC
# MAGIC Next, open 02_build_the_scorer_stack and turn one risk into a repeatable check.
