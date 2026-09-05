# Databricks notebook source
# MAGIC %md
# MAGIC # What should happen after a new failure reaches production?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC A release decision covers the evidence we collected. New behavior can appear after release. This checkpoint sends a new fictional request through the real application, captures its trace, and evaluates it. It demonstrates the feedback loop. It does not silently enable a cloud monitoring job. Production evaluation needs owners, a sampling policy, privacy controls, and a reviewed path back into the evaluation dataset.
# MAGIC
# MAGIC **Watch:** Choose one signal that deserves human review and a place in the next dataset.
# MAGIC
# MAGIC **Build:** Run the checkpoint to trace and score a new fictional customer request.
# MAGIC
# MAGIC **Extend:** Propose who reviews the signal, how it becomes an approved dataset case, and what release gate changes.
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

summary = run_checkpoint(5)
print(json.dumps(summary, indent=2, sort_keys=True))
if summary.get("status") != "passed":
    raise RuntimeError("Checkpoint incomplete. Resolve the readiness or validation issue in the summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the decision
# MAGIC
# MAGIC ![Local MLflow example](https://raw.githubusercontent.com/debu-sinha/mlflow-eval-workshop/west-2026/notebooks/images/west/05-feedback.png)
# MAGIC
# MAGIC Authored teaching feedback is attached to a real defective-item trace. It is not an observed customer report.
# MAGIC
# MAGIC What evidence would make you reopen the release decision?
# MAGIC
# MAGIC Write one application risk, one named example, and one rule you can use in your own release process. To try Phoenix and TruLens next, follow the [optional integration setup](https://github.com/debu-sinha/mlflow-eval-workshop/tree/west-2026#optional-integrations).
