# Databricks notebook source
# MAGIC %md
# MAGIC # Which release earns a pass on this policy?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC A dataset is the set of named examples the team refuses to forget. A gate turns evidence into a ship or block decision. Compare the same questions across releases. Check complete coverage and hard requirements before the regression comparison. The deliberate fault is stale policy retrieval. Repairing that retrieval is the targeted change. Real model behavior decides whether the expected block and pass actually occur.
# MAGIC
# MAGIC **Watch:** Set the rule before revealing baseline, candidate, and repaired candidate results.
# MAGIC
# MAGIC **Build:** Run the checkpoint and read both the block reason and the repaired candidate decision.
# MAGIC
# MAGIC **Extend:** Increase the dataset with boundary and adversarial cases. Decide the minimum evidence before looking at results.
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

summary = run_checkpoint(4)
print(json.dumps(summary, indent=2, sort_keys=True))
if summary.get("status") != "passed":
    raise RuntimeError("Checkpoint incomplete. Resolve the readiness or validation issue in the summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the decision
# MAGIC
# MAGIC ![Local MLflow example](https://raw.githubusercontent.com/debu-sinha/mlflow-eval-workshop/west-2026/notebooks/images/west/04-comparison.png)
# MAGIC
# MAGIC This view compares the stale-policy candidate and repaired application on the same cases. Judge disagreements remain visible. The checkpoint also evaluates both versions against the baseline.
# MAGIC
# MAGIC Does the candidate satisfy our explicit policy, and is every required case accounted for?
# MAGIC
# MAGIC A pass applies only to this teaching policy and dataset. Next, open 05_production_feedback to see how a new failure becomes a future test.
