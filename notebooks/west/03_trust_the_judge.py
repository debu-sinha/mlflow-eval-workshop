# Databricks notebook source
# MAGIC %md
# MAGIC # What if the judge is wrong?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC Calibration means checking a judge against examples that people have already reviewed. These authored examples are teaching material, with explicit expected labels. They are not generated application responses. A tiny agreement check can reveal a mistake. It cannot establish general accuracy. Preserve the judge instructions and version so the next release can be compared fairly.
# MAGIC
# MAGIC **Watch:** Label each calibration example before revealing the judge result.
# MAGIC
# MAGIC **Build:** Run the checkpoint and inspect every disagreement with the authored human labels.
# MAGIC
# MAGIC **Extend:** Try another judge configuration on the same examples. Report disagreements without selecting only favorable cases.
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

summary = run_checkpoint(3)
print(json.dumps(summary, indent=2, sort_keys=True))
if summary.get("status") != "passed":
    raise RuntimeError("Checkpoint incomplete. Resolve the readiness or validation issue in the summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the decision
# MAGIC
# MAGIC ![Local MLflow example](https://raw.githubusercontent.com/debu-sinha/mlflow-eval-workshop/west-2026/notebooks/images/west/03-judge-versions.png)
# MAGIC
# MAGIC These judge definitions were registered locally. A newer version is not automatically a better judge.
# MAGIC
# MAGIC Would you trust this judge to block a release, or do the disagreements need review first?
# MAGIC
# MAGIC Next, open 04_compare_and_gate and make a decision using the same named cases.
