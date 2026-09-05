# Databricks notebook source
# MAGIC %md
# MAGIC # What would you measure before paying for another judge?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC A scorer turns one behavior into a result. A deterministic rule applies code to the same inputs every time. An LLM judge evaluates meaning and can disagree with people. Keep each result visible before combining them. A good average must never hide an unauthorized transaction promise.
# MAGIC
# MAGIC **Watch:** Choose the first risk to test: forbidden refund promises, missing useful guidance, or an unclear answer.
# MAGIC
# MAGIC **Build:** Run the checkpoint and compare the individual scorer results.
# MAGIC
# MAGIC **Extend:** Add one realistic paraphrase that a phrase check could miss. Explain what the LLM judge adds.
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

summary = run_checkpoint(2)
print(json.dumps(summary, indent=2, sort_keys=True))
if summary.get("status") != "passed":
    raise RuntimeError("Checkpoint incomplete. Resolve the readiness or validation issue in the summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the decision
# MAGIC
# MAGIC Which checks are hard requirements, and which help us compare quality?
# MAGIC
# MAGIC Next, open 03_trust_the_judge and check whether the judge deserves our trust.
