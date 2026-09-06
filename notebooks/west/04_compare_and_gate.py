# Databricks notebook source
# MAGIC %md
# MAGIC # Start here: would you ship this assistant?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC Start with the finished release report below. Compare the two decisions and the customer's two answers, then open the evidence to see what changed. The report uses this run's responses and scores.
# MAGIC
# MAGIC ![Recorded release report](https://raw.githubusercontent.com/debu-sinha/mlflow-eval-workshop/main/notebooks/images/west/release-report.png)
# MAGIC
# MAGIC This image is a saved local example. Run the cells below to generate your own report. The application model, questions, scorers, and thresholds stay fixed during each comparison; only the retrieved policy changes. Your result may differ.
# MAGIC
# MAGIC After seeing the result, open **00_ship_or_block** to work through the customer's answer, then **01–03** to explore retrieval, scoring, and judge calibration. Return here to inspect the release rules before continuing to **05_production_feedback**.
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

from west_workshop.notebook_setup import configure_notebook
configure_notebook()

# COMMAND ----------

# Run this cell after completing the setup in README.md.
# It calls the configured provider and can take several minutes.
from west_workshop import run_checkpoint
from west_workshop.report import render_release_report, write_release_report

summary = run_checkpoint(4)
if summary.get("summary_path"):
    report_path = write_release_report(summary)
    print("Saved report:", report_path)
    print("Full results:", summary["summary_path"])

# Databricks renders the report inline. Local scripts save an HTML file to open.
if callable(globals().get("displayHTML")):
    displayHTML(render_release_report(summary))
else:
    print("Checkpoint status:", summary.get("status"))

if summary.get("status") != "passed":
    raise RuntimeError("Checkpoint incomplete. Resolve the readiness or validation issue in the summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Follow the evidence
# MAGIC
# MAGIC Open the report's three expandable sections: the retrieved policies, every scored case, and the release rules. The MLflow run IDs and trace IDs let you follow the same evidence in the experiment UI.
# MAGIC
# MAGIC The judge first has to pass eight separate rubric controls. Then the baseline, stale candidate, and repaired assistant each answer the same ten questions. A missing response or score blocks a complete comparison.
# MAGIC
# MAGIC **Try next:** Add boundary and adversarial cases. Decide the release rules before seeing the new results.
# MAGIC
# MAGIC Which piece of evidence would change your release decision?
# MAGIC
# MAGIC A pass applies only to this teaching policy and dataset. Next, open 05_production_feedback to see how a new failure becomes a future test.
