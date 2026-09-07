# Databricks notebook source
# MAGIC %md
# MAGIC # Explain the release decision
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC ![A saved local release report](https://raw.githubusercontent.com/debu-sinha/mlflow-eval-workshop/main/notebooks/images/west/release-report.png)
# MAGIC
# MAGIC The session opens with the live support app, then previews this report. This image is a saved local example. The cells below generate your own interactive report with actual responses and scores. Run this longer comparison during preparation when possible, so you can inspect its saved evidence during the session.
# MAGIC
# MAGIC After seeing the result, work through **00**, **01**, **02**, and **03** to understand the customer case, traces, scorers, and judge. Then return here to inspect the release rules before continuing to **05**.
# MAGIC
# MAGIC ## What the comparison holds fixed
# MAGIC
# MAGIC | Version | Retrieved policy | Purpose |
# MAGIC |---|---|---|
# MAGIC | Baseline | Current 30-day policy | Reference release |
# MAGIC | Candidate | Stale 90-day policy | Deliberate retrieval fault |
# MAGIC | Repaired | Current 30-day policy | Candidate after the targeted fix |
# MAGIC
# MAGIC The three versions answer the same ten cases with the same application model, scorers, and reference labels. Baseline and repaired use the same policy but make independent model calls, so their scores can differ. The comparison changes the retrieved policy; it does not fine-tune a model.

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
# MAGIC ## Read the release rules before running
# MAGIC
# MAGIC A *release gate* turns the recorded checks into a decision. Each version must have every named case, a successful application response, and all required scores.
# MAGIC
# MAGIC For each case, the gate uses the lower of `deterministic_stack` and `policy_judge`. It requires all deterministic checks to pass, a mean of at least 90% for both baseline and the evaluated version, and no more than 10% regressions versus baseline. The paired comparison can also block a statistically significant loss.
# MAGIC
# MAGIC With ten cases, a mean of 90% can include one failing judge score. It cannot excuse a failed mandatory deterministic check. Read the individual explanations.

# COMMAND ----------

from west_workshop.runtime import QUALITY_FLOOR, REGRESSION_LIMIT

print("Minimum mean evaluation score:", QUALITY_FLOOR)
print("Maximum regression rate versus baseline:", REGRESSION_LIMIT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the report
# MAGIC
# MAGIC The judge first checks eight separate rubric controls. If they pass, the cell evaluates 30 application responses across the three versions. It can take several minutes on Free Edition. Keep one notebook running at a time.
# MAGIC
# MAGIC The exercise completes only if the stale candidate is blocked, the repaired version passes the gate, and the measured mean improves with the dataset and scorers unchanged. An incomplete run is not accepted as a successful comparison.

# COMMAND ----------

from west_workshop import run_checkpoint
from west_workshop.report import render_release_report, write_release_report

summary = run_checkpoint(4)
show_result(summary)
if summary.get("summary_path"):
    report_path = write_release_report(summary)
    print("Saved visual report:", report_path)
if callable(globals().get("displayHTML")):
    displayHTML(render_release_report(summary))
if summary.get("status") != "passed":
    raise RuntimeError("This comparison did not complete. Read the report and saved summary before rerunning.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the result and follow the evidence
# MAGIC
# MAGIC Open the report's expandable sections to inspect the retrieved policies, every case, and the release rules. Locally, open the printed HTML path in a browser. In Databricks, the report appears above.
# MAGIC
# MAGIC The score change compares candidate with repaired. The regression limit compares each version with the baseline. Keep those comparisons distinct.
# MAGIC
# MAGIC The report now separates **correct eligibility** from **combined checks passed**. A correct eligibility label can accompany an incorrect policy explanation. In a recorded Free Edition run on September 6, eligibility improved from 7/10 to 10/10, while combined checks improved from 2/10 to 10/10. Those are that run's observations, not expected values for your run. [Read its two case studies](https://github.com/debu-sinha/mlflow-eval-workshop/blob/main/README.md#read-the-answer-behind-the-score).

# COMMAND ----------

comparison = summary["repair_comparison"]
print("Candidate mean:", comparison["candidate_mean"])
print("Repaired mean:", comparison["repaired_mean"])
print("Improved cases:", comparison["improved_cases"])
print("Regressed cases:", comparison["regressed_cases"])
for name, gate in summary["gates"].items():
    print("\nVersion:", name, "| decision:", gate["decision"])
    print("Reason:", gate["reason"])

# COMMAND ----------

from west_workshop.report import score_breakdown, judge_review_cases

for version in score_breakdown(summary):
    print(version)

# Correct labels with rejected replies need a person to inspect the explanation.
for row in judge_review_cases(summary):
    print("\nReview:", row["version"], row["case_id"])
    print("Actual answer:", row["output"])
    for assessment in row["assessments"]:
        if assessment["name"] == "policy_judge":
            print("Judge's reason:", assessment["rationale"])
    print("Trace:", row["trace_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explain your release decision
# MAGIC
# MAGIC Find one improved case and read both answers. Then inspect any failing judge score, even if the overall gate passes. What evidence would make you change the decision?
# MAGIC
# MAGIC In the recorded example, the baseline judge rejects “visit your account to claim it,” although the rubric permits customer next steps. The stale day-30 reply has a different problem: its refund label is correct, but its explanation cites 90 days. Decide which is an application error and which is an apparent judge error. Keep both recorded scores; a human review does not retroactively change the gate.
# MAGIC
# MAGIC To inspect the implementation, open `west_workshop/runtime.py` and read `_gate` and `_repair_comparison`. The general paired comparison is in `eval_gate.py`; checkpoint 4 adds the mandatory policy checks and quality floor.
# MAGIC
# MAGIC A pass applies to this teaching dataset. Next, open **05_production_feedback** to see how a new review becomes evidence for the next release.
