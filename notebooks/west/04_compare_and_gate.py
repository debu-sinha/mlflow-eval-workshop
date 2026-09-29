# Databricks notebook source
# MAGIC %md
# MAGIC # Better, regressed, or noise?
# MAGIC
# MAGIC ODSC AI West 2026 · Debu Sinha · Prologue · Trace · Score · Trust · **Decide** · Learn · Extend
# MAGIC
# MAGIC ![A saved release report](https://raw.githubusercontent.com/debu-sinha/mlflow-eval-workshop/main/notebooks/images/west/release-report.png)
# MAGIC
# MAGIC We have a trace that explains the failure, checks that run on every answer, and a judge we have tested. Now the release question. Every release decision has to answer three questions:
# MAGIC
# MAGIC 1. **Did it get better?**
# MAGIC 2. **Which cases regressed?**
# MAGIC 3. **Is it real, or noise?**
# MAGIC
# MAGIC The image above is one saved run. The cells below build your own report from real answers and scores. The comparison takes several minutes, so run it before the session when you can and reopen its saved result during the session.
# MAGIC
# MAGIC ## What the comparison holds fixed
# MAGIC
# MAGIC | Version | Retrieved policy | Role |
# MAGIC |---|---|---|
# MAGIC | Baseline | Current, 30 days | The current release |
# MAGIC | Candidate | Stale, 90 days | The version with the retrieval fault |
# MAGIC | Repaired | Current, 30 days | The candidate after the fix |
# MAGIC
# MAGIC All three answer the same ten cases with the same model, scorers, and reference labels. Only the retrieved policy changes, and no model is trained. Baseline and repaired use the same policy but make independent model calls, so their answers and scores can differ.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up
# MAGIC
# MAGIC **Databricks:** in the **Environment** side panel, open **Base environment**, choose **More**, and select **Standard v5**. Add `-r /Workspace/Users/<your-user>/mlflow-eval-workshop/requirements-workshop.txt` with your Git folder's path, click **Apply**, confirm if asked, and wait for Python to restart.
# MAGIC
# MAGIC **Locally:** run `uv run --locked python -m west_workshop --provider openai --checkpoint 4` from the repository, in the terminal where you loaded your API key, and open the printed `report_path`.
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
# MAGIC ## Read the release rules before you see a result
# MAGIC
# MAGIC A *release gate* turns recorded evidence into a decision. The rules are fixed before any version is evaluated:
# MAGIC
# MAGIC - Every case needs a successful answer and every required score. Missing evidence blocks the release.
# MAGIC - Every deterministic check must pass on every case.
# MAGIC - Each case scores the lower of `deterministic_stack` and `policy_judge`. Both the baseline and the evaluated version need a mean of at least 90%.
# MAGIC - No more than 10% of cases may regress against the baseline, and a significant paired loss also blocks.
# MAGIC
# MAGIC With ten cases, a 90% mean can include one judge rejection. It can never excuse a failed deterministic check.

# COMMAND ----------

from west_workshop.runtime import QUALITY_FLOOR, REGRESSION_LIMIT

print("Minimum mean score:", QUALITY_FLOOR)
print("Maximum regression rate against the baseline:", REGRESSION_LIMIT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the report
# MAGIC
# MAGIC The judge must first pass its eight controls. Then the cell makes 30 application requests, ten for each version, and scores every answer with every check and the judge. Progress prints as it goes. Keep one notebook running at a time.
# MAGIC
# MAGIC The exercise passes only if the stale candidate is blocked, the repaired version passes the gate, and the score improves with the dataset and scorers unchanged. An incomplete run is never counted as a success.

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
    raise RuntimeError("This comparison did not pass. Read the run issue printed above and the report, then docs/troubleshooting.md.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Answer the three questions
# MAGIC
# MAGIC Locally, open the printed HTML path in a browser. In Databricks, the report appears above. The three answers sit directly under the two decisions. If your workspace uses dark mode, Databricks inverts the colors of HTML output, so BLOCK and SHIP swap their usual red and green. The words and numbers are unchanged.
# MAGIC
# MAGIC **Did it get better?** Compare the stale candidate with the repaired version. Then read **What actually improved?** A correct eligibility label can still come with a wrong explanation, which is why the two counts can move differently. In one recorded Free Edition run, correct eligibility went from 7/10 to 10/10 while cases passing every check went from 2/10 to 10/10. [Read that run's two case studies](https://github.com/debu-sinha/mlflow-eval-workshop#read-the-answer-behind-the-score).
# MAGIC
# MAGIC **Which cases regressed?** Count against two references. The repair is judged against the stale candidate, and the release is judged against the current baseline.
# MAGIC
# MAGIC **Is it real, or noise?** Ten paired cases can confirm a large change. The exact McNemar test compares the cases that improved with the cases that regressed. With ten cases, it needs at least six of them moving the same way before it can call a change significant. That is why the gate never relies on the test alone: the regression limit and the mandatory checks catch smaller problems.
# MAGIC
# MAGIC The next cell also prints a 95% bootstrap interval. It resamples the ten per-case changes 10,000 times and shows the range of average change these cases support. With only ten cases, expect that range to be wide.

# COMMAND ----------

comparison = summary["repair_comparison"]
evidence = comparison["paired_evidence"]
low, high = evidence["interval"]
print("Candidate mean:", comparison["candidate_mean"])
print("Repaired mean:", comparison["repaired_mean"])
print("Improved cases:", evidence["improved"])
print("Regressed cases:", evidence["regressed"])
print(f"{evidence['test']} p-value: {evidence['p_value']:.4f}")
print(f"95% bootstrap interval for the change: {low:+.0%} to {high:+.0%}")
for name, gate in summary["gates"].items():
    print("\nVersion:", name, "| decision:", gate["decision"])
    print("Reason:", gate["reason"])
    against_baseline = gate.get("paired_evidence")
    if against_baseline:
        print("Regressed against the baseline:", against_baseline["regressed"])

# COMMAND ----------

from west_workshop.report import score_breakdown, judge_review_cases

for version in score_breakdown(summary):
    print(version)

# A correct label with a rejected reply needs a person to read the explanation.
for row in judge_review_cases(summary):
    print("\nReview:", row["version"], row["case_id"])
    print("Actual answer:", row["output"])
    for assessment in row["assessments"]:
        if assessment["name"] == "policy_judge":
            print("Judge's reason:", assessment["rationale"])
    print("Trace:", row["trace_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the call
# MAGIC
# MAGIC Pick one improved case and read both answers. Then read every failing judge score, even when the gate passes. What evidence would change your decision?
# MAGIC
# MAGIC In the recorded run, the baseline judge rejected "visit your account to claim it," although the rubric permits customer next steps. The stale day-30 reply had a different problem: a correct label with an explanation that cites 90 days. One is an application error and one looks like a judge error. Which is which? Keep both recorded scores either way, because a human review does not rewrite the gate after the fact.
# MAGIC
# MAGIC The gate's implementation is in `west_workshop/runtime.py`, in `_gate` and `_repair_comparison`. The paired statistics live in `eval_gate.py`, which you can run as a CI step. It exits with code 1 when regressions pass the limit, when a paired test finds a significant loss, or when evidence is missing. MLflow 3.14 added `@mlflow.test` for the same job inside pytest.
# MAGIC
# MAGIC A pass here applies to this teaching dataset. Next, open **05_production_feedback**: what happens after release?
