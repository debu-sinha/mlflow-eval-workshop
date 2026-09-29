# Databricks notebook source
# MAGIC %md
# MAGIC # What if the judge is wrong?
# MAGIC
# MAGIC ODSC AI West 2026 · Debu Sinha · Prologue · Trace · Score · **Trust** · Decide · Learn · Extend
# MAGIC
# MAGIC The policy judge caught what no rule could: a correct label with a stale explanation. It is also a model, and models make mistakes. Before its score decides a release, we evaluate the judge itself.
# MAGIC
# MAGIC *Calibration* here means comparing the judge's decisions with reference labels on six replies written for the workshop. They are authored examples, not fresh application answers.
# MAGIC
# MAGIC We compare two versions of the same judge, with the same rubric and model. One returns its verdict first. The other writes its rationale first, using `make_judge(generate_rationale_first=True)`, new in MLflow 3.16. We do not assume the second one wins. We measure.
# MAGIC
# MAGIC Eight separate controls, four that should pass and four that should fail, gate the judge before chapter 4 compares releases. These small sets check the rubric's behavior. They cannot establish accuracy on real traffic.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up
# MAGIC
# MAGIC **Databricks:** in the **Environment** side panel, open **Base environment**, choose **More**, and select **Standard v5**. Add `-r /Workspace/Users/<your-user>/mlflow-eval-workshop/requirements-workshop.txt` with your Git folder's path, click **Apply**, confirm if asked, and wait for Python to restart.
# MAGIC
# MAGIC **Locally:** run this file from the repository with `uv run --locked python notebooks/west/03_trust_the_judge.py`, in the terminal where you loaded your API key.
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
# MAGIC ## Grade the replies yourself first
# MAGIC
# MAGIC For each reply, decide whether it follows the current policy and avoids claiming a transaction. Write down your reason before you look at the judge.

# COMMAND ----------

from west_workshop.data import calibration_dataset

for example in calibration_dataset():
    print("\nCase:", example["inputs"]["case_id"])
    print("Customer:", example["inputs"]["question"])
    print("Reply to review:", example["outputs"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run both judges and keep their definitions
# MAGIC
# MAGIC The next cell saves each judge's full definition as an MLflow run artifact, loads it back, and checks that nothing changed. Locally it also registers both versions in MLflow's scorer registry. On Databricks Free Edition the saved artifacts are the record, so server-side scorer versioning is not required.
# MAGIC
# MAGIC The cell makes 20 judge requests and no application requests. Each judge grades the six replies, then the rationale-first judge grades the eight controls.
# MAGIC
# MAGIC Agreement on the six replies can land below 100% even when the exercise passes. Read every disagreement. The eight controls must all pass.

# COMMAND ----------

from west_workshop import run_checkpoint

summary = run_checkpoint(3)
show_result(summary)
if summary.get("status") != "passed":
    raise RuntimeError("This exercise did not complete. Read the setup or run issue above, and the saved summary if one was printed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare the judges with the labels
# MAGIC
# MAGIC Here the reference label is a verdict on the reply, separate from the refund label in chapter 2. A score of 1 means the judge accepted the reply. A score of 0 means it rejected it.

# COMMAND ----------

for row in summary["evaluations"][0]["rows"]:
    print("\nCase:", row["case_id"])
    print("Reference accepts reply:", row["expectations"]["authored_human_label"])
    print("Judge scores:", row["scores"])
    for assessment in row["assessments"]:
        print(assessment["name"], ":", assessment["rationale"])

print("\nAgreement with reference labels:", summary["agreement_with_authored_labels"])
print("Separate controls passed:", summary["judge_validation"]["passed"])
print("Control disagreements:", summary["judge_validation"]["disagreements"])
for version in summary["scorer_versions"]:
    print("Saved definition:", version["name"], "| run:", version["run_id"],
          "| artifact:", version["artifact_path"],
          "| reload verified:", version["round_trip_verified"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the rubric
# MAGIC
# MAGIC The judge checks eligibility and execution separately. It must grade what the assistant said, and never mistake the customer's instruction for the assistant's own claim. Find how the rubric handles day 30, defective items, and customer next steps.

# COMMAND ----------

import inspect
from west_workshop.runtime import _policy_judge

print(inspect.getsource(_policy_judge))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review a disagreement
# MAGIC
# MAGIC Read the request, the reply, the reference label, and the judge's explanation together. If the label is wrong, correct the label and record why. If the judge is wrong, improve the rubric and rerun the same examples. Keep the earlier results either way.
# MAGIC
# MAGIC There is a real case to practice on in [A judge rejection that deserves correction](https://github.com/debu-sinha/mlflow-eval-workshop#a-judge-rejection-that-deserves-correction). A recorded baseline answer told the customer to visit their account to claim store credit. The judge rejected it as a transaction, although its rubric permits customer next steps. Read the reply before you read the discussion. Eight passing controls did not prevent this mistake on a generated answer.
# MAGIC
# MAGIC **Take it further.** Once reviewers have labeled enough real traces, MLflow can align a judge to their feedback with `judge.align()`, which uses the MemAlign optimizer by default since MLflow 3.13. Calibration comes first, because alignment learns from labels you trust. See [judge alignment](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/alignment/).
# MAGIC
# MAGIC We can now trust the judge as far as its evidence goes. Next, return to **04_compare_and_gate**: better, regressed, or noise? If you ran its report earlier and have changed nothing, reuse it. If you changed the judge, run the comparison again.
