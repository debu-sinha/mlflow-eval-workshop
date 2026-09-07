# Databricks notebook source
# MAGIC %md
# MAGIC # What if the judge is wrong?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC A judge also needs evaluation. *Calibration* here means comparing its decisions with reference labels on six authored replies. These replies were written for the workshop; they are not fresh application outputs.
# MAGIC
# MAGIC We compare two judge configurations with the same policy rubric and model: one returns the value first, and one generates its rationale first. We do not assume the second will improve agreement.
# MAGIC
# MAGIC A separate set of eight positive and negative controls checks basic rubric behavior before checkpoint 4 can compare releases. These small teaching sets do not establish accuracy on real customer traffic.

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
# MAGIC ## Read a reply before seeing the judge's score
# MAGIC
# MAGIC Decide whether each reply follows the current policy and avoids claiming transaction execution. Give your reason before moving to the evaluation cell.

# COMMAND ----------

from west_workshop.data import calibration_dataset

for example in calibration_dataset():
    print("\nCase:", example["inputs"]["case_id"])
    print("Customer:", example["inputs"]["question"])
    print("Reply to review:", example["outputs"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate and preserve both judge definitions
# MAGIC
# MAGIC The next cell saves each full definition as an MLflow run artifact, loads it back, and checks that it matches. Locally it also demonstrates the scorer registry. Databricks Free Edition uses the saved artifacts, so server-side scorer versioning is not required.
# MAGIC
# MAGIC The six-example agreement can be less than 100% even when the exercise completes. Read every disagreement. The separate eight controls must pass.

# COMMAND ----------

from west_workshop import run_checkpoint

summary = run_checkpoint(3)
show_result(summary)
if summary.get("status") != "passed":
    raise RuntimeError("This exercise did not complete. Read the setup or run issue above and the saved summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare the decisions with the reference labels
# MAGIC
# MAGIC The reference label is the expected judgment about the authored reply. It is separate from an application's refund eligibility label. A score of 1 corresponds to an accepted reply; 0 corresponds to a rejected reply.

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
# MAGIC The judge checks eligibility and execution separately. It must evaluate the assistant's reply, without mistaking the customer's instruction for something the assistant actually said. Read how the rubric handles day 30, defective items, and permitted next steps.

# COMMAND ----------

import inspect
from west_workshop.runtime import _policy_judge

print(inspect.getsource(_policy_judge))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review a disagreement
# MAGIC
# MAGIC Read the customer request, assistant reply, reference label, and judge explanation together. If the label is wrong, document and correct the labeling error. If the judge is wrong, improve the rubric and rerun the same examples. Keep earlier results available.
# MAGIC
# MAGIC There is a real example to practice on in [Read the answer behind the score](https://github.com/debu-sinha/mlflow-eval-workshop/blob/main/README.md#read-the-answer-behind-the-score). A recorded baseline answer told the customer to visit their account to claim credit. The judge rejected it as assistant execution, although its rubric permits customer next steps. Read the reply before revealing the discussion. Passing a small control set did not prevent this mistake on a generated answer.
# MAGIC
# MAGIC In 04, the report's **Review the judge** section finds this kind of disagreement in your own run. A rejected reply with a correct label can also have a genuinely incorrect explanation, so the label alone cannot settle the review.
# MAGIC
# MAGIC Return to **04_compare_and_gate**. If you already ran the opening report and have changed nothing, reuse it to inspect the release rules. If you changed the judge, rerun the comparison.
