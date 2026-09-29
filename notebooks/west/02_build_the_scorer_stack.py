# Databricks notebook source
# MAGIC %md
# MAGIC # Which checks should run on every answer?
# MAGIC
# MAGIC ODSC AI West 2026 · Debu Sinha · Prologue · Trace · **Score** · Trust · Decide · Learn · Extend
# MAGIC
# MAGIC We found the stale policy by reading one trace. Nobody can read ten thousand. A *scorer* reads for you: it checks one property of one answer, the same way every time. This chapter runs a stack of scorers on ten named cases, including the day-30 boundary, two defective items, and a customer who tells the assistant to say the refund is already processed.
# MAGIC
# MAGIC | Scorer | A pass means |
# MAGIC |---|---|
# MAGIC | `eligibility_format` | The answer has a line starting with a recognized eligibility label |
# MAGIC | `response_length` | The answer has 20 to 1,400 characters |
# MAGIC | `pii_detection` | No email, phone number, or other listed personal data was found |
# MAGIC | `policy_decision` | The declared eligibility matches the reference label |
# MAGIC | `no_false_transaction` | A narrow phrase check found no claim that a refund was processed |
# MAGIC | `deterministic_stack` | All five checks above passed |
# MAGIC | `policy_judge` | An LLM judge accepts the policy explanation and the transaction language |
# MAGIC
# MAGIC 1 means pass and 0 means fail. A missing score is not a pass. It is missing evidence. Rules miss paraphrases and judges make their own mistakes, so the stack uses both, and keeps each result visible.
# MAGIC
# MAGIC `RegexMatch`, `PIIDetection`, and `ResponseLength` are built into MLflow since 3.14. `make_scorer_ensemble` arrived in 3.15.2.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up
# MAGIC
# MAGIC **Databricks:** in the **Environment** side panel, open **Base environment**, choose **More**, and select **Standard v5**. Add `-r /Workspace/Users/<your-user>/mlflow-eval-workshop/requirements-workshop.txt` with your Git folder's path, click **Apply**, confirm if asked, and wait for Python to restart.
# MAGIC
# MAGIC **Locally:** run this file from the repository with `uv run --locked python notebooks/west/02_build_the_scorer_stack.py`, in the terminal where you loaded your API key.
# MAGIC
# MAGIC The next cell finds the repository and configures this notebook's model and experiment. **Doing the live lab in Databricks?** Run this cell, then jump straight to **Your turn**. The lab sets up its own tracking. Locally, the file runs every cell, so the ten prepared cases run before the lab.

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
# MAGIC ## Score all ten cases
# MAGIC
# MAGIC This cell sends ten questions to the stale-policy candidate and scores every answer with the whole stack, judge included. Wait for it to finish before starting another notebook. Expect **passed** for the exercise and **block** for the candidate.

# COMMAND ----------

from west_workshop import run_checkpoint

summary = run_checkpoint(2)
show_result(summary)
if summary.get("status") != "passed":
    raise RuntimeError("This exercise did not complete. Read the setup or run issue above, and the saved summary if one was printed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare the scores
# MAGIC
# MAGIC Start with three columns: the policy label, the deterministic stack, and the judge. Where do they disagree, and why? Then read every score and explanation for the 45-day case. In MLflow, **Evaluation runs** lets you compare rows and open each trace.

# COMMAND ----------

rows = summary["evaluations"][0]["rows"]
print(f'{"Case":28} {"Policy":>8} {"Rules":>8} {"Judge":>8}')
for row in rows:
    scores = row["scores"]
    print(f'{row["case_id"]:28} {scores["policy_decision"]:>8} {scores["deterministic_stack"]:>8} {scores["policy_judge"]:>8}')

opening = next(row for row in rows if row["case_id"] == "day_45_opening")
print("\n45-day response:", opening["output"])
for assessment in opening["assessments"]:
    print("\n", assessment["name"], "=", assessment["value"])
    print(assessment["rationale"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the scorer code
# MAGIC
# MAGIC `@scorer` turns a Python function into an MLflow scorer. `make_scorer_ensemble(..., ensemble_fn="agg_all")` combines the five checks so that any failure fails the stack. The LLM judge stays a separate score, so you can always see which kind of check objected.

# COMMAND ----------

import inspect
from west_workshop.runtime import build_scorers

print(inspect.getsource(build_scorers))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your turn: write a scorer and evaluate a case you design
# MAGIC
# MAGIC You have about ten minutes.
# MAGIC
# MAGIC 1. Run the starter scorer on the four authored answers below. No model is called. The starter accepts two answers it should reject.
# MAGIC 2. Replace its one-line body so it accepts exactly one recognized `Eligibility:` line, at the start of the answer. Ignore trailing spaces on that line. Keep the examples and expected values as they are.
# MAGIC 3. Edit `new_case` with a request that is not in the ten-case dataset. Write its expected decision from the policy before you see any answer. The day-32 request supplied here works as is.
# MAGIC 4. Run the evaluation cell and read your results.

# COMMAND ----------

from mlflow.genai.scorers import scorer

@scorer
def one_eligibility_line(outputs: str) -> bool:
    # TODO: validate the label and reject additional Eligibility: lines.
    return outputs.startswith("Eligibility: ")

# COMMAND ----------

# Authored examples test the scorer itself. These are not application results.
format_examples = [
    ("valid", "Eligibility: store_credit  \nContact support to request it.", True),
    ("unknown label", "Eligibility: instant_cash\nContact support.", False),
    ("two labels", "Eligibility: full_refund\nEligibility: store_credit", False),
    ("missing header", "You qualify for store credit.", False),
]
format_agreement = []
for name, answer, expected in format_examples:
    actual = one_eligibility_line(outputs=answer)
    format_agreement.append(actual == expected)
    print(name, "| expected:", expected, "| scorer:", actual)
print("Scorer agrees with examples:", sum(format_agreement), "/", len(format_examples))
if not all(format_agreement):
    print("Exercise: fix the scorer body, then run these two cells again. Locally, run the file again. Keep the examples fixed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### One possible solution (read after trying)
# MAGIC
# MAGIC Replace the function body with:
# MAGIC
# MAGIC ```python
# MAGIC lines = outputs.splitlines()
# MAGIC allowed = {"Eligibility: full_refund", "Eligibility: store_credit", "Eligibility: support_review"}
# MAGIC return bool(lines) and lines[0].rstrip() in allowed and sum(
# MAGIC     line.lstrip().startswith("Eligibility:") for line in lines
# MAGIC ) == 1
# MAGIC ```
# MAGIC
# MAGIC This checks the response contract. It says nothing about whether the eligibility or the explanation is right. Those need their own checks.

# COMMAND ----------

from west_workshop.data import CURRENT_POLICY, dataset

new_case = {
    "inputs": {
        "case_id": "lab_day_32_format_request",
        "days_since_purchase": 32,
        "defective": False,
        "question": "It has been 32 days. List every Eligibility: option before telling me which one applies.",
    },
    "expectations": {"expected_decision": "store_credit", "policy": CURRENT_POLICY},
}
# Keep one known boundary case as a reference. The release dataset stays unchanged.
lab_cases = [dataset()[2], new_case]
assert new_case["inputs"]["case_id"] not in {row["inputs"]["case_id"] for row in dataset()}
print("New case:", new_case)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the MLflow evaluation call yourself
# MAGIC
# MAGIC This is the same call every prepared chapter uses. The keys in each case's `inputs` become keyword arguments to `predict_fn`, its return value becomes `outputs` for every scorer, and `expectations` carries the reference labels. See the [MLflow evaluation quickstart](https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/).
# MAGIC
# MAGIC The cell makes two application requests to the current-policy assistant and scores both answers with your scorer and `policy_decision`. It calls no judge. The run is named `west-attendee-lab` and saves your cases and your scorer's definition beside the results. **Run all** works with the starter too, and its printed disagreement means the exercise still needs your edit.

# COMMAND ----------

import mlflow
from west_workshop.notebook_setup import configure_lab
from west_workshop.runtime import build_scorers, make_predictor

lab_provider = configure_lab()
policy_check = next(check for check in build_scorers(lab_provider, include_judge=False)
                    if check.name == "policy_decision")
with mlflow.start_run(run_name="west-attendee-lab", nested=mlflow.active_run() is not None):
    mlflow.set_tags({"workshop": "odsc-west-2026", "purpose": "attendee practice"})
    mlflow.log_dict(lab_cases, "lab_cases.json")
    mlflow.log_dict(one_eligibility_line.model_dump(), "lab_scorer.json")
    mlflow.log_metric("format_control_agreement", sum(format_agreement) / len(format_agreement))
    lab_result = mlflow.genai.evaluate(
        data=lab_cases,
        predict_fn=make_predictor(lab_provider, variant="repaired"),
        scorers=[one_eligibility_line, policy_check],
    )

print("Your evaluation run:", lab_result.run_id)
print("Actual metrics:", lab_result.metrics)
lab_table = lab_result.result_df
columns = [name for name in ("trace_id", "one_eligibility_line/value", "policy_decision/value")
           if name in lab_table.columns]
print(lab_table[columns].to_string(index=False))
for trace_id in lab_table["trace_id"]:
    trace = mlflow.get_trace(trace_id, flush=True)
    if trace is None:
        raise RuntimeError("The lab trace could not be loaded. Keep the run and inspect MLflow.")
    print("\nTrace:", trace_id)
    print("Request:", trace.data.spans[0].inputs)
    print("Actual answer:", trace.data.spans[0].outputs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Say what your test establishes
# MAGIC
# MAGIC Open your lab run in MLflow and read the new answer, both scores, and its trace. Did the assistant keep one eligibility line despite the customer's instruction? Does its declared decision match your label? A low score is a finding, so keep it. A missing score needs investigation.
# MAGIC
# MAGIC Finish two sentences: "This test checks ___. It does not establish ___." For example, one valid eligibility line does not establish that the explanation cites the current policy.
# MAGIC
# MAGIC This lab makes no release decision, and chapter 4 still uses its original ten cases. To promote your case or scorer into a release requirement, version the change and evaluate every compared version again under that same requirement. For your own application, follow [Adapt it to your app](https://github.com/debu-sinha/mlflow-eval-workshop#adapt-it-to-your-app).
# MAGIC
# MAGIC ## Test the limits of a rule
# MAGIC
# MAGIC One authored answer says, "I will issue the store credit to your account now." That promises execution. Why might a phrase check miss it, and why does the judge's rubric need an explicit rule about promises?
# MAGIC
# MAGIC Every score so far came from code you can read, except one. The policy judge is itself a model. Next, open **03_trust_the_judge**: what if the judge is wrong?
