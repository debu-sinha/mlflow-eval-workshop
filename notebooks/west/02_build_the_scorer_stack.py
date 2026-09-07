# Databricks notebook source
# MAGIC %md
# MAGIC # Which checks should run on every answer?
# MAGIC
# MAGIC ODSC AI West 2026 | Debu Sinha
# MAGIC
# MAGIC A *scorer* checks one property of a response. This notebook evaluates the stale-policy candidate on ten named cases, including the day-30 boundary, defective items, and an instruction to claim a refund was processed.
# MAGIC
# MAGIC | Scorer | What a passing result means |
# MAGIC |---|---|
# MAGIC | `eligibility_format` | The answer starts with a recognized eligibility identifier |
# MAGIC | `response_length` | The answer has between 20 and 1,400 characters |
# MAGIC | `pii_detection` | The built-in check did not detect personal information |
# MAGIC | `policy_decision` | Declared eligibility matches the reference label |
# MAGIC | `no_false_transaction` | A narrow phrase check did not find a false execution claim |
# MAGIC | `deterministic_stack` | All five checks above pass |
# MAGIC | `policy_judge` | The LLM judge accepts both the policy explanation and transaction language |
# MAGIC
# MAGIC For these checks, 1 means pass and 0 means fail. A missing score is incomplete evidence. Pattern checks have limits: a new paraphrase can escape a phrase check, and a judge can make its own mistake.

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
# MAGIC ## Evaluate all ten cases
# MAGIC
# MAGIC The cell makes fresh application and judge requests. Wait for it to finish before running another notebook. The exercise expects to catch the stale policy; a completed run can therefore show **passed** for the exercise and **block** for the candidate.

# COMMAND ----------

from west_workshop import run_checkpoint

summary = run_checkpoint(2)
show_result(summary)
if summary.get("status") != "passed":
    raise RuntimeError("This exercise did not complete. Read the setup or run issue above and the saved summary.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare the individual scores
# MAGIC
# MAGIC Start with the policy decision, the combined deterministic checks, and the judge. Then inspect every score and explanation for the 45-day case. In MLflow, use **Evaluation runs** to compare rows and open their traces.

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
# MAGIC ## Read the scorer implementation
# MAGIC
# MAGIC `@scorer` wraps a Python check for MLflow. `make_scorer_ensemble(..., ensemble_fn="agg_all")` combines the five checks so that any failure makes the stack fail. The LLM judge remains visible as a separate score.

# COMMAND ----------

import inspect
from west_workshop.runtime import build_scorers

print(inspect.getsource(build_scorers))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your turn: write a scorer and evaluate a new case
# MAGIC
# MAGIC Allow about ten minutes. First run the starter scorer against the four **authored format examples** below; no model calls are needed. It deliberately accepts two malformed answers. Replace its one-line body so it accepts exactly one recognized `Eligibility:` line, at the start of the answer. Ignore trailing spaces on that line. Keep the examples and expected values fixed.
# MAGIC
# MAGIC Run your edited scorer cell and the examples again. Then edit `new_case` with a request that is absent from the ten-case dataset and its policy-based expected label. The supplied day-32 request is a runnable starting point. Write down what behavior your case tests before seeing a model response.
# MAGIC
# MAGIC For the short live exercise, run this notebook's setup cell, then start here. You can skip the prepared ten-case evaluation and source-inspection cells above. The lab configures its own tracking when you reach the evaluation call.

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
    print("Exercise: fix the scorer body, then rerun these two cells. Keep the examples fixed.")

# COMMAND ----------

# MAGIC %md
# MAGIC <details><summary>One possible solution (open after trying)</summary>
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
# MAGIC This checks the response contract. It does not verify eligibility or the explanation. Those need separate checks.
# MAGIC </details>

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
# Keep one known boundary case as a reference; do not modify the release dataset.
lab_cases = [dataset()[2], new_case]
assert new_case["inputs"]["case_id"] not in {row["inputs"]["case_id"] for row in dataset()}
print("New case:", new_case)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the MLflow evaluation call yourself
# MAGIC
# MAGIC `inputs` keys become keyword arguments to `predict_fn`. Its returned text becomes `outputs` for each scorer; `expectations` contains the reference labels. This is the same [MLflow evaluation API](https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/) used inside the prepared checkpoints.
# MAGIC
# MAGIC The next cell makes two fresh application requests with the current policy and runs two Python scorers. It does not call an LLM judge. MLflow may also make a prediction to check the function's tracing. The separate `west-attendee-lab` run preserves the actual cases, scorer definition, and scores. **Run all** also works with the starter scorer; the printed format disagreement means the exercise still needs your edit.

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
# MAGIC ## Explain what your test establishes
# MAGIC
# MAGIC Open your lab run in MLflow. Read the new response, both scores, and its trace. Did the assistant preserve one eligibility line despite the customer's instruction? Does its declared decision match your label? A missing score needs investigation; a low score is a finding to retain.
# MAGIC
# MAGIC This small lab does not make a release decision. Even two passing checks say nothing about the explanation's policy accuracy. Checkpoint 4 still uses its original ten cases and full scorer stack. To promote your new case or scorer into a release requirement, version the change and evaluate **all three versions** again with that same requirement; keep earlier runs.
# MAGIC
# MAGIC For your own application, follow [Adapt this evaluation to your app](https://github.com/debu-sinha/mlflow-eval-workshop/blob/main/README.md#adapt-this-evaluation-to-your-app).
# MAGIC
# MAGIC ## Test the limits of a rule
# MAGIC
# MAGIC Consider the authored example, “I will issue the store credit to your account now.” It promises execution. Explain why a phrase check might miss it and why the semantic judge needs an explicit rule about transaction promises.
# MAGIC
# MAGIC A response can be short, correctly formatted, and still wrong. Which checks would you require for every release?
# MAGIC
# MAGIC Next, open **03_trust_the_judge** to examine the judge itself.
