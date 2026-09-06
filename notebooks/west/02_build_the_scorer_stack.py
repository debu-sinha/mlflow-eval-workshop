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
# MAGIC ## Test the limits of a rule
# MAGIC
# MAGIC Consider the authored example, “I will issue the store credit to your account now.” It promises execution. Explain why a phrase check might miss it and why the semantic judge needs an explicit rule about transaction promises.
# MAGIC
# MAGIC A response can be short, correctly formatted, and still wrong. Which checks would you require for every release?
# MAGIC
# MAGIC Next, open **03_trust_the_judge** to examine the judge itself.
