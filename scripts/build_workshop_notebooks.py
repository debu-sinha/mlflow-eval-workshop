"""Rebuild the three editable primary notebooks from their short teaching cells."""

import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "workshop"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}


def code(source, *, setup=False, hidden=False, cell_id=None):
    metadata = {"tags": ["setup"]} if setup else {}
    if setup or hidden:
        metadata["jupyter"] = {"source_hidden": True}
    cell = {"cell_type": "code", "execution_count": None, "metadata": metadata,
            "outputs": [], "source": textwrap.dedent(source).strip() + "\n"}
    if cell_id:
        cell["id"] = cell_id
    return cell


BOOTSTRAP = """
import importlib
import os, sys, json
from pathlib import Path

# Find the repository folder in local Jupyter or Databricks.
start = Path(os.environ.get("WORKSHOP_PROJECT_ROOT", Path.cwd())).resolve()
ROOT = next((p for p in (start, *start.parents) if (p / "west_workshop" / "data.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Set WORKSHOP_PROJECT_ROOT to the workshop repository folder, then restart this notebook.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
importlib.invalidate_caches()
os.chdir(ROOT)
MODE = os.environ.get("WORKSHOP_MODE", "saved")  # Choose live mode to make new model calls.
"""


def write(name, cells):
    document = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3 (workshop)", "language": "python", "name": "python3"},
                                             "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    for number, cell in enumerate(document["cells"]):
        cell.setdefault("id", name[:2] + "-" + str(number).zfill(2))
    (DESTINATION / name).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


DESTINATION.mkdir(parents=True, exist_ok=True)
write("00_setup.ipynb", [
    markdown("""
    # Start here

    **Would you accept this repair?** Find why the assistant gave a wrong answer, test the fix and decide.

    You can follow along by watching. **Saved mode** shows recorded results without model or network calls. **Live mode** makes new model calls and can produce a different result.
    """),
    code(BOOTSTRAP, setup=True),
    markdown("""
    ## Check your environment

    Run all cells in notebook 00, then 01, then 02. The README explains setup on your computer or in Databricks.
    """),
    code("""
    from IPython.display import display, HTML
    from html import escape
    from west_workshop.repair_workshop import workshop_preflight

    readiness = workshop_preflight(MODE)
    status = "Ready for notebook 01" if readiness["ready"] else "Setup needs attention"
    issues = "".join("<li>" + escape(reason) + "</li>" for reason in readiness["reasons"])
    display(HTML("<h3>" + status + "</h3><p>Mode: " + escape(MODE) + "</p><ul>" + issues + "</ul>"
                 + "<details><summary>Environment details</summary><pre>"
                 + escape(json.dumps(readiness, indent=2)) + "</pre></details>"))
    """, hidden=True),
    markdown("""
    ## What changes?

    | Version | Policy document |
    |---|---|
    | Starting assistant (simulated) | Old 90-day policy |
    | Proposed repair | Current 30-day policy |

    The test cases and checks stay the same. Passing these workshop rules does not mean the app is ready to deploy.

    Next: **01_diagnose_and_evaluate.ipynb**.
    """),
])

write("01_diagnose_and_evaluate.ipynb", [
    markdown("""
    # 1 · Diagnose and evaluate

    **Can a clear answer still be wrong?** The item is not defective. The customer asks for a full refund 45 days after buying it.

    Current rule: full refund through day 30, store credit after day 30. Support must review defective items.
    """),
    code(BOOTSTRAP, setup=True),
    code("""
    from IPython.display import display, HTML, Code
    from west_workshop.repair_workshop import notebook_session
    from west_workshop.mlflow_view import session_header, inspection_card

    session = notebook_session(MODE)
    display(HTML(session_header(session)))
    """, hidden=True),
    markdown("""
    ## Code view 1: one business check

    Does the answer name the right outcome: full refund, store credit or support review? This label check is one of five code checks in `deterministic_stack`. It does not judge the explanation.
    """),
    code("""
    import inspect
    from west_workshop.runtime import policy_decision_value

    display(Code(inspect.getsource(policy_decision_value), language="python"))
    """, hidden=True, cell_id="01-label-check"),
    markdown("""
    ## Code view 2: evaluate the application

    `data` holds the test cases. `predict_fn` runs the app and records its steps. `scorers` lists the checks for each answer.
    Saved mode reads recorded results. Live mode runs new evaluations and reuses completed results from the same output folder.
    """),
    code("""
    if session.needs("deployed"):
        import mlflow
        with session.evaluation("deployed") as run:
            result = mlflow.genai.evaluate(
                data=run.data, predict_fn=run.predict_fn, scorers=run.scorers,
            )
            session.record_evaluation("deployed", result, run)
    """, cell_id="01-evaluate"),
    code("""
    deployed = session.recorded("deployed")
    if not deployed["complete"]:
        raise RuntimeError("This evaluation is incomplete. Read the saved error before trying again in a new output folder.")
    """, hidden=True),
    markdown("""
    ## Inspect in MLflow: request → policy → check results

    A trace shows what happened during one request. Open `day_45_opening`. Which policy did the assistant receive? Which check caught the wrong answer?

    If you cannot open these results in MLflow, expand the recorded case below.
    """),
    code("""
    display(HTML(inspection_card(session, "deployed", ["day_45_opening"])))
    """, hidden=True, cell_id="01-mlflow"),
    markdown("""
    Next: **02_compare_and_decide.ipynb**. Test how answers are graded, then check the repair.
    """),
])

write("02_compare_and_decide.ipynb", [
    markdown("""
    # 2 · Compare and decide

    **What would make the repair acceptable?** Give the app the current policy document. Keep the test cases and checks the same.
    """),
    code(BOOTSTRAP, setup=True),
    code("""
    from IPython.display import display, HTML
    from west_workshop.repair_workshop import notebook_session
    from west_workshop.mlflow_view import (
        session_header, controls_card, policy_card, inspection_card,
        finding_cases, paired_summary, review_card, decision_card, statistics_card,
    )

    session = notebook_session(MODE)
    display(HTML(session_header(session)))
    """, hidden=True),
    markdown("""
    ## Test the judge

    The judge is a model that grades answers. Controls are examples with known grades that we use to test it.

    Can it distinguish a customer's request from an assistant's claim that it processed a refund?
    """),
    code("""
    controls = None
    if not session.has_execution_failure():
        controls = session.qualify_judge()
    display(HTML(controls_card(controls)))
    """, hidden=True, cell_id="02-control-examples"),
    markdown("""
    ## Evaluate the repair

    Use the same MLflow call for both versions. Stop further model calls if the judge fails the controls or execution fails.
    """),
    code("""
    if MODE == "live" and controls and controls["passed"] and not session.has_execution_failure():
        import mlflow
        for variant in ("deployed", "repair"):
            if session.needs(variant):
                with session.evaluation(variant) as run:
                    result = mlflow.genai.evaluate(
                        data=run.data, predict_fn=run.predict_fn, scorers=run.scorers,
                    )
                    session.record_evaluation(variant, result, run)
                if session.has_execution_failure():
                    break

    review_path = os.environ.get("WORKSHOP_REVIEW_JSON")
    reviewed = json.loads(Path(review_path).read_text(encoding="utf-8")) if review_path else None
    comparison = session.policy_inputs(reviewed)
    """, hidden=True),
    markdown("""
    ## Code view 3: apply the acceptance rules

    MLflow records the results. These rules decide whether the repair passes.
    """),
    code("""
    from west_workshop.repair_policy import DEFAULT_POLICY, evaluate_repair

    policy_result = evaluate_repair(**comparison, policy=DEFAULT_POLICY)
    """, cell_id="02-policy"),
    code("""
    report = session.finish(comparison["review"])
    assert report["policy_result"] == policy_result
    display(HTML(policy_card(DEFAULT_POLICY)))
    """, hidden=True),
    markdown("""
    ## Inspect in MLflow: repaired answer → review findings

    Open `day_45_opening` for the repair. Check the policy and answer, then inspect any case flagged in the review.
    MLflow records what happened and the scores. The review records any issue that still needs attention.
    """),
    code("""
    display(HTML(inspection_card(session, "repair", ["day_45_opening", *finding_cases(report)])))
    """, hidden=True, cell_id="02-mlflow"),
    markdown("""
    ## Compare each case and read the review

    The table shows each case's overall result. For each protected check, 1 to 0 is a new failure, even if the overall result stays 0. A check that stays at 0 is a tie.
    A completed review can still have an open issue that affects the decision.
    """),
    code("""
    display(HTML(paired_summary(report)))
    """, hidden=True, cell_id="02-pairs"),
    code("""
    display(HTML(review_card(report)))
    """, hidden=True, cell_id="02-review-evidence"),
    markdown("""
    ## Vote before revealing the decision

    **Pause for 20 seconds.** Which decision fits the evidence?

    - **PASS**: all acceptance rules met.
    - **BLOCK**: an acceptance rule failed.
    - **REVIEW REQUIRED**: review unfinished or an important issue still open.
    - **INVALID**: results cannot be compared.

    Name the evidence that would change your decision.
    """),
    code("""
    display(HTML(decision_card(report)))
    """, hidden=True, cell_id="02-decision"),
    markdown("""
    ## Turn the finding into a future test

    Keep the original answer and score. Ask a person to approve the expected grade for the new test case.
    Test the updated judge on controls, including new examples it was not tuned on. Then use it to grade all the same saved answers again. Keep both sets of scores.

    The README shows where to change the app, cases, checks and acceptance rules. Write down who makes the next decision and what evidence they need.
    """),
    code("""
    display(HTML(statistics_card(report)))
    """, hidden=True),
])

print("Wrote three portable workshop notebooks.")
