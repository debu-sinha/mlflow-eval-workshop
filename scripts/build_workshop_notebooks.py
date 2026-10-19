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

# Find the checkout in local Jupyter or a Databricks Git folder.
start = Path(os.environ.get("WORKSHOP_PROJECT_ROOT", Path.cwd())).resolve()
ROOT = next((p for p in (start, *start.parents) if (p / "west_workshop" / "data.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Set WORKSHOP_PROJECT_ROOT to the workshop checkout, then restart this notebook.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
importlib.invalidate_caches()
os.chdir(ROOT)
MODE = os.environ.get("WORKSHOP_MODE", "saved")  # Explicit opt-in before any model calls.
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

    **Would you accept this repair?** Follow one case through its rule, trace, checks, repair and decision.

    Watching is a complete participation route. **Saved** reads dated real evidence without model or network calls. **Live** makes new model calls and can produce a different result.
    """),
    code(BOOTSTRAP, setup=True),
    markdown("""
    ## Check your environment

    Run all cells in 00, then 01, then 02. Installation and local or Databricks configuration are in the repository README.
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

    | Configuration | Retrieved source |
    |---|---|
    | Simulated deployed assistant | Obsolete 90-day policy |
    | Proposed repair | Current 30-day policy |

    The cases and checks stay fixed. A workshop policy pass is not authorization to deploy.

    Next: **01_diagnose_and_evaluate.ipynb**.
    """),
])

write("01_diagnose_and_evaluate.ipynb", [
    markdown("""
    # 1 · Diagnose and evaluate

    **Can a fluent answer use the wrong policy?** A non-defective item was bought 45 days ago. The customer asks for a full refund.

    Current rule: full refund through day 30, store credit after day 30. Defective items need support review.
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

    Compare the declared eligibility with the reviewed label. A correct label alone cannot prove the explanation is correct.
    """),
    code("""
    import inspect
    from west_workshop.runtime import policy_decision_value

    display(Code(inspect.getsource(policy_decision_value), language="python"))
    """, hidden=True, cell_id="01-label-check"),
    markdown("""
    ## Code view 2: evaluate the application

    `data` holds named cases. `predict_fn` runs the traced application. `scorers` checks its answers.
    Saved mode reads the original result. Live mode generates each missing batch once.
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
        raise RuntimeError("This evaluation is incomplete. Inspect its preserved failure before retrying in a new directory.")
    """, hidden=True),
    markdown("""
    ## Inspect in MLflow: request → source → assessments

    Open `day_45_opening`. Which source did the assistant receive? Which check caught the wrong answer?

    If no matching tracking server is available, expand the recorded case below.
    """),
    code("""
    display(HTML(inspection_card(session, "deployed", ["day_45_opening"])))
    """, hidden=True, cell_id="01-mlflow"),
    markdown("""
    Next: **02_compare_and_decide.ipynb**. Test the judge, change the retrieved policy and inspect what improved.
    """),
])

write("02_compare_and_decide.ipynb", [
    markdown("""
    # 2 · Compare and decide

    **What would make the repair acceptable?** Change the retrieved policy. Keep the cases and checks fixed.
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

    Can it distinguish a customer's request from an assistant's claim that a transaction happened?
    """),
    code("""
    controls = None
    if not session.has_execution_failure():
        controls = session.qualify_judge()
    display(HTML(controls_card(controls)))
    """, hidden=True, cell_id="02-control-examples"),
    markdown("""
    ## Evaluate the repair

    Use the same MLflow call on both configurations. Reuse completed batches. Failed controls or a recorded execution failure stop further provider work.
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
    ## Code view 3: apply the declared policy

    The workshop's acceptance rules are separate from MLflow's measurements.
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
    ## Inspect in MLflow: repair → disputed assessment

    Check the repaired day-45 source and answer. Then inspect any material finding's case.
    MLflow records the trace and scores. Our review records whether an issue remains open.
    """),
    code("""
    display(HTML(inspection_card(session, "repair", ["day_45_opening", *finding_cases(report)])))
    """, hidden=True, cell_id="02-mlflow"),
    markdown("""
    ## Inspect the pairs and review

    Compare the same case before and after. A 0-to-0 score is a tie, even on a protected case.
    A completed review can leave a material finding open.
    """),
    code("""
    display(HTML(paired_summary(report)))
    """, hidden=True, cell_id="02-pairs"),
    code("""
    display(HTML(review_card(report)))
    """, hidden=True, cell_id="02-review-evidence"),
    markdown("""
    ## Vote before revealing the decision

    **Pause for 20 seconds.** PASS, BLOCK, REVIEW REQUIRED, or INVALID EVIDENCE?
    Name the evidence that would change your decision.
    """),
    code("""
    display(HTML(decision_card(report)))
    """, hidden=True, cell_id="02-decision"),
    markdown("""
    ## Turn the finding into a future test

    Preserve the original answer and score. Have a human approve the new case label.
    Qualify a changed judge, then rescore all frozen answers with that same definition.
    Never patch one disagreeable score.

    Adapt the application, cases, checks and policy using the README. Keep the owner and evidence for your next decision explicit.
    """),
    code("""
    display(HTML(statistics_card(report)))
    """, hidden=True),
])

print("Wrote three portable workshop notebooks.")
