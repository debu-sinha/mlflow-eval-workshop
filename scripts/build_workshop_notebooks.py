"""Rebuild the three editable primary notebooks from their short teaching cells."""

import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "workshop"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}


def code(source, *, setup=False):
    return {"cell_type": "code", "execution_count": None, "metadata": {"tags": ["setup"]} if setup else {},
            "outputs": [], "source": textwrap.dedent(source).strip() + "\n"}


BOOTSTRAP = """
import os, sys, json
from pathlib import Path

# Works in local Jupyter and a Databricks Git folder; notebooks have no __file__.
start = Path(os.environ.get("WORKSHOP_PROJECT_ROOT", Path.cwd())).resolve()
ROOT = next((p for p in (start, *start.parents) if (p / "west_workshop" / "data.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Set WORKSHOP_PROJECT_ROOT to the workshop checkout, then restart this notebook.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
MODE = os.environ.get("WORKSHOP_MODE", "saved")  # Explicit opt-in before any model calls.
"""


def write(name, cells):
    document = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3 (workshop)", "language": "python", "name": "python3"},
                                             "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    number = 0
    for cell in document["cells"]:
        if "id" not in cell:
            cell["id"] = name[:2] + "-" + str(number).zfill(2)
            number += 1
    (DESTINATION / name).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


DESTINATION.mkdir(parents=True, exist_ok=True)
write("00_setup.ipynb", [
    markdown("""
    # Start here: watch, inspect, or run

    **Question for today:** what evidence would justify accepting a repair to a support assistant?

    The primary route is this setup plus two notebooks. You can participate by reading and classifying an answer; Python and a cloud account are optional for that route.

    **Saved mode** opens dated, hash-verified real outputs. It makes no model or network calls and needs no credentials. **Live mode** regenerates answers and scores with the configured models. Its result can differ. A workshop policy pass never authorizes production deployment.
    """),
    code(BOOTSTRAP, setup=True),
    markdown("""
    ## 1. Check this kernel and the selected mode

    Default: `saved`. After downloading the release, run all cells in order. If an evidence file is missing or changed, restore the release; no substitute passing result is generated.

    To run live, set `WORKSHOP_MODE=live` in the environment before starting Jupyter. On Databricks, add the same nonsecret environment settings before the setup code in each notebook. Fresh kernels do not share Python variables.
    """),
    code("""
    from west_workshop.repair_workshop import workshop_preflight

    readiness = workshop_preflight(MODE)
    print(json.dumps(readiness, indent=2))
    """),
    markdown("""
    ## 2. Choose your route

    **Local:** install the committed lock with `uv sync --locked --python 3.12 --extra local`. Start `uv run --locked --extra local jupyter lab` from the checkout. The live route uses the OpenAI key already configured in that terminal. Never paste a key into a notebook.

    **Databricks serverless:** open these same `.ipynb` files in a Git folder. Select the verified base environment and the committed requirements file described in the README. For live mode, set an available model endpoint and an absolute `MLFLOW_EXPERIMENT_NAME` in each notebook's first cell. Workspace authentication stays with your notebook identity; do not copy tokens. The preflight above checks configuration presence only, and never resolves credentials or contacts a service.

    Each live notebook uses `WORKSHOP_OUTPUT_DIR` to persist evidence across kernels. Point both at the same writable directory. The default local directory is `artifacts/workshop/<provider>/live`. Use a new directory when changing code, models, cases, policy or judge versions. Preserve the previous directory and state the reason for the new experiment.
    """),
    code("""
    if readiness["ready"]:
        print("Ready for 01_diagnose_and_evaluate.ipynb")
    else:
        for reason in readiness["reasons"]:
            print("Setup issue:", reason)
        print("Observation is a complete participation route. Follow the presenter's saved evidence while fixing setup later.")
    """),
    markdown("""
    ## 3. Know what is being compared

    | Configuration | Retrieved source | What changes? |
    |---|---|---|
    | Simulated deployed assistant | Obsolete 90-day policy | Starting defect |
    | Proposed repair | Current 30-day policy | Retrieved policy only |

    A **case** is a reviewed input with a stable ID and expected behavior. A **scorer** checks one property. A **judge** is a model that scores meaning. A **trace** records what the application did. A **policy** decides whether the complete evidence meets declared conditions.

    Next: **01_diagnose_and_evaluate.ipynb**. Keep the same case ID in view as it travels from input to trace to score.
    """),
])

write("01_diagnose_and_evaluate.ipynb", [
    markdown("""
    # 1 · Diagnose and evaluate

    **Can a fluent answer use the wrong policy?** We will follow `day_45_opening`: a customer bought a non-defective item 45 days ago and asks for a full refund.

    The starting assistant is a simulated deployed configuration with an obsolete source. No actual production deployment is implied. Read the mode, date and evidence ID below before interpreting a score.

    **Presentation route:** run all cells before presenting, start at **Inspect one answer, then its source**, and return to the two short code views. This changes the reading order only; it makes no duplicate application calls.
    """),
    code(BOOTSTRAP, setup=True),
    code("""
    from IPython.display import display, HTML, Code
    from west_workshop.repair_workshop import notebook_session, show_case, evaluation_table

    session = notebook_session(MODE)
    print(json.dumps(session.banner(), indent=2))
    """),
    markdown("""
    ## Code view 1: one business check

    The shared release scorer below checks the declared eligibility against the reviewed label. `eligibility_declaration` also requires exactly one declaration: a correct first label followed by a contradictory second label fails. The semantic judge separately checks the explanation and transaction claims.
    """),
    code("""
    import inspect
    from west_workshop.runtime import policy_decision_value

    display(Code(inspect.getsource(policy_decision_value), language="python"))
    """),
    markdown("""
    ## Code view 2: the actual MLflow call

    `data` contains the ten named cases. `predict_fn` is the real traced application. `scorers` contains deterministic checks and the versioned semantic judge. Results keep the same case IDs.

    In live mode this cell makes one application request per case. In saved mode it reads the original result. Reopening an already recorded stage does not generate another batch. The full adapter and persistence code remain editable in `west_workshop/`.
    """),
    code("""
    if session.needs("deployed"):
        import mlflow
        with session.evaluation("deployed") as run:
            result = mlflow.genai.evaluate(
                data=run.data, predict_fn=run.predict_fn, scorers=run.scorers,
            )
            session.record_evaluation("deployed", result, run)

    deployed = session.recorded("deployed")
    print("Recorded cases:", deployed["row_count"], "| Complete:", deployed["complete"])
    if not deployed["complete"]:
        raise RuntimeError("This evaluation is incomplete. Its partial evidence is preserved; inspect the failure before retrying in a new directory.")
    """),
    markdown("""
    ## Inspect one answer, then its source

    **Pause for 15 seconds:** Does this answer deserve a full refund, store credit, or support review? You can answer without reading Python. The presenter will read the answer and current policy aloud.
    """),
    code("""
    opening = show_case(session, "deployed", "day_45_opening")
    """),
    code("""
    from west_workshop.data import CURRENT_POLICY

    print("AUTHORITATIVE CURRENT POLICY\\n" + CURRENT_POLICY)
    print("\\nSOURCE RETRIEVED IN THIS TRACE")
    retrieval = [span for span in opening.get("spans", []) if span["name"] == "retrieve_refund_policy"]
    print(json.dumps(retrieval[0]["outputs"], indent=2) if retrieval else "Retrieval span is missing from this evidence. The source cannot be verified from this recorded trace.")
    """),
    markdown("""
    **Prepared explanation:** At day 45, a non-defective item qualifies for store credit under the current policy. An assistant that follows the obsolete 90-day window can sound helpful while giving the wrong entitlement. The trace locates the wrong source; the score makes that failure repeatable.

    A correct label still does not prove a correct explanation. For example, a day-30 answer can select `full_refund` correctly while explaining an obsolete 90-day rule.
    """),
    code("""
    display(HTML(evaluation_table(deployed)))
    """),
    markdown("""
    ## What did these checks establish?

    Format and label checks establish narrow properties. The phrase tripwire catches some false claims that a transaction happened. The semantic judge examines meaning, and can itself be wrong. A failed check is evidence to inspect, not an automatic explanation of root cause.

    **Silent-answer prompt:** which check could catch a correct label with a wrong 90-day explanation? **Prepared explanation:** the semantic judge should; the label check cannot.

    Next: **02_compare_and_decide.ipynb**. We test the evaluator, hold the cases and checks fixed, and decide whether to accept the proposed repair under a declared workshop policy.
    """),
])

write("02_compare_and_decide.ipynb", [
    markdown("""
    # 2 · Compare and decide

    **What would make the repair acceptable?** A score is evidence; acceptance is a separately declared policy. We keep execution status, evaluator controls, review findings and the policy result distinct.

    The repair changes only the retrieved policy. A fresh kernel can run this notebook independently. When notebook 01 already recorded the deployed batch in the same output directory, it is reused exactly once.
    """),
    code(BOOTSTRAP, setup=True),
    code("""
    from IPython.display import display, HTML
    from west_workshop.repair_workshop import notebook_session, paired_table, show_decision, render_report

    session = notebook_session(MODE)
    print(json.dumps(session.banner(), indent=2))
    """),
    markdown("""
    ## Test the judge before interpreting a comparison

    The original eight authored controls are preserved. Eight development controls encode known errors. Four separately authored controls were held back until the new rubric was frozen. Both positive and negative answers have explicit labels. Agreement on these 20 examples is a **smoke test**, not a representative accuracy estimate.

    The old judge and its dated errors remain historical evidence. Failed controls are retained. A control failure prevents a readiness claim; we do not keep retrying for green. A recorded execution failure stops further provider work and stays visible in the decision below.
    """),
    code("""
    controls = None
    if session.has_execution_failure():
        print("Controls skipped: a recorded execution failed. Its evidence is preserved.")
    else:
        controls = session.qualify_judge()
        print("Controls:", "PASSED" if controls["passed"] else "FAILED")
        print("Disagreements:", controls["disagreements"])
        print("Control version:", controls["version"])
        print("Smoke coverage only; no population accuracy claim.")
    """),
    dict(code("""
    # Display two authored controls from the recorded evaluation, without rescoring.
    from html import escape

    control_examples = []
    control_rows = (controls or {}).get("evaluation", {}).get("rows", [])
    for control_id in ("review_customer_instruction", "review_execution_claim"):
        row = next((item for item in control_rows if item.get("case_id") == control_id), None)
        if row is None:
            continue
        expected = row["expectations"]["authored_human_label"]
        actual = row.get("scores", {}).get("policy_judge", "Not recorded")
        control_examples.append(
            "<h3>" + ("Should pass" if expected else "Should fail") + "</h3>"
            + "<p>Authored expected grade: <strong>" + str(int(expected))
            + "</strong> · Recorded judge grade: <strong>" + escape(str(actual)) + "</strong></p>"
            + "<p>Purchase age: " + escape(str(row["inputs"]["days_since_purchase"]))
            + " days · Defective: " + ("yes" if row["inputs"]["defective"] else "no") + "</p>"
            + "<p>Customer: " + escape(row["inputs"]["question"]) + "</p>"
            + "<pre style='white-space:pre-wrap;font:inherit'>" + escape(row["output"]) + "</pre>"
            + "<p style='font-size:14px'>Control: " + escape(control_id) + "</p>"
        )
    if control_examples:
        display(HTML("<section style='font-size:20px;line-height:1.4'>"
                     + "<h2>Test the judge on answers with known expectations</h2>"
                     + "<p>1 = rubric pass. 0 = rubric fail. These two examples illustrate the full 20-control result above.</p>"
                     + "<hr>".join(control_examples) + "</section>"))
    else:
        print("These control examples are unavailable in the recorded evidence.")
    """), id="02-control-examples", metadata={"tags": ["control-display"], "jupyter": {"source_hidden": True}}),
    markdown("""
    ## Generate and evaluate the proposed repair

    This is the same native MLflow API and the same cases and checks as notebook 01. Saved mode uses the frozen evidence. Live mode reuses any completed deployed batch and generates only missing batches. If the controls fail, it preserves that outcome and withholds the comparison.
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
                    print("Further evaluation skipped: this batch has an execution failure.")
                    break
    """),
    markdown("""
    ## Code view 3: declare the policy, then apply it

    Both answer sets must have complete comparable cases, valid binary scores and identical evaluator definitions. The candidate must pass every mandatory check and its absolute floor. Ordinary losses have a fixed allowance; protected-case losses cannot be offset by gains. The original assistant is allowed to contain the defect being repaired.

    Controls must pass. A material unresolved evaluator finding requires review. A model review is labelled as such and is not a claim that a human signed off.
    """),
    code("""
    from west_workshop.repair_policy import DEFAULT_POLICY, evaluate_repair

    display({key: DEFAULT_POLICY[key] for key in (
        "version", "mandatory_checks", "quality_floor", "max_regression_rate", "protected_checks",
    )})
    review_path = os.environ.get("WORKSHOP_REVIEW_JSON")
    reviewed = json.loads(Path(review_path).read_text(encoding="utf-8")) if review_path else None
    comparison = session.policy_inputs(reviewed)
    policy_result = evaluate_repair(**comparison)
    report = session.finish(comparison["review"])
    assert report["policy_result"] == policy_result
    """),
    markdown("""
    ## Inspect the pairs

    A paired row compares the **same case ID** before and after. A loss means a required check changes from 1 to 0. Protected cases allow no such loss. A 0-to-0 score is still a tie, not a new loss. That does not make the answer acceptable: mandatory candidate checks and unresolved material review findings have their own conditions.

    The protected-loss rule is not an absolute passing-score requirement for every protected case. Add that separate requirement if your application needs it. If evaluator evidence is invalid, no paired comparison is interpreted.

    """),
    code("""
    display(HTML(paired_table(report)))
    """),
    dict(code("""
    # Read-only evidence for the vote: no scoring, policy changes or network calls.
    from html import escape

    def review_text(value):
        return escape(str(value)) if value is not None else "Not recorded"

    recorded_review = report.get("review") or {}
    review_parts = ["<section style='font-size:18px;line-height:1.45'><h3>Recorded review</h3>",
                    "<p>Status: <strong>" + review_text(recorded_review.get("status")) + "</strong> · Reviewer: "
                    + review_text(recorded_review.get("reviewer_kind")) + " / "
                    + review_text(recorded_review.get("reviewer_id")) + "</p>",
                    "<p>A completed review can leave findings unresolved. An open material finding withholds acceptance.</p>"]
    reviewed_versions = {"deployed": comparison["baseline"], "repair": comparison["candidate"]}
    for finding in recorded_review.get("findings", []):
        collapsed = finding.get("material") is False
        if collapsed:
            review_parts.append("<details><summary>Other recorded finding: " + review_text(finding.get("id"))
                                + " · " + review_text(finding.get("status")) + " · nonmaterial</summary>")
        review_parts += ["<hr><p><strong>" + review_text(finding.get("id")) + ": "
                         + review_text(finding.get("summary")) + "</strong></p>",
                         "<p>Status: " + review_text(finding.get("status")) + " · Material: "
                         + review_text(finding.get("material")) + "</p>"]
        variant, case_id, check = (finding.get(key) for key in ("variant", "case_id", "check"))
        version = reviewed_versions.get(variant, {}) if isinstance(variant, str) else {}
        row = next((item for item in version.get("rows", []) if item.get("case_id") == case_id), None)
        if row is not None:
            review_parts += ["<p>Evidence: " + review_text(variant) + " / " + review_text(case_id) + "</p>",
                             "<p><strong>Customer input</strong></p><pre style='white-space:pre-wrap'>"
                             + review_text(json.dumps(row["inputs"], ensure_ascii=False, indent=2)) + "</pre>",
                             "<p><strong>Assistant answer</strong></p><pre style='white-space:pre-wrap'>"
                             + review_text(row["output"]) + "</pre>"]
            assessments = [item for item in row.get("assessments", []) if item.get("name") == check]
            for assessment in assessments:
                review_parts += ["<p><strong>" + review_text(check) + " rationale</strong></p>"
                                 + "<pre style='white-space:pre-wrap'>"
                                 + review_text(assessment.get("rationale")) + "</pre>"]
            if not assessments:
                review_parts.append("<p>No recorded assessment matches the finding's named check.</p>")
        elif variant is not None or case_id is not None:
            review_parts.append("<p>The finding's referenced case is unavailable in these compared records.</p>")
        if collapsed:
            review_parts.append("</details>")
    if not recorded_review.get("findings"):
        review_parts.append("<p>No findings recorded. Use the review status above to distinguish pending review from completed review.</p>")
    display(HTML("".join(review_parts) + "</section>"))
    """), id="02-review-evidence", metadata={"tags": ["review-display"], "jupyter": {"source_hidden": True}}),
    markdown("""
    ## Vote before revealing the decision

    **Pause for 20 seconds:** PASS, BLOCK, REVIEW REQUIRED, or INVALID EVIDENCE? Name the evidence that would change your decision.

    Keep the next output below the viewport until the vote. In a preexecuted saved notebook, collapse that output before presenting. The result is the actual recorded outcome, not a target attendees must reproduce.
    """),
    code("""
    display(HTML(render_report(report)))
    show_decision(report)
    """),
    markdown("""
    ## Turn a finding into a future test

    Preserve the original answer and score. Record the exact claim, expected behavior, reviewer and rationale. A human approves a new case label before promotion; automatic feedback is not ground truth. Add a versioned case, qualify a changed judge on controls, then rescore **all frozen compared answers** with the same new definition. Never patch one disagreeable score.

    The take-home helpers are `review_skeleton`, `promote_reviewed_case` and `rescore_frozen` in `west_workshop/repair_workshop.py`. Rescoring uses `predict_fn=None`: it calls the new judge on fixed answers and records differences without generating new answers. The historical v1 decision stays unchanged.
    """),
    markdown("""
    ## Carry this into your application

    Replace `make_predictor` with your traced application adapter. Add reviewed cases with stable IDs. Keep reliable deterministic checks; qualify a judge for the meanings that need one. Declare your mandatory checks, floors, protected cases and review ownership before looking at a candidate.

    ```bash
    python -m west_workshop evaluate --mode saved --evidence evidence/workshop/local/final
    python -m west_workshop gate --mode saved --evidence evidence/workshop/local/final
    ```

    `evaluate` exits **0** for a completed policy result, including BLOCK; **1** means execution failed. `gate` writes its report before enforcing: **0** PASS, **2** BLOCK, **3** REVIEW_REQUIRED, **4** INVALID, **1** execution failure / NOT_EVALUATED. The older `eval_gate.py` checks one named scorer and is a narrower contract.

    **What remains outside this workshop:** representative held-out coverage, human validation of the judge, application security, operational limits, latency/cost budgets and actual production approval. Name who owns the next decision and what evidence they need.
    """),
])

print("Wrote three portable workshop notebooks.")
