import ast
from pathlib import Path
import re

import pytest

from west_workshop.runtime import build_scorers

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "notebooks" / "west" / "02_build_the_scorer_stack.py"
DAY_45 = {"expected_decision": "store_credit"}


def _score(check, **kwargs):
    # Scorer.run passes only the arguments each scorer declares, as evaluation does.
    return _value(check.run(**kwargs))


def _value(feedback):
    value = getattr(feedback, "value", feedback)
    return {"yes": True, "no": False}.get(str(value).lower(), value) if isinstance(value, str) else value


@pytest.fixture(scope="module")
def checks():
    return {item.name: item for item in build_scorers("openai", include_judge=False)}


def test_the_stack_has_five_checks_and_one_ensemble(checks):
    assert list(checks) == ["eligibility_format", "response_length", "pii_detection",
                            "policy_decision", "no_false_transaction", "deterministic_stack"]


def test_a_correct_answer_passes_every_check(checks):
    answer = "Eligibility: store_credit\nYour purchase is outside the 30-day window, so store credit is available."
    for name, check in checks.items():
        assert _score(check, outputs=answer, expectations=DAY_45) is True, name


def test_the_stale_answer_fails_policy_but_passes_format(checks):
    answer = "Eligibility: full_refund\nYou're within the 90-day window for a full refund."
    assert _score(checks["eligibility_format"], outputs=answer) is True
    assert _score(checks["policy_decision"], outputs=answer, expectations=DAY_45) is False
    assert _score(checks["deterministic_stack"], outputs=answer, expectations=DAY_45) is False


@pytest.mark.parametrize("claim", [
    "Eligibility: full_refund\nI have processed your refund and the money is on its way.",
    "Eligibility: full_refund\nYour refund has been approved.",
    "Eligibility: store_credit\nWe issued the store credit to your account.",
])
def test_the_phrase_tripwire_catches_explicit_execution_claims(checks, claim):
    assert _score(checks["no_false_transaction"], outputs=claim) is False


def test_the_phrase_tripwire_misses_a_future_promise(checks):
    # The notebook asks attendees why this gap exists. The semantic judge covers it.
    promise = "Eligibility: store_credit\nI will issue the store credit to your account now."
    assert _score(checks["no_false_transaction"], outputs=promise) is True


def test_customer_next_steps_are_not_execution_claims(checks):
    answer = "Eligibility: store_credit\nVisit your account to request it or contact support to claim it."
    assert _score(checks["no_false_transaction"], outputs=answer) is True


def test_pii_and_length_checks(checks):
    assert _score(checks["pii_detection"], outputs="Eligibility: support_review\nEmail jane@example.com.") is False
    assert _score(checks["response_length"], outputs="x" * 19) is False
    assert _score(checks["response_length"], outputs="x" * 20) is True


def _lab_examples():
    tree = ast.parse(LAB.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "format_examples":
            return ast.literal_eval(node.value)
    raise AssertionError("format_examples not found in notebook 02")


def _lab_function(marker):
    source = LAB.read_text(encoding="utf-8")
    if marker == "starter":
        match = re.search(r"def one_eligibility_line\(outputs: str\) -> bool:\n(.*?)\n\n", source, re.S)
        body = match.group(1)
    else:
        block = re.search(r"# MAGIC ```python\n(.*?)# MAGIC ```", source, re.S).group(1)
        lines = [line.removeprefix("# MAGIC ").removeprefix("# MAGIC") for line in block.splitlines()]
        body = "\n".join("    " + line for line in lines)
    namespace = {}
    exec("def one_eligibility_line(outputs: str) -> bool:\n" + body, namespace)
    return namespace["one_eligibility_line"]


def test_lab_starter_disagrees_with_two_examples_and_solution_agrees_with_all():
    examples = _lab_examples()
    assert len(examples) == 4
    starter, solution = _lab_function("starter"), _lab_function("solution")
    assert sum(starter(answer) == expected for _, answer, expected in examples) == 2
    assert sum(solution(answer) == expected for _, answer, expected in examples) == 4
