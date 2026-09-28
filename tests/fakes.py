"""Offline test doubles for the two model calls in the workshop.

The fake chat client stands in for the provider only. Everything else in these
tests is real: MLflow tracing, mlflow.genai.evaluate, the SQLite tracking store,
the deterministic scorers, the release gate, and the report.
"""

import json
import re

from west_workshop.data import CURRENT_POLICY, STALE_POLICY

FAKE_KEY = "sk-test-offline-0000000000000000"


def policy_label(days, defective, window=30):
    """The refund rule written out in code, used to check every authored label."""
    if defective:
        return "support_review"
    return "full_refund" if days <= window else "store_credit"


def canned_answer(days, defective, window):
    label = policy_label(days, defective, window)
    if label == "support_review":
        text = "A defective item needs support review. Please contact support for next steps."
    elif label == "full_refund":
        text = f"You are within the {window}-day window, so a full refund is available. Start a return from your order page."
    else:
        text = f"Your purchase is outside the {window}-day window, so store credit is available. Contact support to request it."
    return f"Eligibility: {label}\n{text}"


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class FakeChatClient:
    """Answers like a model that follows whichever policy it was given."""

    def __init__(self):
        self.chat = self
        self.completions = self
        self.calls = []

    def create(self, *, model, messages, temperature, max_tokens):
        system, user = messages[0]["content"], json.loads(messages[1]["content"])
        if STALE_POLICY in system:
            window = 90
        elif CURRENT_POLICY in system:
            window = 30
        else:
            raise AssertionError("The system prompt must contain exactly one known policy.")
        self.calls.append({"model": model, "window": window, **user})
        return _Response(canned_answer(user["days_since_purchase"], user["defective"], window))


def fake_policy_judge(provider=None, *, rationale_first=True, name="policy_judge"):
    """A deterministic stand-in for the LLM judge with the same name and inputs.

    Authored controls carry their reviewed label, which this double returns so the
    plumbing around the judge can be tested without a model. Generated answers are
    judged by the policy rule and by the stale 90-day window in the explanation.
    """
    from mlflow.genai.scorers import scorer

    @scorer(name=name)
    def judge(inputs, outputs, expectations):
        if "authored_human_label" in expectations:
            return bool(expectations["authored_human_label"])
        declared = re.search(r"(?m)^Eligibility: (\w+)", outputs or "")
        expected = policy_label(inputs["days_since_purchase"], inputs["defective"])
        return bool(declared and declared.group(1) == expected and "90-day" not in outputs)

    return judge
