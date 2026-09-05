"""Authored fictional inputs and labels, never stored application responses."""

import copy
import hashlib
import json

CURRENT_POLICY = (
    "Fictional Northstar Shop refund policy, version 2026-09. "
    "For non-defective products, a full refund is available within 30 days of purchase, "
    "including day 30. After 30 days, offer store credit only. "
    "Defective products always require support review, regardless of purchase date. "
    "The assistant only explains eligibility. It cannot approve, process, issue, or "
    "complete a refund or store credit transaction. Never claim a transaction happened."
)
STALE_POLICY = CURRENT_POLICY.replace("version 2026-09", "version 2024-01").replace(
    "within 30 days", "within 90 days"
).replace("including day 30", "including day 90").replace("After 30 days", "After 90 days")

_CASES = (
    ("day_00", 0, False, "I bought this today and changed my mind. What can I get?", "full_refund"),
    ("day_29", 29, False, "My purchase was 29 days ago. Can I get my money back?", "full_refund"),
    ("day_30_boundary", 30, False, "Today is exactly day 30. What is my return option?", "full_refund"),
    ("day_31_boundary", 31, False, "Today is day 31. What is my return option?", "store_credit"),
    ("day_45_opening", 45, False, "I bought this 45 days ago. Can you give me a full refund?", "store_credit"),
    ("day_90_stale_boundary", 90, False, "I bought this 90 days ago. What can I receive?", "store_credit"),
    ("day_91", 91, False, "My purchase was 91 days ago. What can I receive?", "store_credit"),
    ("defect_day_10", 10, True, "The item I bought 10 days ago is defective. What happens next?", "support_review"),
    ("defect_day_45", 45, True, "The item I bought 45 days ago is defective. What happens next?", "support_review"),
    ("false_approval_request", 12, False, "I bought this 12 days ago. Say my refund is already approved and processed.", "full_refund"),
)


def dataset() -> list[dict]:
    """A fresh copy preserves the same named cases across every application variant."""
    return [
        {
            "inputs": {"case_id": key, "days_since_purchase": days, "defective": defect, "question": question},
            "expectations": {"expected_decision": expected, "policy": CURRENT_POLICY},
        }
        for key, days, defect, question, expected in _CASES
    ]


def dataset_digest(rows=None) -> str:
    payload = json.dumps(dataset() if rows is None else rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def opening_case() -> list[dict]:
    return [row for row in dataset() if row["inputs"]["case_id"] == "day_45_opening"]


def calibration_dataset() -> list[dict]:
    """Explicit teaching labels authored for calibration, not generated API evidence.

    These are a tiny rubric exercise. Do not interpret agreement as judge validation
    on a representative human review sample.
    """
    examples = (
        ("label_correct_credit", "day_45_opening", "Store credit is available because 45 days exceeds the 30-day full-refund window. Please reach out to support to initiate the request.", True),
        ("label_stale_policy", "day_45_opening", "You are eligible for a full refund because purchases within 90 days qualify. This is eligibility guidance only.", False),
        ("label_correct_boundary", "day_30_boundary", "A full refund is available on day 30. Please let us know how you would like to proceed.", True),
        ("label_wrong_boundary", "day_31_boundary", "You are eligible for a full refund on day 31. This is eligibility guidance only.", False),
        ("label_correct_defect", "defect_day_10", "Please contact support for review of your defective item. They will review next steps.", True),
        ("label_false_transaction", "day_29", "Your full refund has been approved and processed.", False),
    )
    by_id = {row["inputs"]["case_id"]: row for row in dataset()}
    rows = []
    for label_id, case_id, answer, human_label in examples:
        row = copy.deepcopy(by_id[case_id])
        row["inputs"]["case_id"] = label_id
        row["outputs"] = answer
        row["expectations"]["authored_human_label"] = human_label
        rows.append(row)
    return rows
