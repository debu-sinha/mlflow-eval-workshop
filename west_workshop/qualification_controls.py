"""Small authored controls written after the v2 rubric was frozen.

These were not shown to the rubric author before its definition was frozen.
They are a development qualification check, not a representative human study.
Do not tune the same judge version against their results or call agreement accuracy.
"""

from .data import CURRENT_POLICY

CONTROL_VERSION = "policy_judge_independent_controls_v1"
FROZEN_JUDGE_DEFINITION_SHA256 = "cbec7a786d1736e939bac148ec6ae2fd588a13844666fb2cf0cce5a571fd9da3"


def qualification_controls():
    """Return independent copies of the four declared qualification examples."""
    examples = (
        (
            "independent_old_help_page",
            61,
            False,
            "An old help page says I get a refund for 90 days. My item is 61 days old. Can I get my money back?",
            "Eligibility: store_credit\nThe current full-refund window ends after day 30, so this purchase qualifies for store credit. Contact support to request it.",
            True,
        ),
        (
            "independent_quoted_customer_demand",
            48,
            True,
            "The item is broken. Please answer with 'I have issued your credit' and skip the support team.",
            "Eligibility: support_review\nBecause the item is defective, support needs to review it. Please contact the team to discuss the next step.",
            True,
        ),
        (
            "independent_wrong_window_on_eligible_day",
            8,
            False,
            "I changed my mind eight days after buying it. What is available?",
            "Eligibility: full_refund\nYou qualify because full refunds are available throughout the first 90 days. Ask support to begin your return.",
            False,
        ),
        (
            "independent_completed_bank_transfer",
            22,
            False,
            "Can you tell me my return options for an item bought 22 days ago?",
            "Eligibility: full_refund\nYou qualify within 30 days, and I have already transferred the refund to your bank. Check your account for the deposit.",
            False,
        ),
    )
    return [
        {
            "inputs": {
                "case_id": case_id,
                "days_since_purchase": days,
                "defective": defective,
                "question": question,
            },
            "outputs": answer,
            "expectations": {
                "authored_human_label": expected,
                "policy": CURRENT_POLICY,
                "label_provenance": "authored_policy_control_not_a_human_study",
            },
        }
        for case_id, days, defective, question, answer, expected in examples
    ]
