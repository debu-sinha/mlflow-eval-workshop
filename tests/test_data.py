from tests.fakes import policy_label
from west_workshop.data import (
    CURRENT_POLICY,
    STALE_POLICY,
    calibration_dataset,
    dataset,
    dataset_digest,
    judge_validation_dataset,
    opening_case,
)


def test_every_reference_label_follows_the_current_policy():
    for row in dataset():
        case = row["inputs"]
        expected = policy_label(case["days_since_purchase"], case["defective"])
        assert row["expectations"]["expected_decision"] == expected, case["case_id"]
        assert row["expectations"]["policy"] == CURRENT_POLICY


def test_ten_named_cases_with_stable_identity():
    rows = dataset()
    ids = [row["inputs"]["case_id"] for row in rows]
    assert len(ids) == len(set(ids)) == 10
    assert dataset_digest() == dataset_digest(dataset())
    rows[0]["inputs"]["question"] = "changed"
    assert dataset_digest() != dataset_digest(rows), "the digest must change when a case changes"
    assert dataset()[0]["inputs"]["question"] != "changed", "dataset() must return a fresh copy"


def test_stale_policy_differs_only_in_version_and_window():
    restored = (STALE_POLICY.replace("version 2024-01", "version 2026-09")
                .replace("within 90 days", "within 30 days")
                .replace("including day 90", "including day 30")
                .replace("After 90 days", "After 30 days"))
    assert restored == CURRENT_POLICY
    assert STALE_POLICY != CURRENT_POLICY


def test_stale_window_changes_exactly_three_labels():
    changed = sorted(
        row["inputs"]["case_id"] for row in dataset()
        if policy_label(row["inputs"]["days_since_purchase"], row["inputs"]["defective"], window=90)
        != row["expectations"]["expected_decision"]
    )
    assert changed == ["day_31_boundary", "day_45_opening", "day_90_stale_boundary"]


def test_opening_case_is_the_45_day_request():
    (row,) = opening_case()
    assert row["inputs"] == {
        "case_id": "day_45_opening",
        "days_since_purchase": 45,
        "defective": False,
        "question": "I bought this 45 days ago. Can you give me a full refund?",
    }
    assert row["expectations"]["expected_decision"] == "store_credit"


def test_calibration_labels_are_authored_and_distinct_from_the_release_cases():
    rows = calibration_dataset()
    ids = [row["inputs"]["case_id"] for row in rows]
    assert len(rows) == len(set(ids)) == 6
    assert not set(ids) & {row["inputs"]["case_id"] for row in dataset()}
    for row in rows:
        assert isinstance(row["outputs"], str) and row["outputs"]
        assert row["expectations"]["authored_human_label"] is row["inputs"]["case_id"].startswith("label_correct")


def test_judge_controls_are_balanced_pairs():
    rows = judge_validation_dataset()
    labels = [row["expectations"]["authored_human_label"] for row in rows]
    assert len(rows) == 8 and labels.count(True) == labels.count(False) == 4
    for accepted, rejected in zip(rows[::2], rows[1::2]):
        assert accepted["expectations"]["authored_human_label"] is True
        assert rejected["expectations"]["authored_human_label"] is False
        assert accepted["inputs"]["question"] == rejected["inputs"]["question"]
        assert accepted["inputs"]["days_since_purchase"] == rejected["inputs"]["days_since_purchase"]
    release_ids = {row["inputs"]["case_id"] for row in dataset()}
    assert not {row["inputs"]["case_id"] for row in rows} & release_ids
