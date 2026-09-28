import threading

import pytest

from west_workshop.support_app import ORDERS, SupportService, create_app

CHAT = {"Content-Type": "application/json", "X-Northstar-Request": "chat"}


class FakeService:
    def __init__(self, answer=None, error=None, report=None):
        self.provider = "openai"
        self.lock = threading.Lock()
        self._answer, self._error, self._report = answer, error, report

    def answer(self, order, question, variant):
        if self._error:
            raise self._error
        return {"answer": self._answer, "order_key": order, "variant": variant, "question_seen": question}

    def report(self):
        return self._report


@pytest.fixture
def client():
    return create_app(FakeService(answer="Eligibility: store_credit\nHello.")).test_client()


def test_pages_and_headers(client):
    page = client.get("/")
    assert page.status_code == 200 and b"Ask Northstar" in page.data
    assert page.headers["X-Content-Type-Options"] == "nosniff"
    assert page.headers["Cache-Control"] == "no-store"
    assert client.get("/health").get_json() == {"status": "ok"}
    config = client.get("/api/config").get_json()
    assert set(config["orders"]) == set(ORDERS) and set(config["policies"]) == {"repaired", "candidate"}


def test_chat_requires_the_page_header_and_same_origin(client):
    body = {"order": "day_45", "variant": "repaired", "question": "Refund?"}
    assert client.post("/api/chat", json=body).status_code == 403
    cross = dict(CHAT, Origin="https://attacker.example")
    assert client.post("/api/chat", json=body, headers=cross).status_code == 403


@pytest.mark.parametrize("body", [
    {"order": "day_99", "variant": "repaired", "question": "Refund?"},
    {"order": "day_45", "variant": "baseline", "question": "Refund?"},
    {"order": "day_45", "variant": "repaired", "question": "   "},
    {"order": "day_45", "variant": "repaired", "question": "x" * 1201},
])
def test_chat_rejects_invalid_requests(client, body):
    assert client.post("/api/chat", json=body, headers=CHAT).status_code == 400


def test_chat_answers_one_request_at_a_time():
    service = FakeService(answer="Eligibility: store_credit\nHello.")
    client = create_app(service).test_client()
    body = {"order": "day_45", "variant": "candidate", "question": "  Refund?  "}
    response = client.post("/api/chat", json=body, headers=CHAT)
    assert response.status_code == 200
    assert response.get_json()["question_seen"] == "Refund?"
    assert service.lock.acquire(blocking=False), "the lock is released after the answer"
    busy = client.post("/api/chat", json=body, headers=CHAT)
    assert busy.status_code == 429
    service.lock.release()


def test_provider_errors_never_reach_the_browser():
    secret = "sk-live-secret-value"
    client = create_app(FakeService(error=RuntimeError(secret))).test_client()
    response = client.post("/api/chat", json={"order": "day_45", "variant": "repaired", "question": "Refund?"}, headers=CHAT)
    assert response.status_code == 503 and secret.encode() not in response.data


def test_report_page_states():
    assert create_app(FakeService(report=None)).test_client().get("/report").status_code == 404
    ok = create_app(FakeService(report="<html>report</html>")).test_client().get("/report")
    assert ok.status_code == 200 and ok.data == b"<html>report</html>"


def test_the_real_service_returns_verified_trace_evidence(offline_openai, tmp_path):
    from west_workshop.publish_report import publish
    from west_workshop.runtime import run_checkpoint

    service = SupportService()
    stale = service.answer("day_45", "I bought this 45 days ago. Can you give me a full refund?", "candidate")
    assert stale["eligibility"] == "full_refund" and stale["policy_version"] == "2024-01"
    assert "within 90 days" in stale["retrieved_policy"] and stale["trace_id"].startswith("tr-")
    current = service.answer("day_45", "I bought this 45 days ago. Can you give me a full refund?", "repaired")
    assert current["eligibility"] == "store_credit" and current["policy_version"] == "2026-09"
    assert current["evaluated"] is False
    assert current["trace_url"] == (f"http://127.0.0.1:5000/#/experiments/{service.experiment_id}"
                                    f"/traces?selectedEvaluationId={current['trace_id']}")

    assert service.report() is None
    summary = run_checkpoint(4, "openai")
    assert summary["status"] == "passed"
    published = publish(summary["summary_path"], provider="openai")
    assert published["model_calls"] == 0
    assert ">SHIP<" in service.report()


@pytest.mark.parametrize(("value", "expected"), [
    ("http://127.0.0.1:5000", "http://127.0.0.1:5000"),
    ("http://localhost:5001/", "http://localhost:5001"),
    ("https://127.0.0.1:5000", None),
    ("http://example.com:5000", None),
    ("http://127.0.0.1:5000/path", None),
    ("http://user:pass@127.0.0.1:5000", None),
])
def test_local_trace_links_only_point_at_loopback(value, expected):
    from west_workshop.support_app import _local_ui_url

    assert _local_ui_url(value) == expected
