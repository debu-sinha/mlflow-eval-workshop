"""A small web interface for live, independently traced refund questions."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from queue import Empty, Queue
import threading
import time
from urllib.parse import urlsplit
import uuid

from flask import Flask, jsonify, render_template, request

from .config import application_model, preflight, selected_provider
from .data import CURRENT_POLICY, STALE_POLICY
from .runtime import WorkshopExecutionError, _ELIGIBILITY_PATTERN, _configure_timeouts, _setup_tracking, make_predictor

ROOT = Path(__file__).resolve().parent
ORDERS = {
    "day_45": {"id": "NS-2045", "days": 45, "defective": False, "label": "45 days ago", "condition": "Working as expected"},
    "day_30": {"id": "NS-2030", "days": 30, "defective": False, "label": "30 days ago", "condition": "Working as expected"},
    "day_31": {"id": "NS-2031", "days": 31, "defective": False, "label": "31 days ago", "condition": "Working as expected"},
    "defect_45": {"id": "NS-2145", "days": 45, "defective": True, "label": "45 days ago", "condition": "Defective item"},
}


class SupportService:
    def __init__(self):
        self.provider = selected_provider()
        self.experiment_id = None
        self.predictors = {}
        # One model request at a time across users, with one server worker.
        self.lock = threading.Lock()

    def configure(self):
        if self.experiment_id is not None:
            return
        readiness = preflight(self.provider)
        if not readiness["ready"]:
            raise WorkshopExecutionError("Complete the app's model and MLflow resource setup.")
        output = Path(os.environ.get("WORKSHOP_OUTPUT_DIR", "artifacts/west-live")).resolve()
        output.mkdir(parents=True, exist_ok=True)
        _configure_timeouts(self.provider)
        os.environ.setdefault("MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT", "20")
        os.environ.setdefault("MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT", "30")
        experiment_id = _setup_tracking(self.provider, output)
        predictors = {variant: make_predictor(self.provider, variant) for variant in ("candidate", "repaired")}
        self.experiment_id, self.predictors = experiment_id, predictors

    def answer(self, order_key, question, variant):
        import mlflow

        self.configure()
        order = ORDERS[order_key]
        request_id = "support-" + uuid.uuid4().hex
        started = time.monotonic()
        answer = self.predictors[variant](case_id=request_id, days_since_purchase=order["days"], defective=order["defective"], question=question)
        trace_id = mlflow.get_last_active_trace_id()
        trace = mlflow.get_trace(trace_id, flush=True) if trace_id else None
        evidence = None
        if trace is not None:
            retrieval = next((span for span in trace.data.spans if span.name == "retrieve_refund_policy"), None)
            root_span = next((span for span in trace.data.spans if span.name == "refund_assistant"), None)
            if (retrieval is not None and root_span is not None and root_span.outputs == answer
                    and isinstance(root_span.inputs, dict) and root_span.inputs.get("case_id") == request_id):
                documents = retrieval.outputs
                if isinstance(documents, list) and documents and isinstance(documents[0], dict):
                    evidence = documents[0]
        # Never substitute an expected policy for missing trace evidence.
        if evidence is None or not isinstance(evidence.get("page_content"), str):
            raise WorkshopExecutionError("The answer's saved retrieval trace could not be verified.")
        eligibility = re.search(_ELIGIBILITY_PATTERN, answer)
        workspace_host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        trace_url = None
        if self.provider == "databricks" and re.fullmatch(r"https://[A-Za-z0-9.-]+", workspace_host):
            trace_url = f"{workspace_host}/ml/experiments/{self.experiment_id}/traces?selectedEvaluationId={trace_id}"
        return {
            "answer": answer,
            "body": answer[eligibility.end():].lstrip(" .:\n\r") if eligibility and eligibility.start() == 0 else answer,
            "eligibility": eligibility.group(1) if eligibility else None,
            "order_key": order_key,
            "variant": variant,
            "trace_id": trace_id,
            "trace_url": trace_url,
            "experiment_id": self.experiment_id,
            "retrieved_policy": evidence.get("page_content"),
            "policy_version": evidence.get("metadata", {}).get("policy_version"),
            "model": application_model(self.provider),
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256((ROOT / "runtime.py").read_text(encoding="utf-8").encode()).hexdigest(),
            "evaluated": False,
        }

    def report(self):
        import mlflow
        from .report import render_release_report

        self.configure()
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="tags.`workshop.report` = 'checkpoint_4'",
            order_by=["attributes.start_time DESC"], max_results=1, output_format="list",
        )
        if not runs:
            return None
        # The newest published comparison is used, regardless of whether it passed.
        summary = mlflow.artifacts.load_dict(f"runs:/{runs[0].info.run_id}/release-summary.json")
        return render_release_report(summary)


def create_app(service=None):
    app = Flask(__name__, template_folder="support_ui", static_folder="support_ui", static_url_path="/assets")
    if os.environ.get("DATABRICKS_APP_NAME"):
        # Databricks terminates HTTPS and forwards the original hostname.
        # Trust its single platform boundary only inside the Apps runtime.
        from werkzeug.middleware.proxy_fix import ProxyFix

        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=0, x_proto=1, x_host=1)
    app.config["MAX_CONTENT_LENGTH"] = 8192
    app.config["REQUEST_TIMEOUT_SECONDS"] = 180
    service = service or SupportService()

    def call_with_deadline(function):
        result = Queue(maxsize=1)

        def work():
            try:
                outcome = (True, function())
            except Exception:
                outcome = (False, None)
            finally:
                service.lock.release()
            result.put(outcome)

        # The worker retains the lock after a timeout so a retry cannot overlap it.
        threading.Thread(target=work, daemon=True).start()
        try:
            success, value = result.get(timeout=app.config["REQUEST_TIMEOUT_SECONDS"])
        except Empty:
            raise TimeoutError from None
        if not success:
            raise WorkshopExecutionError("The request could not be completed.")
        return value

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        # Serving the interface is independent of a model request's readiness.
        return jsonify(status="ok")

    @app.get("/api/config")
    def config():
        return jsonify(orders=ORDERS, provider=service.provider,
                       policies={"repaired": CURRENT_POLICY, "candidate": STALE_POLICY})

    @app.post("/api/chat")
    def chat():
        # A JSON-only endpoint with a custom header rejects cross-site form posts.
        if not request.is_json or request.headers.get("X-Northstar-Request") != "chat":
            return jsonify(error="Please send your question from the support page."), 403
        origin = request.headers.get("Origin")
        if origin and urlsplit(origin).netloc != request.host:
            return jsonify(error="Please send your question from the support page."), 403
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="Enter a question and choose an order."), 400
        order = body.get("order")
        variant = body.get("variant", "repaired")
        question = body.get("question")
        if not isinstance(order, str) or order not in ORDERS or not isinstance(variant, str) or variant not in {"candidate", "repaired"}:
            return jsonify(error="Choose one of the available orders and policies."), 400
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1200:
            return jsonify(error="Enter a question between 1 and 1,200 characters."), 400
        if not service.lock.acquire(blocking=False):
            return jsonify(error="Another question is being answered. Please try again in a moment."), 429
        try:
            return jsonify(call_with_deadline(lambda: service.answer(order, question.strip(), variant)))
        except TimeoutError:
            return jsonify(error="This request took too long. It may still be finishing. Wait a moment before trying again; if the app stays busy, restart it from Databricks Apps."), 504
        except Exception:
            # Raw provider errors and credentials never reach the browser or logs.
            return jsonify(error="We couldn't complete this request with a verified trace. Please retry. If it continues, check model access, MLflow permissions, and available quota."), 503

    @app.get("/report")
    def report():
        if not service.lock.acquire(blocking=False):
            return render_template("report_pending.html", message="An answer is being generated. Open the report again in a moment."), 429
        try:
            result = call_with_deadline(service.report)
            if result is None:
                return render_template("report_pending.html", message="No release report has been published to this experiment yet. Run checkpoint 4, then publish its saved summary using the instructions in the README."), 404
            return result
        except Exception:
            return render_template("report_pending.html", message="The saved release evidence is unavailable. Check the app's MLflow experiment permission and the published report artifact."), 503

    return app
