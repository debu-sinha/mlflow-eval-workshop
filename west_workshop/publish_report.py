"""Publish an existing checkpoint 4 summary without generating new responses."""

import argparse
import json
import os
from pathlib import Path

from .config import selected_provider
from .report import render_release_report
from .runtime import _setup_tracking


def publish(path, provider=None, experiment_id=None):
    import mlflow

    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    report = render_release_report(summary)
    output = Path(os.environ.get("WORKSHOP_OUTPUT_DIR", "artifacts/west-live")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    default_experiment = _setup_tracking(selected_provider(provider), output)
    experiment_id = str(experiment_id or default_experiment)
    with mlflow.start_run(experiment_id=experiment_id, run_name="northstar-recorded-release-report", tags={"workshop.report": "checkpoint_4", "workshop.source_gate_run": str(summary.get("gate_run_id", "unavailable"))}) as run:
        mlflow.log_dict(summary, "release-summary.json")
        mlflow.log_text(report, "release-report.html")
    return {"experiment_id": experiment_id, "report_run_id": run.info.run_id, "source_gate_run_id": summary.get("gate_run_id"), "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="Path printed by checkpoint 4")
    parser.add_argument("--provider", choices=["openai", "databricks"])
    parser.add_argument("--experiment-id", help="App experiment ID, if different from the notebook experiment")
    args = parser.parse_args()
    print(json.dumps(publish(args.summary, args.provider, args.experiment_id), indent=2))
