"""Notebook-native Free Edition defaults; local scripts keep terminal settings."""

import os

FREE_EDITION_MODEL = "databricks-qwen3-next-80b-a3b-instruct"


def configure_notebook():
    """Configure each independent Databricks notebook without copying credentials.

    Override the non-secret WORKSHOP_DATABRICKS_MODEL / JUDGE_MODEL or experiment
    settings in a cell before this call when using a different endpoint or path.
    """
    if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return
    os.environ.pop("DATABRICKS_CONFIG_PROFILE", None)
    os.environ["WORKSHOP_PROVIDER"] = "databricks"
    os.environ.setdefault("WORKSHOP_DATABRICKS_MODEL", FREE_EDITION_MODEL)
    if not os.environ.get("MLFLOW_EXPERIMENT_ID") and not os.environ.get("MLFLOW_EXPERIMENT_NAME"):
        from databricks.sdk import WorkspaceClient

        username = WorkspaceClient().current_user.me().user_name
        if not username:
            raise RuntimeError("Set MLFLOW_EXPERIMENT_NAME to your absolute workspace experiment path.")
        os.environ["MLFLOW_EXPERIMENT_NAME"] = f"/Users/{username}/odsc-west-2026"


def configure_lab():
    """Prepare tracking for the short lab without running a checkpoint or model."""
    from pathlib import Path
    from .config import preflight, selected_provider
    from .runtime import _configure_timeouts, _setup_tracking

    configure_notebook()
    provider = selected_provider()
    readiness = preflight(provider)
    if not readiness["ready"]:
        raise RuntimeError("Lab setup is incomplete: " + "; ".join(readiness["reasons"]))
    output = Path(os.environ.get("WORKSHOP_OUTPUT_DIR", "artifacts/west-live")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _configure_timeouts(provider)
    _setup_tracking(provider, output)
    return provider


def show_result(summary):
    """Print the outcome and evidence locations without flooding a notebook with JSON."""
    print("Exercise status:", summary.get("status", "unknown"))
    if summary.get("decision"):
        print("Decision:", summary["decision"])
    for reason in summary.get("preflight", {}).get("reasons", []):
        print("Setup issue:", reason)
    if summary.get("error"):
        print("Run issue:", summary["error"])
    if summary.get("experiment_id"):
        print("MLflow experiment ID:", summary["experiment_id"])
    for evaluation in summary.get("evaluations", []):
        print("Evaluation run:", evaluation["run_id"],
              "| cases:", evaluation.get("row_count", 0),
              "| complete:", evaluation.get("complete", False))
    if summary.get("summary_path"):
        print("Full saved results:", summary["summary_path"])
