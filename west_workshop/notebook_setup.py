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
