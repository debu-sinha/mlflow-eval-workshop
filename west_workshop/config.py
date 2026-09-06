"""Readiness checks inspect local configuration without constructing API clients."""

import configparser
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import re

TARGET_MLFLOW = "3.16.0"
APP_TIMEOUT_SECONDS = 45
APP_MAX_RETRIES = 1


def selected_provider(provider=None) -> str:
    return (provider or os.environ.get("WORKSHOP_PROVIDER", "openai")).strip().lower()


def application_model(provider: str) -> str:
    name = "WORKSHOP_OPENAI_MODEL" if provider == "openai" else "WORKSHOP_DATABRICKS_MODEL"
    default = "gpt-4o-mini" if provider == "openai" else ""
    return os.environ.get(name, default).strip()


def judge_model(provider: str):
    if provider == "openai":
        return "openai:/" + os.environ.get("WORKSHOP_OPENAI_JUDGE_MODEL", "gpt-4o-mini")
    # Pin an accessible endpoint instead of relying on an opaque managed judge.
    return "databricks:/" + os.environ.get("WORKSHOP_DATABRICKS_JUDGE_MODEL", application_model(provider)).strip()


def _safe_model_name(value: str) -> str:
    return value if re.fullmatch(r"[a-zA-Z0-9_.:/-]{1,160}", value) and "://" not in value else "configured"


def _databricks_auth_hint() -> bool:
    """Inspect presence only. SDK resolution could refresh OAuth, so is not preflight."""
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return True
    if os.environ.get("DATABRICKS_HOST") and (
        os.environ.get("DATABRICKS_TOKEN")
        or (os.environ.get("DATABRICKS_CLIENT_ID") and os.environ.get("DATABRICKS_CLIENT_SECRET"))
    ):
        return True
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
    config_path = Path(os.environ.get("DATABRICKS_CONFIG_FILE", str(Path.home() / ".databrickscfg")))
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(config_path, encoding="utf-8")
        values = parser.defaults() if profile == "DEFAULT" else dict(parser[profile]) if parser.has_section(profile) else {}
        return bool(values.get("host") and (values.get("token") or values.get("auth_type") or values.get("client_id")))
    except (OSError, configparser.Error, UnicodeError):
        return False


def preflight(provider=None) -> dict:
    """No network calls, no API client construction, no credential or host values."""
    provider = selected_provider(provider)
    reasons = []
    versions = {}
    for package in ("mlflow", "openai", "databricks-sdk", "databricks-agents", "litellm"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
            if package in {"mlflow", "openai"} or (package in {"databricks-sdk", "databricks-agents"} and provider == "databricks"):
                reasons.append(f"Install the pinned workshop dependencies. Missing {package}.")
    if versions["mlflow"] and versions["mlflow"] != TARGET_MLFLOW:
        reasons.append(f"Use MLflow {TARGET_MLFLOW} from the committed dependency lock.")
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            reasons.append("Set OPENAI_API_KEY using a fresh credential.")
        if os.environ.get("OPENAI_BASE_URL"):
            reasons.append("Unset OPENAI_BASE_URL to exercise the required real OpenAI route.")
        if os.environ.get("MLFLOW_TRACKING_URI") and not os.environ["MLFLOW_TRACKING_URI"].startswith("sqlite:///"):
            reasons.append("Use local SQLite tracking for the OpenAI path.")
    elif provider == "databricks":
        if not _databricks_auth_hint():
            reasons.append("Configure Databricks SDK authentication through environment settings, a CLI profile, or Databricks Runtime.")
        if not application_model(provider):
            reasons.append("Set WORKSHOP_DATABRICKS_MODEL to an available Foundation Model API endpoint name.")
        if not os.environ.get("MLFLOW_EXPERIMENT_NAME", "").startswith("/") and not os.environ.get("MLFLOW_EXPERIMENT_ID"):
            reasons.append("Set MLFLOW_EXPERIMENT_NAME to an absolute workspace experiment path, or set MLFLOW_EXPERIMENT_ID.")
    else:
        reasons.append("WORKSHOP_PROVIDER must be openai or databricks.")
    return {
        "ready": not reasons,
        "provider": provider if provider in {"openai", "databricks"} else "unsupported",
        "reasons": reasons,
        "versions": versions,
        "application_model": _safe_model_name(application_model(provider)),
        "judge_model": _safe_model_name(judge_model(provider) or "managed-databricks-default"),
        "credential_values_included": False,
        "network_checked": False,
        "note": "Configuration presence does not validate authentication or model access.",
        "live_validation": "not_run",
    }
