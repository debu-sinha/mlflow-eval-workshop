import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

# Settings a developer might have exported. Each test starts without them.
_INHERITED = (
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "WORKSHOP_PROVIDER", "WORKSHOP_OPENAI_MODEL",
    "WORKSHOP_OPENAI_JUDGE_MODEL", "WORKSHOP_DATABRICKS_MODEL", "WORKSHOP_DATABRICKS_JUDGE_MODEL",
    "WORKSHOP_OUTPUT_DIR", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME",
    "DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET",
    "DATABRICKS_CONFIG_PROFILE", "DATABRICKS_CONFIG_FILE", "DATABRICKS_RUNTIME_VERSION", "DATABRICKS_APP_NAME",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    for key in _INHERITED:
        monkeypatch.delenv(key, raising=False)
    # Never read a real Databricks profile from the developer's home directory.
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(tmp_path / "no-databrickscfg"))


@pytest.fixture
def offline_openai(monkeypatch, tmp_path):
    """The local OpenAI route with SQLite tracking and a fake model provider."""
    from tests.fakes import FAKE_KEY, FakeChatClient, fake_policy_judge
    from west_workshop import runtime

    client = FakeChatClient()
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("WORKSHOP_OUTPUT_DIR", str(tmp_path / "west-live"))
    monkeypatch.setattr(runtime, "_client", lambda provider: client)
    monkeypatch.setattr(runtime, "_policy_judge", fake_policy_judge)
    return client
