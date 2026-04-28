# Databricks notebook source
# MAGIC %md
# MAGIC # Bonus: Locking Dependencies with uv
# MAGIC
# MAGIC **Objective:** Understand how MLflow uses uv lockfiles to pin every
# MAGIC dependency (including transitive ones) when logging a model.
# MAGIC
# MAGIC **Time:** 5 minutes
# MAGIC
# MAGIC This is a standalone bonus module. It does not depend on Modules 1-4.

# COMMAND ----------

import os

if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    _nb_path = (
        dbutils.notebook.entry_point.getDbutils()  # noqa: F821
        .notebook()
        .getContext()
        .notebookPath()
        .get()
    )
    _repo_root = "/Workspace" + "/".join(_nb_path.split("/")[:-2])
    REQ_PATH = f"{_repo_root}/requirements-workshop.txt"
    print(f"Installing workshop requirements from: {REQ_PATH}")
else:
    REQ_PATH = None
    print(
        "Local run detected. Skip the next two cells (Databricks "
        "%pip install + %run helper) and run `pip install '.[all]'` "
        "from the repo root before continuing."
    )

# COMMAND ----------

# MAGIC %pip install -q -r $REQ_PATH 'mlflow[genai]>=3.11' uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_verify_environment

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os

ON_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------

import mlflow

if ON_DATABRICKS:
    _user = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .userName()
        .get()
    )  # noqa: F821
    mlflow.set_experiment(f"/Users/{_user}/odsc-workshop-bonus")
else:
    mlflow.set_tracking_uri("sqlite:///mlflow_workshop.db")
    mlflow.set_experiment("odsc-eval-workshop")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check the working directory
# MAGIC
# MAGIC MLflow looks for `uv.lock` in the current working directory. If the
# MAGIC notebook launched from `notebooks/` the lockfile is one level up,
# MAGIC so we hop to the repo root before logging the model.

# COMMAND ----------

import shutil
from pathlib import Path

_repo_root = Path.cwd()
if not (_repo_root / "uv.lock").exists():
    _candidate = _repo_root.parent
    if (_candidate / "uv.lock").exists():
        os.chdir(_candidate)
        _repo_root = _candidate

print(f"Working directory: {_repo_root}")
print(f"uv.lock present:   {(_repo_root / 'uv.lock').exists()}")

if shutil.which("uv") is None:
    raise RuntimeError(
        "uv binary not found on PATH after install. The lockfile-pinning "
        "demo below cannot run because MLflow requires the uv CLI to "
        "export from uv.lock. Verify the install cell at the top of this "
        "notebook completed successfully and that the kernel restart "
        "picked up the new venv bin directory."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## How uv lockfiles work with MLflow
# MAGIC
# MAGIC **Requires MLflow >= 3.11.** uv lockfile auto-detection was added in
# MAGIC [PR #20344](https://github.com/mlflow/mlflow/pull/20344) and shipped in
# MAGIC MLflow 3.11.0.
# MAGIC
# MAGIC When you log a model from a uv-managed project, MLflow reads your
# MAGIC `uv.lock` file and pins every dependency (including transitive ones)
# MAGIC to the exact version you tested with.
# MAGIC
# MAGIC Without uv, MLflow infers dependencies from the Python runtime. With uv,
# MAGIC MLflow gets exact pins from the lockfile. The result is a `requirements.txt`
# MAGIC inside the model artifact that precisely matches your development environment.
# MAGIC
# MAGIC MLflow auto-detects `uv.lock` and `pyproject.toml` in the current working
# MAGIC directory. On Databricks, the notebook's default cwd may not be the repo
# MAGIC root, so we explicitly set it below.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set working directory to repo root
# MAGIC
# MAGIC MLflow's uv auto-detection checks `Path.cwd()` for `uv.lock` and
# MAGIC `pyproject.toml`. On Databricks, the default cwd is the driver directory,
# MAGIC not the repo root. We fix that here.

# COMMAND ----------

import os
from pathlib import Path

if ON_DATABRICKS:
    _repo_root = Path("/Workspace" + "/".join(_nb_path.split("/")[:-2]))
else:
    _repo_root = Path(__file__).resolve().parent.parent

if (_repo_root / "uv.lock").exists():
    os.chdir(_repo_root)
    print(f"Working directory: {Path.cwd()}")
    print(f"uv.lock found: {(_repo_root / 'uv.lock').exists()}")
    print(f"pyproject.toml found: {(_repo_root / 'pyproject.toml').exists()}")
else:
    print(f"WARNING: uv.lock not found at {_repo_root}")
    print("uv auto-detection will not work. Model will use runtime inference.")

# COMMAND ----------

# Verify uv is installed and MLflow can find it
import shutil
import subprocess

uv_bin = shutil.which("uv")
if uv_bin:
    result = subprocess.run([uv_bin, "--version"], capture_output=True, text=True)
    print(f"uv binary: {uv_bin}")
    print(f"uv version: {result.stdout.strip()}")
else:
    print("WARNING: uv not found on PATH. Install with: pip install uv")

# Verify MLflow version has uv support
try:
    from mlflow.utils.uv_utils import detect_uv_project

    project = detect_uv_project()
    if project:
        print(f"MLflow detected uv project at: {project.uv_lock.parent}")
    else:
        print("MLflow did not detect a uv project at current directory")
except ImportError:
    print(f"ERROR: mlflow {mlflow.__version__} does not have uv support.")
    print("uv lockfile auto-detection requires mlflow >= 3.11.0")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: Log a sklearn model

# COMMAND ----------

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=200).fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2%}")

# COMMAND ----------

# Log the model. If uv.lock exists in the working directory,
# MLflow auto-detects it and exports pinned requirements.
with mlflow.start_run(run_name="sklearn-with-uv") as run:
    model_info = mlflow.sklearn.log_model(model, name="iris-model")
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model logged: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect captured requirements

# COMMAND ----------

import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run.info.run_id}/iris-model",
        dst_path=tmp,
    )
    reqs_path = Path(local_path) / "requirements.txt"
    if reqs_path.exists():
        reqs = reqs_path.read_text()
        lines = [line for line in reqs.strip().split("\n") if line.strip()]
        print(f"Captured {len(lines)} requirements:")
        for line in lines[:15]:
            print(f"  {line}")
        if len(lines) > 15:
            print(f"  ... and {len(lines) - 15} more")
    else:
        print("No requirements.txt found in model artifact")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why this matters
# MAGIC
# MAGIC Without lockfile support, the model's `requirements.txt` contains versions
# MAGIC inferred from the Python runtime at log time. These are often loose pins
# MAGIC (e.g., `scikit-learn==1.4.0`) that do not capture transitive dependencies.
# MAGIC
# MAGIC With uv, every dependency in the lockfile is included with its exact version
# MAGIC and hash. When you serve the model in a different environment, `pip install`
# MAGIC recreates the exact same dependency tree.
# MAGIC
# MAGIC This closes the gap between "works on my machine" and "works in production."
