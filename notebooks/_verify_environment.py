# Databricks notebook source
# MAGIC %md
# MAGIC # Shared West environment verification
# MAGIC
# MAGIC Run after installing `requirements-workshop.txt` in the Serverless
# MAGIC Environment panel and applying the environment, as described in README.md.
# MAGIC This helper checks the core MLflow 3.16 stack and real deterministic scoring.
# MAGIC Set `WORKSHOP_VERIFY_ECOSYSTEM = True` before `%run` to also construct the
# MAGIC optional Phoenix/TruLens scorers (requires the `ecosystem` extra).
# MAGIC Constructors never call an LLM judge. Provider access is a separate preflight.

# COMMAND ----------

import runpy
from pathlib import Path

_verifier_path = next(
    (
        _root / "scripts" / "verify_west_environment.py"
        for _root in (Path.cwd(), *Path.cwd().parents)
        if (_root / "scripts" / "verify_west_environment.py").is_file()
    ),
    None,
)
if _verifier_path is None:
    raise RuntimeError(
        "Cannot locate scripts/verify_west_environment.py. Open this notebook "
        "inside the imported workshop Git folder, preserving its directory layout."
    )

_verifier = runpy.run_path(str(_verifier_path))
_verifier["verify_environment"](
    ecosystem=bool(globals().get("WORKSHOP_VERIFY_ECOSYSTEM", False)),
    databricks=True,
)
