# Verified workshop routes

The current revision uses MLflow 3.17.1, OpenAI 3.8.0 and the committed dependency lock. Local setup uses Python 3.12.13. The native Databricks route uses Standard v5 with Python 3.12.3 and `requirements-workshop.txt`.

Final provider execution and saved bundles are being prepared. This page will record the actual results before this workshop branch is published. A completed notebook can produce BLOCK, REVIEW_REQUIRED or INVALID. Execution completion and acceptance are separate results.

Saved mode will contain dated, verified outputs from actual runs. Local and Databricks model routes remain separate. Do not pool their scores or treat ten teaching cases as production coverage.

The original optional integration checks exercised Phoenix Hallucination and TruLens Coherence on one answer per route. Guardrails AI and TruLens Groundedness were not qualified. External scores do not enter the repair policy.

Native notebook jobs and manual interactive setup are different checks. The interactive Git-folder and Environment-panel steps still need a manual walkthrough. Follow the [setup instructions](../README.md) and choose saved mode if you prefer to inspect results without credentials.
