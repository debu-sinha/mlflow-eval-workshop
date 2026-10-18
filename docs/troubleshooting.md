# Troubleshooting the two-notebook workshop

Read the mode, evidence ID and named result first. A completed BLOCK is a valid evaluation outcome. An INVALID result means the evidence cannot support the comparison. Neither is an instruction to rerun until the result passes.

| Symptom | What to do |
|---|---|
| Saved evidence is missing or its hash differs | Restore the matching workshop release. Saved mode never invents replacement answers or scores. |
| Saved results differ from a new live run | Check model, source, case, judge, policy and dependency identities. Live generation and grading can vary. Keep both records. |
| A directory belongs to another measurement identity | Preserve it. Choose a new `WORKSHOP_OUTPUT_DIR` and record what changed and why. |
| Live preflight is not ready | Read its specific reasons. Check the selected provider, required packages, model configuration and tracking location. |
| Preflight passes but a model call fails | Preflight checks presence only. Check access, endpoint availability, quota and connectivity. Preserve the execution failure before a documented retry. |
| Judge controls disagree with their labels | Inspect each answer, label and rationale. Keep the failed control record. A justified rubric or label correction needs a new version and a new experiment. |
| REVIEW_REQUIRED | Complete an explicit review tied to this evidence. Do not mark a material finding resolved without a supported explanation. |
| BLOCK | Read all failed conditions and named case IDs. Fix the application or investigate the measurement, then run a declared new comparison. |
| Notebook 02 has no paired table | Invalid controls or incomplete evidence withhold the comparison. Inspect the named reasons. |
| Notebook 01 and 02 generated separate batches unexpectedly | Check that both kernels use the same output directory and measurement configuration. Kernels do not share variables. |

## Local

Use Python 3.12 and `uv sync --locked --python 3.12 --extra local`. Start Jupyter from the checkout in that environment. Load `OPENAI_API_KEY` in the terminal only when using live mode. A `.env` file is not loaded automatically.

The local tracking URI is a `sqlite:///` database URI. If you open an MLflow UI separately, point it at that same database. Check inherited `MLFLOW_EXPERIMENT_ID`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_TRACKING_URI` and `WORKSHOP_OUTPUT_DIR` when results appear somewhere unexpected.

## Databricks

Use the qualified base environment and committed requirements from the README. Select an endpoint your notebook identity can query. Use an absolute experiment path. Give both notebooks the same writable output directory.

For rate limits, stop overlapping model-calling notebooks and inspect the failure before a retry. A daily workspace quota may require waiting. Do not change model, judge, cases or policy inside a partially completed evidence directory.

Set `WORKSHOP_PROJECT_ROOT` when a notebook is outside the checkout. The notebooks locate source from their working directory or that explicit path. They do not require `__file__`.

HTML colors can change with notebook themes. Read the written status and reasons, rather than infer acceptance from color. Native notebook job validation and manual interactive setup validation are recorded separately in [workshop-validation.md](workshop-validation.md).

## Optional integrations

Install `--extra ecosystem` locally or use `requirements-ecosystem.txt` for the optional reference chapter. Keep the committed package versions together. Phoenix Hallucination and TruLens Coherence measure different properties, and their scores do not enter the repair policy. Check the validation record before claiming another integration or environment was exercised.

The original seven-chapter files retain their v1 teaching assumptions. Use `notebooks/workshop/` for the current route.
