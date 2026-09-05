# Evaluating LLM Applications with MLflow

ODSC AI West 2026 | Debu Sinha

Run a support assistant, inspect its traces, evaluate its answers, and compare a candidate with a baseline before making a release decision. The examples use a fictional refund policy and real model APIs.

The local OSS MLflow path was verified on September 5, 2026 with real OpenAI application and judge calls. All six notebooks passed in sequence and in independent processes. The separate Phoenix and TruLens integration check also passed. Databricks execution has not been verified.

## Setup

Use Python 3.10, 3.11, or 3.12. Python 3.12 is recommended for the local environment. Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/) before running these commands. uv can install Python 3.12 during setup if it is not already available.

```bash
git clone --branch west-2026 https://github.com/debu-sinha/mlflow-eval-workshop.git
cd mlflow-eval-workshop
uv sync --locked --python 3.12
uv run --locked python scripts/verify_west_environment.py
```

The committed lock pins MLflow 3.16.0 and its dependencies. Keep credentials out of notebook cells and source files.

## Run locally with OpenAI

Configure `OPENAI_API_KEY` in your environment using your usual credential manager. Application and judge calls use `gpt-4o-mini` by default.

```bash
uv run --locked python -m west_workshop --provider openai --check
uv run --locked python -m west_workshop --provider openai --checkpoint 0
```

Change `--checkpoint` from `0` through `5` to run another notebook checkpoint. Each checkpoint can run independently. Model calls use your provider account and may incur charges. `--check` inspects configuration without making model calls.

Optional local settings:

| Variable | Purpose |
|---|---|
| `WORKSHOP_OPENAI_MODEL` | Application model name |
| `WORKSHOP_OPENAI_JUDGE_MODEL` | Judge model name without a provider prefix |
| `MLFLOW_TRACKING_URI` | Local SQLite tracking URI |
| `MLFLOW_EXPERIMENT_NAME` | Evaluation experiment name |
| `WORKSHOP_OUTPUT_DIR` | Directory for local checkpoint outputs |

Open MLflow against the same tracking database to inspect traces and scores. For the default local output directory, run this command from the checkout and open `http://127.0.0.1:5000`:

```bash
uv run --locked mlflow server --backend-store-uri sqlite:///artifacts/west-live/mlflow-west.db --host 127.0.0.1 --port 5000 --workers 1
```

If you configure a different tracking URI or output directory, use its database instead.

## Run on Databricks

Create a Databricks Git folder from this repository and select the `west-2026` branch. Open `notebooks/west/00_ship_or_block` on standard serverless compute. In the Environment side panel, select Standard environment version 5, then add this dependency using the absolute path to your Git folder:

```text
-r /Workspace/Users/<your-user>/mlflow-eval-workshop/requirements-workshop.txt
```

Click Apply. Repeat this environment configuration for every notebook. Standard serverless dependencies are notebook-scoped. The separate Git Folder Serverless Beta uses a shared `pyproject.toml` environment and has a different setup. See the official [serverless environment instructions](https://docs.databricks.com/aws/en/compute/serverless/dependencies) and [environment version 5](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/five).

Set the following non-secret configuration in each notebook session, replacing the placeholders:

```python
import os
os.environ["WORKSHOP_PROVIDER"] = "databricks"
os.environ["WORKSHOP_DATABRICKS_MODEL"] = "<available-foundation-model-endpoint>"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Users/<your-user>/odsc-west-2026"
```

Alternatively, use an existing `MLFLOW_EXPERIMENT_ID`. The runtime uses the managed Databricks judge. Your workspace identity needs model serving, judge, and tracking access. The [official evaluation guide](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor) describes these capabilities.

To run from your laptop against a workspace, configure an authenticated Databricks SDK profile and install the Databricks extra:

```bash
uv sync --locked --python 3.12 --extra databricks
uv run --locked --extra databricks python -m west_workshop --provider databricks --check
uv run --locked --extra databricks python -m west_workshop --provider databricks --checkpoint 0
```

Set `DATABRICKS_CONFIG_PROFILE` for a named profile. A configuration check does not establish authentication or endpoint availability.

## Notebooks

| Notebook | Exercise |
|---|---|
| [00 Ship or block](notebooks/west/00_ship_or_block.py) | Evaluate an answer against the current refund policy |
| [01 Trace the failure](notebooks/west/01_trace_the_failure.py) | Inspect the retrieved policy and response |
| [02 Build the scorer stack](notebooks/west/02_build_the_scorer_stack.py) | Combine deterministic checks and an LLM judge |
| [03 Trust the judge](notebooks/west/03_trust_the_judge.py) | Compare judge output with authored reference labels |
| [04 Compare and gate](notebooks/west/04_compare_and_gate.py) | Compare application variants using the same cases |
| [05 Production feedback](notebooks/west/05_production_feedback.py) | Trace and evaluate a new request |

The files use Databricks Python notebook format. They also run as Python scripts from this checkout. Each prepared execution cell calls the configured provider. A failed request or incomplete evaluation stops the checkpoint.

The small dataset demonstrates evaluation mechanics. Passing its gate does not establish broad production safety. A new deployment needs representative cases and a policy suited to its risks.

## What a local run looks like

These are unmodified screenshots from the local MLflow 3.16.0 UI using real OpenAI responses and evaluations. The examples use fictional customer inputs. Calibration labels and the follow-up feedback are explicitly authored teaching material, not observed customer feedback. Your responses, scores, timing, and run IDs can differ.

The stale-policy candidate was blocked and the repaired version passed in both execution modes. Some valid answers still received an incorrect judge score. Inspect the rationale and keep deterministic policy checks alongside the judge.

| View | What to inspect |
|---|---|
| [Application answer](notebooks/images/west/00-answer.png) | The 45-day request received an incorrect full-refund answer |
| [Retrieved policy](notebooks/images/west/01-retrieval.png) | The retrieval span contains the stale 90-day policy |
| [Scorer stack](notebooks/images/west/02-scorer-stack.png) | Individual checks and the semantic judge remain inspectable |
| [Judge versions](notebooks/images/west/03-judge-versions.png) | Registered definitions have explicit versions |
| [Run comparison](notebooks/images/west/04-comparison.png) | The stale candidate and repaired results use the same ten cases |
| [Follow-up feedback](notebooks/images/west/05-feedback.png) | Authored review feedback is attached to a real trace |
| [Phoenix and TruLens](notebooks/images/west/06-integrations.png) | Both third-party scorers evaluated a fresh response |

![Real model response in local MLflow](notebooks/images/west/00-answer.png)

![Stale candidate and repaired comparison in local MLflow](notebooks/images/west/04-comparison.png)

## Optional integrations

Phoenix and TruLens require the ecosystem extra:

```bash
uv sync --locked --python 3.12 --extra ecosystem
uv run --locked --extra ecosystem python -m west_workshop --provider openai --integrations
```

Phoenix remains below version 3 because the MLflow 3.16 integration uses its earlier evaluator API. See the [official Phoenix migration notes](https://arize.com/docs/phoenix/release-notes/04-2026/04-07-2026-phoenix-v14-breaking-changes) and [TruLens provider reference](https://www.trulens.org/reference/trulens/providers/litellm/provider/).

For the APIs used in these exercises, see [MLflow scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/), [trace evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/), [scorer versioning](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/versioning/), and the [OpenAI SDK documentation](https://developers.openai.com/api/docs/libraries).

## Earlier edition

The [ODSC AI East 2026 release](https://github.com/debu-sinha/mlflow-eval-workshop/tree/odsc-east-2026-final) preserves the previous workshop.

## Author and license

Workshop by Debu Sinha. [LinkedIn](https://linkedin.com/in/debusinha) | [GitHub](https://github.com/debu-sinha)

[Apache 2.0](LICENSE). Libraries retain their respective licenses and project attribution.
