# Evaluating LLM Applications with MLflow

ODSC AI West 2026 | Debu Sinha

Run a support assistant, inspect its traces, evaluate its answers, and compare a candidate with a baseline before making a release decision. The examples use a fictional refund policy and real model APIs.

The local OSS MLflow path was verified on September 5, 2026 with real OpenAI application and judge calls. All six notebooks passed in sequence and in independent processes. The separate Phoenix and TruLens integration check also passed. Databricks execution has not been verified.

## Setup

Use Python 3.10, 3.11, or 3.12. Python 3.12 is recommended for the local environment. Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/) before running these commands. uv can install Python 3.12 during setup if it is not already available.

The local route does not require a Databricks account or a notebook server. Open a terminal and confirm that `git --version` and `uv --version` work. If you just installed uv and the command is not found, reopen the terminal. The commands below work in PowerShell or Bash unless a shell is named explicitly.

```bash
git clone --branch west-2026 https://github.com/debu-sinha/mlflow-eval-workshop.git
cd mlflow-eval-workshop
uv sync --locked --python 3.12
uv run --locked python scripts/verify_west_environment.py
```

The committed lock pins MLflow 3.16.0 and its dependencies. `uv run` uses the checkout's `.venv`, so manual environment activation is not needed. Run every command below from this same checkout. Initial dependency downloads and imports can take several minutes.

## Run locally with OSS MLflow

MLflow and its tracking database run locally. Application and judge calls use the OpenAI API with `gpt-4o-mini` by default, so this is not an offline model demo. You need internet access and an OpenAI API account with access and quota for that model. Calls may incur charges.

If `OPENAI_API_KEY` is already configured in your terminal, skip the next step. Otherwise create a key using the [official OpenAI quickstart](https://developers.openai.com/api/docs/quickstart) and load it with your credential manager, or use the hidden-input example for your shell. Never paste the key into a notebook, source file, screenshot, or chat.

PowerShell:

```powershell
$workshopKey = Read-Host "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $workshopKey).Password
Remove-Variable workshopKey
```

Bash: run the first command, paste the key, and press Enter. Input is hidden. Then run the second command.

```bash
read -r -s OPENAI_API_KEY
export OPENAI_API_KEY
```

These examples set the key only for the current terminal and its child processes. Keep using that terminal for model calls. The workshop does not automatically load a `.env` file.

Check configuration first:

```bash
uv run --locked python -m west_workshop --provider openai --check
```

Look for `"ready": true`. If it is false, follow the printed `reasons`. This check makes no model calls and does not verify authentication, model access, or quota.

Run the first checkpoint:

```bash
uv run --locked python -m west_workshop --provider openai --checkpoint 0
```

Then run the remaining checkpoints in order for the full workshop:

```bash
uv run --locked python -m west_workshop --provider openai --checkpoint 1
uv run --locked python -m west_workshop --provider openai --checkpoint 2
uv run --locked python -m west_workshop --provider openai --checkpoint 3
uv run --locked python -m west_workshop --provider openai --checkpoint 4
uv run --locked python -m west_workshop --provider openai --checkpoint 5
```

You can also select a single checkpoint independently. Each execution creates fresh results and makes model calls. Do not start the next command until the current one finishes.

A successful exercise prints `"status": "passed"` and `"live_validation": "completed"`. It can also print `"decision": "block"`, which means it successfully detected the intentionally incorrect candidate. Checkpoint 4 is expected to block the stale candidate and pass the repaired version. A failed command exits with code 2. Inspect the printed summary before retrying, and do not lower the gate thresholds to force a pass.

The printed `summary_path` identifies the saved results under `artifacts/west-live/`. The default tracking database is `artifacts/west-live/mlflow-west.db`, and the default experiment is `odsc-west-2026`. Keep generated results out of Git.

Optional local settings:

| Variable | Purpose |
|---|---|
| `WORKSHOP_OPENAI_MODEL` | Application model name |
| `WORKSHOP_OPENAI_JUDGE_MODEL` | Judge model name without a provider prefix |
| `MLFLOW_TRACKING_URI` | Local SQLite tracking URI |
| `MLFLOW_EXPERIMENT_NAME` | Evaluation experiment name |
| `WORKSHOP_OUTPUT_DIR` | Directory for local checkpoint outputs |

### Open the local MLflow UI

After running a checkpoint, open a second terminal in the same checkout. Start MLflow against the same tracking database and leave this terminal running:

```bash
uv run --locked mlflow server --backend-store-uri sqlite:///artifacts/west-live/mlflow-west.db --host 127.0.0.1 --port 5000 --workers 1
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), select the `odsc-west-2026` experiment, then choose Traces or Evaluation runs. Stop the server with Ctrl+C when finished. The API key is not required just to view existing results.

This workshop writes directly to SQLite. Do not set `MLFLOW_TRACKING_URI` to the HTTP UI address. If you configure a different tracking URI or output directory, start the UI against that database instead. See the [official MLflow server documentation](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) for backend-store options.

### Local troubleshooting

| Symptom | What to check |
|---|---|
| Python or package version error | Run `uv sync --locked --python 3.12` again, then the environment verifier. Do not install the latest packages over the lock. |
| Configuration check is blocked | Load `OPENAI_API_KEY` in the terminal making model calls. Remove an inherited `OPENAI_BASE_URL` override. Leave tracking overrides unset for the default local route. |
| Configuration is ready but a live request fails | Check API account access, model permissions, quota, and connectivity. A configuration check does not contact OpenAI. |
| MLflow opens with no workshop results | Run a checkpoint first, start the server from the same checkout, and check the database path and experiment name. Do not use bare `mlflow server`, which may open a different database. |
| Port 5000 is already in use | Change only `--port 5000` to `--port 5001` in the server command, then open `http://127.0.0.1:5001`. Keep the SQLite URI unchanged. |
| Old or unexpected experiment appears | Check inherited `MLFLOW_EXPERIMENT_ID`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_TRACKING_URI`, and `WORKSHOP_OUTPUT_DIR` settings. |

## Run on Databricks

These instructions have been checked against official documentation, but live Databricks execution is not yet verified. Use [Databricks Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition), which replaced Community Edition in 2025, or an existing supported workspace. Free Edition is serverless-only and quota-limited. Model and managed-judge availability must be checked in your workspace.

In Workspace, choose Create, then Git folder, clone this public repository, and select the `west-2026` branch. See the [Git folder setup guide](https://docs.databricks.com/aws/en/repos/repos-setup). Open `notebooks/west/00_ship_or_block` on standard serverless compute. In the Environment side panel, open Base environment, use More if needed, and select Standard environment version 5. Then add this dependency using the absolute path to your Git folder:

```text
-r /Workspace/Users/<your-user>/mlflow-eval-workshop/requirements-workshop.txt
```

Click Apply and wait for dependency installation and the Python restart. Repeat this environment configuration for every notebook. Version 5 alone does not supply the workshop's required MLflow 3.16.0, so do not skip the dependency file. Standard serverless dependencies are notebook-scoped. The separate Git Folder Serverless Beta uses a shared `pyproject.toml` environment and has a different setup. See the official [serverless environment instructions](https://docs.databricks.com/aws/en/compute/serverless/dependencies) and [environment version 5](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/five).

After Apply finishes, set the following non-secret configuration in a new code cell before the checkpoint execution cell in each notebook. Replace the placeholders with an available chat-completion endpoint and your workspace experiment path. Notebook-native SDK authentication must not use a laptop configuration profile:

```python
import os
os.environ.pop("DATABRICKS_CONFIG_PROFILE", None)
os.environ["WORKSHOP_PROVIDER"] = "databricks"
os.environ["WORKSHOP_DATABRICKS_MODEL"] = "<available-foundation-model-endpoint>"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Users/<your-user>/odsc-west-2026"
```

Alternatively, use an existing `MLFLOW_EXPERIMENT_ID` from this workspace, not a local SQLite experiment ID. The application endpoint and managed Databricks judge are separate dependencies. Changing `WORKSHOP_DATABRICKS_MODEL` changes only the application model. Your identity needs model serving, judge, and tracking access. See the [official evaluation guide](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor), [judge restrictions](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers), and [notebook SDK authentication](https://docs.databricks.com/aws/en/dev-tools/sdk-python).

Run checkpoint 0 before proceeding to the remaining notebooks. Free Edition excludes some models and enforces fair-use quotas. Repeated full runs can exhaust the available quota and suspend compute until it resets. See [Free Edition limitations](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations). Do not treat an unavailable endpoint or judge as a passing evaluation.

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
uv run --locked --extra ecosystem python scripts/verify_west_environment.py --ecosystem
uv run --locked --extra ecosystem python -m west_workshop --provider openai --integrations
```

The environment verifier checks local package and constructor compatibility without model calls. The integration command makes real provider calls. This extra is not required for checkpoints 0 through 5.

Phoenix remains below version 3 because the MLflow 3.16 integration uses its earlier evaluator API. See the [official Phoenix migration notes](https://arize.com/docs/phoenix/release-notes/04-2026/04-07-2026-phoenix-v14-breaking-changes) and [TruLens provider reference](https://www.trulens.org/reference/trulens/providers/litellm/provider/).

For the APIs used in these exercises, see [MLflow scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/), [trace evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/), [scorer versioning](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/versioning/), and the [OpenAI SDK documentation](https://developers.openai.com/api/docs/libraries).

## Earlier edition

The [ODSC AI East 2026 release](https://github.com/debu-sinha/mlflow-eval-workshop/tree/odsc-east-2026-final) preserves the previous workshop.

## Author and license

Workshop by Debu Sinha. [LinkedIn](https://linkedin.com/in/debusinha) | [GitHub](https://github.com/debu-sinha)

[Apache 2.0](LICENSE). Libraries retain their respective licenses and project attribution.
