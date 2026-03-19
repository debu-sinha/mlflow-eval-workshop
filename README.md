# Evaluating LLM Applications with MLflow

**ODSC AI East 2026 Workshop** | April 28-30, Boston | 60 minutes

## What you'll build

A production evaluation pipeline with built-in and third-party scorers, statistical regression detection, and a deployment gate that blocks bad models from production.

## Modules

| # | Module | Time | Notebook |
|---|--------|------|----------|
| 1 | MLflow Evaluation Ecosystem | 20 min | `01_mlflow_evaluation_ecosystem` |
| 2 | Production Evaluation Infrastructure | 8 min | `02_production_infrastructure` |
| 3 | Comparing Runs and Detecting Regressions | 12 min | `03_comparison_and_regression` |
| 4 | The Evaluation Gate | 15 min | `04_evaluation_gate` |
| - | Bonus: uv Dependency Management | 5 min | `bonus_uv_dependencies` |

## Setup

### Option A: Databricks Workspace (recommended for the workshop)

Sign up for the [Databricks Free Edition](https://www.databricks.com/try-databricks-free) if you don't have a workspace. The Free Edition includes serverless compute, MLflow tracking, and Git Folders.

**Clone this repo into your workspace:**

1. In the Databricks sidebar, click **Workspace**
2. Expand **Workspace > Repos** and click into your user folder
3. Click the **Create Git folder** button (top right) or use **New > Git folder**
4. Paste the repository URL:
   ```
   https://github.com/debu-sinha/mlflow-eval-workshop
   ```
5. Select **GitHub** as the Git provider and click **Create Git folder**
6. Open `notebooks/01_mlflow_evaluation_ecosystem` and attach to any serverless cluster

No Git credentials needed (public repo). No API keys needed. The notebooks auto-detect the Databricks environment and use built-in Foundation Model APIs (e.g., `databricks-claude-sonnet-4`). MLflow tracking is built in. Zero configuration.

**For Guardrails AI scorers on Databricks:** Set your Guardrails Hub API token as an environment variable on the cluster (`GUARDRAILS_API_KEY`), or store it in a Databricks secret scope. Sign up at [hub.guardrailsai.com](https://hub.guardrailsai.com) if you don't have a token.

### Option B: Run locally

```bash
git clone https://github.com/debu-sinha/mlflow-eval-workshop.git
cd mlflow-eval-workshop
pip install mlflow[genai] arize-phoenix trulens guardrails-ai trulens-providers-litellm scikit-learn numpy
export OPENAI_API_KEY="sk-..."
mlflow ui &   # starts tracking server at http://localhost:5000
```

Then open each notebook in Jupyter or VS Code and run the cells.

## What you'll learn

1. Run MLflow's built-in scorers (Correctness, Safety) and third-party scorers (Phoenix, TruLens, Guardrails AI) from a single `mlflow.genai.evaluate()` call
2. Write custom scorers and combine them with built-in and third-party scorers
3. Call a real LLM, evaluate its response, and view results in the MLflow UI
4. Configure judge parameters and scorer concurrency for reproducible evaluations
5. Compare two evaluation runs with statistical significance testing (McNemar's test, bootstrap CI, Cohen's d, win rate)
6. Build an evaluation gate that blocks model promotion on regression

## Background

This workshop covers MLflow's GenAI evaluation ecosystem, including features
contributed by the speaker to MLflow core and the broader evaluation ecosystem:

- **Third-party scorer integrations** (Phoenix, TruLens, Guardrails AI): Connect
  external evaluation libraries to MLflow's unified `evaluate()` API.
  PRs [#19473](https://github.com/mlflow/mlflow/pull/19473),
  [#19492](https://github.com/mlflow/mlflow/pull/19492),
  [#20038](https://github.com/mlflow/mlflow/pull/20038)

- **LLM judge inference parameters**: Temperature, max tokens, and other controls
  for deterministic evaluation scoring.
  PR [#19152](https://github.com/mlflow/mlflow/pull/19152)

- **Scorer parallelism control**: `MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS` for
  managing API rate limits during concurrent evaluation.
  PR [#19248](https://github.com/mlflow/mlflow/pull/19248)

- **Inspect AI MLflow tracking**: Logging hook for the UK AI Safety Institute's
  evaluation framework.
  PRs [#3433](https://github.com/UKGovernmentBEIS/inspect_ai/pull/3433),
  [#3483](https://github.com/UKGovernmentBEIS/inspect_ai/pull/3483)

MLflow is downloaded over 30 million times per month from PyPI.

Presented at [ODSC AI East 2026](https://odsc.com/boston/), Boston, April 28-30.

## Prerequisites

- Python 3.10+
- An OpenAI API key (for LLM-based scorers when running locally)
- Basic familiarity with MLflow (experiment tracking, model logging)
- No prior experience with Phoenix, TruLens, or Guardrails AI required

## Speaker

**Debu Sinha** | Lead Applied AI/ML Engineer @ Databricks

Built MLflow's first third-party scorer integrations ([Phoenix](https://github.com/mlflow/mlflow/pull/19473), [TruLens](https://github.com/mlflow/mlflow/pull/19492), [Guardrails](https://github.com/mlflow/mlflow/pull/20038)) and the first MLflow tracking integration for [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai/pull/3433). Author of *Practical Machine Learning on Databricks* (Packt, 2023).

## License

Apache 2.0
