# Evaluating LLM Applications with MLflow

**ODSC AI East 2026 Workshop** | April 28-30, Boston | 60 minutes

## What you'll build

A production evaluation pipeline that connects external evaluation libraries to MLflow, runs reproducible evaluations, and detects model regressions with statistical significance testing.

## Modules

| # | Module | Time | Notebook |
|---|--------|------|----------|
| 1 | Connecting External Evaluation Libraries | 15 min | `01_third_party_scorers` |
| 2 | Evaluation Infrastructure for Production | 15 min | `02_eval_infrastructure` |
| 3 | Comparing Runs and Detecting Regressions | 20 min | `03_comparison_and_regression` |
| 4 | End-to-End Pipeline and Deployment | 10 min | `04_pipeline_and_deployment` |

## Setup

### Option A: Databricks Free Serverless (recommended for the workshop)

1. Sign up at [community.cloud.databricks.com](https://community.cloud.databricks.com/)
2. Create a new cluster (any default serverless cluster works)
3. Import this repo: **Workspace > Repos > Add Repo** and paste this URL:
   ```
   https://github.com/debu-sinha/mlflow-eval-workshop
   ```
4. Open `notebooks/01_third_party_scorers` and run each cell

The notebooks auto-detect the Databricks environment and use Databricks Foundation Model APIs (e.g., `databricks-claude-sonnet-4`). No API keys needed. MLflow tracking is also built in. Zero configuration required.

### Option B: Run locally

```bash
git clone https://github.com/debu-sinha/mlflow-eval-workshop.git
cd mlflow-eval-workshop
pip install mlflow[genai] arize-phoenix trulens guardrails-ai scikit-learn numpy
export OPENAI_API_KEY="sk-..."
mlflow ui &   # starts tracking server at http://localhost:5000
```

Then open each notebook in Jupyter or VS Code and run the cells.

## Prerequisites

- Python 3.10+
- An OpenAI API key (for LLM-based scorers in Modules 1 and 2)
- Basic familiarity with MLflow (experiment tracking, model logging)
- No prior experience with Phoenix, TruLens, or Guardrails AI required

## What you'll learn

1. Connect Phoenix (hallucination), TruLens (groundedness), and Guardrails AI (toxicity, PII) to `mlflow.genai.evaluate()`
2. Configure LLM judge parameters and scorer concurrency for reproducible, rate-limit-safe evaluations
3. Use uv lockfiles to pin model dependencies across training and serving environments
4. Compare two evaluation runs with sample alignment, regression detection, and statistical significance testing
5. Build an evaluation gate that blocks model promotion on regression

## Speaker

**Debu Sinha** | Lead Applied AI/ML Engineer @ Databricks

Built MLflow's first third-party scorer integrations ([Phoenix](https://github.com/mlflow/mlflow/pull/19473), [TruLens](https://github.com/mlflow/mlflow/pull/19492), [Guardrails](https://github.com/mlflow/mlflow/pull/20038)) and the first MLflow tracking integration for [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai/pull/3433). Author of *Practical Machine Learning on Databricks* (Packt, 2023).

## License

Apache 2.0
