# Evaluating LLM Applications with MLflow

**ODSC AI East 2026 Workshop** | April 28-30, Boston | 60 minutes, hands-on

## What you'll build

A production evaluation pipeline that connects external evaluation libraries to MLflow, runs reproducible evaluations on Databricks, and detects model regressions with statistical significance testing.

## Modules

| # | Module | Time | Notebook |
|---|--------|------|----------|
| 1 | Connecting External Evaluation Libraries | 15 min | `01_third_party_scorers` |
| 2 | Evaluation Infrastructure for Production | 15 min | `02_eval_infrastructure` |
| 3 | Comparing Runs and Detecting Regressions | 20 min | `03_comparison_and_regression` |
| 4 | End-to-End Pipeline and Deployment | 10 min | `04_pipeline_and_deployment` |

## Prerequisites

- Python proficiency (pip/uv, virtual environments, notebooks)
- Basic familiarity with MLflow (experiment tracking, model logging)
- Understanding of LLM applications (prompts, completions, evaluation)
- No prior experience with Phoenix, TruLens, or Guardrails AI required

## Setup

Databricks workspace access will be provided during the session. To run locally:

```bash
pip install mlflow[genai] arize-phoenix trulens guardrails-ai scikit-learn numpy
```

## Speaker

**Debu Sinha** | Lead Applied AI/ML Engineer @ Databricks

Built MLflow's first third-party scorer integrations (Phoenix, TruLens, Guardrails AI) and the first MLflow tracking integration for Inspect AI. Author of *Practical Machine Learning on Databricks* (Packt, 2023).

## License

Apache 2.0
