# Evaluating LLM Applications with MLflow

**ODSC AI East 2026 Workshop** | April 28-30, Boston | 60 minutes

## What you'll build

A production evaluation pipeline that goes from scorer selection to deployment gating in one continuous story: evaluate a real LLM with built-in and third-party scorers, compare two evaluation runs with statistical rigor, and block bad models from production with an automated gate.

![MLflow Experiment Overview](notebooks/images/mlflow-experiment-overview.png)

## Modules

| # | Module | Time | What you'll do |
|---|--------|------|----------------|
| 1 | MLflow Evaluation Ecosystem | 20 min | Run built-in, third-party, and custom scorers in one `evaluate()` call |
| 2 | Production Infrastructure | 8 min | Control judge temperature for determinism, manage scorer concurrency |
| 3 | Comparing Runs and Regressions | 12 min | Align samples across runs, detect regressions, test significance |
| 4 | The Evaluation Gate | 15 min | Build a gate that blocks deployment on regression, wire into CI/CD |
| - | Bonus: uv Dependencies | 5 min | Pin transitive deps with uv lockfiles for reproducible serving |

## Setup

### Option A: Databricks Workspace (recommended for the live workshop)

Sign up for the [Databricks Free Edition](https://login.databricks.com/signup) if you don't have a workspace. Free Edition includes serverless compute, MLflow tracking, and Git Folders.

1. In the sidebar, click **Workspace > Repos** and open your user folder
2. Click **Create Git folder**, paste `https://github.com/debu-sinha/mlflow-eval-workshop`
3. Open `notebooks/01_mlflow_evaluation_ecosystem` and attach to any serverless cluster

No API keys needed. The notebooks auto-detect Databricks and use Foundation Model APIs. MLflow tracking is built in.

**Model availability:**
- **Free Edition**: `databricks-gpt-oss-120b` (sufficient for the workshop)
- **Enterprise**: Premium models like `databricks-gpt-5-4`. Set `WORKSHOP_JUDGE_MODEL` on the cluster to override.

### Option B: Run locally

```bash
git clone https://github.com/debu-sinha/mlflow-eval-workshop.git
cd mlflow-eval-workshop

# Install dependencies (pick one)
pip install -e .                    # from pyproject.toml
# or
pip install mlflow[genai]>=3.1 arize-phoenix-evals trulens trulens-providers-litellm scikit-learn numpy openai

# Set your OpenAI key (needed for LLM-based scorers)
export OPENAI_API_KEY="sk-..."

# Start tracking server with the same SQLite backend the notebooks use
mlflow server --backend-store-uri sqlite:///mlflow_workshop.db --port 5000 &
```

Open the notebooks as Python scripts in VS Code, or convert to Jupyter:

```bash
# The notebooks are Databricks-exported .py files. Each "# COMMAND ----------"
# line is a cell boundary. VS Code and Databricks both handle this natively.
# To convert to .ipynb for Jupyter, use jupytext:
pip install jupytext
jupytext --to notebook notebooks/01_mlflow_evaluation_ecosystem.py
```

### Guardrails AI setup (optional)

The `DetectPII` scorer in Module 1 uses a Guardrails Hub validator that requires an API token. The rest of the workshop works without it.

1. Create a free account at [hub.guardrailsai.com](https://hub.guardrailsai.com)
2. After signing in, go to **Settings > API Keys** and copy your token
3. Set it as an environment variable before running the notebooks:

```bash
# Local
export GUARDRAILS_API_KEY="your-token-here"

# Databricks: set as a cluster environment variable or store in a secret scope
# Cluster env var: GUARDRAILS_API_KEY=your-token-here
# Secret scope: dbutils.secrets.get(scope="guardrails-hub", key="api-token")
```

If you skip this step, Module 1 will print a warning when it tries to install the `detect_pii` validator. All other scorers (Correctness, Safety, Hallucination, Groundedness, and custom scorers) work without it.

## What you'll learn

1. Run built-in scorers (`Correctness`, `Safety`) and third-party scorers (Phoenix `Hallucination`, TruLens `Groundedness`, Guardrails `DetectPII`) from a single `mlflow.genai.evaluate()` call
2. Write custom scorers with the `@scorer` decorator and mix them with built-in and third-party scorers
3. Call a real LLM, evaluate its response, and inspect traces in the MLflow UI
4. Configure judge parameters (`temperature=0.0`) and scorer concurrency (`MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS`) for reproducible evaluations
5. Compare two evaluation runs with McNemar's test, bootstrap CI, Cohen's d, and win rate
6. Build an evaluation gate that blocks model promotion on regression and integrate it into CI/CD

### MLflow evaluation capabilities beyond the workshop

MLflow's evaluation surface is broader than what fits in 60 minutes. Features worth exploring after the workshop:

- **Evaluation datasets**: First-class dataset objects for evaluation-driven development ([docs](https://mlflow.org/docs/latest/genai/datasets/))
- **Built-in RAG judges**: `Groundedness`, `RelevanceToQuery`, `ChunkRelevance` without third-party deps ([docs](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/))
- **Judge Builder UI**: Visual judge creation in the MLflow UI (requires MLflow >= 3.9) ([docs](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/))
- **Trace-based evaluation**: Pass `mlflow.search_traces()` output directly into `evaluate()` ([docs](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/))
- **Scheduled scorers**: Automatically evaluate production traces on a schedule ([docs](https://mlflow.org/docs/latest/python_api/mlflow.genai.html))
- **Agent evaluation**: `ToolCallCorrectness` and `ToolCallEfficiency` for evaluating agent tool usage ([docs](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/))

## The evaluation gate

The repo includes `eval_gate.py`, a standalone script that compares two MLflow evaluation runs and exits with code 1 if the candidate regresses. It aligns samples by hashing the input content (not by trace ID, which differs across runs), performs an inner join on the shared keys, and reports overlap statistics.

```bash
python eval_gate.py \
    --baseline-run-id <BASELINE_RUN_ID> \
    --candidate-run-id <CANDIDATE_RUN_ID> \
    --scorer correctness \
    --threshold 0.10
```

The GitHub Actions workflow (`.github/workflows/eval-gate.yml`) wraps this for CI/CD. Point `MLFLOW_TRACKING_URI` at your tracking server and trigger manually with run IDs.

![MLflow Traces with Assessment Data](notebooks/images/mlflow-traces-with-data.png)

## Background

This workshop covers MLflow's GenAI evaluation ecosystem, including features contributed by the speaker to MLflow core and the broader evaluation community:

- **Third-party scorer integrations** (Phoenix, TruLens, Guardrails AI): Connect external evaluation libraries to MLflow's unified `evaluate()` API. PRs [#19473](https://github.com/mlflow/mlflow/pull/19473), [#19492](https://github.com/mlflow/mlflow/pull/19492), [#20038](https://github.com/mlflow/mlflow/pull/20038)
- **LLM judge inference parameters**: Temperature, max tokens, and other controls for deterministic evaluation scoring. PR [#19152](https://github.com/mlflow/mlflow/pull/19152)
- **Scorer parallelism control**: `MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS` for managing API rate limits during concurrent evaluation. PR [#19248](https://github.com/mlflow/mlflow/pull/19248)
- **Inspect AI MLflow tracking**: Logging hook for the UK AI Safety Institute's evaluation framework. PRs [#3433](https://github.com/UKGovernmentBEIS/inspect_ai/pull/3433), [#3483](https://github.com/UKGovernmentBEIS/inspect_ai/pull/3483)

MLflow is downloaded over 30 million times per month from PyPI.

## Prerequisites

- Python 3.10+
- An OpenAI API key (for LLM-based scorers when running locally)
- Basic familiarity with MLflow (experiment tracking, model logging)
- No prior experience with Phoenix, TruLens, or Guardrails AI required

## Speaker

**Debu Sinha** | Lead Applied AI/ML Engineer @ Databricks

Built MLflow's first third-party scorer integrations ([Phoenix](https://github.com/mlflow/mlflow/pull/19473), [TruLens](https://github.com/mlflow/mlflow/pull/19492), [Guardrails](https://github.com/mlflow/mlflow/pull/20038)) and the first MLflow tracking integration for [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai/pull/3433). Author of *Practical Machine Learning on Databricks* (Packt, 2023).

- [LinkedIn](https://linkedin.com/in/debusinha)
- [GitHub](https://github.com/debu-sinha)

Presented at [ODSC AI East 2026](https://odsc.com/boston/), Boston, April 28-30.

## License

[Apache 2.0](LICENSE)
