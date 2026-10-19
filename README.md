# Would you accept this repair?

**Evaluating LLM Applications with MLflow · ODSC AI West 2026 · Debu Sinha**

A customer bought a non-defective item 45 days ago and asks for a full refund. The assistant uses an obsolete 90-day policy. The current rule says store credit after day 30.

I use this small example to make the evaluation process visible: inspect the answer, find the source in its trace, test the checks, and decide what the evidence supports. You can participate by watching and classifying an answer. Running code during the session is optional.

The main route is **setup plus two notebooks**. They contain real `mlflow.genai.evaluate` calls and an editable application adapter. Saved mode opens dated results from actual executions. Live mode makes new model calls and can produce a different decision.

| Open in order | What you will learn |
|---|---|
| [00_setup.ipynb](notebooks/workshop/00_setup.ipynb) | Choose saved or live mode and check the environment |
| [01_diagnose_and_evaluate.ipynb](notebooks/workshop/01_diagnose_and_evaluate.ipynb) | Follow a case through its source, answer, trace and checks |
| [02_compare_and_decide.ipynb](notebooks/workshop/02_compare_and_decide.ipynb) | Test the evaluator, inspect paired cases, and apply the declared policy |

Start with saved mode. It requires no credentials and makes no model calls. The evidence manifest verifies the recorded files before displaying them. Those hashes detect accidental changes. They are not signatures or a substitute for trusting the source of a download.

## Local Jupyter

Use the branch or source archive that accompanies this workshop. This revision is prepared on `odsc-west-2026-workshop`. The repository's default branch may contain the earlier seven-chapter workshop until this revision is merged.

```bash
git clone --branch odsc-west-2026-workshop https://github.com/debu-sinha/mlflow-eval-workshop.git
cd mlflow-eval-workshop
uv sync --locked --python 3.12 --extra local
uv run --locked --extra local python scripts/verify_west_environment.py
uv run --locked --extra local jupyter lab
```

Open `notebooks/workshop/00_setup.ipynb`. Run all cells, then do the same in 01 and 02. Each notebook starts a fresh session and reads the same evidence directory. You do not need variables from another kernel. The committed lock defines the tested dependency versions. See [verified routes](docs/verified-routes.md) for qualified versions and execution scope.

Before presenting a completed notebook 02 in JupyterLab, click the triangle beside **Vote before revealing the decision**. The **1 cell hidden** indicator confirms that the following verdict cell, including its HTML and text outputs, is hidden. Click **1 cell hidden** after the audience vote to reveal it. Leave the control and review outputs visible with their long helper source collapsed.

To make new model calls, configure `OPENAI_API_KEY` in the terminal that starts Jupyter. Never paste a key into a notebook or commit it. Set these nonsecret options before starting Jupyter, or in the setup cell of each notebook:

```python
import os
os.environ["WORKSHOP_MODE"] = "live"
os.environ["WORKSHOP_PROVIDER"] = "openai"
os.environ["WORKSHOP_OUTPUT_DIR"] = "artifacts/my-first-repair-run"
```

The default local application and judge use `gpt-4o-mini`. A complete new comparison requests 20 application answers, their semantic assessments, and 20 authored judge controls. SDK retries may add requests. Usage is billed by your provider. A failed control run withholds the repair comparison.

Use a new output directory when changing code, cases, model settings, dependencies, judge or policy versions. A completed stage is read back instead of silently generating it again. Preserve a failed run and record the reason for a retry or new experiment.

## Databricks serverless

Use the same source revision and notebooks in a Databricks Git folder. Saved mode reads the bundled local results unless you select the separate Databricks evidence folder. Live mode uses workspace authentication and the configured serving endpoint. No personal access token is required by these notebooks.

1. Create a Git folder for this repository and select the workshop revision.
2. Open `notebooks/workshop/00_setup.ipynb`. In the **Environment** panel, choose **Base environment > More > Standard v5** and add the committed requirements file using its absolute workspace path, for example `-r /Workspace/Users/<you>/<repo>/requirements-workshop.txt`.
3. Click **Apply**, confirm the environment reset, and wait for installation to finish. Repeat these Environment-panel steps in `01_diagnose_and_evaluate.ipynb` and `02_compare_and_decide.ipynb`. Each notebook has its own settings. Run all cells in 00, then 01, then 02.

For the first run, leave the default saved mode. To make new model calls, add the configuration cell below before the bootstrap cell in **each** notebook. Fresh kernels do not share Python variables or environment settings.

```python
import os
os.environ["WORKSHOP_MODE"] = "live"
os.environ["WORKSHOP_PROVIDER"] = "databricks"
os.environ["WORKSHOP_DATABRICKS_MODEL"] = "databricks-qwen3-next-80b-a3b-instruct"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Users/<you>/odsc-repair-workshop"
os.environ["WORKSHOP_OUTPUT_DIR"] = "/Workspace/Users/<you>/odsc-repair-evidence"
```

Replace `<you>` and choose an endpoint you can access. The preflight checks configuration presence, not endpoint availability or quota. The Free Edition's workspace limits still apply. [Verified routes](docs/verified-routes.md) distinguishes native notebook job execution from manual interactive setup checks.

To inspect the saved Databricks result, put this cell before the bootstrap cell in **each** notebook, replacing `<you>` and `<repo>` with your Git-folder path:

```python
import os
os.environ["WORKSHOP_MODE"] = "saved"
os.environ["WORKSHOP_EVIDENCE_DIR"] = "/Workspace/Users/<you>/<repo>/evidence/workshop/databricks/final"
```

Without this selection, saved mode opens the local bundle even when the notebook runs on Databricks. Check the displayed provider and evidence ID before comparing results. Set `WORKSHOP_PROJECT_ROOT` explicitly only if your notebook is outside the repository.

If Databricks reports that its HTML-output host is unreachable and offers **Run in strict sandbox**, that supported option rendered the workshop tables in the October 18, 2026 Chrome walkthrough. After execution finishes, reload the page if an HTML output is blank. Reloading the page displayed completed saved and live outputs without another model run.

Before presenting a completed notebook, check its visible state. The long control-example and review-display source cells in 02 should be collapsed while their outputs remain visible. Leave the material finding visible and secondary findings collapsed. Use the **Collapse cell** control beside the final decision cell before the audience vote, then **Expand cell** to reveal the recorded result. This hides both its HTML and text verdicts. The two short code views in 01 remain part of the explanation.

## Read a decision without rerunning a model

```bash
uv run --locked python -m west_workshop evaluate --mode saved --evidence evidence/workshop/local/final
uv run --locked python -m west_workshop gate --mode saved --evidence evidence/workshop/local/final
```

`evaluate` exits 0 when the measurement/report completes, including a valid BLOCK. Exit 1 means execution failed. `gate` writes the report and then enforces its result:

| Exit | Meaning |
|---|---|
| 0 | PASS under the named workshop policy |
| 2 | BLOCK: a policy condition failed |
| 3 | REVIEW_REQUIRED: evidence needs review |
| 4 | INVALID: the recorded evidence cannot support a comparison |
| 1 | Execution failed, including NOT_EVALUATED |

The earlier `eval_gate.py` checks one named scorer. It is a narrower contract and must not be described as this full policy.

## What the policy means

The simulated deployed version receives obsolete policy. The proposed repair receives current policy. The cases and evaluator definitions stay fixed. The faulty version can contain the defect being repaired.

[Repair policy v2](west_workshop/repair_policy.py) requires complete comparable evidence, valid binary scores, passing judge controls, and an explicit review bound to these results. The repair must pass every mandatory deterministic check and reach a nine-of-ten all-required-check floor. At most one ordinary paired loss is allowed. Regressions on the predeclared transaction-authority and defective-item cases block independently of gains elsewhere.

These are teaching thresholds, not universal production thresholds. A material unresolved finding cannot be silently cleared. Paired statistics describe this small sample. They do not establish safety or isolate judge variation. **A workshop policy pass is not authorization to deploy.**

## Adapt it to your app

| Your change | Where to start |
|---|---|
| Call your application and record its trace | `make_predictor` in [runtime.py](west_workshop/runtime.py), passed as `predict_fn` |
| Add reviewed inputs and expected behavior | [data.py](west_workshop/data.py), retaining stable case IDs |
| Change deterministic business checks | [runtime.py](west_workshop/runtime.py) |
| Define and qualify a semantic judge | [judge_v2.py](west_workshop/judge_v2.py) and [qualification_controls.py](west_workshop/qualification_controls.py) |
| Declare acceptance rules and review ownership | [repair_policy.py](west_workshop/repair_policy.py) |
| Preserve evidence across notebooks and enforce the policy | [repair_workshop.py](west_workshop/repair_workshop.py) |

Begin with one new case whose expected behavior you can explain. Test both application versions on the same cases. Review the answer and rationale when a score looks wrong.

Use `review-template` to create a pending review tied to a recorded bundle. `promote-case` requires an explicit human approval record before adding a labelled case. `rescore` evaluates all frozen compared answers with the current versioned judge, using `predict_fn=None`. It does not regenerate application answers. `rescore-legacy` handles all three historical variants and preserves the original v1 verdicts separately. Run `python -m west_workshop --help` for the commands. Never patch a single inconvenient score or rename an old decision as a new pass.

For your production application, add representative held-out cases, domain-owner review, evaluator validation, security tests, operational limits, latency/cost budgets, monitoring and a release owner. The workshop gives you the evaluation structure. Those requirements depend on your system.

## Attach a review before freezing a saved result

After recording the comparison, create a bound review form:

```bash
python -m west_workshop review-template --evidence artifacts/my-first-repair-run --output artifacts/my-review
```

Inspect the answers, rationales, controls and any failures. Complete `artifacts/my-review/review.json` with the actual reviewer identity and kind (`human`, `model` or `mixed`), findings, and supporting evidence for any resolution. Keep material unresolved findings open. A complete review can still require further work.

Append the review using the same configured environment and original live directory:

```bash
python -m west_workshop evaluate --mode live --output artifacts/my-first-repair-run --review-json artifacts/my-review/review.json
```

When both answer batches and controls are already recorded, this reuses them and appends the reviewed decision without new model calls. It preserves the initial pending decision. To freeze a shareable saved bundle after review, copy `manifest.json` and only the files listed in its `files` mapping, retaining their relative paths. The raw output directory can also contain a local tracking store and private runtime artifacts. Those are not part of the portable bundle. Verify the copied bundle in saved mode. A fresh unreviewed run intentionally ends at REVIEW_REQUIRED.

## Optional material and historical evidence

The original [seven chapter files](notebooks/west/) remain reference material. Their v1 three-run comparison and historical results are distinct from the two-version workshop path. [Original instructions](docs/legacy-seven-chapter-readme.md) are archived for provenance, not the current starting route.

The original audits ran real Phoenix Hallucination and TruLens Coherence adapters on one answer in both environments. Neither metric enters the repair policy. The public session abstract also named Guardrails AI. This revision does not claim a qualified Guardrails execution. The core phrase check is a custom check, not that library. See [verified routes](docs/verified-routes.md) for precise integration scope.

Current MLflow references: [GenAI evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/), [RAG scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/), and [tool-call scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/). This example explains policy lookup and response evaluation. It does not execute financial transactions or demonstrate a multi-agent system.
