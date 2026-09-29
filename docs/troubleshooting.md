# Troubleshooting

Most problems come from one of three places: the environment, the model provider, or the tracking location. Each checkpoint prints the step it is on. A run that does not pass prints a setup issue or a run issue that names the likely cause. Every run that passes the configuration check also saves a `summary.json` that records the run issue, without keys or hosts.

## When an exercise does not pass

| The run issue starts with | What to check |
|---|---|
| At least one answer or score is missing | A model request failed or timed out. Locally, check your OpenAI key, model access, and quota. In Databricks, check that the endpoint is available and that no other notebook is calling it. Then rerun. |
| No answer failed the policy check | Chapters 0 to 2 expect the stale candidate to give at least one wrong eligibility. Fresh answers vary, and this time every label was right. Read the answers, then rerun once. |
| The judge could not score all eight rubric controls | Chapter 4 checks the judge on eight authored controls before it compares releases. A judge request failed, so check the judge model's access and quota, then rerun. |
| The judge disagreed with at least one of its eight rubric controls | The next line names each control the judge got wrong. Open the `west-authored_judge_validation` run in MLflow and read that control's reply, the judge's explanation, and the authored label. Rerun only after you understand the disagreement. |
| The comparison finished, but the gates did not block the stale candidate and ship the repair | Read each version's reason in the report and open **Review the judge**. Every deterministic check must pass in the baseline and the evaluated version, and both need at least 90% of cases passing every required check. So one failed check, or two judge rejections in the baseline or the repair, can block the repair. Keep the thresholds fixed. |
| The live checkpoint did not complete | The run stopped with an error before it finished. Check authentication, model access, package versions, and tracking, then rerun. |

## Local route

| Symptom | What to check |
|---|---|
| Python or package version error | Run `uv sync --locked --python 3.12` again, then `uv run --locked python scripts/verify_west_environment.py`. Do not install newer packages over the lock. |
| The configuration check is not ready | Load `OPENAI_API_KEY` in the terminal that runs the command. Remove an inherited `OPENAI_BASE_URL`. Leave `MLFLOW_TRACKING_URI` unset for the default route. |
| The check is ready but a live call fails | Check your OpenAI account access, model permissions, quota, and connection. The configuration check never contacts OpenAI. |
| The MLflow UI shows no workshop results | Run a checkpoint first, start the server from the same directory, and check the database path and experiment name. A bare `mlflow server` opens a different database. |
| Port 5000 is in use | Change only `--port 5000` to `--port 5001` and open `http://127.0.0.1:5001`. Keep the SQLite URI unchanged. If you use the support app, also set `WORKSHOP_MLFLOW_UI_URL` to `http://127.0.0.1:5001`. |
| An old or unexpected experiment appears | Check inherited `MLFLOW_EXPERIMENT_ID`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_TRACKING_URI`, and `WORKSHOP_OUTPUT_DIR` values. |

The workshop writes to SQLite directly, so pointing `MLFLOW_TRACKING_URI` at the UI's HTTP address will not work. If you choose a different database, start the UI with that same path. See the [MLflow tracking server documentation](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) for other setups.

## Databricks Free Edition route

| Symptom | What to check |
|---|---|
| A package or version error | Apply the requirements file with the base environment from the README, in this notebook. Dependencies belong to each notebook. |
| Python fails to start after **Apply**, with a protobuf version error | Select **Standard v5** under **Base environment > More** and apply again. Standard v6 needs a newer protobuf than the workshop pins. |
| The **Environment** panel warns that core Python package versions changed | This is expected. The requirements file replaces some base environment packages with the versions the workshop was tested with, such as NumPy 1.26.4. The notebooks run normally with this warning. |
| A package is missing after pulling a Git update | Run the notebook's setup cell again. It refreshes Python's import cache before loading the workshop package. |
| Serverless compute capacity is reached | In a notebook you no longer need, open the **Serverless** menu, then its **Serverless** submenu, and choose **Terminate**. Then click **Apply** again in the notebook you want. |
| The endpoint is missing from Playground | Use the endpoint listing in the README to see which chat endpoints you can query. |
| A rate limit or a missing answer | Stop overlapping notebook runs, let the limit reset, read the saved failure, then rerun. A daily fair-use quota can mean waiting until it resets. |
| The judge disagrees with a reference label | Read the answer and the judge's explanation before changing either. Keep the label unless you find an error in it. |
| The report shows a light background with BLOCK in teal and SHIP in red | Databricks dark mode inverts the colors of HTML output. The decisions and numbers are unchanged. |
| MLflow warns that `extraContext` is not whitelisted | This concerns optional notebook metadata. The lab filters exactly this warning and keeps every other warning and error visible. |
| An unexpected experiment | Check inherited `MLFLOW_EXPERIMENT_ID` and `MLFLOW_EXPERIMENT_NAME`. An ID from a local SQLite store does not identify a workspace experiment. |

A missing answer or score leaves a comparison incomplete, and the gate blocks it. Read the failure before rerunning, and keep the release thresholds fixed so every comparison stays meaningful.

## Optional integrations

| Symptom | What to check |
|---|---|
| The environment check prints `Package jsonschema not present in requirements.` | TruLens logs this when it loads an optional JSON schema validator. The workshop does not use that validator, so the message is harmless. |
| Phoenix or TruLens fails to import | Install the `ecosystem` extra locally, or apply `requirements-ecosystem.txt` in Databricks. MLflow 3.16 needs the 2.x series of `arize-phoenix-evals`, so do not upgrade it on its own. |
