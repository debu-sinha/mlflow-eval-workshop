# Troubleshooting

Most problems come from one of three places: the environment, the model provider, or the tracking location. Each checkpoint prints the step it is on, and every run saves a `summary.json` that names the failure without exposing keys or hosts.

## Local route

| Symptom | What to check |
|---|---|
| Python or package version error | Run `uv sync --locked --python 3.12` again, then `uv run --locked python scripts/verify_west_environment.py`. Do not install newer packages over the lock. |
| The configuration check is not ready | Load `OPENAI_API_KEY` in the terminal that runs the command. Remove an inherited `OPENAI_BASE_URL`. Leave `MLFLOW_TRACKING_URI` unset for the default route. |
| The check is ready but a live call fails | Check your OpenAI account access, model permissions, quota, and connection. The configuration check never contacts OpenAI. |
| The MLflow UI shows no workshop results | Run a checkpoint first, start the server from the same directory, and check the database path and experiment name. A bare `mlflow server` opens a different database. |
| Port 5000 is in use | Change only `--port 5000` to `--port 5001` and open `http://127.0.0.1:5001`. Keep the SQLite URI unchanged. |
| An old or unexpected experiment appears | Check inherited `MLFLOW_EXPERIMENT_ID`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_TRACKING_URI`, and `WORKSHOP_OUTPUT_DIR` values. |

The workshop writes to SQLite directly, so pointing `MLFLOW_TRACKING_URI` at the UI's HTTP address will not work. If you choose a different database, start the UI with that same path. See the [MLflow tracking server documentation](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) for other setups.

## Databricks Free Edition route

| Symptom | What to check |
|---|---|
| A package or version error | Apply the requirements file with the base environment from the README, in this notebook. Dependencies belong to each notebook. |
| A package is missing after pulling a Git update | Run the notebook's setup cell again. It refreshes Python's import cache before loading the workshop package. |
| Serverless compute capacity is reached | In a notebook you no longer need, open the **Serverless** menu, then its **Serverless** submenu, and choose **Terminate**. Then click **Apply** again in the notebook you want. |
| The endpoint is missing from Playground | Use the endpoint listing in the README to see which chat endpoints you can query. |
| A rate limit or a missing answer | Stop overlapping notebook runs, let the limit reset, read the saved failure, then rerun. A daily fair-use quota can mean waiting until it resets. |
| The judge disagrees with a reference label | Read the answer and the judge's explanation before changing either. Keep the label unless you find an error in it. |
| MLflow warns that `extraContext` is not whitelisted | This concerns optional notebook metadata. The lab filters exactly this warning and keeps every other warning and error visible. |
| An unexpected experiment | Check inherited `MLFLOW_EXPERIMENT_ID` and `MLFLOW_EXPERIMENT_NAME`. An ID from a local SQLite store does not identify a workspace experiment. |

A missing answer or score leaves a comparison incomplete, and the gate blocks it. Read the failure before rerunning, and keep the release thresholds fixed so every comparison stays meaningful.
