"""Northstar support app: the same predictor used by the evaluation notebooks."""

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

from west_workshop.support_app import create_app

app = create_app()

if __name__ == "__main__":
    # A local browser demo. Databricks Apps starts Gunicorn through app.yaml.
    if os.name == "nt":
        from waitress import serve

        serve(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")), threads=4)
    else:
        from gunicorn.app.base import BaseApplication

        class LocalServer(BaseApplication):
            def load_config(self):
                for name, value in {"bind": "127.0.0.1:" + os.environ.get("PORT", "8000"), "workers": 1, "threads": 4, "timeout": 180}.items():
                    self.cfg.set(name, value)

            def load(self):
                return app

        LocalServer().run()
