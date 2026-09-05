"""Check the locked West stack and real scorer constructors without provider calls.

Core:      uv run --locked python scripts/verify_west_environment.py
Advanced:  uv run --locked --extra ecosystem python scripts/verify_west_environment.py --ecosystem

The CLI blocks outbound socket access. Constructor checks do not establish that
credentials, a hosted tracking server, or paid judge execution work.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import ipaddress
import os
import platform
import sys
import time

from packaging.specifiers import SpecifierSet

CORE_REQUIREMENTS = {
    "mlflow": "==3.16.0",
    "mlflow-skinny": "==3.16.0",
    "mlflow-tracing": "==3.16.0",
    "openai": ">=2.54",
    "numpy": ">=1.26,<2",
    "protobuf": ">=5.26.1,<6",
    "scikit-learn": ">=1.4",
}
ECOSYSTEM_REQUIREMENTS = {
    "arize-phoenix-evals": ">=2.11,<3",
    "trulens": "==2.14.0",
    "trulens-core": "==2.14.0",
    "trulens-feedback": "==2.14.0",
    "trulens-providers-litellm": "==2.14.0",
    "litellm": ">=1.82.6,!=1.82.7,!=1.82.8",
    "nltk": ">=3.9",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _check_versions(requirements: dict[str, str]) -> dict[str, str]:
    versions = {}
    for package, constraint in requirements.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"{package} is missing. Sync the locked environment and requested extras."
            ) from error
        _require(
            installed in SpecifierSet(constraint),
            f"{package} {installed} does not satisfy the West contract {constraint}.",
        )
        versions[package] = installed
        print(f"  {package}: {installed}", flush=True)
    return versions


def _check_core_contracts() -> None:
    from mlflow.genai import make_judge
    from mlflow.genai.scorers import (
        Correctness,
        RegexMatch,
        RelevanceToQuery,
        Safety,
        make_scorer_ensemble,
    )

    # No call to these LLM judges: construction validates API/config compatibility.
    for scorer_class in (Correctness, RelevanceToQuery, Safety):
        constructed = scorer_class(model="openai:/gpt-4o-mini")
        _require(bool(constructed.name), f"{scorer_class.__name__} has no name.")
    judge = make_judge(
        name="west_contract_judge",
        instructions="Return true if {{ outputs }} answers {{ inputs }}.",
        model="openai:/gpt-4o-mini",
        feedback_value_type=bool,
    )
    _require(judge.name == "west_contract_judge", "make_judge lost the configured name.")

    # Exercise real deterministic scoring, including a negative control.
    citation = RegexMatch(name="citation", pattern=r"\[KB-\d+\]")
    _require(citation(outputs="Answer [KB-42]").value == "yes", "Regex positive failed.")
    _require(citation(outputs="Unsupported answer").value == "no", "Regex negative failed.")
    ensemble = make_scorer_ensemble(
        name="west_contract_ensemble",
        scorers=[citation, RegexMatch(name="prefix", pattern=r"^Answer")],
        ensemble_fn="agg_all",
    )
    _require(ensemble(outputs="Answer [KB-42]").value is True, "Ensemble positive failed.")
    _require(
        ensemble(outputs="Answer without citation").value is False,
        "Ensemble negative failed. Categorical 'no' must not be treated as truthy.",
    )
    print("Core constructors, RegexMatch and ensemble positive/negative controls: OK", flush=True)


def _check_ecosystem_contracts() -> None:
    # Use the bundled cost table to keep LiteLLM imports independent of network access.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ.setdefault("OLLAMA_API_BASE", "http://127.0.0.1:11434")
    from mlflow.genai.scorers.phoenix import Hallucination
    from mlflow.genai.scorers.trulens import Groundedness
    from phoenix.evals import HallucinationEvaluator, LiteLLMModel
    from trulens.providers.litellm import LiteLLM

    # ollama_chat is a LiteLLM provider absent from MLflow's native registry.
    # Its constructor requires no API credentials or running Ollama server.
    for model in ("databricks", "ollama_chat:/llama3.1:8b"):
        started = time.perf_counter()
        phoenix = Hallucination(model=model)
        _require(
            isinstance(phoenix._evaluator, HallucinationEvaluator),
            "Phoenix did not construct its legacy HallucinationEvaluator.",
        )
        if model.startswith("ollama_chat:"):
            _require(
                isinstance(phoenix._evaluator._model, LiteLLMModel),
                "Phoenix did not construct its LiteLLM fallback model.",
            )
        print(f"Phoenix Hallucination constructor ({model}): OK", flush=True)
        trulens = Groundedness(model=model)
        _require(
            callable(getattr(trulens._provider, trulens._method_name, None)),
            "TruLens provider does not implement the groundedness feedback method.",
        )
        if model.startswith("ollama_chat:"):
            _require(
                isinstance(trulens._provider, LiteLLM),
                "TruLens did not construct its LiteLLM fallback provider.",
            )
        print(
            f"TruLens Groundedness constructor ({model}): OK "
            f"({time.perf_counter() - started:.1f}s for pair)",
            flush=True,
        )


def verify_environment(*, ecosystem: bool = False, databricks: bool = False) -> dict[str, str]:
    """Validate installed versions and constructors without executing an LLM judge."""
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    _require(
        (3, 10) <= sys.version_info[:2] < (3, 13),
        "West supports Python 3.10-3.12 with NumPy <2. Use Python 3.12 locally.",
    )
    print(f"Python: {platform.python_version()}", flush=True)
    versions = _check_versions(CORE_REQUIREMENTS)
    # Check the exclusions even when LiteLLM is installed through another extra.
    try:
        litellm_version = metadata.version("litellm")
    except metadata.PackageNotFoundError:
        pass
    else:
        _require(
            litellm_version in SpecifierSet("!=1.82.7,!=1.82.8"),
            "The installed LiteLLM version is explicitly excluded by this workshop.",
        )
    if databricks:
        versions.update(_check_versions({"databricks-agents": ">=0.14"}))
    _check_core_contracts()
    if ecosystem:
        versions.update(_check_versions(ECOSYSTEM_REQUIREMENTS))
        _check_ecosystem_contracts()
    print("READY: environment contracts passed. No LLM judge was called.", flush=True)
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecosystem", action="store_true", help="Check optional Phoenix/TruLens.")
    parser.add_argument("--databricks", action="store_true", help="Require the Databricks extra.")
    args = parser.parse_args()
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    attempts: list[str] = []

    def block_network(event: str, audit_args: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            # Windows implements asyncio's internal socketpair using loopback TCP.
            host = audit_args[1][0] if event == "socket.connect" else audit_args[0]
            try:
                if ipaddress.ip_address(host).is_loopback:
                    return
            except ValueError:
                if host == "localhost":
                    return
            attempts.append(event)
            raise RuntimeError(f"Environment verification blocked {event} to {host}.")

    # This process ends after verification. The notebook function intentionally does
    # not install an irreversible audit hook in an interactive Python session.
    sys.addaudithook(block_network)
    verify_environment(ecosystem=args.ecosystem, databricks=args.databricks)
    _require(not attempts, f"Verification attempted {len(attempts)} network operations.")
    print("Outbound network operations: 0", flush=True)


if __name__ == "__main__":
    main()
