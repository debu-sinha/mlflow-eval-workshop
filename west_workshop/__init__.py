"""Independent live checkpoints for Debu Sinha's ODSC AI West workshop."""

from .config import preflight


def run_checkpoint(index: int, provider=None, output_dir=None) -> dict:
    """Run one live checkpoint, or return a safe blocked readiness report."""
    from .runtime import run_checkpoint as run

    return run(index, provider=provider, output_dir=output_dir)


def run_integrations(provider=None, output_dir=None) -> dict:
    """Run the optional real Phoenix and TruLens integration acceptance path."""
    from .runtime import run_integrations as run

    return run(provider=provider, output_dir=output_dir)


__all__ = ["preflight", "run_checkpoint", "run_integrations"]
