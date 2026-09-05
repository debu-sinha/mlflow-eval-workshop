"""Run an audience checkpoint against a configured provider."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "databricks"), default="openai")
    parser.add_argument("--checkpoint", type=int, choices=range(6), default=0)
    parser.add_argument("--check", action="store_true", help="Check setup without model calls")
    parser.add_argument("--integrations", action="store_true", help="Run the optional Phoenix and TruLens scorers")
    args = parser.parse_args()
    os.environ["WORKSHOP_PROVIDER"] = args.provider
    from west_workshop import preflight, run_checkpoint, run_integrations

    if args.check:
        result = preflight(args.provider)
    elif args.integrations:
        result = run_integrations(args.provider)
    else:
        result = run_checkpoint(args.checkpoint, args.provider)
    print(json.dumps(result, indent=2, sort_keys=True))
    passed = result.get("ready") if args.check else result.get("status") == "passed"
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
