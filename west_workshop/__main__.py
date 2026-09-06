"""Run an audience checkpoint against a configured provider."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "databricks"), default="openai")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--checkpoint", type=int, choices=range(6), default=0)
    action.add_argument("--check", action="store_true", help="Check setup without model calls")
    action.add_argument("--integrations", action="store_true", help="Run the optional Phoenix and TruLens scorers")
    action.add_argument("--report", type=Path, metavar="SUMMARY_JSON", help="Create a visual report from a saved checkpoint 4 run, without model calls")
    args = parser.parse_args()
    if args.report:
        from .report import write_release_report

        result = json.loads(args.report.read_text(encoding="utf-8"))
        result["summary_path"] = str(args.report.resolve())
        report = write_release_report(result)
        print(json.dumps({"report_path": str(report), "recorded_status": result.get("status")}, indent=2))
        return 0
    os.environ["WORKSHOP_PROVIDER"] = args.provider
    from west_workshop import preflight, run_checkpoint, run_integrations

    if args.check:
        result = preflight(args.provider)
    elif args.integrations:
        result = run_integrations(args.provider)
    else:
        result = run_checkpoint(args.checkpoint, args.provider)
        if args.checkpoint == 4 and result.get("summary_path"):
            from .report import write_release_report

            result["report_path"] = str(write_release_report(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    passed = result.get("ready") if args.check else result.get("status") == "passed"
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
