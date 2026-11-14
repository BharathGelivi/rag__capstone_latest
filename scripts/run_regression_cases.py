"""Run captured RAG regressions; exit nonzero for failures or unavailable judgments."""
import argparse
import json
from pathlib import Path
import sys

from src.regression_cases import load_cases, run_case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="eval/regression_cases")
    parser.add_argument("--mode", choices=("generate", "retrieve", "snapshot"), default="generate")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", default=".cache/regression-results.json")
    args = parser.parse_args(argv)
    try:
        if not Path(args.cases).is_dir():
            raise ValueError("Regression case directory is missing")
        cases = load_cases(args.cases)
        if args.validate:
            print(f"Validated {len(cases)} regression cases")
            return 0
        results = []
        for case in cases:
            result = run_case(case, args.mode)
            results.append(result)
            print(f"{'PASS' if result['passed'] else 'FAIL'} {case.case_id}: {case.name}")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"mode": args.mode, "passed": all(r["passed"] for r in results),
                                      "cases": results}, indent=2) + "\n", encoding="utf-8")
        print(f"{len(results)} cases; report: {output}")
        return int(any(not r["passed"] for r in results))
    except (ValueError, OSError) as exc:
        print(f"Regression suite error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
