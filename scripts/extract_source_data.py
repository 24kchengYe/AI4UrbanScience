from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai4us.source_data import validate_source_data, write_validation_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and optionally extract the 16 quantitative Source Data sheets."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_source_data(args.output_dir)
    if args.report:
        write_validation_report(report, args.report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
