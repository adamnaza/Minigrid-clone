"""Combine evaluation summaries without loading models or training dependencies."""

import argparse
import csv
from pathlib import Path

from minigrid.custom_scripts import PROJECT_ROOT

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"


def collect(results_dir, output, include_instructions=False):
    summaries = sorted(results_dir.glob("*/summary.csv"))
    if not summaries:
        raise ValueError(f"No */summary.csv files found in {results_dir}")

    rows = []
    columns = ["run"]
    for summary in summaries:
        with summary.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "instruction_id" not in reader.fieldnames:
                raise ValueError(f"Missing instruction_id column: {summary}")
            if "run" in reader.fieldnames:
                raise ValueError(f"Reserved column 'run' already exists: {summary}")
            for column in reader.fieldnames:
                if column not in columns:
                    columns.append(column)
            run_rows = list(reader)
        overall = [row for row in run_rows if row["instruction_id"] == "overall"]
        if len(overall) != 1:
            raise ValueError(f"Expected exactly one overall row: {summary}")
        rows.extend({"run": summary.parent.name, **row}
                    for row in (run_rows if include_instructions else overall))

    # Validate every input before creating or replacing the combined output.
    if output.resolve() in {summary.resolve() for summary in summaries}:
        raise ValueError("Output must not overwrite an input summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(summaries), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Directory containing one subdirectory per run")
    parser.add_argument("--output", type=Path,
                        help="Output CSV (default: <results-dir>/combined_summary.csv)")
    parser.add_argument("--include-instructions", action="store_true",
                        help="Include per-instruction rows as well as overall rows")
    args = parser.parse_args()
    output = args.output or args.results_dir / "combined_summary.csv"
    try:
        runs, rows = collect(args.results_dir, output, args.include_instructions)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Collected {runs} runs ({rows} rows) into {output}")


if __name__ == "__main__":
    main()
