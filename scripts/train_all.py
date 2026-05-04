"""Run the full Prefect training flow against real MOABB data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/train_all.py` from repo root without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from neurodrift.flows.training_flow import neurodrift_training_flow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the NeuroDrift training flow")
    parser.add_argument(
        "--global-subjects",
        type=str,
        default=None,
        help="Comma-separated subject ids for global acceptance training",
    )
    parser.add_argument(
        "--held-out-subject",
        type=int,
        default=None,
        help="Subject id to calibrate (Stage 2)",
    )
    args = parser.parse_args()

    global_subjects = [int(x) for x in args.global_subjects.split(",")] if args.global_subjects else None

    result = neurodrift_training_flow(
        global_subjects=global_subjects,
        held_out_subject=args.held_out_subject,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
