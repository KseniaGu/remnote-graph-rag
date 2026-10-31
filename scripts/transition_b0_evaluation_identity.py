"""Explicitly bind the canonical B0 ledger to a reviewed evaluator identity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.evaluation.portfolio_budget import (  # noqa: E402
    EvaluationBudgetUnavailable,
    transition_b0_evaluation_identity,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--expected-ledger-sha256", required=True)
    parser.add_argument("--previous-source-root", type=Path, required=True)
    parser.add_argument("--previous-inventory", type=Path, required=True)
    parser.add_argument("--expected-previous-inventory-sha256", required=True)
    parser.add_argument("--current-inventory", type=Path, required=True)
    parser.add_argument("--expected-current-inventory-sha256", required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--confirm-b0-evaluator-transition", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_b0_evaluator_transition:
        print("Explicit --confirm-b0-evaluator-transition is required", file=sys.stderr)
        return 2
    try:
        transition_b0_evaluation_identity(
            checkout=ROOT,
            ledger=args.budget_ledger,
            expected_ledger_sha256=args.expected_ledger_sha256,
            previous_source_root=args.previous_source_root,
            previous_inventory=args.previous_inventory,
            expected_previous_inventory_sha256=args.expected_previous_inventory_sha256,
            current_inventory=args.current_inventory,
            expected_current_inventory_sha256=args.expected_current_inventory_sha256,
            configuration=json.loads(args.configuration.read_text(encoding="utf-8")),
        )
    except (EvaluationBudgetUnavailable, OSError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("B0 evaluator identity transition applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
