"""Explicitly apply the authorized B0 completion budget policy amendment."""

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
    amend_b0_completion_budget,
    amend_daily_policy_budget,
    capture_prospective_boundary,
    recover_fixed_judge_setup,
    register_prospective_observation,
    transition_final_evaluation_identity,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--expected-ledger-sha256", required=True)
    parser.add_argument("--previous-source-root", type=Path)
    parser.add_argument("--configuration", type=Path)
    confirmation = parser.add_mutually_exclusive_group()
    confirmation.add_argument("--confirm-b0-completion-amendment", action="store_true")
    confirmation.add_argument("--confirm-daily-policy-amendment", action="store_true")
    confirmation.add_argument(
        "--confirm-final-evaluator-transition", action="store_true"
    )
    confirmation.add_argument(
        "--confirm-fixed-judge-successor-transition", action="store_true"
    )
    confirmation.add_argument("--recover-fixed-judge-setup", type=Path)
    confirmation.add_argument("--capture-prospective-boundary", type=Path)
    confirmation.add_argument("--register-prospective-observation", type=Path)
    day_choice = parser.add_mutually_exclusive_group()
    day_choice.add_argument("--preserve-current-day-limit", action="store_true")
    day_choice.add_argument("--reviewed-day-proof", type=Path)
    day_choice.add_argument("--prospective-authority-proof", type=Path)
    parser.add_argument("--prospective-authority-proof-sha256", default="")
    parser.add_argument("--expected-account-id", default="")
    parser.add_argument("--expected-key-id", default="")
    parser.add_argument("--reviewed-day-proof-sha256", default="")
    parser.add_argument("--previous-inventory", type=Path)
    parser.add_argument("--recovery-input", type=Path)
    parser.add_argument("--recovery-contract-pair", type=Path)
    parser.add_argument("--previous-inventory-sha256", default="")
    parser.add_argument("--current-inventory", type=Path)
    parser.add_argument("--current-inventory-sha256", default="")
    parser.add_argument("--approved-patch", type=Path)
    parser.add_argument("--approved-patch-sha256", default="")
    args = parser.parse_args(argv)
    if args.capture_prospective_boundary is not None:
        try:
            capture_prospective_boundary(
                checkout=ROOT,
                ledger=args.budget_ledger,
                expected_ledger_sha256=args.expected_ledger_sha256,
                destination=args.capture_prospective_boundary,
            )
        except (EvaluationBudgetUnavailable, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print("Read-only quota capture boundary recorded")
        return 0
    if args.register_prospective_observation is not None:
        try:
            register_prospective_observation(
                checkout=ROOT,
                ledger=args.budget_ledger,
                expected_ledger_sha256=args.expected_ledger_sha256,
                proof_path=args.register_prospective_observation,
                expected_proof_sha256=args.prospective_authority_proof_sha256,
            )
        except (EvaluationBudgetUnavailable, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print("Prospective quota observation registered")
        return 0
    if args.recover_fixed_judge_setup is not None:
        if not all(
            (
                args.previous_source_root,
                args.previous_inventory,
                args.recovery_input,
                args.recovery_contract_pair,
            )
        ):
            print(
                "recovery requires parent source/inventory, input and contract pair",
                file=sys.stderr,
            )
            return 2
        try:
            recover_fixed_judge_setup(
                checkout=ROOT,
                ledger=args.budget_ledger,
                expected_ledger_sha256=args.expected_ledger_sha256,
                evidence_dir=args.recover_fixed_judge_setup,
                parent_source=args.previous_source_root,
                parent_inventory=args.previous_inventory,
                input_file=args.recovery_input,
                contract_pair=args.recovery_contract_pair,
            )
        except (EvaluationBudgetUnavailable, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print("Fixed judge no-dispatch recovery recorded")
        return 0
    if args.previous_source_root is None or args.configuration is None:
        print(
            "--previous-source-root and --configuration are required for amendment",
            file=sys.stderr,
        )
        return 2
    if (
        args.confirm_final_evaluator_transition
        or args.confirm_fixed_judge_successor_transition
    ):
        if not all(
            (
                args.previous_inventory,
                args.previous_inventory_sha256,
                args.current_inventory,
                args.current_inventory_sha256,
                args.approved_patch,
                args.approved_patch_sha256,
            )
        ):
            print("reviewed source inventories and patch are required", file=sys.stderr)
            return 2
        try:
            changed = transition_final_evaluation_identity(
                checkout=ROOT,
                ledger=args.budget_ledger,
                previous_source_root=args.previous_source_root,
                expected_ledger_sha256=args.expected_ledger_sha256,
                configuration=json.loads(
                    args.configuration.read_text(encoding="utf-8")
                ),
                previous_inventory=args.previous_inventory,
                expected_previous_inventory_sha256=args.previous_inventory_sha256,
                current_inventory=args.current_inventory,
                expected_current_inventory_sha256=args.current_inventory_sha256,
                approved_patch=args.approved_patch,
                expected_patch_sha256=args.approved_patch_sha256,
                fixed_judge_successor=args.confirm_fixed_judge_successor_transition,
            )
        except (EvaluationBudgetUnavailable, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            "Final evaluator identity transition"
            + (" applied" if changed else " already applied")
        )
        return 0
    if (
        not args.confirm_b0_completion_amendment
        and not args.confirm_daily_policy_amendment
    ):
        print("Explicit --confirm-b0-completion-amendment is required", file=sys.stderr)
        return 2
    if args.confirm_daily_policy_amendment and not (
        args.preserve_current_day_limit
        or args.reviewed_day_proof
        or args.prospective_authority_proof
    ):
        print(
            "Explicit --preserve-current-day-limit or --reviewed-day-proof is required",
            file=sys.stderr,
        )
        return 2
    try:
        amendment = (
            amend_daily_policy_budget
            if args.confirm_daily_policy_amendment
            else amend_b0_completion_budget
        )
        extra = (
            {
                "prospective_proof": args.prospective_authority_proof,
                "expected_prospective_sha256": args.prospective_authority_proof_sha256,
                "expected_account_id": args.expected_account_id,
                "expected_key_id": args.expected_key_id,
                "preserve_current_day": args.preserve_current_day_limit,
                "quota_provenance": args.reviewed_day_proof,
                "expected_provenance_sha256": args.reviewed_day_proof_sha256,
                "previous_inventory": args.previous_inventory,
                "expected_previous_inventory_sha256": args.previous_inventory_sha256,
                "current_inventory": args.current_inventory,
                "expected_current_inventory_sha256": args.current_inventory_sha256,
                "approved_patch": args.approved_patch,
                "expected_patch_sha256": args.approved_patch_sha256,
            }
            if args.confirm_daily_policy_amendment
            else {}
        )
        changed = amendment(
            checkout=ROOT,
            ledger=args.budget_ledger,
            previous_source_root=args.previous_source_root,
            expected_ledger_sha256=args.expected_ledger_sha256,
            configuration=json.loads(args.configuration.read_text(encoding="utf-8")),
            **extra,
        )
    except (EvaluationBudgetUnavailable, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        (
            "Daily policy amendment"
            if args.confirm_daily_policy_amendment
            else "B0 budget amendment"
        )
        + (" applied" if changed else " already applied")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
