"""Rebuild consolidated Graph RAG evaluation reports from immutable history."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.evaluation.evaluation_reporting import (  # noqa: E402
    DEFAULT_EVALUATION_ROOT,
    DEFAULT_REPORTS_ROOT,
    build_evaluation_reports,
    publish_evaluation_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild scorecards from completed Graph RAG evaluation runs."
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=DEFAULT_EVALUATION_ROOT,
    )
    parser.add_argument(
        "--fingerprint",
        help="Optionally rebuild only one exact evaluation fingerprint.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish a sanitized scorecard snapshot to the commit-friendly reports tree.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Commit-friendly publication root used with --publish.",
    )
    parser.add_argument(
        "--confirm-complete",
        action="store_true",
        help="Publish only when the selected campaign passes completeness checks.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Explicitly permit publication of a visibly incomplete campaign.",
    )
    parser.add_argument(
        "--audit-completeness",
        action="store_true",
        help="Print machine-readable completeness diagnostics for --fingerprint.",
    )
    parser.add_argument(
        "--verify-run-fingerprints",
        action="store_true",
        help="Verify that every repeated --run-id has one fingerprint.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Exact campaign run ID used by --verify-run-fingerprints.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_run_fingerprints:
        if not args.run_id:
            print(
                "Fingerprint verification requires at least one --run-id.",
                file=sys.stderr,
            )
            return 2
        runs = []
        missing = []
        for run_id in args.run_id:
            manifest_path = args.evaluation_root / "runs" / run_id / "manifest.json"
            if not manifest_path.is_file():
                missing.append(run_id)
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            runs.append(
                {
                    "run_id": run_id,
                    "status": manifest.get("status"),
                    "fingerprint": manifest.get("evaluation_fingerprint"),
                }
            )
        fingerprints = sorted(
            {str(item["fingerprint"]) for item in runs if item.get("fingerprint")}
        )
        result = {
            "runs": runs,
            "missing_run_ids": missing,
            "fingerprints": fingerprints,
            "all_share_one_fingerprint": not missing and len(fingerprints) == 1,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["all_share_one_fingerprint"] else 1

    if args.audit_completeness:
        if not args.fingerprint:
            print("Completeness audit requires --fingerprint.", file=sys.stderr)
            return 2
        reports = build_evaluation_reports(
            args.evaluation_root, fingerprint=args.fingerprint
        )
        scorecard = reports.get(args.fingerprint)
        if scorecard is None:
            print("No scorecard exists for the requested fingerprint.", file=sys.stderr)
            return 2
        coverage = scorecard.get("coverage", {})
        print(json.dumps(coverage, indent=2, ensure_ascii=False))
        return 0 if coverage.get("complete") else 1

    if args.publish:
        try:
            published = publish_evaluation_report(
                args.evaluation_root,
                args.reports_root,
                fingerprint=args.fingerprint,
                allow_incomplete=args.allow_incomplete,
                confirm_complete=args.confirm_complete,
            )
        except ValueError as exc:
            print(f"Evaluation report publication failed: {exc}", file=sys.stderr)
            return 2
        print(
            f"Published {published['fingerprint']} to "
            f"{published['latest_markdown']} "
            f"({published['contributing_run_count']} contributing run(s))."
        )
        return 0

    reports = build_evaluation_reports(
        args.evaluation_root,
        fingerprint=args.fingerprint,
    )
    print(
        f"Rebuilt {len(reports)} scorecard(s) under "
        f"{args.evaluation_root / 'fingerprints'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
