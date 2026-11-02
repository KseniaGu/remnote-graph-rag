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
    publish_saved_evaluation,
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

    parser.add_argument(
        "--saved-runtime", type=Path, help="Reviewed runtime observation summary JSON."
    )
    parser.add_argument(
        "--saved-retrieval", type=Path, help="Sanitized retrieval aggregate JSON."
    )
    parser.add_argument("--saved-judge", type=Path, help="Sanitized judge status JSON.")
    parser.add_argument(
        "--saved-report",
        type=Path,
        help="Optional sanitized detailed findings in Markdown.",
    )

    parser.add_argument(
        "--derive-saved-runtime",
        action="store_true",
        help="Aggregate already-scored live records without rescoring.",
    )
    parser.add_argument(
        "--scored-run-dir",
        action="append",
        type=Path,
        default=[],
        help="Completed live run directory; repeat for each saved observation.",
    )
    parser.add_argument(
        "--derived-output",
        type=Path,
        help="Destination for sanitized derived runtime metrics JSON.",
    )

    parser.add_argument(
        "--system-august-scorecard", type=Path, help="Preserved August scorecard JSON."
    )
    parser.add_argument(
        "--system-august-provenance",
        type=Path,
        help="Preserved August provenance JSON.",
    )
    parser.add_argument(
        "--system-runtime", type=Path, help="Sanitized October runtime summary JSON."
    )
    parser.add_argument(
        "--system-metrics",
        type=Path,
        help="Derived metrics from already-scored October runs.",
    )
    parser.add_argument(
        "--system-judge", type=Path, help="Sanitized October judge status JSON."
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    system_inputs = (
        args.system_august_scorecard,
        args.system_august_provenance,
        args.system_runtime,
        args.system_metrics,
        args.system_judge,
    )
    if any(system_inputs):
        if (
            not args.publish
            or not all(system_inputs)
            or args.derive_saved_runtime
            or args.fingerprint
            or args.saved_runtime
            or args.saved_retrieval
            or args.saved_judge
            or args.saved_report
        ):
            print(
                "System publication requires --publish and all five system inputs.",
                file=sys.stderr,
            )
            return 2
        from backend.evaluation.system_report import publish_system_evaluation

        try:
            result = publish_system_evaluation(
                *system_inputs, reports_root=args.reports_root
            )
        except (OSError, ValueError, KeyError) as exc:
            print(f"System publication failed: {exc}", file=sys.stderr)
            return 2
        print(f"Published {result['snapshot']} to {result['history_directory']}.")
        return 0
    if args.derive_saved_runtime:
        if not args.scored_run_dir or args.derived_output is None or args.publish:
            print(
                "Derivation requires --scored-run-dir and --derived-output without --publish.",
                file=sys.stderr,
            )
            return 2
        from backend.evaluation.system_report import derive_saved_runs

        try:
            derived = derive_saved_runs(args.scored_run_dir)
            args.derived_output.parent.mkdir(parents=True, exist_ok=True)
            args.derived_output.write_text(
                json.dumps(derived, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except (OSError, ValueError) as exc:
            print(f"Saved runtime derivation failed: {exc}", file=sys.stderr)
            return 2
        print(f"Derived saved runtime metrics to {args.derived_output}.")
        return 0
    saved = (args.saved_runtime, args.saved_retrieval, args.saved_judge)
    if any(saved) or args.saved_report:
        if (
            not args.publish
            or not all(saved)
            or args.fingerprint
            or args.audit_completeness
            or args.verify_run_fingerprints
        ):
            print(
                "Saved publication requires --publish and all three saved JSON inputs.",
                file=sys.stderr,
            )
            return 2
        try:
            result = publish_saved_evaluation(
                *saved,
                reports_root=args.reports_root,
                source_report_path=args.saved_report,
            )
        except (ValueError, OSError) as exc:
            print(f"Saved publication failed: {exc}", file=sys.stderr)
            return 2
        print(f"Published {result['snapshot']} to {result['history_directory']}.")
        return 0
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
