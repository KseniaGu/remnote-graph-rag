"""Output helpers for the optimized RemNote parser IR."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.utils.common_funcs import write_json, write_jsonl

if TYPE_CHECKING:
    from backend.data_processing.parser_optimized import OptimizedParseResult


SUMMARY_FILENAME = "summary.json"
SOURCE_DOCUMENTS_FILENAME = "source_documents.jsonl"
BLOCKS_FILENAME = "blocks.jsonl"
EXTERNAL_RESOURCES_FILENAME = "external_resources.jsonl"
PARSED_ARTIFACTS_FILENAME = "parsed_artifacts.jsonl"
ARTIFACT_GATE_DECISIONS_FILENAME = "artifact_gate_decisions.jsonl"
RETRIEVAL_CHUNKS_FILENAME = "retrieval_chunks.jsonl"
COMPARISON_FILENAME = "comparison.md"


def result_to_jsonable(result: OptimizedParseResult) -> dict[str, Any]:
    """Converts optimized parser IR dataclasses to plain JSON-compatible records."""

    return {
        "source_documents": [asdict(item) for item in result.source_documents],
        "blocks": [asdict(item) for item in result.blocks],
        "external_resources": [asdict(item) for item in result.external_resources],
        "parsed_artifacts": [asdict(item) for item in result.parsed_artifacts],
        "artifact_gate_decisions": [
            asdict(item) for item in result.artifact_gate_decisions
        ],
        "retrieval_chunks": [asdict(item) for item in result.retrieval_chunks],
        "summary": result.summary,
    }


def write_optimized_parser_ir(output_root: Path, result: OptimizedParseResult) -> Path:
    """Writes all optimized parser IR sidecars and return the output directory."""

    jsonable = result_to_jsonable(result)
    write_json(output_root / SUMMARY_FILENAME, result.summary)
    write_jsonl(output_root / SOURCE_DOCUMENTS_FILENAME, jsonable["source_documents"])
    write_jsonl(output_root / BLOCKS_FILENAME, jsonable["blocks"])
    write_jsonl(
        output_root / EXTERNAL_RESOURCES_FILENAME, jsonable["external_resources"]
    )
    write_jsonl(output_root / PARSED_ARTIFACTS_FILENAME, jsonable["parsed_artifacts"])
    write_jsonl(
        output_root / ARTIFACT_GATE_DECISIONS_FILENAME,
        jsonable["artifact_gate_decisions"],
    )
    write_jsonl(output_root / RETRIEVAL_CHUNKS_FILENAME, jsonable["retrieval_chunks"])
    write_comparison_markdown(output_root / COMPARISON_FILENAME, result.summary)
    return output_root


def write_comparison_markdown(path: Path, summary: dict[str, Any]) -> None:
    """Writes the human-readable optimized parser comparison report."""

    criteria = summary["success_criteria"]
    baseline = summary["baseline_comparison"]
    lines = [
        "# Optimized Shadow Ingestion Comparison",
        "",
        "## Counts",
        "",
        "| Metric | Optimized | Baseline |",
        "|---|---:|---:|",
        (
            f"| Raw URL occurrences | {summary['raw_url_occurrences']} | "
            f"{baseline.get('baseline_raw_url_total_in_selected_files')} |"
        ),
        (
            f"| Parser-visible URL resources | {summary['parser_visible_url_resources']} | "
            f"{baseline.get('baseline_parser_visible_url_candidate_nodes')} |"
        ),
        (
            f"| Multi-URL gap | {summary['raw_url_occurrences'] - summary['parser_visible_url_resources']} | "
            f"{baseline.get('baseline_multi_url_line_gap_count')} |"
        ),
        (
            f"| Tiny retrieval chunks / nodes | {summary['standalone_tiny_chunk_count']} | "
            f"{baseline.get('baseline_tiny_node_count_len_1_to_3')} |"
        ),
        (
            f"| Duplicate retrieval text keys | {summary['duplicate_retrieval_chunk_text_keys']} | "
            f"{baseline.get('baseline_duplicate_source_text_keys')} |"
        ),
        f"| Header-only retrieval chunks | {summary['header_only_chunk_count']} | n/a |",
        f"| Orphan list-parent chunks | {summary['orphan_list_parent_chunk_count']} | n/a |",
        f"| Split list-item subtrees | {summary['split_list_item_subtree_count']} | n/a |",
        f"| Resource-only retrieval chunks | {summary['resource_only_chunk_count']} | n/a |",
        f"| Mixed-source retrieval chunks | {summary['mixed_source_retrieval_chunk_count']} | n/a |",
        (
            f"| Image binaries selected despite md sibling | "
            f"{summary['image_binary_selected_despite_md_sibling_count']} | n/a |"
        ),
        f"| Code-fence marker lines | {summary['code_fence_marker_line_count']} | n/a |",
        f"| Dataset artifacts metadata-only | {summary['dataset_artifact_metadata_only_count']} | n/a |",
        f"| URL mismatch artifacts quarantined | {summary['url_mismatch_quarantine_count']} | n/a |",
        f"| Duplicate artifacts metadata-only | {summary['duplicate_artifact_metadata_only_count']} | n/a |",
        f"| Low-quality OCR artifacts quarantined | {summary['low_quality_ocr_quarantine_count']} | n/a |",
        f"| External artifact chunks | {summary['external_artifact_chunk_count']} | n/a |",
        (
            f"| External artifact chunks with RemNote context | "
            f"{summary['external_artifact_chunks_with_context_count']} | n/a |"
        ),
        (
            f"| External artifact chunks without RemNote context | "
            f"{summary['external_artifact_chunks_without_context_count']} | n/a |"
        ),
        (
            f"| External artifact embedding support-label chunks | "
            f"{summary['external_artifact_embedding_support_label_count']} | n/a |"
        ),
        "",
        "## Success Criteria",
        "",
        "| Criterion | Passed |",
        "|---|---:|",
    ]
    for key, passed in criteria.items():
        lines.append(f"| `{key}` | {passed} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a shadow pipeline output only; it does not write production storage.",
            "- `not_resolved` external resources are explicit resource records, not silent parse failures.",
            "- Retrieval chunks are separate from raw RemNote blocks and retain source block provenance.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
