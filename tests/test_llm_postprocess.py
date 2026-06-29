import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.data_processing.concept_registry import (
    ConceptAdjudicationResponse,
    ConceptMention,
    ConceptRegistryEntry,
    ConceptResolution,
    apply_concept_adjudications,
    build_concept_resolution,
    build_concept_resolution_from_mentions,
    mention_id_for,
    validate_concept_adjudication_response,
)
from backend.data_processing.llm_postprocess import (
    CONCEPT_ADJUDICATIONS_FILENAME,
    CONCEPT_MERGE_REVIEW_FILENAME,
    CONCEPT_REGISTRY_FILENAME,
    DECISIONS_FILENAME,
    DEFAULT_SMOKE_LIMIT,
    DEFAULT_PROMPT_VERSION,
    FAILURES_FILENAME,
    GRAPH_PREVIEW_FILENAME,
    RELATION_REGISTRY_FILENAME,
    ChunkPostprocessBatch,
    ChunkPostprocessInput,
    LLMResponseCache,
    build_graph_projection,
    build_relation_registry,
    cache_key_for_batch,
    detect_preflags,
    fake_llm_response_for_batch,
    parse_llm_response,
    response_schema_hint,
    sanitize_markup_for_embedding,
    validate_and_enrich_response,
    write_jsonl,
    write_sidecar_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_llm_postprocess import load_excluded_chunk_ids, resolve_run_limit, select_run_inputs
from run_optimized_postprocess_pipeline import (
    load_prompt_for_pass,
    run_postprocess_pass,
    parse_args as parse_optimized_pipeline_args,
    resolved_prompt_name_for_pass,
    validate_args as validate_optimized_pipeline_args,
)
from run_llm_postprocess import (
    effective_num_predict,
    generation_settings_for_pass,
    is_empty_llm_response,
    is_usage_limit_error,
)


def make_input(text: str, *, chunk_id: str = "chunk_1") -> ChunkPostprocessInput:
    return ChunkPostprocessInput(
        chunk_id=chunk_id,
        chunk_type="remnote_section",
        chunk_role="paragraph_group",
        source="AI Research/Test.md",
        path=["AI Research", "Test"],
        heading_path=["AI Research", "Test"],
        line_start=1,
        line_end=3,
        source_block_ids=["block_1"],
        external_resource_ids=[],
        text=text,
        embedding_text=text,
        display_text=text,
        context_text="Test context",
    )


def valid_backprop_response(chunk_id: str = "chunk_1", predicate: str = "PRODUCES") -> str:
    return json.dumps(
        {
            "decisions": [
                {
                    "chunk_id": chunk_id,
                    "action": "keep",
                    "issue_types": [],
                    "educational_usefulness": 0.95,
                    "confidence": 0.92,
                    "warnings": [],
                    "cleaned_embedding_text": None,
                    "cleaned_display_text": None,
                    "chunk_summary": "Backpropagation produces gradients for each layer during training.",
                    "concepts": [
                        {
                            "local_id": "c1",
                            "canonical_name": "Backpropagation",
                            "display_name": "Backpropagation",
                            "type": "METHOD",
                            "aliases": [],
                            "salience": 0.98,
                            "description": "A method for computing training gradients.",
                            "evidence_spans": ["Backpropagation"],
                        },
                        {
                            "local_id": "c2",
                            "canonical_name": "Gradient",
                            "display_name": "Gradient",
                            "type": "CONCEPT",
                            "aliases": ["gradients"],
                            "salience": 0.9,
                            "description": None,
                            "evidence_spans": ["gradients"],
                        },
                    ],
                    "relations": [
                        {
                            "source_concept_id": "c1",
                            "target_concept_id": "c2",
                            "raw_predicate": predicate,
                            "canonical_predicate": predicate,
                            "predicate_status": "proposed" if predicate != "USES" else "existing",
                            "predicate_family": "computation",
                            "predicate_definition": "The source creates or yields the target.",
                            "relation_phrase": "produces gradients for each layer",
                            "generality_score": 0.82,
                            "retrieval_usefulness": 0.88,
                            "visualization_usefulness": 0.76,
                            "confidence": 0.9,
                            "evidence_chunk_ids": [chunk_id],
                            "evidence_spans": ["Backpropagation produces gradients"],
                        }
                    ],
                    "reason": "The relation is explicitly stated.",
                }
            ]
        }
    )


def make_concept_mention(
    name: str,
    *,
    mention_id: str,
    concept_type: str = "CONCEPT",
    aliases: list[str] | None = None,
    description: str | None = None,
) -> ConceptMention:
    return ConceptMention(
        mention_id=mention_id,
        decision_id=f"decision_{mention_id}",
        chunk_id=f"chunk_{mention_id}",
        local_id="c1",
        canonical_name=name,
        display_name=name,
        type=concept_type,
        aliases=aliases or [],
        salience=0.8,
        description=description,
        evidence_spans=[name],
    )


class LLMPostprocessTests(unittest.TestCase):
    def test_optimized_pipeline_prepares_external_artifacts_by_default(self) -> None:
        argv = [
            "run_optimized_postprocess_pipeline.py",
            "--raw-data-dir",
            "raw",
            "--output-root",
            "out",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_optimized_pipeline_args()

        self.assertTrue(args.prepare_external_artifacts)

        with patch.object(sys, "argv", [*argv, "--skip-external-artifacts"]):
            args = parse_optimized_pipeline_args()

        self.assertFalse(args.prepare_external_artifacts)

    def test_optimized_pipeline_limit_defaults_follow_full_run_flag(self) -> None:
        argv = [
            "run_optimized_postprocess_pipeline.py",
            "--raw-data-dir",
            "raw",
            "--output-root",
            "out",
        ]
        with patch.object(sys, "argv", argv):
            smoke_args = parse_optimized_pipeline_args()
        validate_optimized_pipeline_args(smoke_args)

        self.assertEqual(DEFAULT_SMOKE_LIMIT, smoke_args.limit)
        self.assertEqual(DEFAULT_SMOKE_LIMIT, smoke_args.concept_resolution_limit)

        with patch.object(sys, "argv", [*argv, "--allow-full-run"]):
            full_args = parse_optimized_pipeline_args()
        validate_optimized_pipeline_args(full_args)

        self.assertEqual(0, full_args.limit)
        self.assertEqual(0, full_args.concept_resolution_limit)

    def test_optimized_pipeline_pass_specific_num_predict_args(self) -> None:
        argv = [
            "run_optimized_postprocess_pipeline.py",
            "--raw-data-dir",
            "raw",
            "--output-root",
            "out",
            "--quality-num-predict",
            "1536",
            "--graph-num-predict",
            "3072",
            "--concept-resolution-num-predict",
            "1024",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_optimized_pipeline_args()
        validate_optimized_pipeline_args(args)

        self.assertEqual(1536, effective_num_predict(args, "quality"))
        self.assertEqual(3072, effective_num_predict(args, "graph"))
        self.assertEqual(1024, effective_num_predict(args, "concept_resolution"))
        self.assertEqual(3072, generation_settings_for_pass(args, "graph")["num_predict"])

    def test_generation_settings_are_included_in_batch_cache_key(self) -> None:
        chunk = make_input("Backpropagation produces gradients for each layer during training.")
        batch = ChunkPostprocessBatch(batch_id="batch_1", chunks=[chunk])

        key_2048 = cache_key_for_batch(
            batch,
            model_name="model",
            prompt_version="v6:graph",
            prompt_content_hash="prompt",
            generation_settings={"num_predict": 2048},
        )
        key_3072 = cache_key_for_batch(
            batch,
            model_name="model",
            prompt_version="v6:graph",
            prompt_content_hash="prompt",
            generation_settings={"num_predict": 3072},
        )

        self.assertNotEqual(key_2048, key_3072)

    def test_usage_limit_and_empty_response_helpers(self) -> None:
        self.assertTrue(is_usage_limit_error("ResponseError: weekly usage limit (status code: 429)"))
        self.assertTrue(is_usage_limit_error("Rate limit exceeded"))
        self.assertFalse(is_usage_limit_error("JSONDecodeError: Expecting value"))
        self.assertTrue(is_empty_llm_response(""))
        self.assertTrue(is_empty_llm_response("  \n"))
        self.assertFalse(is_empty_llm_response('{"decisions": []}'))

    def _optimized_pass_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            max_batch_chunks=1,
            max_batch_chars=9000,
            prompt_dir=ROOT / "backend" / "llm" / "prompts",
            prompt_version=DEFAULT_PROMPT_VERSION,
            fake_llm=False,
            force_refresh_cache=True,
            model_name="test-model",
            temperature=0.0,
            top_k=10,
            top_p=0.1,
            num_ctx=8192,
            num_predict=4096,
            quality_num_predict=1536,
            graph_num_predict=3072,
            concept_resolution_num_predict=1024,
            base_url="https://ollama.com",
        )

    def test_optimized_pipeline_uses_split_v7_and_v9_prompts_with_v6_fallback(self) -> None:
        args = self._optimized_pass_args()

        self.assertEqual("remnote_postprocess", resolved_prompt_name_for_pass(args, "quality"))
        self.assertEqual("remnote_postprocess", resolved_prompt_name_for_pass(args, "graph"))

        args.prompt_version = "v7"
        self.assertEqual("remnote_postprocess_quality", resolved_prompt_name_for_pass(args, "quality"))
        self.assertEqual("remnote_postprocess_graph", resolved_prompt_name_for_pass(args, "graph"))

        quality_prompt, quality_name = load_prompt_for_pass(args, "quality")
        graph_prompt, graph_name = load_prompt_for_pass(args, "graph")

        self.assertEqual("remnote_postprocess_quality", quality_name)
        self.assertEqual("remnote_postprocess_graph", graph_name)
        self.assertIn("Do not extract graph concepts or relations", quality_prompt[1])
        self.assertIn("Existing predicates are preferred", graph_prompt[1])

        args.prompt_version = "v9"
        self.assertEqual("remnote_postprocess_quality", resolved_prompt_name_for_pass(args, "quality"))
        self.assertEqual("remnote_postprocess_graph", resolved_prompt_name_for_pass(args, "graph"))

        quality_prompt, quality_name = load_prompt_for_pass(args, "quality")
        graph_prompt, graph_name = load_prompt_for_pass(args, "graph")

        self.assertEqual("remnote_postprocess_quality", quality_name)
        self.assertEqual("remnote_postprocess_graph", graph_name)
        self.assertIn("Do not extract graph concepts or relations", quality_prompt[1])
        self.assertIn("Existing predicates are preferred", graph_prompt[1])

    def test_optimized_pass_skips_repair_for_empty_response(self) -> None:
        chunk = make_input("Backpropagation produces gradients for each layer during training.")
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("run_optimized_postprocess_pipeline.build_ollama_llm", return_value=object()),
                patch("run_optimized_postprocess_pipeline.invoke_llm_batch", return_value=("", {})),
                patch("run_optimized_postprocess_pipeline.invoke_llm_repair") as repair_mock,
            ):
                decisions, failures, hits, misses, aborted = run_postprocess_pass(
                    self._optimized_pass_args(),
                    pass_name="graph",
                    inputs=[chunk],
                    output_dir=Path(tmp),
                )

        self.assertEqual([], decisions)
        self.assertEqual(1, len(failures))
        self.assertEqual("empty_llm_response", failures[0].error_type)
        self.assertEqual(0, hits)
        self.assertEqual(1, misses)
        self.assertFalse(aborted)
        repair_mock.assert_not_called()

    def test_optimized_pass_aborts_on_usage_limit_error(self) -> None:
        chunk = make_input("Backpropagation produces gradients for each layer during training.")
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("run_optimized_postprocess_pipeline.build_ollama_llm", return_value=object()),
                patch(
                    "run_optimized_postprocess_pipeline.invoke_llm_batch",
                    side_effect=RuntimeError("weekly usage limit (status code: 429)"),
                ),
            ):
                decisions, failures, _hits, misses, aborted = run_postprocess_pass(
                    self._optimized_pass_args(),
                    pass_name="graph",
                    inputs=[chunk],
                    output_dir=Path(tmp),
                )

        self.assertEqual([], decisions)
        self.assertEqual(1, len(failures))
        self.assertEqual("llm_usage_limit_error", failures[0].error_type)
        self.assertEqual(1, misses)
        self.assertTrue(aborted)

    def test_preflag_detection_finds_html_caption_boilerplate_and_source_cards(self) -> None:
        flags = detect_preflags(
            {
                "text": '<img src="chart.png"> Figure 1: model accuracy.',
                "embedding_text": "Title: Video\nURL: https://youtube.com/watch?v=123\nTerms of Service",
                "display_text": "",
                "context_text": "",
                "chunk_quality_flags": ["small_but_kept"],
            }
        )

        self.assertTrue(flags.html_fragment_detected)
        self.assertTrue(flags.caption_only_candidate)
        self.assertTrue(flags.source_card_candidate)
        self.assertTrue(flags.boilerplate_candidate)
        self.assertTrue(flags.visual_content_missing_candidate)
        self.assertTrue(flags.small_but_kept)

    def test_sanitize_markup_drops_generic_image_and_wrapper_tags(self) -> None:
        result = sanitize_markup_for_embedding(
            '<div><img src="x.jpg" alt="Image"></div> Figure 4. Extensions connect Agents...'
        )

        self.assertEqual("Figure 4. Extensions connect Agents...", result.text)
        self.assertEqual(1, result.removed_image_count)
        self.assertGreaterEqual(result.removed_tag_count, 3)
        self.assertEqual([], result.preserved_alt_texts)

    def test_sanitize_markup_preserves_meaningful_alt_text(self) -> None:
        result = sanitize_markup_for_embedding('Before <img src="x.jpg" alt="Architecture diagram"> after')

        self.assertEqual("Before Architecture diagram after", result.text)
        self.assertEqual(["Architecture diagram"], result.preserved_alt_texts)

    def test_sanitize_markup_makes_tables_readable(self) -> None:
        result = sanitize_markup_for_embedding(
            "<table><tr><td>Extensions</td><td>Function Calling</td></tr></table>"
        )

        self.assertNotIn("<td", result.text)
        self.assertIn("Extensions", result.text)
        self.assertIn("Function Calling", result.text)
        self.assertIn("|", result.text)

    def test_sanitize_markup_handles_malformed_tags_and_preserves_math(self) -> None:
        malformed = sanitize_markup_for_embedding('<div style="text-align: center;" Figure 1. Caption')
        math_text = sanitize_markup_for_embedding("A threshold x < 0.5 > y should remain readable.")

        self.assertEqual("Figure 1. Caption", malformed.text)
        self.assertEqual("A threshold x < 0.5 > y should remain readable.", math_text.text)

    def test_valid_response_produces_decision_projection_and_registry(self) -> None:
        chunk = make_input("Backpropagation produces gradients for each layer during training.")
        batch = ChunkPostprocessBatch(batch_id="batch_1", chunks=[chunk])
        parsed = parse_llm_response(valid_backprop_response())

        decisions, failures = validate_and_enrich_response(
            parsed,
            batch,
            model_name="nemotron-3-super:cloud",
            prompt_version="v1",
            raw_response=valid_backprop_response(),
        )

        self.assertEqual([], failures)
        self.assertEqual(1, len(decisions))
        projection = build_graph_projection(decisions)
        registry = build_relation_registry(decisions)
        self.assertEqual(2, len(projection.nodes))
        self.assertEqual(1, len(projection.edges))
        self.assertEqual(["produces gradients for each layer"], projection.edges[0]["relation_phrases"])
        self.assertEqual(0.82, projection.edges[0]["max_generality_score"])
        self.assertEqual(0.88, projection.edges[0]["max_retrieval_usefulness"])
        self.assertEqual(0.76, projection.edges[0]["max_visualization_usefulness"])
        self.assertEqual("PRODUCES", registry[0]["canonical_predicate"])
        self.assertEqual(["produces gradients for each layer"], registry[0]["relation_phrases"])

    def test_old_relation_payload_defaults_enrichment_fields(self) -> None:
        payload = json.loads(valid_backprop_response())
        relation = payload["decisions"][0]["relations"][0]
        for key in ("relation_phrase", "generality_score", "retrieval_usefulness", "visualization_usefulness"):
            relation.pop(key, None)

        parsed = parse_llm_response(json.dumps(payload))

        parsed_relation = parsed.decisions[0].relations[0]
        self.assertIsNone(parsed_relation.relation_phrase)
        self.assertEqual(0.5, parsed_relation.generality_score)
        self.assertEqual(0.5, parsed_relation.retrieval_usefulness)
        self.assertEqual(0.5, parsed_relation.visualization_usefulness)

    def test_response_schema_hint_includes_relation_enrichment_fields(self) -> None:
        relation_schema = response_schema_hint()["decisions"][0]["relations"][0]

        self.assertIn("relation_phrase", relation_schema)
        self.assertIn("generality_score", relation_schema)
        self.assertIn("retrieval_usefulness", relation_schema)
        self.assertIn("visualization_usefulness", relation_schema)
        self.assertIn("listed existing labels", relation_schema["predicate_status"])

    def test_quality_response_schema_hint_keeps_graph_fields_empty(self) -> None:
        decision_schema = response_schema_hint(pass_name="quality")["decisions"][0]

        self.assertEqual([], decision_schema["concepts"])
        self.assertEqual([], decision_schema["relations"])
        self.assertIn("cleaned_embedding_text", decision_schema)
        self.assertIn("educational_usefulness", decision_schema)

    def test_concept_resolution_auto_merges_singular_plural_names(self) -> None:
        resolution = build_concept_resolution_from_mentions(
            [
                make_concept_mention("Data Store", mention_id="m1", concept_type="COMPONENT"),
                make_concept_mention("Data Stores", mention_id="m2", concept_type="COMPONENT"),
            ]
        )

        self.assertEqual(1, len(resolution.registry_entries))
        self.assertEqual("auto_merged", resolution.registry_entries[0].merge_status)
        self.assertEqual("Data Store", resolution.registry_entries[0].canonical_name)
        self.assertEqual([], resolution.review_clusters)

    def test_concept_resolution_marks_derivational_variant_uncertain(self) -> None:
        resolution = build_concept_resolution_from_mentions(
            [
                make_concept_mention("Function Call", mention_id="m1", concept_type="METHOD"),
                make_concept_mention("Function Calling", mention_id="m2", concept_type="METHOD"),
            ]
        )

        self.assertEqual(2, len(resolution.registry_entries))
        self.assertEqual(1, len(resolution.review_clusters))
        self.assertIn("derivational_variant", resolution.review_clusters[0].risk_flags)

    def test_concept_resolution_marks_modifier_containment_uncertain(self) -> None:
        resolution = build_concept_resolution_from_mentions(
            [
                make_concept_mention("Convolution", mention_id="m1", concept_type="METHOD"),
                make_concept_mention("Convolution Filter", mention_id="m2", concept_type="METHOD"),
            ]
        )

        self.assertEqual(2, len(resolution.registry_entries))
        self.assertEqual(1, len(resolution.review_clusters))
        self.assertIn("modifier_scope_difference", resolution.review_clusters[0].risk_flags)

    def test_concept_resolution_does_not_merge_related_specific_model(self) -> None:
        resolution = build_concept_resolution_from_mentions(
            [
                make_concept_mention("Word Embeddings", mention_id="m1", concept_type="CONCEPT"),
                make_concept_mention("Word2Vec", mention_id="m2", concept_type="MODEL"),
            ]
        )

        self.assertEqual(2, len(resolution.registry_entries))
        self.assertEqual([], resolution.review_clusters)

    def test_concept_adjudication_can_merge_uncertain_cluster(self) -> None:
        resolution = build_concept_resolution_from_mentions(
            [
                make_concept_mention("Function Call", mention_id="m1", concept_type="METHOD"),
                make_concept_mention("Function Calling", mention_id="m2", concept_type="METHOD"),
            ]
        )
        cluster = resolution.review_clusters[0]
        adjudication = ConceptAdjudicationResponse.model_validate(
            {
                "cluster_id": cluster.cluster_id,
                "action": "merge_all",
                "groups": [
                    {
                        "mention_ids": cluster.mention_ids,
                        "canonical_name": "Function Call",
                        "display_name": "Function Call",
                        "type": "METHOD",
                        "aliases": ["Function Calling"],
                        "confidence": 0.85,
                    }
                ],
                "rationale": "Same operation phrasing.",
                "warnings": [],
            }
        )

        self.assertEqual([], validate_concept_adjudication_response(adjudication, cluster))
        applied = apply_concept_adjudications(resolution, [adjudication])
        self.assertEqual(1, len(applied.registry_entries))
        self.assertEqual([], applied.review_clusters)
        self.assertEqual(1, len(applied.adjudications))

    def test_concept_adjudication_rejects_duplicate_or_missing_mentions(self) -> None:
        resolution = build_concept_resolution_from_mentions(
            [
                make_concept_mention("Function Call", mention_id="m1", concept_type="METHOD"),
                make_concept_mention("Function Calling", mention_id="m2", concept_type="METHOD"),
            ]
        )
        cluster = resolution.review_clusters[0]
        adjudication = ConceptAdjudicationResponse.model_validate(
            {
                "cluster_id": cluster.cluster_id,
                "action": "merge_groups",
                "groups": [
                    {
                        "mention_ids": [cluster.mention_ids[0], cluster.mention_ids[0]],
                        "canonical_name": "Function Call",
                        "display_name": "Function Call",
                        "type": "METHOD",
                        "aliases": [],
                        "confidence": 0.7,
                    }
                ],
                "rationale": "Bad duplicate output.",
                "warnings": [],
            }
        )

        errors = validate_concept_adjudication_response(adjudication, cluster)
        self.assertTrue(any("duplicate" in error for error in errors))
        self.assertTrue(any("missing" in error for error in errors))

    def test_singleton_llm_split_preserves_deterministic_canonical_name_and_audit(self) -> None:
        resolution = build_concept_resolution_from_mentions(
            [
                make_concept_mention("Convolution", mention_id="m1", concept_type="METHOD"),
                make_concept_mention(
                    "Convolution Filter",
                    mention_id="m2",
                    concept_type="METHOD",
                    aliases=["Convolution Filter for Text Classification"],
                ),
            ]
        )
        cluster = resolution.review_clusters[0]
        adjudication = ConceptAdjudicationResponse.model_validate(
            {
                "cluster_id": cluster.cluster_id,
                "action": "split_all",
                "groups": [
                    {
                        "mention_ids": ["m1"],
                        "canonical_name": "Convolution",
                        "display_name": "Convolution",
                        "type": "METHOD",
                        "aliases": [],
                        "confidence": 0.9,
                    },
                    {
                        "mention_ids": ["m2"],
                        "canonical_name": "Convolution Filter for Text Classification",
                        "display_name": "Convolution Filter",
                        "type": "METHOD",
                        "aliases": ["Convolution Filter"],
                        "confidence": 0.9,
                    },
                ],
                "rationale": "The operation and filter are distinct concepts.",
                "warnings": [],
            }
        )

        applied = apply_concept_adjudications(resolution, [adjudication])
        filter_entry = next(entry for entry in applied.registry_entries if "m2" in entry.mention_ids)

        self.assertEqual("Convolution Filter", filter_entry.canonical_name)
        self.assertIn("Convolution Filter for Text Classification", filter_entry.aliases)
        self.assertIn("Convolution Filter for Text Classification", filter_entry.source_names)
        self.assertEqual(["METHOD"], filter_entry.source_types)
        self.assertEqual("llm_adjudicated", filter_entry.resolution_source)
        self.assertEqual([cluster.cluster_id], filter_entry.adjudication_cluster_ids)
        self.assertEqual(["split_all"], filter_entry.adjudication_actions)
        self.assertEqual(["The operation and filter are distinct concepts."], filter_entry.adjudication_rationales)

    def test_invalid_evidence_span_is_rejected(self) -> None:
        chunk = make_input("Backpropagation produces gradients for each layer during training.")
        batch = ChunkPostprocessBatch(batch_id="batch_1", chunks=[chunk])
        payload = json.loads(valid_backprop_response())
        payload["decisions"][0]["concepts"][0]["evidence_spans"] = ["not present in the chunk"]

        decisions, failures = validate_and_enrich_response(
            parse_llm_response(json.dumps(payload)),
            batch,
            model_name="nemotron-3-super:cloud",
            prompt_version="v1",
            raw_response=json.dumps(payload),
        )

        self.assertEqual([], decisions)
        self.assertEqual(1, len(failures))
        self.assertIn("source-grounded", failures[0].message)

    def test_cleaned_text_with_debug_label_is_rejected(self) -> None:
        chunk = make_input("Backpropagation produces gradients for each layer during training.")
        batch = ChunkPostprocessBatch(batch_id="batch_1", chunks=[chunk])
        payload = json.loads(valid_backprop_response())
        payload["decisions"][0]["cleaned_embedding_text"] = "artifact_path: /tmp/raw/file.md"

        decisions, failures = validate_and_enrich_response(
            parse_llm_response(json.dumps(payload)),
            batch,
            model_name="nemotron-3-super:cloud",
            prompt_version="v1",
            raw_response=json.dumps(payload),
        )

        self.assertEqual([], decisions)
        self.assertEqual(1, len(failures))
        self.assertIn("cleaned_embedding_text", failures[0].message)

    def test_related_to_with_specific_relation_cue_is_rejected(self) -> None:
        chunk = make_input("Backpropagation produces gradients for each layer during training.")
        batch = ChunkPostprocessBatch(batch_id="batch_1", chunks=[chunk])
        payload = json.loads(valid_backprop_response(predicate="RELATED_TO"))

        decisions, failures = validate_and_enrich_response(
            parse_llm_response(json.dumps(payload)),
            batch,
            model_name="nemotron-3-super:cloud",
            prompt_version="v1",
            raw_response=json.dumps(payload),
        )

        self.assertEqual([], decisions)
        self.assertEqual(1, len(failures))
        self.assertIn("too generic", failures[0].message)

    def test_duplicate_decision_for_same_chunk_is_rejected(self) -> None:
        chunk = make_input("Backpropagation produces gradients for each layer during training.")
        batch = ChunkPostprocessBatch(batch_id="batch_1", chunks=[chunk])
        payload = json.loads(valid_backprop_response())
        payload["decisions"].append(payload["decisions"][0])

        decisions, failures = validate_and_enrich_response(
            parse_llm_response(json.dumps(payload)),
            batch,
            model_name="nemotron-3-super:cloud",
            prompt_version="v1",
            raw_response=json.dumps(payload),
        )

        self.assertEqual(1, len(decisions))
        self.assertEqual(1, len(failures))
        self.assertEqual("duplicate_decision", failures[0].error_type)

    def test_unknown_existing_relation_label_is_downgraded_to_proposed(self) -> None:
        chunk = make_input("Backpropagation produces gradients for each layer during training.")
        batch = ChunkPostprocessBatch(batch_id="batch_1", chunks=[chunk])
        payload = json.loads(valid_backprop_response(predicate="SHOWS"))
        payload["decisions"][0]["relations"][0]["predicate_status"] = "existing"
        payload["decisions"][0]["relations"][0]["predicate_definition"] = None

        decisions, failures = validate_and_enrich_response(
            parse_llm_response(json.dumps(payload)),
            batch,
            model_name="nemotron-3-super:cloud",
            prompt_version="v1",
            raw_response=json.dumps(payload),
        )

        self.assertEqual([], failures)
        self.assertEqual(1, len(decisions))
        relation = decisions[0].relations[0]
        self.assertEqual("SHOWS", relation.canonical_predicate)
        self.assertEqual("proposed", relation.predicate_status)
        self.assertEqual("Proposed grounded relation.", relation.predicate_definition)
        self.assertIn("unknown_existing_predicate_downgraded", decisions[0].warnings)

    def test_is_part_of_relation_alias_is_normalized_to_part_of(self) -> None:
        chunk = make_input("Backpropagation produces gradients for each layer during training.")
        batch = ChunkPostprocessBatch(batch_id="batch_1", chunks=[chunk])
        payload = json.loads(valid_backprop_response(predicate="IS_PART_OF"))
        payload["decisions"][0]["relations"][0]["predicate_status"] = "existing"
        payload["decisions"][0]["relations"][0]["predicate_family"] = "composition"
        payload["decisions"][0]["relations"][0]["predicate_definition"] = None

        decisions, failures = validate_and_enrich_response(
            parse_llm_response(json.dumps(payload)),
            batch,
            model_name="nemotron-3-super:cloud",
            prompt_version="v1",
            raw_response=json.dumps(payload),
        )

        self.assertEqual([], failures)
        self.assertEqual(1, len(decisions))
        relation = decisions[0].relations[0]
        self.assertEqual("IS_PART_OF", relation.raw_predicate)
        self.assertEqual("PART_OF", relation.canonical_predicate)
        self.assertEqual("existing", relation.predicate_status)
        self.assertIn("predicate_alias_normalized", decisions[0].warnings)

    def test_malformed_relation_predicate_is_rejected(self) -> None:
        payload = json.loads(valid_backprop_response())
        payload["decisions"][0]["relations"][0]["canonical_predicate"] = "!!!"

        with self.assertRaises(ValueError):
            parse_llm_response(json.dumps(payload))

    def test_graph_projection_skips_post_resolution_self_loop_edges(self) -> None:
        chunk = make_input("Backpropagation produces gradients for each layer during training.")
        batch = ChunkPostprocessBatch(batch_id="batch_1", chunks=[chunk])
        decisions, failures = validate_and_enrich_response(
            parse_llm_response(valid_backprop_response()),
            batch,
            model_name="nemotron-3-super:cloud",
            prompt_version="v1",
        )

        self.assertEqual([], failures)
        decision = decisions[0]
        concept_id = "concept_same"
        resolution = ConceptResolution(
            registry_entries=[
                ConceptRegistryEntry(
                    concept_id=concept_id,
                    canonical_name="Backpropagation Gradient Merge",
                    display_name="Backpropagation Gradient Merge",
                    type="CONCEPT",
                    aliases=[],
                    descriptions=[],
                    source_names=["Backpropagation", "Gradient"],
                    source_types=["METHOD", "CONCEPT"],
                    source_chunk_ids=[chunk.chunk_id],
                    evidence_spans=["Backpropagation", "gradients"],
                    mention_ids=[
                        mention_id_for(decision.decision_id, chunk.chunk_id, "c1"),
                        mention_id_for(decision.decision_id, chunk.chunk_id, "c2"),
                    ],
                    source_decision_ids=[decision.decision_id],
                    resolution_source="test",
                    resolution_sources=["test"],
                    merge_status="test",
                    merge_statuses=["test"],
                    max_salience=1.0,
                )
            ],
            mention_to_concept_id={
                mention_id_for(decision.decision_id, chunk.chunk_id, "c1"): concept_id,
                mention_id_for(decision.decision_id, chunk.chunk_id, "c2"): concept_id,
            },
        )

        projection = build_graph_projection(decisions, concept_resolution=resolution)

        self.assertEqual(1, len(projection.nodes))
        self.assertEqual([], projection.edges)

    def test_prompt_v6_documents_existing_and_proposed_predicate_contract(self) -> None:
        prompt_text = (
            ROOT
            / "backend"
            / "llm"
            / "prompts"
            / "learner_workflow"
            / "orchestrator"
            / "remnote_postprocess"
            / "v6.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn('Use predicate_status "existing" only for the exact existing predicate labels', prompt_text)
        self.assertIn("Use PART_OF, not IS_PART_OF", prompt_text)
        self.assertIn("do not automatically make those words existing predicate labels", prompt_text)

    def test_prompt_v7_quality_and_graph_prompts_have_separate_contracts(self) -> None:
        prompt_root = ROOT / "backend" / "llm" / "prompts" / "learner_workflow" / "orchestrator"
        quality_prompt = (prompt_root / "remnote_postprocess_quality" / "v7.yaml").read_text(encoding="utf-8")
        graph_prompt = (prompt_root / "remnote_postprocess_graph" / "v7.yaml").read_text(encoding="utf-8")

        self.assertIn("Do not extract graph concepts or relations in the quality pass", quality_prompt)
        self.assertIn('"concepts": []', quality_prompt)
        self.assertIn('"relations": []', quality_prompt)
        self.assertNotIn("Existing predicates are preferred", quality_prompt)
        self.assertIn("Existing predicates are preferred", graph_prompt)
        self.assertIn("predicate_status", graph_prompt)
        self.assertIn("relation_phrase", graph_prompt)
        self.assertIn("retrieval_usefulness", graph_prompt)
        self.assertIn("visualization_usefulness", graph_prompt)

    def test_prompt_v9_keeps_split_contract_and_simpler_graph_budget(self) -> None:
        prompt_root = ROOT / "backend" / "llm" / "prompts" / "learner_workflow" / "orchestrator"
        quality_prompt = (prompt_root / "remnote_postprocess_quality" / "v9.yaml").read_text(encoding="utf-8")
        graph_prompt = (prompt_root / "remnote_postprocess_graph" / "v9.yaml").read_text(encoding="utf-8")

        self.assertIn("Do not extract graph concepts or relations in the quality pass", quality_prompt)
        self.assertIn('"concepts": []', quality_prompt)
        self.assertIn('"relations": []', quality_prompt)
        self.assertIn("relation_phrase", graph_prompt)
        self.assertIn("predicate_family", graph_prompt)
        self.assertIn("generality_score", graph_prompt)
        self.assertIn("retrieval_usefulness", graph_prompt)
        self.assertIn("visualization_usefulness", graph_prompt)
        self.assertIn("Normal useful chunks: 3-5 concepts and 2-3 relations", graph_prompt)
        self.assertIn("Dense technical, list, or table-like chunks: 4-6 concepts and 2-4 relations", graph_prompt)
        self.assertIn("Do not create standalone nodes for local variables, indexed tensors", graph_prompt)
        self.assertNotIn("connect at least half", graph_prompt)

    def test_default_prompt_version_is_v6(self) -> None:
        self.assertEqual("v6", DEFAULT_PROMPT_VERSION)

    def test_parse_repairs_missing_decisions_array_closure(self) -> None:
        raw = valid_backprop_response()
        broken = raw[:-2] + "}"

        parsed = parse_llm_response(broken)

        self.assertEqual(1, len(parsed.decisions))
        self.assertEqual("chunk_1", parsed.decisions[0].chunk_id)

    def test_evidence_normalization_repairs_ocr_lookalike_span(self) -> None:
        text = (
            "Extensions bridge the gap between an APl and an agent. "
            "You know that you want to use the Google Flights APl to retrieve flight information."
        )
        chunk = make_input(text)
        batch = ChunkPostprocessBatch(batch_id="batch_1", chunks=[chunk])
        payload = {
            "decisions": [
                {
                    "chunk_id": "chunk_1",
                    "action": "keep",
                    "issue_types": [],
                    "educational_usefulness": 0.8,
                    "confidence": 0.9,
                    "warnings": [],
                    "cleaned_embedding_text": None,
                    "cleaned_display_text": None,
                    "chunk_summary": "Extensions connect agents to APIs.",
                    "concepts": [
                        {
                            "local_id": "c1",
                            "canonical_name": "Extension",
                            "display_name": "Extension",
                            "type": "METHOD",
                            "aliases": ["Extensions"],
                            "salience": 0.8,
                            "description": None,
                            "evidence_spans": ["Extensions"],
                        },
                        {
                            "local_id": "c2",
                            "canonical_name": "Google Flights API",
                            "display_name": "Google Flights API",
                            "type": "TOOL",
                            "aliases": ["Google Flights API"],
                            "salience": 0.8,
                            "description": None,
                            "evidence_spans": ["Google Flights API"],
                        },
                    ],
                    "relations": [
                        {
                            "source_concept_id": "c1",
                            "target_concept_id": "c2",
                            "raw_predicate": "USES",
                            "canonical_predicate": "USES",
                            "predicate_status": "existing",
                            "predicate_family": "method",
                            "predicate_definition": None,
                            "confidence": 0.8,
                            "evidence_chunk_ids": ["chunk_1"],
                            "evidence_spans": ["use the Google Flights API"],
                        }
                    ],
                    "reason": "Explicit API usage.",
                }
            ]
        }

        decisions, failures = validate_and_enrich_response(
            parse_llm_response(json.dumps(payload)),
            batch,
            model_name="test",
            prompt_version="v3",
        )

        self.assertEqual([], failures)
        self.assertEqual(["Google Flights APl"], decisions[0].concepts[1].evidence_spans)
        self.assertEqual(["use the Google Flights APl"], decisions[0].relations[0].evidence_spans)
        self.assertIn("postprocess_normalized_llm_output", decisions[0].warnings)

    def test_visual_reparse_issue_type_is_normalized(self) -> None:
        chunk = make_input('<img src="diagram.png" alt="Image" />', chunk_id="chunk_visual")
        batch = ChunkPostprocessBatch(batch_id="batch_1", chunks=[chunk])
        payload = {
            "decisions": [
                {
                    "chunk_id": "chunk_visual",
                    "action": "needs_visual_reparse",
                    "issue_types": [],
                    "educational_usefulness": 0.1,
                    "confidence": 0.8,
                    "warnings": [],
                    "cleaned_embedding_text": None,
                    "cleaned_display_text": None,
                    "chunk_summary": None,
                    "concepts": [],
                    "relations": [],
                    "reason": "Image only.",
                }
            ]
        }

        decisions, failures = validate_and_enrich_response(
            parse_llm_response(json.dumps(payload)),
            batch,
            model_name="test",
            prompt_version="v3",
        )

        self.assertEqual([], failures)
        self.assertIn("visual_content_missing", decisions[0].issue_types)

    def test_cache_hit_and_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LLMResponseCache(Path(tmp))
            self.assertIsNone(cache.get("abc"))
            cache.set("abc", '{"decisions": []}', metadata={"latency_seconds": 0.1})
            self.assertEqual('{"decisions": []}', cache.get("abc"))

    def test_sidecar_outputs_with_fake_llm_response(self) -> None:
        chunk = make_input("Backpropagation produces gradients for each layer during training.")
        batch = ChunkPostprocessBatch(batch_id="batch_1", chunks=[chunk])
        decisions, failures = validate_and_enrich_response(
            parse_llm_response(fake_llm_response_for_batch(batch)),
            batch,
            model_name="fake",
            prompt_version="v1",
        )

        with tempfile.TemporaryDirectory() as tmp:
            concept_resolution = build_concept_resolution(decisions)
            report = write_sidecar_outputs(
                Path(tmp),
                inputs=[chunk],
                decisions=decisions,
                failures=failures,
                concept_resolution=concept_resolution,
            )
            self.assertEqual(1, report["decision_count"])
            self.assertEqual(2, report["concept_registry_count"])
            self.assertEqual(1, report["relation_registry_count"])
            self.assertTrue((Path(tmp) / DECISIONS_FILENAME).exists())
            self.assertTrue((Path(tmp) / FAILURES_FILENAME).exists())
            self.assertTrue((Path(tmp) / RELATION_REGISTRY_FILENAME).exists())
            self.assertTrue((Path(tmp) / CONCEPT_REGISTRY_FILENAME).exists())
            self.assertTrue((Path(tmp) / CONCEPT_MERGE_REVIEW_FILENAME).exists())
            self.assertTrue((Path(tmp) / CONCEPT_ADJUDICATIONS_FILENAME).exists())
            self.assertTrue((Path(tmp) / GRAPH_PREVIEW_FILENAME).exists())

    def test_select_run_inputs_supports_offset(self) -> None:
        values = list(range(10))

        self.assertEqual([3, 4, 5], select_run_inputs(values, limit=3, offset=3, allow_full_run=False))
        self.assertEqual([7, 8, 9], select_run_inputs(values, limit=0, offset=7, allow_full_run=True))

    def test_select_run_inputs_resolves_omitted_limit_from_full_run_flag(self) -> None:
        values = list(range(DEFAULT_SMOKE_LIMIT + 5))

        self.assertEqual(values[:DEFAULT_SMOKE_LIMIT], select_run_inputs(values, limit=None, allow_full_run=False))
        self.assertEqual(values, select_run_inputs(values, limit=None, allow_full_run=True))
        self.assertEqual(DEFAULT_SMOKE_LIMIT, resolve_run_limit(None, allow_full_run=False))
        self.assertEqual(0, resolve_run_limit(None, allow_full_run=True))

    def test_load_excluded_chunk_ids_from_previous_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "previous"
            write_jsonl(
                output_dir / "llm_postprocess_inputs.jsonl",
                [
                    {"chunk_id": "chunk_a"},
                    {"chunk_id": "chunk_b"},
                ],
            )

            self.assertEqual({"chunk_a", "chunk_b"}, load_excluded_chunk_ids([output_dir]))

    def test_cli_fake_llm_smoke_writes_sidecars(self) -> None:
        long_text = (
            "Backpropagation produces gradients for each layer during training. "
            "The gradient signal is used to update model parameters and explain how loss changes. "
            "This chunk is intentionally long enough to pass candidate selection."
        )
        chunk = {
            "id": "chunk_cli",
            "chunk_type": "remnote_section",
            "chunk_role": "paragraph_group",
            "source": "AI Research/Test.md",
            "path": ["AI Research", "Test"],
            "heading_path": ["AI Research", "Test"],
            "line_start": 1,
            "line_end": 4,
            "source_block_ids": ["block_1"],
            "external_resource_ids": [],
            "text": long_text,
            "embedding_text": long_text,
            "display_text": long_text,
            "context_text": "Backpropagation context",
            "chunk_quality_flags": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "ir"
            output_dir = Path(tmp) / "out"
            input_dir.mkdir()
            write_jsonl(input_dir / "retrieval_chunks.jsonl", [chunk])
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_llm_postprocess.py"),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--fake-llm",
                    "--limit",
                    "1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue((output_dir / DECISIONS_FILENAME).exists())
            self.assertTrue((output_dir / CONCEPT_REGISTRY_FILENAME).exists())
            self.assertTrue((output_dir / GRAPH_PREVIEW_FILENAME).exists())

    def test_llm_postprocess_cli_concept_resolution_only_reads_existing_sidecars(self) -> None:
        chunk = make_input("Backpropagation produces gradients for each layer during training.")
        batch = ChunkPostprocessBatch(batch_id="batch_1", chunks=[chunk])
        decisions, failures = validate_and_enrich_response(
            parse_llm_response(valid_backprop_response()),
            batch,
            model_name="fake-model",
            prompt_version="v3",
        )
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            resolved_dir = Path(tmp) / "resolved"
            write_sidecar_outputs(
                source_dir,
                inputs=[chunk],
                decisions=decisions,
                failures=failures,
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_llm_postprocess.py"),
                    "--concept-resolution-only",
                    "--input-dir",
                    str(source_dir),
                    "--output-dir",
                    str(resolved_dir),
                    "--concept-resolution-mode",
                    "deterministic",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue((resolved_dir / DECISIONS_FILENAME).exists())
            self.assertTrue((resolved_dir / CONCEPT_REGISTRY_FILENAME).exists())
            self.assertTrue((resolved_dir / CONCEPT_MERGE_REVIEW_FILENAME).exists())
            self.assertTrue((resolved_dir / GRAPH_PREVIEW_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
