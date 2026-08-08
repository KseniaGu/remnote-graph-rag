from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LLAMA_INDEX_AVAILABLE = importlib.util.find_spec("llama_index") is not None


from backend.data_processing.parser_optimized import (  # noqa: E402
    IMAGE_PLACEHOLDER,
    OptimizedRemNoteParser,
    RemNoteParserOptimized,
    extract_url_matches,
    normalize_nfc,
    stable_id,
)
from backend.data_processing.parser_outputs import (  # noqa: E402
    ARTIFACT_GATE_DECISIONS_FILENAME,
    BLOCKS_FILENAME,
    COMPARISON_FILENAME,
    EXTERNAL_RESOURCES_FILENAME,
    PARSED_ARTIFACTS_FILENAME,
    RETRIEVAL_CHUNKS_FILENAME,
    SOURCE_DOCUMENTS_FILENAME,
    SUMMARY_FILENAME,
    write_optimized_parser_ir,
)


class OptimizedParserTest(unittest.TestCase):
    def test_extract_url_matches_returns_each_url_occurrence(self) -> None:
        line = (
            "- see ![First](https://example.test/one.png) and "
            "[Second](https://example.test/two.pdf) plus https://example.test/raw"
        )

        matches = extract_url_matches(line)

        self.assertEqual([match.kind for match in matches], ["image", "link", "raw"])
        self.assertEqual([match.ordinal for match in matches], [0, 1, 2])
        self.assertEqual([match.name for match in matches], ["First", "Second", None])

    def test_nfc_normalization_makes_ids_stable(self) -> None:
        decomposed = "Cafe\u0301"
        composed = "Caf\u00e9"

        self.assertEqual(normalize_nfc(decomposed), composed)
        self.assertEqual(stable_id("demo", decomposed), stable_id("demo", composed))

    def test_shadow_parser_creates_resources_and_clean_retrieval_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "AI Research"
            parsed_images = root / "parsed_images"
            raw_dir.mkdir(parents=True)
            parsed_images.mkdir(parents=True)

            long_note = (
                "This paragraph is intentionally long enough to become a coherent retrieval unit. "
                "It explains the surrounding RemNote context, keeps useful words together, and "
                "ensures the optimized parser does not need to embed placeholder-only fragments."
            )
            artifact_text = (
                "Parsed artifact section with enough semantic content to be useful in retrieval. "
                "It should remain linked to its parent RemNote block and expose the artifact line span."
            )
            (parsed_images / "Screenshot.md").write_text(
                artifact_text, encoding="utf-8"
            )
            (raw_dir / "Source.md").write_text(
                "\n".join(
                    [
                        "# Topic",
                        f"- {long_note}",
                        "  - ![Screenshot](https://remnote-user-data.s3.amazonaws.com/a.png) "
                        "and ![Diagram](https://remnote-user-data.s3.amazonaws.com/b.png)",
                        f"  - {IMAGE_PLACEHOLDER}",
                        "  - ```",
                        "  - ok",
                    ]
                ),
                encoding="utf-8",
            )

            result = OptimizedRemNoteParser(raw_dir, parsed_roots=[parsed_images]).run()

        self.assertTrue(result.summary["url_count_match"])
        self.assertEqual(result.summary["raw_url_occurrences"], 2)
        self.assertEqual(result.summary["parser_visible_url_resources"], 2)
        self.assertEqual(result.summary["standalone_tiny_chunk_count"], 0)
        self.assertEqual(result.summary["placeholder_only_chunk_count"], 0)
        self.assertEqual(result.summary["code_fence_only_chunk_count"], 0)
        self.assertEqual(result.summary["header_only_chunk_count"], 0)
        self.assertEqual(result.summary["orphan_list_parent_chunk_count"], 0)
        self.assertEqual(result.summary["split_list_item_subtree_count"], 0)
        self.assertEqual(result.summary["resource_only_chunk_count"], 0)
        self.assertEqual(result.summary["chunks_missing_provenance_count"], 0)
        self.assertEqual(result.summary["failed_path_current_dir_count"], 0)
        self.assertEqual(result.summary["mixed_source_retrieval_chunk_count"], 0)

        for resource in result.external_resources:
            self.assertNotEqual(resource.artifact_path, ".")
            self.assertNotEqual(resource.artifact_path, "")
            self.assertTrue(resource.parent_block_id)
            self.assertTrue(resource.url_hash)

        for chunk in result.retrieval_chunks:
            self.assertGreater(len(chunk.text.strip()), 3)
            self.assertNotEqual(chunk.text.strip(), IMAGE_PLACEHOLDER)
            self.assertTrue(chunk.source_block_ids)
            self.assertTrue(chunk.source)
            self.assertTrue(chunk.path)
            self.assertIsInstance(chunk.line_start, int)
            self.assertIsInstance(chunk.line_end, int)

        artifact_chunks = [
            chunk
            for chunk in result.retrieval_chunks
            if chunk.chunk_type == "external_artifact"
        ]
        self.assertEqual(len(artifact_chunks), 1)
        self.assertIsNotNone(artifact_chunks[0].artifact_path)
        self.assertIsNotNone(artifact_chunks[0].artifact_line_start)
        self.assertIsNotNone(artifact_chunks[0].artifact_line_end)

    def test_numbered_list_example_stays_with_list_context_and_resource(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "AI Research"
            raw_dir.mkdir(parents=True)
            (raw_dir / "Text Classification.md").write_text(
                "\n".join(
                    [
                        "- #### Naive Bayes Classifier",
                        "- A Naive Bayes classifier explanation that gives enough context for the section.",
                        "5. **Variants**",
                        "    - **Multinomial NB**: models word counts directly.",
                        "    - **Bernoulli NB**: uses binary features and models absent words.",
                        "6. **Example (Gaussian Naive Bayes classifier)**",
                        "    - ![](https://remnote-user-data.s3.amazonaws.com/example.png)",
                    ]
                ),
                encoding="utf-8",
            )

            result = OptimizedRemNoteParser(raw_dir).run()

        chunks_with_example = [
            chunk for chunk in result.retrieval_chunks if "6. Example" in chunk.text
        ]
        self.assertEqual(len(chunks_with_example), 1)
        example_chunk = chunks_with_example[0]
        self.assertIn("5. Variants", example_chunk.text)
        self.assertIn("Bernoulli NB", example_chunk.text)
        self.assertTrue(example_chunk.external_resource_ids)
        self.assertIn("resource_attached", example_chunk.chunk_quality_flags)
        self.assertEqual(example_chunk.chunk_role, "section_with_list_items")
        self.assertEqual(result.summary["header_only_chunk_count"], 0)
        self.assertEqual(result.summary["orphan_list_parent_chunk_count"], 0)
        self.assertEqual(result.summary["split_list_item_subtree_count"], 0)

    def test_chunker_never_merges_across_source_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "AI Research"
            raw_dir.mkdir(parents=True)
            (raw_dir / "A Short.md").write_text(
                "[Short](https://example.test/short)\n",
                encoding="utf-8",
            )
            (raw_dir / "B Long.md").write_text(
                "\n".join(
                    [
                        "This second source has enough content to form a retrieval chunk. "
                        "It must not be merged backward into the first source just because "
                        "the first source is shorter than the minimum chunk target.",
                        "Another line stays in the second source and gives the chunker room to aggregate safely.",
                    ]
                ),
                encoding="utf-8",
            )

            result = OptimizedRemNoteParser(raw_dir).run()

        blocks_by_id = {block.id: block for block in result.blocks}
        self.assertEqual(result.summary["mixed_source_retrieval_chunk_count"], 0)
        self.assertTrue(result.summary["success_criteria"]["no_mixed_source_chunks"])
        for chunk in result.retrieval_chunks:
            block_sources = {
                blocks_by_id[block_id].source for block_id in chunk.source_block_ids
            }
            self.assertEqual(block_sources, {chunk.source})

    def test_code_fence_markers_are_removed_but_code_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "AI Research"
            raw_dir.mkdir(parents=True)
            (raw_dir / "SQL Notes.md").write_text(
                "\n".join(
                    [
                        "- **Quick SQL example:**",
                        "- .",
                        "    ```",
                        "    ```sql",
                        "CREATE TABLE users (",
                        "    id SERIAL PRIMARY KEY,",
                        "    email VARCHAR(255) UNIQUE,",
                        "    name VARCHAR(100)",
                        ");",
                        "```",
                        "    ```",
                    ]
                ),
                encoding="utf-8",
            )

            result = OptimizedRemNoteParser(raw_dir).run()

        chunk_text = "\n".join(chunk.text for chunk in result.retrieval_chunks)
        embedded_text = "\n".join(
            chunk.embedding_text or "" for chunk in result.retrieval_chunks
        )
        display_text = "\n".join(
            chunk.display_text or "" for chunk in result.retrieval_chunks
        )
        self.assertIn("CREATE TABLE users", chunk_text)
        self.assertIn("id SERIAL PRIMARY KEY", chunk_text)
        self.assertNotIn("```sql", chunk_text)
        self.assertNotIn("```", chunk_text)
        self.assertNotIn("```sql", embedded_text)
        self.assertNotIn("```", embedded_text)
        self.assertNotIn("```sql", display_text)
        self.assertNotIn("```", display_text)
        self.assertEqual(result.summary["code_fence_marker_line_count"], 0)
        self.assertTrue(
            result.summary["success_criteria"]["no_code_fence_marker_lines"]
        )

    def test_deterministic_external_artifact_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "AI Research"
            parsed_texts = root / "parsed_texts"
            parsed_images = root / "parsed_images"
            raw_dir.mkdir(parents=True)
            parsed_texts.mkdir(parents=True)
            parsed_images.mkdir(parents=True)

            dataset_text = "\n".join(
                [
                    "title: fancyzhx/ag_news · Datasets at Hugging Face",
                    "url: https://huggingface.co/datasets/fancyzhx/ag_news",
                    "description: Classification dataset with training samples and testing samples.",
                    "| example text | label |",
                    "| Market story one | Business |",
                ]
            )
            mismatch_text = "\n".join(
                [
                    "title: Trending Papers - Hugging Face",
                    "url: https://huggingface.co/papers/trending",
                    "description: Your daily dose of AI research from AK",
                ]
            )
            useful_text = (
                "This external explanation is long enough to be embedded once. "
                "It describes a focused educational concept with clear prose, stable provenance, "
                "and no dataset rows or unrelated navigation content. "
                "The same content appears in a second artifact to test duplicate gating."
            )
            bad_ocr = " ".join(
                [
                    "PaccMOTpM IIpocTo pıMep Ha puc oKa3aHa IByMepHa Bbi6opKa",
                    "c IByM Kjacca aa oepb i Hk e mt bib a repiococTbIo",
                ]
                * 10
            )
            formula_ocr = r"$$Q(a)=\frac{1}{2}N a-y N^{2}+\frac{1}{2}a^{2}k a$$"
            valid_english_ocr = (
                "A more resilient approach would be to use an Extension. An Extension bridges the gap "
                "between an agent and an API by teaching the agent how to use the endpoint with examples, "
                "and by teaching what arguments are needed to successfully call the API."
            )

            (parsed_texts / "AG_News.md").write_text(dataset_text, encoding="utf-8")
            (parsed_texts / "Yahoo_Answers.md").write_text(
                mismatch_text, encoding="utf-8"
            )
            (parsed_texts / "Good_One.md").write_text(useful_text, encoding="utf-8")
            (parsed_texts / "Good_Two.md").write_text(useful_text, encoding="utf-8")
            (parsed_images / "BadOCR.md").write_text(bad_ocr, encoding="utf-8")
            (parsed_images / "Formula.md").write_text(formula_ocr, encoding="utf-8")
            (parsed_images / "ValidEnglish.md").write_text(
                valid_english_ocr, encoding="utf-8"
            )

            (raw_dir / "English Source.md").write_text(
                "\n".join(
                    [
                        "[AG News](https://paperswithcode.com/dataset/ag-news?utm_source=chatgpt.com)",
                        "[Yahoo Answers](https://paperswithcode.com/dataset/yahoo-answers)",
                        "[Good One](https://example.test/good-one)",
                        "[Good Two](https://example.test/good-two)",
                    ]
                ),
                encoding="utf-8",
            )
            (raw_dir / "Русский источник.md").write_text(
                "\n".join(
                    [
                        "![BadOCR](https://remnote-user-data.s3.amazonaws.com/bad.png)",
                        "![Formula](https://remnote-user-data.s3.amazonaws.com/formula.png)",
                        "![ValidEnglish](https://remnote-user-data.s3.amazonaws.com/valid-english.png)",
                    ]
                ),
                encoding="utf-8",
            )

            result = OptimizedRemNoteParser(
                raw_dir, parsed_roots=[parsed_texts, parsed_images]
            ).run()

        decisions = {
            Path(decision.artifact_path).name: decision
            for decision in result.artifact_gate_decisions
        }
        self.assertEqual(decisions["AG_News.md"].policy, "metadata_only")
        self.assertIn("dataset_artifact", decisions["AG_News.md"].reason_codes)
        self.assertEqual(decisions["AG_News.md"].emitted_chunk_count, 0)

        self.assertEqual(decisions["Yahoo_Answers.md"].policy, "quarantine")
        self.assertIn("url_mismatch", decisions["Yahoo_Answers.md"].reason_codes)
        self.assertIn(
            "generic_navigation_artifact", decisions["Yahoo_Answers.md"].reason_codes
        )
        self.assertEqual(decisions["Yahoo_Answers.md"].emitted_chunk_count, 0)

        self.assertEqual(decisions["Good_One.md"].policy, "embed_full")
        self.assertGreater(decisions["Good_One.md"].emitted_chunk_count, 0)
        self.assertEqual(decisions["Good_Two.md"].policy, "metadata_only")
        self.assertIn("duplicate_content_hash", decisions["Good_Two.md"].reason_codes)
        self.assertEqual(decisions["Good_Two.md"].emitted_chunk_count, 0)

        self.assertEqual(decisions["BadOCR.md"].policy, "quarantine")
        self.assertIn("low_quality_ocr", decisions["BadOCR.md"].reason_codes)
        self.assertEqual(decisions["Formula.md"].policy, "embed_full")
        self.assertEqual(decisions["ValidEnglish.md"].policy, "embed_full")

        embedded_artifacts = {
            Path(chunk.artifact_path).name
            for chunk in result.retrieval_chunks
            if chunk.chunk_type == "external_artifact" and chunk.artifact_path
        }
        self.assertNotIn("AG_News.md", embedded_artifacts)
        self.assertNotIn("Yahoo_Answers.md", embedded_artifacts)
        self.assertNotIn("Good_Two.md", embedded_artifacts)
        self.assertNotIn("BadOCR.md", embedded_artifacts)
        self.assertIn("Good_One.md", embedded_artifacts)
        self.assertIn("Formula.md", embedded_artifacts)
        self.assertIn("ValidEnglish.md", embedded_artifacts)

        self.assertEqual(result.summary["embedded_dataset_dump_chunk_count"], 0)
        self.assertEqual(result.summary["embedded_url_mismatch_chunk_count"], 0)
        self.assertEqual(result.summary["embedded_duplicate_artifact_chunk_count"], 0)
        self.assertEqual(result.summary["embedded_low_quality_ocr_chunk_count"], 0)
        self.assertEqual(result.summary["dataset_artifact_metadata_only_count"], 1)
        self.assertEqual(result.summary["url_mismatch_quarantine_count"], 1)
        self.assertEqual(result.summary["duplicate_artifact_metadata_only_count"], 1)
        self.assertEqual(result.summary["low_quality_ocr_quarantine_count"], 1)
        self.assertTrue(result.summary["success_criteria"]["no_embedded_dataset_dumps"])
        self.assertTrue(
            result.summary["success_criteria"]["no_embedded_url_mismatch_artifacts"]
        )
        self.assertTrue(
            result.summary["success_criteria"]["no_embedded_duplicate_artifacts"]
        )
        self.assertTrue(
            result.summary["success_criteria"]["no_embedded_low_quality_ocr"]
        )

    def test_external_artifact_chunks_inherit_remnote_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "AI Research"
            parsed_images = root / "parsed_images"
            raw_dir.mkdir(parents=True)
            parsed_images.mkdir(parents=True)

            (parsed_images / "nn-one.md").write_text(
                "Neural network parsed diagram text explaining hidden layers, logits, "
                "and nonlinear feature extraction for text classification.",
                encoding="utf-8",
            )
            (parsed_images / "proc.md").write_text(
                "Stored procedure parsed screenshot showing CREATE PROCEDURE syntax, "
                "parameters, and execution flow inside a SQL database.",
                encoding="utf-8",
            )
            (parsed_images / "proc-next.md").write_text(
                "A second stored procedure image with SQL syntax and procedure execution details.",
                encoding="utf-8",
            )
            (raw_dir / "Text Classification.md").write_text(
                "\n".join(
                    [
                        "#### Neural Networks",
                        "![](https://remnote-user-data.s3.amazonaws.com/nn-one.png)",
                    ]
                ),
                encoding="utf-8",
            )
            (
                raw_dir
                / "Tutorial 1- MySQL With Python And Data Science- MySQL Installation Steps - YouTube 4.md"
            ).write_text(
                "\n".join(
                    [
                        "### Stored procedures",
                        "- Типа функции видимо",
                        "- ![](https://remnote-user-data.s3.amazonaws.com/proc.png)",
                        "- ![](https://remnote-user-data.s3.amazonaws.com/proc-next.png)",
                    ]
                ),
                encoding="utf-8",
            )

            result = OptimizedRemNoteParser(raw_dir, parsed_roots=[parsed_images]).run()

        artifact_chunks = {
            Path(chunk.artifact_path).name: chunk
            for chunk in result.retrieval_chunks
            if chunk.chunk_type == "external_artifact" and chunk.artifact_path
        }
        self.assertIn("nn-one.md", artifact_chunks)
        self.assertIn("proc.md", artifact_chunks)
        self.assertIn("proc-next.md", artifact_chunks)

        nn_chunk = artifact_chunks["nn-one.md"]
        self.assertIn("#### Neural Networks", nn_chunk.path)
        self.assertIn("#### Neural Networks", nn_chunk.heading_path)
        self.assertIn("#### Neural Networks", nn_chunk.context_text or "")
        self.assertIn("Neural Networks", nn_chunk.embedding_text or "")
        self.assertNotIn("#### Neural Networks", nn_chunk.embedding_text or "")
        self.assertNotIn("external:", nn_chunk.embedding_text or "")
        self.assertNotIn("RemNote context:", nn_chunk.embedding_text or "")
        self.assertNotIn("Parsed external content:", nn_chunk.embedding_text or "")
        self.assertEqual(nn_chunk.source_relation, "parsed_external_resource")
        self.assertIn("context_attached", nn_chunk.chunk_quality_flags)

        proc_chunk = artifact_chunks["proc.md"]
        self.assertIn("### Stored procedures", proc_chunk.path)
        self.assertIn("### Stored procedures", proc_chunk.heading_path)
        self.assertIn("Типа функции видимо", proc_chunk.context_text or "")
        self.assertIn("Stored procedures", proc_chunk.embedding_text or "")
        self.assertIn("Типа функции видимо", proc_chunk.embedding_text or "")
        self.assertNotIn("### Stored procedures", proc_chunk.embedding_text or "")
        self.assertNotIn("external:", proc_chunk.embedding_text or "")
        self.assertNotIn("RemNote context:", proc_chunk.embedding_text or "")
        self.assertNotIn("Parsed external content:", proc_chunk.embedding_text or "")
        self.assertEqual(proc_chunk.source_relation, "parsed_external_resource")
        self.assertIn("context_attached", proc_chunk.chunk_quality_flags)

        proc_next_chunk = artifact_chunks["proc-next.md"]
        self.assertIn("### Stored procedures", proc_next_chunk.context_text or "")
        self.assertNotIn("Типа функции видимо", proc_next_chunk.context_text or "")
        self.assertIn("Stored procedures", proc_next_chunk.embedding_text or "")
        self.assertNotIn("Типа функции видимо", proc_next_chunk.embedding_text or "")

        self.assertEqual(result.summary["external_artifact_chunk_count"], 3)
        self.assertEqual(
            result.summary["external_artifact_chunks_with_context_count"], 3
        )
        self.assertEqual(
            result.summary["external_artifact_chunks_without_context_count"], 0
        )
        self.assertEqual(
            result.summary["external_artifact_embedding_support_label_count"], 0
        )
        self.assertTrue(
            result.summary["success_criteria"][
                "no_external_artifact_embedding_support_labels"
            ]
        )

    def test_image_url_prefers_parsed_markdown_sibling_over_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "AI Research"
            parsed_images = root / "parsed_images"
            raw_dir.mkdir(parents=True)
            parsed_images.mkdir(parents=True)

            full_image_name = (
                "prefix_that_gets_trimmed_" + "A" * 90 + "_CorrectOCRArtifact.png"
            )
            image_name = full_image_name[-100:]
            image_url = f"https://remnote-user-data.s3.amazonaws.com/{full_image_name}"
            parsed_text = (
                "# Gaussian Naive Bayes: Training via Maximum Likelihood Estimation\n\n"
                "This parsed image text explains how to estimate class priors, feature means, "
                "and variances for Gaussian Naive Bayes. It should be embedded as an external "
                "artifact instead of being hidden behind the binary image file."
            )
            (parsed_images / image_name).write_bytes(b"fake image bytes")
            (parsed_images / Path(image_name).with_suffix(".md").name).write_text(
                parsed_text, encoding="utf-8"
            )
            (raw_dir / "Text Classification.md").write_text(
                f"6. **Example (Gaussian Naive Bayes classifier)**\n    - ![]({image_url})\n",
                encoding="utf-8",
            )

            result = OptimizedRemNoteParser(raw_dir, parsed_roots=[parsed_images]).run()

        resource = result.external_resources[0]
        self.assertEqual(resource.artifact_type, "markdown")
        self.assertTrue(resource.artifact_path.endswith("CorrectOCRArtifact.md"))
        self.assertEqual(
            result.summary["image_binary_selected_despite_md_sibling_count"], 0
        )
        self.assertTrue(
            result.summary["success_criteria"][
                "no_image_binary_selected_when_md_sibling_exists"
            ]
        )
        artifact_chunks = [
            chunk
            for chunk in result.retrieval_chunks
            if chunk.chunk_type == "external_artifact"
        ]
        self.assertEqual(len(artifact_chunks), 1)
        self.assertTrue(
            artifact_chunks[0].artifact_path.endswith("CorrectOCRArtifact.md")
        )
        self.assertIn("Gaussian Naive Bayes", artifact_chunks[0].text)

    def test_drop_in_wrapper_accepts_original_constructor_args_without_llamaindex_import(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path_settings = SimpleNamespace(
                raw_data_dir=root / "raw",
                parsed_images_dir=root / "parsed_images",
                parsed_pdfs_dir=root / "parsed_pdfs",
                parsed_texts_dir=root / "parsed_texts",
                local_storage_dir=root / "storage",
            )
            storage_settings = SimpleNamespace(
                document_storage=SimpleNamespace(storage_type="local")
            )

            parser = RemNoteParserOptimized(
                path_settings,
                storage_settings,
                prepare_external_artifacts=False,
                copy_existing_artifacts=False,
                write_ir=False,
            )

        self.assertIs(parser.path_settings, path_settings)
        self.assertIs(parser.storage_settings, storage_settings)
        self.assertFalse(parser.prepare_external_artifacts_enabled)
        self.assertFalse(parser.copy_existing_artifacts_enabled)
        self.assertIsNone(parser.existing_artifacts_dir)
        self.assertFalse(parser.write_ir)
        self.assertIsNone(parser.kg_storage)

    def test_copy_existing_artifacts_copies_only_markdown_to_isolated_cache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "reviewed_parsed_images"
            target_dir = root / "isolated" / "parsed_images"
            source_dir.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            (source_dir / "copy-me.md").write_text(
                "parsed image text", encoding="utf-8"
            )
            (source_dir / "ignore.png").write_bytes(b"image bytes")
            (source_dir / "nested").mkdir()
            (source_dir / "nested" / "copy-nested.md").write_text(
                "nested parsed image text", encoding="utf-8"
            )
            (target_dir / "already.md").write_text("keep this target", encoding="utf-8")
            (source_dir / "already.md").write_text("do not overwrite", encoding="utf-8")

            path_settings = SimpleNamespace(
                raw_data_dir=root / "raw",
                parsed_images_dir=target_dir,
                parsed_pdfs_dir=root / "isolated" / "parsed_pdfs",
                parsed_texts_dir=root / "isolated" / "parsed_texts",
                local_storage_dir=root / "isolated" / "storage",
            )
            storage_settings = SimpleNamespace(
                document_storage=SimpleNamespace(storage_type="local")
            )
            parser = RemNoteParserOptimized(
                path_settings,
                storage_settings,
                prepare_external_artifacts=False,
                copy_existing_artifacts=True,
                existing_artifacts_dir=source_dir,
                write_ir=False,
            )

            copied = parser.copy_existing_artifacts()

            self.assertEqual(copied, 2)
            self.assertEqual(parser.last_copied_artifact_count, 2)
            self.assertTrue((target_dir / "copy-me.md").exists())
            self.assertTrue((target_dir / "nested" / "copy-nested.md").exists())
            self.assertFalse((target_dir / "ignore.png").exists())
            self.assertEqual(
                (target_dir / "already.md").read_text(encoding="utf-8"),
                "keep this target",
            )

    def test_default_artifact_copy_ignores_unrelated_raw_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "parsed_images"
            path_settings = SimpleNamespace(
                raw_data_dir=root / "unrelated_raw" / "AI Research",
                parsed_images_dir=target_dir,
                parsed_pdfs_dir=root / "parsed_pdfs",
                parsed_texts_dir=root / "parsed_texts",
                local_storage_dir=root / "storage",
            )
            storage_settings = SimpleNamespace(
                document_storage=SimpleNamespace(storage_type="local")
            )
            parser = RemNoteParserOptimized(
                path_settings,
                storage_settings,
                prepare_external_artifacts=False,
                copy_existing_artifacts=True,
                write_ir=False,
            )

            copied = parser.copy_existing_artifacts()

            self.assertEqual(copied, 0)
            self.assertFalse(target_dir.exists() and list(target_dir.rglob("*.md")))

    def test_write_optimized_parser_ir_writes_expected_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "AI Research"
            raw_dir.mkdir(parents=True)
            (raw_dir / "Source.md").write_text(
                "#### Topic\n- This source has enough content to become a retrieval chunk.\n",
                encoding="utf-8",
            )

            result = OptimizedRemNoteParser(raw_dir).run()
            output_dir = write_optimized_parser_ir(root / "optimized_parser_ir", result)

            expected_files = {
                SUMMARY_FILENAME,
                SOURCE_DOCUMENTS_FILENAME,
                BLOCKS_FILENAME,
                EXTERNAL_RESOURCES_FILENAME,
                PARSED_ARTIFACTS_FILENAME,
                ARTIFACT_GATE_DECISIONS_FILENAME,
                RETRIEVAL_CHUNKS_FILENAME,
                COMPARISON_FILENAME,
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()}, expected_files
            )
            summary = json.loads(
                (output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8")
            )
            chunks = (
                (output_dir / RETRIEVAL_CHUNKS_FILENAME)
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(summary["retrieval_chunk_count"], len(chunks))

    def test_three_source_regression_preserves_core_parser_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "AI Research"
            parsed_images = root / "parsed_images"
            parsed_pdfs = root / "parsed_pdfs"
            parsed_texts = root / "parsed_texts"
            raw_dir.mkdir(parents=True)
            parsed_images.mkdir(parents=True)
            parsed_pdfs.mkdir(parents=True)
            parsed_texts.mkdir(parents=True)

            (parsed_pdfs / "Alice_paper.md").write_text(
                "Differentiable Wonderland paper notes about neural networks and optimization.",
                encoding="utf-8",
            )
            (parsed_images / "Embedding_Diagram.md").write_text(
                "Vector embedding diagram explaining word contexts and dense semantic spaces.",
                encoding="utf-8",
            )
            (parsed_texts / "Decision_tree_notes.md").write_text(
                "Decision tree artifact text about feature splits, entropy, and information gain.",
                encoding="utf-8",
            )
            alice_source = (
                raw_dir
                / "[2404.17625] Alice's Adventures in a Differentiable Wonderland -- Volume I, A Tour of the Land.md"
            )
            alice_source.write_text(
                "\n".join(
                    [
                        "#### Alice's Adventures in a Differentiable Wonderland",
                        (
                            "- Read the [Alice paper](https://example.test/Alice_paper.pdf) "
                            "for neural network optimization examples."
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            (raw_dir / "Word Embeddings.md").write_text(
                "\n".join(
                    [
                        "#### Word Embeddings",
                        "- Dense vector spaces represent words by context.",
                        "- ![Embedding Diagram](https://remnote-user-data.s3.amazonaws.com/Embedding_Diagram.png)",
                    ]
                ),
                encoding="utf-8",
            )
            (raw_dir / "почитать про деревья решений.md").write_text(
                "\n".join(
                    [
                        "#### Деревья решений",
                        (
                            "- Дерево решений выбирает признаки для разбиения "
                            "и объясняет структуру классификации."
                        ),
                        (
                            "- Нужно почитать про критерии разбиения и "
                            "[Decision tree notes](https://example.test/Decision_tree_notes.html)."
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            result = OptimizedRemNoteParser(
                raw_dir,
                parsed_roots=[parsed_images, parsed_pdfs, parsed_texts],
            ).run()

        self.assertEqual(len(result.source_documents), 3)
        self.assertEqual(result.summary["raw_url_occurrences"], 3)
        self.assertEqual(result.summary["parser_visible_url_resources"], 3)
        self.assertEqual(result.summary["parsed_artifact_count"], 3)
        self.assertTrue(
            result.summary["success_criteria"]["raw_url_count_equals_parser_visible"]
        )
        self.assertTrue(
            result.summary["success_criteria"]["all_chunks_have_provenance"]
        )
        self.assertTrue(result.summary["success_criteria"]["no_mixed_source_chunks"])
        self.assertEqual(result.summary["placeholder_only_chunk_count"], 0)
        self.assertEqual(result.summary["resource_only_chunk_count"], 0)
        self.assertEqual(result.summary["header_only_chunk_count"], 0)
        self.assertEqual(result.summary["external_artifact_chunk_count"], 3)
        self.assertTrue(
            any("Дерево решений" in chunk.text for chunk in result.retrieval_chunks)
        )
        self.assertTrue(
            any(
                "Деревья решений" in " ".join(chunk.path)
                for chunk in result.retrieval_chunks
            )
        )
        artifact_paths = {
            Path(artifact.artifact_path).name for artifact in result.parsed_artifacts
        }
        self.assertEqual(
            artifact_paths,
            {"Alice_paper.md", "Embedding_Diagram.md", "Decision_tree_notes.md"},
        )

    @unittest.skipUnless(
        LLAMA_INDEX_AVAILABLE, "LlamaIndex is not installed in this environment"
    )
    def test_drop_in_wrapper_writes_retrieval_chunks_to_local_docstore(self) -> None:
        from backend.configs.paths import PathSettings
        from backend.configs.storage import LocalStorageSettings, StorageSettings

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "AI Research"
            parsed_images = root / "parsed_images"
            raw_dir.mkdir(parents=True)
            parsed_images.mkdir(parents=True)
            (parsed_images / "Diagram.md").write_text(
                "Parsed diagram content about neural network hidden layers and logits for classification.",
                encoding="utf-8",
            )
            (raw_dir / "Text Classification.md").write_text(
                "\n".join(
                    [
                        "#### Neural Networks",
                        "- This section explains why hidden layers are useful in text classification models.",
                        "- ![Diagram](https://remnote-user-data.s3.amazonaws.com/diagram.png)",
                    ]
                ),
                encoding="utf-8",
            )

            storage_dir = root / "storage"
            local_storage = LocalStorageSettings(
                storage_type="local", storage_path=storage_dir
            )
            storage_settings = StorageSettings(
                document_storage=local_storage,
                index_storage=local_storage,
                vector_storage=local_storage,
                property_graph_storage=local_storage,
            )
            path_settings = PathSettings(
                raw_data_dir=raw_dir,
                parsed_images_dir=parsed_images,
                parsed_pdfs_dir=root / "parsed_pdfs",
                parsed_texts_dir=root / "parsed_texts",
                local_storage_dir=storage_dir,
            )
            parser = RemNoteParserOptimized(
                path_settings,
                storage_settings,
                prepare_external_artifacts=False,
                copy_existing_artifacts=False,
                force_rebuild=True,
                write_ir=True,
            )

            parser.run()

            docs = parser.kg_storage.storage_context.docstore.docs
            result = parser.last_result
            self.assertIsNotNone(result)
            self.assertEqual(len(docs), len(result.retrieval_chunks))
            for chunk in result.retrieval_chunks:
                self.assertIn(chunk.id, docs)
                node = docs[chunk.id]
                self.assertEqual(node.text, chunk.embedding_text or chunk.text)
                self.assertEqual(
                    set(node.excluded_embed_metadata_keys), set(node.metadata.keys())
                )
                self.assertNotIn("external:", node.text)
                self.assertNotIn("Parsed external content:", node.text)
                self.assertNotIn("RemNote context:", node.text)

            artifact_nodes = [
                node
                for node in docs.values()
                if node.metadata["chunk_type"] == "external_artifact"
            ]
            self.assertEqual(len(artifact_nodes), 1)
            artifact_node = artifact_nodes[0]
            self.assertTrue(
                artifact_node.metadata["artifact_path"].endswith("Diagram.md")
            )
            self.assertIn("#### Neural Networks", artifact_node.metadata["path"])
            self.assertIn(
                "#### Neural Networks", artifact_node.metadata["context_text"]
            )
            self.assertTrue(
                (root / "optimized_parser_ir" / "retrieval_chunks.jsonl").exists()
            )


if __name__ == "__main__":
    unittest.main()
