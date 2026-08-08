import unittest

from backend.data_processing.embedding_passages import build_embedding_passages


class EmbeddingPassageTests(unittest.TestCase):
    def test_short_text_stays_single_passage_with_path_prefix(self) -> None:
        passages = build_embedding_passages(
            parent_chunk_id="chunk_1",
            text="Backpropagation computes gradients using the chain rule.",
            source_path="Neural Networks > Backpropagation",
            summary="Gradient computation algorithm.",
            target_tokens=80,
            overlap_tokens=10,
        )

        self.assertEqual(1, len(passages))
        self.assertEqual("chunk_1::passage_000", passages[0].passage_id)
        self.assertEqual("chunk_1", passages[0].parent_chunk_id)
        self.assertIn("Neural Networks > Backpropagation", passages[0].text)
        self.assertNotIn("Path:", passages[0].text)
        self.assertNotIn("Summary:", passages[0].text)
        self.assertNotIn("Gradient computation algorithm.", passages[0].text)
        self.assertIn("chain rule", passages[0].text)

    def test_embedding_text_preserves_markdown_anchor_text_but_strips_urls(
        self,
    ) -> None:
        passages = build_embedding_passages(
            parent_chunk_id="chunk_links",
            text="Read [AG News](https://example.com/ag-news) and https://example.com/raw for datasets.",
            source_path="Text Classification > ## Common datasets > external:abc123",
            target_tokens=80,
            overlap_tokens=10,
        )

        self.assertEqual(1, len(passages))
        self.assertIn("Text Classification > Common datasets", passages[0].text)
        self.assertIn("AG News", passages[0].text)
        self.assertNotIn("https://", passages[0].text)
        self.assertNotIn("external:abc123", passages[0].text)

    def test_sentence_boundaries_are_preferred_over_hard_truncation(self) -> None:
        text = " ".join(
            f"Sentence {idx} explains optimizer behavior carefully."
            for idx in range(1, 26)
        )

        passages = build_embedding_passages(
            parent_chunk_id="chunk_optimizers",
            text=text,
            token_counter=lambda value: len(value.split()),
            target_tokens=90,
            overlap_tokens=0,
        )

        self.assertGreater(len(passages), 1)
        combined_bodies = "\n".join(passage.text for passage in passages)
        self.assertIn(
            "Sentence 1 explains optimizer behavior carefully.", combined_bodies
        )
        self.assertIn(
            "Sentence 25 explains optimizer behavior carefully.", combined_bodies
        )
        self.assertTrue(all("Sentence" in passage.text for passage in passages))

    def test_oversized_sentence_uses_hard_token_fallback(self) -> None:
        text = " ".join(f"word{idx}" for idx in range(180))

        passages = build_embedding_passages(
            parent_chunk_id="chunk_long_sentence",
            text=text,
            token_counter=lambda value: len(value.split()),
            target_tokens=90,
            overlap_tokens=0,
        )

        self.assertGreater(len(passages), 1)
        self.assertTrue(
            any(passage.split_strategy == "hard_token_fallback" for passage in passages)
        )
        self.assertTrue(
            all(
                passage.parent_chunk_id == "chunk_long_sentence" for passage in passages
            )
        )


if __name__ == "__main__":
    unittest.main()
