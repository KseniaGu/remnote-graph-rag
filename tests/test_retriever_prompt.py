import unittest
from pathlib import Path

RETRIEVER_V5_PROMPT = Path("backend/llm/prompts/learner_workflow/retriever/v5.yaml")


class RetrieverPromptTests(unittest.TestCase):
    def test_visualization_query_generation_examples_preserve_facets_and_comparisons(
        self,
    ) -> None:
        prompt = RETRIEVER_V5_PROMPT.read_text(encoding="utf-8")

        self.assertIn('["Text Classification methods"]', prompt)
        self.assertIn('["Text Classification datasets"]', prompt)
        self.assertIn('["CLIP architecture", "CLIP training objective"]', prompt)
        self.assertIn('["Naive Bayes vs Logistic Regression"]', prompt)
        self.assertIn("Keep visualization queries focused", prompt)
        self.assertIn("Resolve pronouns from recent conversation history", prompt)


if __name__ == "__main__":
    unittest.main()
