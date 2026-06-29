from pathlib import Path
import unittest


RETRIEVER_V4_PROMPT = Path("backend/llm/prompts/learner_workflow/retriever/v4.yaml")


class RetrieverPromptTests(unittest.TestCase):
    def test_visualization_query_generation_examples_preserve_facets_and_comparisons(self) -> None:
        prompt = RETRIEVER_V4_PROMPT.read_text(encoding="utf-8")

        self.assertIn('["Text Classification methods"]', prompt)
        self.assertIn('["Text Classification datasets"]', prompt)
        self.assertIn('["CLIP architecture"]', prompt)
        self.assertIn('["CLIP training objective"]', prompt)
        self.assertIn('["Naive Bayes vs Logistic Regression"]', prompt)
        self.assertIn("Do not add broad parent topics", prompt)
        self.assertIn("using the provided Conversation History", prompt)


if __name__ == "__main__":
    unittest.main()
