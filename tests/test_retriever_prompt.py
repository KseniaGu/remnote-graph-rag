import unittest
from pathlib import Path

RETRIEVER_V6_PROMPT = Path("backend/llm/prompts/learner_workflow/retriever/v6.yaml")


class RetrieverPromptTests(unittest.TestCase):
    def test_active_prompt_requires_explicit_topic_policy(self) -> None:
        prompt_path = Path("backend/llm/prompts/learner_workflow/retriever/v7.yaml")
        self.assertTrue(prompt_path.is_file(), "Retriever v7 prompt must exist")
        prompt = prompt_path.read_text(encoding="utf-8")
        self.assertIn('"topic_policy":"broad","required_topics":[]', prompt)
        self.assertIn(
            '"topic_policy":"named","required_topics":[["RetNet","Retentive Network"]]',
            prompt,
        )
        self.assertIn("only after an in-scope route", prompt)
        self.assertIn("Never guess or invent an expansion", prompt)
        self.assertIn(
            "Never pass topic_policy or required_topics to get_subgraphs_to_visualize",
            prompt,
        )

    def test_named_topics_have_explicit_adequacy_requirements(self) -> None:
        prompt = RETRIEVER_V6_PROMPT.read_text(encoding="utf-8")

        self.assertIn("required_topics", prompt)
        self.assertIn('["RetNet","Retentive Network"]', prompt)
        self.assertIn('["Multi-Head Latent Attention","MLA"]', prompt)
        self.assertIn('[["EAGLT"]]', prompt)
        self.assertIn('[["RetNet","Retentive Network"],["Mamba"]]', prompt)
        self.assertIn("Never guess or invent an expansion", prompt)
        self.assertIn("Leave required_topics empty for broad categories", prompt)

    def test_visualization_contract_does_not_accept_required_topics(self) -> None:
        prompt = RETRIEVER_V6_PROMPT.read_text(encoding="utf-8")

        self.assertIn(
            "Never pass required_topics to get_subgraphs_to_visualize", prompt
        )
        self.assertIn(
            'get_subgraphs_to_visualize({"queries":["Text Classification methods"]})',
            prompt,
        )

    def test_visualization_query_generation_examples_preserve_facets_and_comparisons(
        self,
    ) -> None:
        prompt = RETRIEVER_V6_PROMPT.read_text(encoding="utf-8")

        self.assertIn('["Text Classification methods"]', prompt)
        self.assertIn('["Text Classification datasets"]', prompt)
        self.assertIn('["CLIP architecture", "CLIP training objective"]', prompt)
        self.assertIn('["Naive Bayes vs Logistic Regression"]', prompt)
        self.assertIn("Keep visualization queries focused", prompt)
        self.assertIn("Resolve pronouns from recent conversation history", prompt)


if __name__ == "__main__":
    unittest.main()
