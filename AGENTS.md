# AGENTS.md

Guidance for AI coding agents working in this repository.

## General Code Style

- Prefer the existing project patterns over new abstractions.
- Keep changes closely scoped to the task.
- Avoid speculative helpers, compatibility layers, or configuration unless the current code path needs them.
- Use clear names and straightforward control flow before adding comments.
- Add comments only when they explain intent, boundary conditions, or non-obvious tradeoffs.

## Docstrings

Write docstrings when they help a future maintainer understand the role or contract of code. Do not add a docstring to every small helper by default.

Preferred style:

- Use concise triple-quoted docstrings.
- Start with a short summary sentence.
- Add a second paragraph only when it explains why the function exists or how it fits into the pipeline.
- Use `Args:` and `Returns:` when the argument contract is not obvious.
- Keep docstrings accurate after refactors. An outdated docstring is worse than no docstring.

Examples:

```python
class PaddleOCRPipeline:
    """Pipeline for OCR processing of PDFs and images using PaddleOCR."""

    def parse_pdf(self, input_file: str | Path) -> Path | None:
        """Parses a PDF file and converts it to Markdown format.

        Args:
            input_file: Path to the input PDF file.

        Returns:
            Path to the generated markdown file, or None if the PDF has too many pages.
        """
```

Another existing style example:

```python
class KnowledgeGraphIndexer:
    """Manages knowledge graph indexing, processing, and retrieval operations.

    This class handles the creation, processing, and querying of property graph indexes,
    including implicit graph processing, entity/relation extraction, and vector embeddings.
    """
```

For new optimized-pipeline code, prefer docstrings that explain boundaries and invariants, for example:

```python
def run_postprocess_pass(...) -> PostprocessPassResult:
    """Runs one post-processing pass over selected chunks.

    The pass can be the legacy single-pass runner or one phase of the optimized
    quality/graph pipeline. Successful model responses are cached only after
    schema validation succeeds, which keeps failed or partial generations from
    becoming future replay baselines.
    """
```

## External Library Documentation

Consider using Context7 when code review, refactoring, feature development, or another task depends on the current or version-specific behavior of an external library.

It is especially useful for checking API contracts, return types, configuration options, deprecations, and recommended framework usage. When practical, identify the installed dependency version from `pyproject.toml` or `poetry.lock` and prefer matching documentation.

Use repository evidence, tests, and runtime behavior as the primary source of truth. Context7 should support decisions, not override established project patterns or justify unnecessary changes.
