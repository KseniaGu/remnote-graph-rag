from app.state import _normalize_math_delimiters


def test_normalize_math_delimiters_escapes_pipes_inside_table_math() -> None:
    markdown = (
        "| Component | Role | Typical Formulation |\n"
        "|-----------|------|---------------------|\n"
        "| **Encoder** \\(q_{\\phi}(z|x)\\) | Maps input \\(x\\) | \\(p_{\\theta}(x|z)\\) |\n"
    )

    normalized = _normalize_math_delimiters(markdown)
    row = normalized.splitlines()[2]

    assert "z\\mid x" in row
    assert "x\\mid z" in row
    assert "z|x" not in row
    assert "x|z" not in row
    assert row.count("|") == 4


def test_normalize_math_delimiters_collapses_display_math_inside_table_row() -> None:
    markdown = (
        "| Component | Description | Source |\n"
        "|-----------|-------------|--------|\n"
        "| **Contrastive Objective** | The symmetric loss is: \\[\n"
        "\\mathcal{L}=\\log p(x|z)\n"
        "\\] where the model contrasts pairs. | Internal KG |\n"
    )

    normalized = _normalize_math_delimiters(markdown)
    lines = normalized.splitlines()

    assert len(lines) == 3
    assert lines[2].startswith("| **Contrastive Objective** |")
    assert "$\\mathcal{L}=\\log p(x\\mid z)$" in lines[2]
    assert "$$" not in lines[2]
    assert lines[2].count("|") == 4
