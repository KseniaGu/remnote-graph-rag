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


def test_normalize_math_delimiters_escapes_hashes_only_inside_math() -> None:
    markdown = (
        "# Training compute\n\n"
        "| Aspect | Impact |\n"
        "|---|---|\n"
        "| FLOPs | For a dense Transformer, training FLOPs is approximately "
        "$6 \\times \\text{#parameters} \\times \\text{#tokens}$. |\n"
    )

    normalized = _normalize_math_delimiters(markdown)

    assert "# Training compute" in normalized
    assert r"\text{\#parameters}" in normalized
    assert r"\text{\#tokens}" in normalized
    assert r"\text{#parameters}" not in normalized


def test_normalize_math_delimiters_repairs_concatenated_table_rows() -> None:
    markdown = (
        "## Summary\n\n"
        "| Feature | Sohl-Dickstein | DDPM"
        " | | :--- | :--- | :---"
        " | | **Primary Goal** | Theory | High-quality synthesis"
        " | | **Noise Type** | Various | Gaussian |"
    )

    normalized = _normalize_math_delimiters(markdown)

    assert normalized.splitlines() == [
        "## Summary",
        "",
        "| Feature | Sohl-Dickstein | DDPM |",
        "| :--- | :--- | :--- |",
        "| **Primary Goal** | Theory | High-quality synthesis |",
        "| **Noise Type** | Various | Gaussian |",
    ]


def test_normalize_math_delimiters_preserves_valid_table_with_empty_cell() -> None:
    markdown = (
        "| A | B | C |\n"
        "|---|---|---|\n"
        "| value | | final |"
    )

    assert _normalize_math_delimiters(markdown) == markdown
