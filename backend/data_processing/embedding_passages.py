from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DEFAULT_TARGET_TOKENS = 220
DEFAULT_OVERLAP_TOKENS = 40
MIN_BODY_TOKENS = 80
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)]\((?:[^)]*)\)")
RAW_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


@dataclass(frozen=True)
class EmbeddingPassage:
    """Embedding-sized child record for a stable parent retrieval chunk."""

    passage_id: str
    parent_chunk_id: str
    passage_index: int
    text: str
    split_strategy: str
    token_count: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class _TextUnit:
    text: str
    start: int
    end: int
    strategy: str


TokenCounter = Callable[[str], int]


def build_embedding_passages(
    *,
    parent_chunk_id: str,
    text: str,
    source_path: str = "",
    summary: str = "",
    token_counter: TokenCounter | None = None,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[EmbeddingPassage]:
    """Splits one parent chunk into semantic, token-budgeted embedding passages.

    Parent chunks remain the evidence units. These child passages are only the
    vector-search representation and are mapped back to the parent chunk at
    retrieval time.
    """

    clean_text = _clean_embedding_text(str(text or "").strip())
    if not clean_text:
        return []

    count_tokens = token_counter or approximate_token_count
    prefix = _passage_prefix(source_path=source_path, summary=summary)
    prefix_tokens = count_tokens(prefix) if prefix else 0
    body_budget = max(MIN_BODY_TOKENS, target_tokens - prefix_tokens)
    overlap_budget = max(0, min(overlap_tokens, body_budget // 3))

    units = _semantic_units(clean_text)
    normalized_units: list[_TextUnit] = []
    for unit in units:
        normalized_units.extend(_split_oversized_unit(unit, body_budget, count_tokens))

    passages: list[EmbeddingPassage] = []
    current: list[_TextUnit] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        _append_passage(
            passages,
            parent_chunk_id=parent_chunk_id,
            units=current,
            prefix=prefix,
            count_tokens=count_tokens,
        )
        current = _overlap_tail(current, overlap_budget, count_tokens)

    for unit in normalized_units:
        candidate_units = [*current, unit]
        if current and _units_token_count(candidate_units, count_tokens) > body_budget:
            flush()
            candidate_units = [*current, unit]
        if (
            not current
            or _units_token_count(candidate_units, count_tokens) <= body_budget
        ):
            current.append(unit)
        else:
            _append_passage(
                passages,
                parent_chunk_id=parent_chunk_id,
                units=[unit],
                prefix=prefix,
                count_tokens=count_tokens,
            )
            current = []

    if current:
        _append_passage(
            passages,
            parent_chunk_id=parent_chunk_id,
            units=current,
            prefix=prefix,
            count_tokens=count_tokens,
        )

    return passages


def tokenizer_token_counter(tokenizer: Any | None) -> TokenCounter:
    """Creates a token counter from a HuggingFace-style tokenizer."""

    if tokenizer is None:
        return approximate_token_count

    def count(text: str) -> int:
        if not text:
            return 0
        try:
            encoded = tokenizer(text, add_special_tokens=True, truncation=False)
            input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
            if input_ids is not None:
                return len(input_ids)
        except TypeError:
            pass
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                return len(encode(text, add_special_tokens=True, truncation=False))
            except TypeError:
                return len(encode(text))
        return approximate_token_count(text)

    return count


def approximate_token_count(text: str) -> int:
    """Cheap fallback token estimate used in unit tests and non-HF contexts."""

    if not text:
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)))


def _passage_prefix(*, source_path: str, summary: str) -> str:
    del summary
    return _clean_source_path(source_path)


def _clean_source_path(source_path: str) -> str:
    path = _clean_embedding_text(source_path)
    parts = []
    for part in path.split(">"):
        cleaned = re.sub(r"^\s*#{1,6}\s*", "", part).strip()
        if not cleaned or cleaned.startswith("external:"):
            continue
        parts.append(cleaned)
    return " > ".join(parts)


def _clean_embedding_text(text: str) -> str:
    text = MARKDOWN_LINK_PATTERN.sub(r"\1", str(text or ""))
    text = RAW_URL_PATTERN.sub("", text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _semantic_units(text: str) -> list[_TextUnit]:
    units: list[_TextUnit] = []
    for paragraph_match in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text):
        paragraph = paragraph_match.group(0).strip()
        if not paragraph:
            continue
        paragraph_start = (
            paragraph_match.start()
            + len(paragraph_match.group(0))
            - len(paragraph_match.group(0).lstrip())
        )
        paragraph_end = paragraph_start + len(paragraph)

        line_units = _list_or_line_units(paragraph, paragraph_start)
        if line_units:
            units.extend(line_units)
            continue

        units.extend(_sentence_units(paragraph, paragraph_start, paragraph_end))
    return units


def _list_or_line_units(paragraph: str, offset: int) -> list[_TextUnit]:
    lines = paragraph.splitlines()
    nonempty = [line for line in lines if line.strip()]
    if len(nonempty) <= 1 or not any(_looks_like_list_item(line) for line in nonempty):
        return []

    units: list[_TextUnit] = []
    cursor = 0
    for line in lines:
        line_start = paragraph.find(line, cursor)
        cursor = line_start + len(line)
        stripped = line.strip()
        if not stripped:
            continue
        start = offset + line_start + len(line) - len(line.lstrip())
        units.append(_TextUnit(stripped, start, start + len(stripped), "list_item"))
    return units


def _sentence_units(paragraph: str, start: int, end: int) -> list[_TextUnit]:
    sentence_pattern = re.compile(r"[^.!?\n]+(?:[.!?](?=\s|$)|$)", re.MULTILINE)
    units: list[_TextUnit] = []
    for match in sentence_pattern.finditer(paragraph):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        sentence_start = (
            start + match.start() + len(match.group(0)) - len(match.group(0).lstrip())
        )
        units.append(
            _TextUnit(
                sentence, sentence_start, sentence_start + len(sentence), "sentence"
            )
        )
    if units:
        return units
    return [_TextUnit(paragraph.strip(), start, end, "paragraph")]


def _split_oversized_unit(
    unit: _TextUnit, body_budget: int, count_tokens: TokenCounter
) -> list[_TextUnit]:
    if count_tokens(unit.text) <= body_budget:
        return [unit]

    words = list(re.finditer(r"\S+", unit.text))
    if not words:
        return [unit]

    chunks: list[_TextUnit] = []
    current_words: list[re.Match[str]] = []

    def flush() -> None:
        nonlocal current_words
        if not current_words:
            return
        text = " ".join(match.group(0) for match in current_words)
        chunks.append(
            _TextUnit(
                text=text,
                start=unit.start + current_words[0].start(),
                end=unit.start + current_words[-1].end(),
                strategy="hard_token_fallback",
            )
        )
        current_words = []

    for word in words:
        candidate = " ".join(
            [*(match.group(0) for match in current_words), word.group(0)]
        )
        if current_words and count_tokens(candidate) > body_budget:
            flush()
        current_words.append(word)
    flush()
    return chunks or [unit]


def _append_passage(
    passages: list[EmbeddingPassage],
    *,
    parent_chunk_id: str,
    units: list[_TextUnit],
    prefix: str,
    count_tokens: TokenCounter,
) -> None:
    body = "\n".join(_clean_embedding_text(unit.text) for unit in units).strip()
    if not body:
        return
    text = f"{prefix}\n\n{body}" if prefix else body
    strategies = {unit.strategy for unit in units}
    strategy = (
        "hard_token_fallback"
        if "hard_token_fallback" in strategies
        else units[0].strategy
    )
    passage_index = len(passages)
    passages.append(
        EmbeddingPassage(
            passage_id=f"{parent_chunk_id}::passage_{passage_index:03d}",
            parent_chunk_id=parent_chunk_id,
            passage_index=passage_index,
            text=text,
            split_strategy=strategy,
            token_count=count_tokens(text),
            char_start=min(unit.start for unit in units),
            char_end=max(unit.end for unit in units),
        )
    )


def _overlap_tail(
    units: list[_TextUnit], overlap_budget: int, count_tokens: TokenCounter
) -> list[_TextUnit]:
    if overlap_budget <= 0:
        return []
    tail: list[_TextUnit] = []
    for unit in reversed(units):
        candidate = [unit, *tail]
        if _units_token_count(candidate, count_tokens) > overlap_budget:
            break
        tail.insert(0, unit)
    return tail


def _units_token_count(units: list[_TextUnit], count_tokens: TokenCounter) -> int:
    return count_tokens("\n".join(unit.text for unit in units))


def _looks_like_list_item(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line))
