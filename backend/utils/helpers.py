import json
import re
from typing import Any

from backend.configs.constants import WORKFLOW_LOGGING
from backend.utils.common_funcs import get_logger

logger = get_logger(WORKFLOW_LOGGING)


def clean_json_markdown(text: str) -> str:
    """Removes Markdown code blocks from JSON text.
    
    Args:
        text: Input text potentially wrapped in Markdown code blocks.
        
    Returns:
        Cleaned text without Markdown code block markers.
    """
    # Remove `json at start and ``` at end, handling newlines
    pattern = r"^`{3}(?:json)?\s*(.*?)\s*`{3}$"
    match = re.search(pattern, text.strip(), re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()


def make_json_serializable(node: Any, attribute: str = "property") -> Any:
    """Converts non-serializable values in a node attribute to JSON strings.
    
    Args:
        node: Node object with attributes to serialize.
        attribute: Name of the attribute to make JSON serializable.
        
    Returns:
        Node with serialized attribute values.
    """
    attribute_value = getattr(node, attribute).copy()
    for key, value in attribute_value.items():
        if isinstance(value, (dict, list, bool)) or (
                value is not None and not isinstance(value, (str, int, float, bytes))
        ):
            attribute_value[key] = json.dumps(value)

    setattr(node, attribute, attribute_value)
    return node
