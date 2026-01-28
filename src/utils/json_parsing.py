"""
Robust JSON parsing utilities for LLM responses.

LLMs often return JSON wrapped in markdown code fences, with extra explanatory
text, or with inconsistent whitespace. This module provides utilities to
extract valid JSON from such responses.
"""
import json
import re
from typing import Any


class JSONParseError(Exception):
    """Raised when JSON extraction fails from LLM response."""

    def __init__(self, message: str, raw_response: str):
        self.raw_response = raw_response
        super().__init__(message)


def extract_json_from_llm_response(response: str) -> dict[str, Any] | list[Any]:
    """
    Extract JSON from an LLM response that may have extra text or formatting.

    Handles common LLM output patterns:
    - Clean JSON (direct parsing)
    - JSON wrapped in markdown code fences (```json ... ```)
    - JSON with explanatory text before/after
    - JSON with extra whitespace/newlines

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON as dict or list

    Raises:
        JSONParseError: If no valid JSON found in response

    Examples:
        >>> extract_json_from_llm_response('{"key": "value"}')
        {'key': 'value'}

        >>> extract_json_from_llm_response('```json\\n{"key": "value"}\\n```')
        {'key': 'value'}

        >>> extract_json_from_llm_response('Here is the result: {"key": "value"}')
        {'key': 'value'}
    """
    if not response or not response.strip():
        raise JSONParseError("Empty response provided", response or "")

    # Try direct parsing first (most common case for well-behaved LLMs)
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    # Matches: ```json ... ``` or ``` ... ```
    code_fence_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_fence_pattern, response)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try finding JSON object pattern (handles extra text before/after)
    # Match from first { to last }
    object_match = re.search(r'\{[\s\S]*\}', response)
    if object_match:
        try:
            return json.loads(object_match.group().strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON array pattern
    # Match from first [ to last ]
    array_match = re.search(r'\[[\s\S]*\]', response)
    if array_match:
        try:
            return json.loads(array_match.group().strip())
        except json.JSONDecodeError:
            pass

    # Last resort: try progressively smaller substrings for nested JSON
    # This handles cases where there's trailing text after valid JSON
    for pattern in [r'\{[^{}]*\}', r'\[[^\[\]]*\]']:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Nothing worked
    raise JSONParseError(
        f"No valid JSON found in response. Response preview: {response[:200]}...",
        response
    )


def extract_json_with_default(
    response: str,
    default: dict[str, Any] | list[Any]
) -> dict[str, Any] | list[Any]:
    """
    Extract JSON from LLM response, returning default on failure.

    Use this when you want graceful degradation instead of exceptions.

    Args:
        response: Raw LLM response string
        default: Value to return if JSON extraction fails

    Returns:
        Parsed JSON or default value

    Examples:
        >>> extract_json_with_default('invalid', {"status": "failed"})
        {'status': 'failed'}
    """
    try:
        return extract_json_from_llm_response(response)
    except (JSONParseError, Exception):
        return default


def safe_get_field(
    data: dict[str, Any],
    field: str,
    default: Any = None,
    expected_type: type | None = None
) -> Any:
    """
    Safely extract a field from parsed JSON with type validation.

    Args:
        data: Parsed JSON dict
        field: Field name to extract
        default: Default value if field missing or wrong type
        expected_type: Optional type to validate against

    Returns:
        Field value or default

    Examples:
        >>> safe_get_field({"count": 5}, "count", 0, int)
        5
        >>> safe_get_field({"count": "five"}, "count", 0, int)
        0
    """
    value = data.get(field, default)

    if expected_type is not None and value is not default:
        if not isinstance(value, expected_type):
            return default

    return value
