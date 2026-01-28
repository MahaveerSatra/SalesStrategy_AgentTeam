"""
Tests for the robust JSON parsing utility module.

These tests verify that extract_json_from_llm_response can handle
various LLM output formats including markdown, extra text, and whitespace.
"""
import pytest

from src.utils.json_parsing import (
    extract_json_from_llm_response,
    extract_json_with_default,
    safe_get_field,
    JSONParseError,
)


class TestExtractJsonFromLlmResponse:
    """Tests for the main JSON extraction function."""

    def test_clean_json_object(self):
        """Clean JSON object parses directly."""
        response = '{"key": "value", "number": 42}'
        result = extract_json_from_llm_response(response)
        assert result == {"key": "value", "number": 42}

    def test_clean_json_array(self):
        """Clean JSON array parses directly."""
        response = '[1, 2, 3, "four"]'
        result = extract_json_from_llm_response(response)
        assert result == [1, 2, 3, "four"]

    def test_json_with_leading_whitespace(self):
        """JSON with leading whitespace parses correctly."""
        response = '   \n\n  {"key": "value"}'
        result = extract_json_from_llm_response(response)
        assert result == {"key": "value"}

    def test_json_with_trailing_whitespace(self):
        """JSON with trailing whitespace parses correctly."""
        response = '{"key": "value"}   \n\n  '
        result = extract_json_from_llm_response(response)
        assert result == {"key": "value"}

    def test_json_in_markdown_code_fence(self):
        """JSON wrapped in markdown code fence parses correctly."""
        response = '''```json
{"key": "value", "nested": {"a": 1}}
```'''
        result = extract_json_from_llm_response(response)
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_json_in_markdown_code_fence_no_language(self):
        """JSON wrapped in code fence without language hint parses correctly."""
        response = '''```
{"key": "value"}
```'''
        result = extract_json_from_llm_response(response)
        assert result == {"key": "value"}

    def test_json_with_text_before(self):
        """JSON with explanatory text before it parses correctly."""
        response = '''Here is the analysis result:

{"requirements": ["need A", "need B"], "confidence": 0.8}'''
        result = extract_json_from_llm_response(response)
        assert result == {"requirements": ["need A", "need B"], "confidence": 0.8}

    def test_json_with_text_after(self):
        """JSON with explanatory text after it parses correctly."""
        response = '''{"requirements": ["need A", "need B"]}

I hope this helps with your analysis!'''
        result = extract_json_from_llm_response(response)
        assert result == {"requirements": ["need A", "need B"]}

    def test_json_with_text_before_and_after(self):
        """JSON with text both before and after parses correctly."""
        response = '''Based on my analysis, here is the result:

{"status": "success", "items": [1, 2, 3]}

Let me know if you need anything else!'''
        result = extract_json_from_llm_response(response)
        assert result == {"status": "success", "items": [1, 2, 3]}

    def test_json_in_markdown_with_surrounding_text(self):
        """JSON in markdown with surrounding explanation parses correctly."""
        response = '''I've analyzed the data. Here are the results:

```json
{
    "opportunities": [
        {"product": "Widget A", "score": 0.9},
        {"product": "Widget B", "score": 0.7}
    ]
}
```

These opportunities look promising!'''
        result = extract_json_from_llm_response(response)
        assert result["opportunities"][0]["product"] == "Widget A"
        assert len(result["opportunities"]) == 2

    def test_complex_nested_json(self):
        """Complex nested JSON structures parse correctly."""
        response = '''```json
{
    "analysis": {
        "company": "Acme Corp",
        "signals": [
            {"type": "hiring", "confidence": 0.9},
            {"type": "tech", "confidence": 0.8}
        ]
    },
    "metadata": {
        "version": 1,
        "tags": ["enterprise", "saas"]
    }
}
```'''
        result = extract_json_from_llm_response(response)
        assert result["analysis"]["company"] == "Acme Corp"
        assert len(result["analysis"]["signals"]) == 2
        assert result["metadata"]["tags"] == ["enterprise", "saas"]

    def test_json_array_in_markdown(self):
        """JSON array in markdown parses correctly."""
        response = '''Here are the requirements:
```json
["requirement 1", "requirement 2", "requirement 3"]
```'''
        result = extract_json_from_llm_response(response)
        assert result == ["requirement 1", "requirement 2", "requirement 3"]

    def test_empty_response_raises_error(self):
        """Empty response raises JSONParseError."""
        with pytest.raises(JSONParseError) as exc_info:
            extract_json_from_llm_response("")
        assert "Empty response" in str(exc_info.value)

    def test_whitespace_only_raises_error(self):
        """Whitespace-only response raises JSONParseError."""
        with pytest.raises(JSONParseError) as exc_info:
            extract_json_from_llm_response("   \n\t  ")
        assert "Empty response" in str(exc_info.value)

    def test_no_json_raises_error(self):
        """Response with no JSON raises JSONParseError."""
        with pytest.raises(JSONParseError) as exc_info:
            extract_json_from_llm_response("This is just plain text with no JSON at all.")
        assert "No valid JSON found" in str(exc_info.value)

    def test_malformed_json_raises_error(self):
        """Malformed JSON raises JSONParseError."""
        with pytest.raises(JSONParseError) as exc_info:
            extract_json_from_llm_response('{"key": "value", missing_quotes: bad}')
        assert "No valid JSON found" in str(exc_info.value)

    def test_truncated_json_raises_error(self):
        """Truncated JSON raises JSONParseError."""
        with pytest.raises(JSONParseError):
            extract_json_from_llm_response('{"key": "value", "nested": {"incomplete":')

    def test_json_parse_error_contains_raw_response(self):
        """JSONParseError includes the raw response for debugging."""
        raw = "invalid json content here"
        with pytest.raises(JSONParseError) as exc_info:
            extract_json_from_llm_response(raw)
        assert exc_info.value.raw_response == raw

    def test_json_with_unicode(self):
        """JSON with unicode characters parses correctly."""
        response = '{"message": "Hello 世界! 🎉", "emoji": "✅"}'
        result = extract_json_from_llm_response(response)
        assert result["message"] == "Hello 世界! 🎉"
        assert result["emoji"] == "✅"

    def test_json_with_escaped_characters(self):
        """JSON with escaped characters parses correctly."""
        response = '{"path": "C:\\\\Users\\\\test", "quote": "He said \\"hello\\""}'
        result = extract_json_from_llm_response(response)
        assert result["path"] == "C:\\Users\\test"
        assert result["quote"] == 'He said "hello"'


class TestExtractJsonWithDefault:
    """Tests for the default-returning JSON extraction function."""

    def test_valid_json_returns_parsed(self):
        """Valid JSON returns parsed result."""
        result = extract_json_with_default('{"key": "value"}', {"default": True})
        assert result == {"key": "value"}

    def test_invalid_json_returns_default(self):
        """Invalid JSON returns default value."""
        result = extract_json_with_default("not json", {"default": True})
        assert result == {"default": True}

    def test_empty_returns_default(self):
        """Empty response returns default value."""
        result = extract_json_with_default("", {"status": "failed"})
        assert result == {"status": "failed"}

    def test_default_can_be_list(self):
        """Default value can be a list."""
        result = extract_json_with_default("invalid", [])
        assert result == []


class TestSafeGetField:
    """Tests for the safe field extraction function."""

    def test_existing_field(self):
        """Existing field is returned."""
        data = {"name": "test", "count": 5}
        assert safe_get_field(data, "name") == "test"
        assert safe_get_field(data, "count") == 5

    def test_missing_field_returns_default(self):
        """Missing field returns default value."""
        data = {"name": "test"}
        assert safe_get_field(data, "missing") is None
        assert safe_get_field(data, "missing", "default") == "default"

    def test_type_validation_passes(self):
        """Field with correct type is returned."""
        data = {"count": 5, "name": "test"}
        assert safe_get_field(data, "count", 0, int) == 5
        assert safe_get_field(data, "name", "", str) == "test"

    def test_type_validation_fails_returns_default(self):
        """Field with wrong type returns default."""
        data = {"count": "not a number", "items": "not a list"}
        assert safe_get_field(data, "count", 0, int) == 0
        assert safe_get_field(data, "items", [], list) == []

    def test_none_value_with_type_check(self):
        """None value with type check returns default."""
        data = {"value": None}
        # None is not an int, but it matches the default
        assert safe_get_field(data, "value", 0, int) == 0

    def test_list_type_validation(self):
        """List type validation works correctly."""
        data = {"items": [1, 2, 3]}
        assert safe_get_field(data, "items", [], list) == [1, 2, 3]

    def test_dict_type_validation(self):
        """Dict type validation works correctly."""
        data = {"config": {"nested": True}}
        assert safe_get_field(data, "config", {}, dict) == {"nested": True}


class TestRealWorldLlmResponses:
    """Tests with realistic LLM response patterns."""

    def test_gatherer_analysis_response(self):
        """Typical GathererAgent analysis response parses correctly."""
        response = '''Based on my analysis of the webpage, here is the structured assessment:

```json
{
    "confidence": 0.85,
    "summary": "Acme Corp is expanding their cloud infrastructure with a focus on Kubernetes.",
    "source_type": "official_company_blog",
    "key_facts": [
        "Migrating to Kubernetes",
        "Hiring 50 engineers",
        "Q2 2024 target"
    ],
    "keywords": ["kubernetes", "cloud", "infrastructure", "hiring"],
    "relevance": "high"
}
```

This appears to be a high-quality source from the company's official blog.'''

        result = extract_json_from_llm_response(response)
        assert result["confidence"] == 0.85
        assert "Kubernetes" in result["summary"]
        assert len(result["key_facts"]) == 3
        assert result["relevance"] == "high"

    def test_identifier_requirements_response(self):
        """Typical IdentifierAgent requirements response parses correctly."""
        response = '''I've analyzed the signals and job postings. Here are the identified requirements:

{
    "requirements": [
        "Need for container orchestration platform",
        "Requirement for CI/CD pipeline automation",
        "Looking for observability and monitoring solution",
        "Data pipeline and ETL tooling needed",
        "API gateway and service mesh requirements"
    ]
}

These requirements are based on their job postings and technology signals.'''

        result = extract_json_from_llm_response(response)
        assert len(result["requirements"]) == 5
        assert "container orchestration" in result["requirements"][0]

    def test_validator_risks_response(self):
        """Typical ValidatorAgent risks response parses correctly."""
        response = '''After analyzing the competitive landscape:

```json
{
    "risks": [
        "Strong competitor presence: Evidence shows active use of Competitor X based on job postings",
        "Budget timing: Recent infrastructure investment may limit new tool purchases",
        "Integration complexity: Legacy Oracle systems may complicate deployment"
    ]
}
```'''

        result = extract_json_from_llm_response(response)
        assert len(result["risks"]) == 3
        assert "competitor" in result["risks"][0].lower()

    def test_coordinator_validation_response(self):
        """Typical CoordinatorAgent validation response parses correctly."""
        response = '''```json
{
    "is_valid": true,
    "errors": [],
    "suggested_corrections": {},
    "concerns": ["Company name is quite generic, might want to verify"]
}
```'''

        result = extract_json_from_llm_response(response)
        assert result["is_valid"] is True
        assert result["errors"] == []
        assert len(result["concerns"]) == 1

    def test_coordinator_feedback_routing_response(self):
        """Typical CoordinatorAgent feedback routing response parses correctly."""
        response = '''Based on the user feedback, here is my classification:

{
    "route": "GATHERER",
    "reasoning": "User wants more information about the company's cloud initiatives",
    "context_for_retry": "Focus on cloud and infrastructure signals, look for AWS/Azure/GCP mentions"
}'''

        result = extract_json_from_llm_response(response)
        assert result["route"] == "GATHERER"
        assert "cloud" in result["reasoning"].lower()
