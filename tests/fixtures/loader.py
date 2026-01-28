"""
Fixture loader utility for test fixtures.
Provides easy access to realistic test data with various formats.
"""
import json
from pathlib import Path
from typing import Any


class FixtureLoader:
    """
    Load test fixtures from the fixtures directory.

    Supports loading:
    - LLM responses (with format variants: clean, markdown, extra_text)
    - Search results (DuckDuckGo format)
    - Job postings (various HTML structures)
    """

    def __init__(self):
        self.fixtures_dir = Path(__file__).parent
        self.llm_dir = self.fixtures_dir / "llm_responses"
        self.search_dir = self.fixtures_dir / "search_results"
        self.jobs_dir = self.fixtures_dir / "job_postings"

    def get_llm_response(self, fixture_name: str, variant: str = "clean") -> str:
        """
        Get an LLM response fixture.

        Args:
            fixture_name: Name of the fixture (e.g., "gatherer_analysis", "identifier_requirements")
            variant: Response format variant:
                - "clean": Perfect JSON as expected
                - "markdown": JSON wrapped in markdown code fences
                - "extra_text": JSON with explanatory text before/after
                - "whitespace": JSON with extra whitespace/newlines
                - "partial": Malformed JSON (for error handling tests)

        Returns:
            LLM response string (may or may not be valid JSON)
        """
        fixture_path = self.llm_dir / f"{fixture_name}.json"

        if not fixture_path.exists():
            raise FileNotFoundError(f"LLM fixture not found: {fixture_path}")

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Get the variant
        if variant not in data.get("variants", {}):
            raise ValueError(f"Variant '{variant}' not found in fixture '{fixture_name}'")

        return data["variants"][variant]

    def get_llm_response_parsed(self, fixture_name: str) -> dict[str, Any]:
        """
        Get the expected parsed data from an LLM fixture.
        This is the "ground truth" for what the JSON should parse to.

        Args:
            fixture_name: Name of the fixture

        Returns:
            Dict of expected parsed data
        """
        fixture_path = self.llm_dir / f"{fixture_name}.json"

        if not fixture_path.exists():
            raise FileNotFoundError(f"LLM fixture not found: {fixture_path}")

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("expected_parsed", {})

    def get_search_results(self, fixture_name: str) -> list[dict[str, Any]]:
        """
        Get search result fixtures.

        Args:
            fixture_name: Name of the fixture (e.g., "acme_corp", "tech_startup")

        Returns:
            List of search result dicts matching SearchResult model
        """
        fixture_path = self.search_dir / f"{fixture_name}.json"

        if not fixture_path.exists():
            raise FileNotFoundError(f"Search fixture not found: {fixture_path}")

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("results", [])

    def get_search_results_raw(self, fixture_name: str) -> str:
        """
        Get raw MCP response format for search results.
        This simulates what the MCP client actually receives.

        Args:
            fixture_name: Name of the fixture

        Returns:
            Raw text response as MCP would return it
        """
        fixture_path = self.search_dir / f"{fixture_name}.json"

        if not fixture_path.exists():
            raise FileNotFoundError(f"Search fixture not found: {fixture_path}")

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("raw_mcp_response", "")

    def get_job_postings(self, fixture_name: str) -> list[dict[str, Any]]:
        """
        Get job posting fixtures.

        Args:
            fixture_name: Name of the fixture (e.g., "greenhouse", "lever", "generic")

        Returns:
            List of job posting dicts matching JobPosting model
        """
        fixture_path = self.jobs_dir / f"{fixture_name}.json"

        if not fixture_path.exists():
            raise FileNotFoundError(f"Job posting fixture not found: {fixture_path}")

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("job_postings", [])

    def get_job_postings_html(self, fixture_name: str) -> str:
        """
        Get raw HTML for job posting pages (for scraper testing).

        Args:
            fixture_name: Name of the fixture

        Returns:
            HTML string of career page
        """
        fixture_path = self.jobs_dir / f"{fixture_name}.json"

        if not fixture_path.exists():
            raise FileNotFoundError(f"Job posting fixture not found: {fixture_path}")

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("html", "")

    def get_news_items(self, fixture_name: str) -> list[dict[str, Any]]:
        """
        Get news item fixtures.

        Args:
            fixture_name: Name of the fixture

        Returns:
            List of news item dicts matching NewsItem model
        """
        fixture_path = self.search_dir / f"{fixture_name}.json"

        if not fixture_path.exists():
            raise FileNotFoundError(f"News fixture not found: {fixture_path}")

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("news_items", [])

    def list_fixtures(self, category: str) -> list[str]:
        """
        List available fixtures in a category.

        Args:
            category: "llm", "search", or "jobs"

        Returns:
            List of fixture names (without .json extension)
        """
        dir_map = {
            "llm": self.llm_dir,
            "search": self.search_dir,
            "jobs": self.jobs_dir
        }

        if category not in dir_map:
            raise ValueError(f"Unknown category: {category}")

        target_dir = dir_map[category]
        if not target_dir.exists():
            return []

        return [f.stem for f in target_dir.glob("*.json")]


def extract_json_from_llm_response(response: str) -> dict[str, Any]:
    """
    Extract JSON from an LLM response that may have extra text or formatting.

    This is a utility function that can be used to make the actual code more robust.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON dict

    Raises:
        json.JSONDecodeError: If no valid JSON found
    """
    import re

    # Try direct parsing first
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    code_fence_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_fence_pattern, response)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try finding JSON object/array patterns
    json_patterns = [
        r'\{[\s\S]*\}',  # Object
        r'\[[\s\S]*\]'   # Array
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Last resort: raise error
    raise json.JSONDecodeError("No valid JSON found in response", response, 0)
