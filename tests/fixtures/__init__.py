"""
Test fixtures for realistic integration testing.

This package provides:
- Realistic LLM response fixtures (varied formats)
- DuckDuckGo search result fixtures
- Job posting fixtures
- Fixture loader utilities

Usage:
    from tests.fixtures import FixtureLoader

    loader = FixtureLoader()
    llm_response = loader.get_llm_response("gatherer_analysis", variant="markdown_wrapped")
    search_results = loader.get_search_results("acme_corp")
"""
from tests.fixtures.loader import FixtureLoader

__all__ = ["FixtureLoader"]
