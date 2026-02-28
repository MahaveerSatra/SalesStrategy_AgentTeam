"""
Live integration test for the 2-tier SearchClient (DDG MCP -> Tavily fallback).
Tests web search, news search, and job board scraping.
Not a pytest file -- run directly: python tests/manual_test_search_client.py
"""
import asyncio
import sys
import io
from pathlib import Path

# Force UTF-8 output so Unicode in result titles does not crash on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.data_sources.search_client import SearchClient
from src.data_sources.job_boards import JobBoardScraper


def print_section(title: str):
    print()
    print("-" * 65)
    print(f"  {title}")
    print("-" * 65)


def print_result(idx: int, title: str, url: str, source: str, extra: str = ""):
    print(f"  {idx}. [{source}] {title[:60]}")
    print(f"       {str(url)[:70]}")
    if extra:
        print(f"       {extra}")


async def test_web_search(client: SearchClient, query: str):
    print(f"\n  Query: \"{query}\"")
    results = await client.search(query, max_results=5)
    if results:
        for i, r in enumerate(results, 1):
            print_result(i, r.title, r.url, r.source)
    else:
        print("  [!] No results returned")
    return len(results)


async def test_news_search(client: SearchClient, query: str):
    print(f"\n  Query: \"{query}\"")
    news = await client.search_news(query, max_results=5)
    if news:
        for i, n in enumerate(news, 1):
            print_result(i, n.title, n.url, n.source, f"score={n.relevance_score:.2f}")
    else:
        print("  [!] No news returned")
    return len(news)


async def test_jobs(company: str, domain: str):
    scraper = JobBoardScraper()
    print(f"\n  Company: {company}  Domain: {domain}")
    jobs = await scraper.fetch(company_name=company, company_domain=domain)
    if jobs:
        for i, j in enumerate(jobs[:5], 1):
            techs = ", ".join(j.technologies[:3]) if j.technologies else "--"
            print(f"  {i}. {j.title[:55]}  [{j.location or 'Remote'}]")
            print(f"       tech: {techs}")
    else:
        print("  [!] No job postings found")
    return len(jobs)


async def main():
    print_section("2-TIER SEARCH CLIENT -- LIVE INTEGRATION TEST")

    async with SearchClient() as client:

        # Boeing (large company, DDG often blocked -> Tavily fallback)
        print_section("BOEING -- Web Search")
        c1 = await test_web_search(client, "Boeing AI machine learning hiring 2025")
        c2 = await test_web_search(client, "Boeing digital transformation technology")

        print_section("BOEING -- News Search")
        n1 = await test_news_search(client, "Boeing technology investment announcement")
        n2 = await test_news_search(client, "Boeing aerospace engineering news 2025")

        print_section("BOEING -- Job Postings (JobBoardScraper)")
        j1 = await test_jobs("Boeing", "boeing.com")

        # Remora Carbon (niche company, DDG works reliably)
        print_section("REMORA CARBON -- Web Search")
        c3 = await test_web_search(client, "Remora Carbon technology carbon capture")
        c4 = await test_web_search(client, "Remora Carbon engineering hiring")

        print_section("REMORA CARBON -- News Search")
        n3 = await test_news_search(client, "Remora Carbon news funding")

        print_section("REMORA CARBON -- Job Postings (JobBoardScraper)")
        j2 = await test_jobs("Remora Carbon", "remoracarbon.com")

    print_section("SUMMARY")
    print(f"  Boeing  web search   : {c1 + c2} results across 2 queries")
    print(f"  Boeing  news search  : {n1 + n2} results across 2 queries")
    print(f"  Boeing  job postings : {j1} postings")
    print(f"  Remora  web search   : {c3 + c4} results across 2 queries")
    print(f"  Remora  news search  : {n3} results")
    print(f"  Remora  job postings : {j2} postings")
    print()
    total = c1 + c2 + n1 + n2 + c3 + c4 + n3
    if total > 0:
        print(f"  [OK] {total} total results -- search pipeline operational")
    else:
        print("  [!!] Zero results across all queries -- check API keys / MCP server")


if __name__ == "__main__":
    asyncio.run(main())
