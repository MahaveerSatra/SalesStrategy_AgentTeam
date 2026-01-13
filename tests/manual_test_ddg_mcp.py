"""Quick manual test for MCP connection - not a pytest file"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_sources.mcp_ddg_client import DuckDuckGoMCPClient


async def main():
    print("Testing MCP DuckDuckGo connection...")

    try:
        async with DuckDuckGoMCPClient() as client:
            # Test search
            print("\n1. Testing search...")
            results = await client.search("MathWorks careers", max_results=3)
            print(f"[OK] Search: Found {len(results)} results")
            if results:
                print(f"  First result: {results[0].title}")
                print(f"  URL: {results[0].url}")

            # Test fetch_content
            if results:
                print("\n2. Testing fetch_content...")
                content = await client.fetch_content(str(results[0].url))
                print(f"[OK] Fetch: Retrieved {len(content)} characters")

            # Test caching
            print("\n3. Testing cache...")
            cached_results = await client.search("MathWorks careers", max_results=3)
            print(f"[OK] Cache: Second search returned {len(cached_results)} results")

            # Test metrics
            print("\n4. Testing metrics...")
            metrics = client.get_metrics()
            print(f"[OK] Metrics:")
            print(f"  Requests: {metrics['request_count']}")
            print(f"  Errors: {metrics['error_count']}")
            print(f"  Cache hit rate: {metrics['cache_stats']['hit_rate']:.2%}")
            print(f"  Avg latency: {metrics['avg_latency_ms']:.0f}ms")

        print("\n[OK] All tests passed!")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
