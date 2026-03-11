"""
One-time script to scrape MathWorks solutions pages and index into ChromaDB.

This enriches the product catalog with industry/application context from
mathworks.com/solutions.html and all linked solution pages (~76 URLs).

Usage:
    python scripts/index_mathworks_solutions.py

Run once before using the agent system. Safe to re-run (idempotent — skips
if >= 50 solution docs already exist in the collection).

After running, the ProductMatcher will automatically use solution docs to
boost hybrid BM25+vector recall without any additional configuration.
"""
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_sources.product_catalog import ProductCatalogIndexer


async def main() -> None:
    print("=" * 60)
    print("  MathWorks Solutions Indexer")
    print("=" * 60)
    print()

    indexer = ProductCatalogIndexer(company_name="MathWorks")

    # Ensure product catalog is indexed first
    print("  Step 1/2 — Building product catalog...")
    products = await indexer.build_catalog()
    await indexer.index_products(products)
    print(f"  Product catalog: {len(products)} products indexed")

    # Fetch and index solution pages via Tavily Extract
    print()
    print("  Step 2/2 — Fetching MathWorks solution pages via Tavily Extract...")
    print("  (batched at 20 URLs/call — typically completes in seconds)")
    print()

    count = await indexer.scrape_and_index_solutions()

    total = indexer.collection.count()
    print()
    print("=" * 60)
    if count is None:
        print("  Solutions already indexed — skipping.")
        print("  (delete solution docs from ChromaDB to force re-index)")
    elif count == 0:
        print("  WARNING: 0 solution pages indexed.")
        print("  Check logs above — TAVILY_API_KEY may be missing or all URLs failed.")
    else:
        print(f"  Indexed {count} solution pages.")
    print(f"  Total documents in collection: {total}")
    print("  (products + solution enrichment docs)")
    print()
    print("  ProductMatcher will now use hybrid BM25+vector search")
    print("  with solution context for improved product recall.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
