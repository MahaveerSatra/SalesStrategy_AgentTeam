"""
One-time script to scrape MathWorks product pages and index into ChromaDB.

Enriches the product catalog with rich per-product content so BM25/vector
search can bridge domain abbreviations (e.g., "GNC" → "Guidance, Navigation
and Control") to the correct toolboxes.

Usage:
    python scripts/index_mathworks_products.py

Run once before using the agent system. Safe to re-run (idempotent — skips
if >= 50 product_page docs already exist in the collection).
"""
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_sources.product_catalog import ProductCatalogIndexer


async def main() -> None:
    print("=" * 60)
    print("  MathWorks Product Page Indexer")
    print("=" * 60)
    print()

    indexer = ProductCatalogIndexer(company_name="MathWorks")

    # Ensure product catalog is indexed first
    print("  Step 1/2 — Building product catalog...")
    products = await indexer.build_catalog()
    await indexer.index_products(products)
    print(f"  Product catalog: {len(products)} products indexed")

    # Fetch and index individual product pages via Tavily Extract
    print()
    print("  Step 2/2 — Fetching MathWorks product pages via Tavily Extract...")
    print("  (Step 1: discover URLs from products.html)")
    print("  (Step 2: batch-scrape each product page at 20 URLs/call)")
    print()

    count = await indexer.scrape_and_index_product_pages()

    total = indexer.collection.count()
    print()
    print("=" * 60)
    if count is None:
        print("  Product pages already indexed — skipping.")
        print("  (delete product_page docs from ChromaDB to force re-index)")
    elif count == 0:
        print("  WARNING: 0 product pages indexed.")
        print("  Check logs above — TAVILY_API_KEY may be missing or all URLs failed.")
    else:
        print(f"  Indexed {count} product pages.")
    print(f"  Total documents in collection: {total}")
    print("  (products + solution enrichment + product page content)")
    print()
    print("  ProductMatcher will now use rich product page content for")
    print("  improved domain abbreviation matching (e.g., GNC, MBSE, HIL).")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
