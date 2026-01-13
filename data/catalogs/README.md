# Product Catalogs

This directory contains product catalog JSON files for different companies.

## Creating a Product Catalog

To use the system with your company's products, create a JSON file with the following structure:

```json
[
  {
    "name": "Product Name",
    "category": "Product Category",
    "description": "Brief description of what the product does",
    "key_features": ["feature 1", "feature 2", "feature 3"],
    "use_cases": ["use case 1", "use case 2"],
    "target_personas": ["persona 1", "persona 2"]
  }
]
```

### Field Descriptions

- **name** (required): The product or service name
- **category** (required): Product category (e.g., "Software", "Cloud Services", "Hardware")
- **description** (required): Clear, concise description of the product
- **key_features** (required): List of main features/capabilities
- **use_cases** (required): List of common use cases or applications
- **target_personas** (required): List of target customer personas (job titles, roles)

## Usage Example

```python
from src.data_sources.product_catalog import ProductCatalogIndexer, ProductMatcher

# Option 1: Load from JSON file
indexer = ProductCatalogIndexer(
    company_name="YourCompany",
    catalog_file="./data/catalogs/your_company_catalog.json"
)

# Option 2: Provide fallback products directly
fallback_products = [
    {
        "name": "Product A",
        "category": "Software",
        "description": "Amazing product",
        "key_features": ["fast", "secure"],
        "use_cases": ["data processing"],
        "target_personas": ["data engineers"]
    }
]

indexer = ProductCatalogIndexer(company_name="YourCompany")
products = await indexer.build_catalog(fallback_products=fallback_products)

# Index products for semantic search
await indexer.index_products(products)

# Match requirements to products
matcher = ProductMatcher(company_name="YourCompany")
matches = await matcher.match_requirements_to_products(
    requirements=["real-time data processing", "cloud deployment"],
    top_k=5
)

for product_name, confidence in matches:
    print(f"{product_name}: {confidence:.2f}")
```

## Built-in Catalogs

The following companies have built-in default catalogs:
- **MathWorks**: 20+ products including MATLAB, Simulink, and toolboxes

For other companies, provide a JSON catalog file or fallback products.

## Best Practices

1. **Be specific**: Detailed descriptions and features improve matching accuracy
2. **Use customer language**: Include terms and phrases customers actually use
3. **Cover personas**: List all relevant job titles and roles
4. **Keep updated**: Regularly update your catalog as products evolve
5. **Test matching**: Run test queries to verify semantic matching quality

## Example Catalogs

- `mathworks_catalog.json` - Sample MathWorks products (subset)
- Create your own: `your_company_catalog.json`
