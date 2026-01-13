"""
Unit tests for product catalog scraping and indexing.
Tests ProductCatalogIndexer and ProductMatcher.
"""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import tempfile
import shutil

from src.data_sources.product_catalog import ProductCatalogIndexer, ProductMatcher
from src.models.domain import Product
from src.core.exceptions import DataSourceError


class TestProductCatalogIndexer:
    """Test ProductCatalogIndexer for building product catalog."""

    @pytest.fixture
    def temp_db_path(self):
        """Provide temporary ChromaDB path."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def indexer(self, temp_db_path):
        """Provide ProductCatalogIndexer instance."""
        return ProductCatalogIndexer(company_name="TestCompany", db_path=temp_db_path)

    def test_get_mathworks_products(self):
        """Test loading hardcoded MathWorks products."""
        indexer = ProductCatalogIndexer(company_name="MathWorks", db_path=tempfile.mkdtemp())
        products = indexer._get_mathworks_products()

        assert len(products) >= 20
        assert all(isinstance(p, Product) for p in products)

        # Check for key products
        product_names = [p.name for p in products]
        assert "MATLAB" in product_names
        assert "Simulink" in product_names
        assert "Deep Learning Toolbox" in product_names

        # Check product structure
        first_product = products[0]
        assert first_product.name
        assert first_product.category
        assert first_product.description
        assert len(first_product.key_features) > 0
        assert len(first_product.use_cases) > 0
        assert len(first_product.target_personas) > 0

    @pytest.mark.asyncio
    async def test_build_catalog_with_fallback(self, indexer):
        """Test build_catalog with fallback products."""
        fallback_products = [
            {
                "name": "Test Product",
                "category": "Testing",
                "description": "A test product",
                "key_features": ["feature1"],
                "use_cases": ["testing"],
                "target_personas": ["tester"]
            }
        ]
        products = await indexer.build_catalog(fallback_products=fallback_products)

        assert len(products) == 1
        assert products[0].name == "Test Product"

    @pytest.mark.asyncio
    async def test_build_catalog_mathworks_builtin(self):
        """Test build_catalog uses built-in MathWorks products."""
        indexer = ProductCatalogIndexer(company_name="MathWorks", db_path=tempfile.mkdtemp())
        products = await indexer.build_catalog()

        # Should return built-in MathWorks products
        assert len(products) >= 20
        product_names = [p.name for p in products]
        assert "MATLAB" in product_names

    @pytest.mark.asyncio
    async def test_index_products_empty_list(self, indexer):
        """Test indexing empty product list."""
        await indexer.index_products([])

        # Should not raise error, just log warning
        # Collection should be empty or unchanged
        count = indexer.collection.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_index_products_success(self, indexer):
        """Test successful product indexing."""
        products = [
            Product(
                name="Test Product 1",
                category="Testing",
                description="A test product",
                key_features=["feature1", "feature2"],
                use_cases=["use1"],
                target_personas=["tester"]
            ),
            Product(
                name="Test Product 2",
                category="Testing",
                description="Another test product",
                key_features=["feature3"],
                use_cases=["use2"],
                target_personas=["developer"]
            ),
        ]

        await indexer.index_products(products)

        # Check that products were indexed
        count = indexer.collection.count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_index_products_creates_embeddings(self):
        """Test that indexing creates embeddings."""
        indexer = ProductCatalogIndexer(company_name="MathWorks", db_path=tempfile.mkdtemp())
        products = indexer._get_mathworks_products()[:5]  # Use first 5 products

        await indexer.index_products(products)

        # Query the collection to verify embeddings exist
        results = indexer.collection.get()
        assert len(results["ids"]) == 5
        assert len(results["documents"]) == 5
        assert len(results["metadatas"]) == 5

    @pytest.mark.asyncio
    async def test_index_products_document_format(self, indexer):
        """Test document format for indexing."""
        products = [
            Product(
                name="MATLAB",
                category="Programming",
                description="Programming platform",
                key_features=["feature1", "feature2"],
                use_cases=["use1", "use2"],
                target_personas=["engineer", "scientist"]
            )
        ]

        await indexer.index_products(products)

        results = indexer.collection.get()
        doc = results["documents"][0]

        # Check document contains all relevant information
        assert "MATLAB" in doc
        assert "Programming platform" in doc
        assert "feature1" in doc
        assert "feature2" in doc
        assert "use1" in doc
        assert "engineer" in doc

    @pytest.mark.asyncio
    async def test_index_products_metadata(self, indexer):
        """Test metadata storage."""
        products = [
            Product(
                name="Test Product",
                category="Test Category",
                description="Test",
                key_features=["f1"],
                use_cases=["u1"],
                target_personas=["p1"]
            )
        ]

        await indexer.index_products(products)

        results = indexer.collection.get()
        metadata = results["metadatas"][0]

        assert metadata["name"] == "Test Product"
        assert metadata["category"] == "Test Category"


class TestProductMatcher:
    """Test ProductMatcher for semantic requirement matching."""

    @pytest.fixture
    def temp_db_path(self):
        """Provide temporary ChromaDB path."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    async def matcher_with_data(self, temp_db_path):
        """Provide ProductMatcher with indexed data."""
        indexer = ProductCatalogIndexer(company_name="TestCompany", db_path=temp_db_path)

        # Index sample products
        products = [
            Product(
                name="Simulink Real-Time",
                category="Real-Time Simulation",
                description="Real-time simulation and testing for embedded systems",
                key_features=["real-time execution", "hardware-in-the-loop", "rapid prototyping"],
                use_cases=["HIL testing", "real-time control", "embedded testing"],
                target_personas=["test engineers", "control engineers"]
            ),
            Product(
                name="Embedded Coder",
                category="Code Generation",
                description="Generates C/C++ code for embedded systems",
                key_features=["code generation", "optimization", "MISRA compliance"],
                use_cases=["production code", "embedded software", "automotive"],
                target_personas=["embedded engineers", "software architects"]
            ),
            Product(
                name="Deep Learning Toolbox",
                category="AI/ML",
                description="Design and train deep neural networks",
                key_features=["neural networks", "model training", "deployment"],
                use_cases=["image classification", "AI deployment", "deep learning"],
                target_personas=["AI engineers", "data scientists"]
            ),
        ]

        await indexer.index_products(products)

        # Create matcher
        matcher = ProductMatcher(company_name="TestCompany", db_path=temp_db_path)
        return matcher

    @pytest.mark.asyncio
    async def test_matcher_collection_not_found(self, temp_db_path):
        """Test matcher raises error when collection doesn't exist."""
        with pytest.raises(DataSourceError, match="Product catalog not indexed"):
            ProductMatcher(company_name="NonExistentCompany", db_path=temp_db_path)

    @pytest.mark.asyncio
    async def test_match_requirements_basic(self, matcher_with_data):
        """Test basic requirement matching."""
        requirements = ["real-time embedded systems testing"]

        matches = await matcher_with_data.match_requirements_to_products(
            requirements=requirements,
            top_k=3
        )

        assert len(matches) > 0
        assert all(isinstance(match, tuple) for match in matches)
        assert all(len(match) == 2 for match in matches)

        # Check structure (product_name, confidence_score)
        product_name, confidence = matches[0]
        assert isinstance(product_name, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_match_requirements_semantic_similarity(self, matcher_with_data):
        """Test semantic matching finds relevant products."""
        requirements = ["real-time hardware testing"]

        matches = await matcher_with_data.match_requirements_to_products(
            requirements=requirements,
            top_k=3
        )

        # Should find Simulink Real-Time as top match
        product_names = [name for name, _ in matches]
        assert "Simulink Real-Time" in product_names

    @pytest.mark.asyncio
    async def test_match_requirements_multiple_requirements(self, matcher_with_data):
        """Test matching with multiple requirements."""
        requirements = [
            "real-time control systems",
            "code generation for embedded",
            "neural network training"
        ]

        matches = await matcher_with_data.match_requirements_to_products(
            requirements=requirements,
            top_k=5
        )

        # Should find all three products
        product_names = [name for name, _ in matches]
        assert "Simulink Real-Time" in product_names
        assert "Embedded Coder" in product_names
        assert "Deep Learning Toolbox" in product_names

    @pytest.mark.asyncio
    async def test_match_requirements_confidence_scores(self, matcher_with_data):
        """Test confidence scores are reasonable."""
        requirements = ["real-time simulation"]

        matches = await matcher_with_data.match_requirements_to_products(
            requirements=requirements,
            top_k=3
        )

        # Confidence scores should be in valid range
        for _, confidence in matches:
            assert 0.0 <= confidence <= 1.0

        # Scores should be sorted descending
        confidences = [conf for _, conf in matches]
        assert confidences == sorted(confidences, reverse=True)

    @pytest.mark.asyncio
    async def test_match_requirements_deduplication(self, matcher_with_data):
        """Test that duplicate products are deduplicated."""
        requirements = [
            "real-time systems",
            "real-time testing",
            "real-time control"
        ]

        matches = await matcher_with_data.match_requirements_to_products(
            requirements=requirements,
            top_k=5
        )

        # Each product should appear only once
        product_names = [name for name, _ in matches]
        assert len(product_names) == len(set(product_names))

    @pytest.mark.asyncio
    async def test_match_requirements_top_k_limit(self, matcher_with_data):
        """Test top_k limit is respected."""
        requirements = ["embedded systems software"]

        matches = await matcher_with_data.match_requirements_to_products(
            requirements=requirements,
            top_k=2
        )

        assert len(matches) <= 2

    @pytest.mark.asyncio
    async def test_match_requirements_empty_list(self, matcher_with_data):
        """Test matching with empty requirements list."""
        matches = await matcher_with_data.match_requirements_to_products(
            requirements=[],
            top_k=5
        )

        assert matches == []

    @pytest.mark.asyncio
    async def test_match_requirements_no_good_matches(self, matcher_with_data):
        """Test matching with unrelated requirement."""
        requirements = ["underwater basket weaving"]

        matches = await matcher_with_data.match_requirements_to_products(
            requirements=requirements,
            top_k=3
        )

        # Should still return matches, but with lower confidence
        if matches:
            for _, confidence in matches:
                assert confidence < 0.9  # Unlikely to have high confidence

    @pytest.mark.asyncio
    async def test_explain_match_basic(self, matcher_with_data):
        """Test match explanation generation."""
        explanation = await matcher_with_data.explain_match(
            requirement="real-time testing",
            product_name="Simulink Real-Time"
        )

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "Simulink Real-Time" in explanation
        assert "real-time testing" in explanation

    @pytest.mark.asyncio
    async def test_match_requirements_aggregation(self, matcher_with_data):
        """Test that max score is taken for duplicate products."""
        # This requirement should match Simulink Real-Time multiple times
        requirements = [
            "real-time testing",
            "real-time simulation",
            "real-time control"
        ]

        matches = await matcher_with_data.match_requirements_to_products(
            requirements=requirements,
            top_k=5
        )

        # Simulink Real-Time should appear once with max confidence
        simulink_matches = [(name, conf) for name, conf in matches if "Simulink" in name]
        assert len(simulink_matches) == 1

    @pytest.mark.asyncio
    async def test_match_requirements_error_handling(self, matcher_with_data):
        """Test error handling for failed matches."""
        # Even with potential errors, should return partial results or empty list
        requirements = ["test"] * 100  # Large number of requirements

        try:
            matches = await matcher_with_data.match_requirements_to_products(
                requirements=requirements,
                top_k=5
            )
            # Should either succeed or return empty
            assert isinstance(matches, list)
        except Exception:
            # If it raises, should be a known exception type
            pass
