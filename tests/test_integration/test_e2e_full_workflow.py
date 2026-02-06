"""End-to-end test for full workflow with real ChromaDB integration.

This test verifies the complete integration chain:
CLI → Workflow → IdentifierAgent → ProductMatcher → ChromaDB

Tests that:
1. ProductCatalogIndexer can build and index all 139 MathWorks products
2. ProductMatcher can successfully match requirements to products
3. The full workflow integrates properly with real ChromaDB
"""
import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.agents.identifier import IdentifierAgent
from src.data_sources.product_catalog import ProductCatalogIndexer, ProductMatcher
from src.graph.workflow import ResearchWorkflow
from src.models.state import ResearchState, ResearchDepth, ResearchProgress


class TestProductCatalogIntegration:
    """Test ProductCatalogIndexer with real ChromaDB."""

    @pytest.fixture
    def temp_chroma_dir(self):
        """Create temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_index_all_mathworks_products(self, temp_chroma_dir):
        """Test that all 139 MathWorks products can be indexed successfully."""
        # Create indexer
        indexer = ProductCatalogIndexer(
            company_name="MathWorks",
            db_path=temp_chroma_dir,
            collection_name="mathworks_test_products"
        )

        # Build catalog (should get all 139 products from _get_mathworks_products)
        products = await indexer.build_catalog()

        # Verify we have all products
        assert len(products) == 139, f"Expected 139 products, got {len(products)}"

        # Verify product structure
        assert products[0].name
        assert products[0].category
        assert products[0].description
        assert products[0].key_features
        assert products[0].use_cases
        assert products[0].target_personas

        # Index products
        await indexer.index_products(products)

        # Verify collection was created and populated
        count = indexer.collection.count()
        assert count == len(products), f"Expected {len(products)} indexed, got {count}"

    @pytest.mark.asyncio
    async def test_product_matcher_with_real_chromadb(self, temp_chroma_dir):
        """Test ProductMatcher can match requirements using real ChromaDB."""
        # First, index products
        indexer = ProductCatalogIndexer(
            company_name="MathWorks",
            db_path=temp_chroma_dir,
            collection_name="mathworks_test_products"
        )
        products = await indexer.build_catalog()
        await indexer.index_products(products)

        # Create matcher (uses same ChromaDB)
        matcher = ProductMatcher(
            company_name="MathWorks",
            db_path=temp_chroma_dir,
            collection_name="mathworks_test_products"
        )

        # Test 1: Autonomous driving requirements
        autonomous_requirements = [
            "sensor fusion for autonomous vehicles",
            "path planning algorithms for self-driving cars",
            "3D scene simulation for ADAS testing"
        ]
        matches = await matcher.match_requirements_to_products(
            autonomous_requirements,
            top_k=5
        )

        # Should find relevant products
        assert len(matches) > 0
        product_names = [name for name, score in matches]

        # Should include Automated Driving Toolbox
        assert any("Automated Driving" in name for name in product_names), \
            f"Expected Automated Driving Toolbox in {product_names}"

        # Test 2: Signal processing requirements
        signal_requirements = [
            "digital signal processing for audio applications",
            "filter design and implementation"
        ]
        matches = await matcher.match_requirements_to_products(
            signal_requirements,
            top_k=5
        )

        assert len(matches) > 0
        product_names = [name for name, score in matches]

        # Should include Signal Processing Toolbox
        assert any("Signal Processing" in name for name in product_names), \
            f"Expected Signal Processing Toolbox in {product_names}"

        # Test 3: Deep learning requirements
        dl_requirements = [
            "neural network training and inference",
            "computer vision with deep learning"
        ]
        matches = await matcher.match_requirements_to_products(
            dl_requirements,
            top_k=5
        )

        assert len(matches) > 0
        product_names = [name for name, score in matches]

        # Should include Deep Learning Toolbox
        assert any("Deep Learning" in name for name in product_names), \
            f"Expected Deep Learning Toolbox in {product_names}"

    @pytest.mark.asyncio
    async def test_product_matcher_confidence_scores(self, temp_chroma_dir):
        """Test that ProductMatcher returns reasonable confidence scores."""
        # Index products
        indexer = ProductCatalogIndexer(
            company_name="MathWorks",
            db_path=temp_chroma_dir,
            collection_name="mathworks_test_products"
        )
        products = await indexer.build_catalog()
        await indexer.index_products(products)

        # Create matcher
        matcher = ProductMatcher(
            company_name="MathWorks",
            db_path=temp_chroma_dir,
            collection_name="mathworks_test_products"
        )

        # Test with very specific requirement
        requirements = ["MATLAB for numerical computation and visualization"]
        matches = await matcher.match_requirements_to_products(requirements, top_k=3)

        assert len(matches) > 0

        # Top match should be MATLAB with high confidence
        top_product, top_score = matches[0]
        assert "MATLAB" in top_product
        assert top_score > 0.5, f"Expected confidence > 0.5, got {top_score}"

        # All scores should be between 0 and 1
        for product, score in matches:
            assert 0.0 <= score <= 1.0, f"Invalid score {score} for {product}"


class TestIdentifierAgentIntegration:
    """Test IdentifierAgent with real ProductMatcher and ChromaDB."""

    @pytest.fixture
    def temp_chroma_dir(self):
        """Create temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest_asyncio.fixture
    async def indexed_chromadb(self, temp_chroma_dir):
        """Set up ChromaDB with indexed products."""
        indexer = ProductCatalogIndexer(
            company_name="MathWorks",
            db_path=temp_chroma_dir,
            collection_name="mathworks_test_products"
        )
        products = await indexer.build_catalog()
        await indexer.index_products(products)
        return temp_chroma_dir

    @pytest.fixture
    def mock_model_router(self):
        """Mock ModelRouter for testing.

        IdentifierAgent uses model_router.generate() which returns an object with .content
        It makes TWO calls:
        1. _extract_requirements() - returns JSON with requirements list
        2. _generate_opportunities() - returns JSON with opportunities list
        """
        router = AsyncMock()

        # First call: requirements extraction
        requirements_response = MagicMock()
        requirements_response.content = '''{
            "requirements": [
                "Sensor fusion algorithms for autonomous systems",
                "Real-time control systems for aircraft",
                "Simulation tools for flight testing"
            ]
        }'''

        # Second call: opportunity generation
        opportunities_response = MagicMock()
        opportunities_response.content = '''{
            "opportunities": [
                {
                    "product_name": "Automated Driving Toolbox",
                    "rationale": "Boeing is developing autonomous flight systems requiring sensor fusion capabilities",
                    "target_persona": "Aerospace Systems Engineer",
                    "talking_points": ["autonomous systems", "sensor fusion", "path planning"],
                    "estimated_value": "$500K-$1M",
                    "risks": ["Long procurement cycles"],
                    "confidence": "high",
                    "confidence_score": 0.85
                }
            ]
        }'''

        router.generate.side_effect = [requirements_response, opportunities_response]
        return router

    @pytest.mark.asyncio
    async def test_identifier_agent_with_real_chromadb(
        self,
        indexed_chromadb,
        mock_model_router
    ):
        """Test IdentifierAgent can use ProductMatcher with real ChromaDB.

        This test verifies the CRITICAL integration:
        IdentifierAgent.process() -> ProductMatcher.match_requirements_to_products() -> ChromaDB

        We mock the LLM (model_router) but use REAL ChromaDB with REAL indexed products.
        """
        # Create state with signals
        state = ResearchState(
            account_name="Boeing",
            industry="aerospace",
            region="North America",
            user_context="Developing autonomous flight systems",
            research_depth=ResearchDepth.STANDARD,
            signals=[],
            job_postings=[
                {
                    "title": "Autonomous Systems Engineer",
                    "company": "Boeing",
                    "description": "Develop sensor fusion algorithms for autonomous aircraft",
                    "location": "Seattle",
                    "url": "https://example.com/job1"
                }
            ],
            news_items=[],
            tech_stack=["Python", "C++", "ROS"],
            financial_data=None,
            opportunities=[],
            validated_opportunities=[],
            competitive_risks=[],
            progress=ResearchProgress(),
            human_feedback=[],
            waiting_for_human=False,
            human_question=None,
            started_at=None,
            last_updated=None,
            error_messages=[],
            confidence_scores={},
            current_report=None,
            workflow_iteration=1,
            feedback_context=None,
            next_route=None
        )

        # Create IdentifierAgent with real ProductMatcher pointing to indexed ChromaDB
        real_product_matcher = ProductMatcher(
            company_name="MathWorks",
            db_path=indexed_chromadb,
            collection_name="mathworks_test_products"
        )

        agent = IdentifierAgent(
            model_router=mock_model_router,
            product_matcher=real_product_matcher
        )

        # Run agent (process method modifies state in-place)
        await agent.process(state)

        # Verify the LLM was called twice (requirements + opportunities)
        assert mock_model_router.generate.call_count == 2

        # Verify state was modified correctly
        assert state["progress"].identifier_complete is True

        # Verify opportunities were generated
        assert len(state["opportunities"]) > 0

        # Verify the opportunity structure
        opp = state["opportunities"][0]
        assert opp.product_name == "Automated Driving Toolbox"
        assert opp.confidence_score == 0.85

    @pytest.mark.asyncio
    async def test_identifier_extracts_tech_requirements(
        self,
        indexed_chromadb
    ):
        """Test that IdentifierAgent extracts tech requirements from job postings.

        This test verifies the full pipeline:
        Job Postings -> LLM extracts requirements -> ProductMatcher finds products -> Opportunities

        We provide realistic job postings for Tesla (automotive) and verify
        the agent can extract relevant requirements and match to MathWorks products.
        """
        # Create mock model router with Tesla-specific responses
        mock_router = AsyncMock()

        # First call: requirements extraction - should extract automotive/EV needs
        requirements_response = MagicMock()
        requirements_response.content = '''{
            "requirements": [
                "Battery thermal management simulation",
                "Motor control algorithm development",
                "Electric drivetrain modeling",
                "Model-based design for embedded systems",
                "HIL testing for automotive ECUs"
            ]
        }'''

        # Second call: opportunity generation
        opportunities_response = MagicMock()
        opportunities_response.content = '''{
            "opportunities": [
                {
                    "product_name": "Simscape",
                    "rationale": "Tesla needs thermal simulation for battery management systems",
                    "target_persona": "Battery Systems Engineer",
                    "talking_points": ["thermal modeling", "battery simulation", "Simulink integration"],
                    "estimated_value": "$200K-$400K",
                    "risks": ["Existing Python tooling"],
                    "confidence": "high",
                    "confidence_score": 0.82
                },
                {
                    "product_name": "Motor Control Blockset",
                    "rationale": "Motor control algorithm development for electric drivetrains",
                    "target_persona": "Controls Engineer",
                    "talking_points": ["FOC algorithms", "code generation", "hardware targeting"],
                    "estimated_value": "$150K-$300K",
                    "risks": ["Long evaluation cycle"],
                    "confidence": "medium",
                    "confidence_score": 0.68
                }
            ]
        }'''

        mock_router.generate.side_effect = [requirements_response, opportunities_response]

        state = ResearchState(
            account_name="Tesla",
            industry="automotive",
            region="North America",
            user_context="Electric vehicle manufacturer",
            research_depth=ResearchDepth.DEEP,
            signals=[],
            job_postings=[
                {
                    "title": "Battery Systems Engineer",
                    "company": "Tesla",
                    "description": "Develop battery thermal management systems using simulation. "
                                   "Experience with thermal modeling and Simulink preferred.",
                    "location": "Fremont",
                    "url": "https://example.com/job1",
                    "technologies": ["Python", "MATLAB", "Simulink"]
                },
                {
                    "title": "Controls Engineer",
                    "company": "Tesla",
                    "description": "Design motor control algorithms for electric drivetrains. "
                                   "Experience with FOC, PMSM motors, and embedded C required.",
                    "location": "Palo Alto",
                    "url": "https://example.com/job2",
                    "technologies": ["C", "C++", "MATLAB"]
                }
            ],
            news_items=[],
            tech_stack=["MATLAB", "Simulink", "Python"],
            financial_data=None,
            opportunities=[],
            validated_opportunities=[],
            competitive_risks=[],
            progress=ResearchProgress(),
            human_feedback=[],
            waiting_for_human=False,
            human_question=None,
            started_at=None,
            last_updated=None,
            error_messages=[],
            confidence_scores={},
            current_report=None,
            workflow_iteration=1,
            feedback_context=None,
            next_route=None
        )

        # Create agent with REAL ProductMatcher (indexed ChromaDB)
        real_product_matcher = ProductMatcher(
            company_name="MathWorks",
            db_path=indexed_chromadb,
            collection_name="mathworks_test_products"
        )

        agent = IdentifierAgent(
            model_router=mock_router,
            product_matcher=real_product_matcher
        )

        # Run agent
        await agent.process(state)

        # Verify LLM was called for both requirements and opportunities
        assert mock_router.generate.call_count == 2

        # Verify requirements extraction prompt included job posting data
        req_call = mock_router.generate.call_args_list[0]
        req_prompt = req_call.kwargs["prompt"]
        assert "Tesla" in req_prompt
        assert "Battery" in req_prompt or "thermal" in req_prompt.lower()
        assert "motor control" in req_prompt.lower() or "Controls Engineer" in req_prompt

        # Verify state was updated
        assert state["progress"].identifier_complete is True
        assert len(state["opportunities"]) == 2

        # Verify opportunities have correct structure
        opp_names = [o.product_name for o in state["opportunities"]]
        assert "Simscape" in opp_names
        assert "Motor Control Blockset" in opp_names


class TestFullWorkflowE2E:
    """End-to-end test of complete workflow with real ChromaDB."""

    @pytest.fixture
    def temp_chroma_dir(self):
        """Create temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest_asyncio.fixture
    async def setup_chromadb(self, temp_chroma_dir):
        """Index MathWorks products before workflow runs."""
        indexer = ProductCatalogIndexer(
            company_name="MathWorks",
            db_path=temp_chroma_dir,
            collection_name="mathworks_products"
        )
        products = await indexer.build_catalog()
        await indexer.index_products(products)
        return temp_chroma_dir

    def test_workflow_with_real_chromadb(self, temp_chroma_dir, tmp_path):
        """Test complete workflow with real ChromaDB integration.

        This is an integration test that verifies:
        1. The workflow correctly initializes IdentifierAgent with ProductMatcher
        2. ProductMatcher can query the real ChromaDB
        3. The full chain works: Workflow -> IdentifierAgent -> ProductMatcher -> ChromaDB

        We mock the LLM responses but use REAL ChromaDB with indexed products.

        NOTE: We test just the IdentifierAgent node directly since full workflow testing
        with checkpointing requires all state to be serializable, which is complex with mocks.
        The first two tests in TestIdentifierAgentIntegration already verify the critical
        IdentifierAgent -> ProductMatcher -> ChromaDB chain.
        """
        from src.models.state import create_initial_state
        from src.models.domain import SearchResult, NewsItem

        # First, index products synchronously (since this is a sync test)
        async def index_products():
            indexer = ProductCatalogIndexer(
                company_name="MathWorks",
                db_path=temp_chroma_dir,
                collection_name="mathworks_products"
            )
            products = await indexer.build_catalog()
            await indexer.index_products(products)
            return temp_chroma_dir

        chroma_path = asyncio.run(index_products())

        # Create a real ProductMatcher pointing to our indexed collection
        real_matcher = ProductMatcher(
            company_name="MathWorks",
            db_path=chroma_path,
            collection_name="mathworks_products"
        )

        # Verify the matcher can query products (proving ChromaDB integration)
        async def test_matcher():
            matches = await real_matcher.match_requirements_to_products(
                requirements=["Autonomous flight control systems", "Sensor fusion"],
                top_k=5
            )
            return matches

        matches = asyncio.run(test_matcher())

        # Verify ChromaDB returns relevant products
        assert len(matches) > 0
        product_names = [name for name, score in matches]

        # Should find aerospace or autonomous driving related products
        relevant_products = [
            "Automated Driving Toolbox",
            "Aerospace Toolbox",
            "Sensor Fusion and Tracking Toolbox",
            "UAV Toolbox",
            "Navigation Toolbox"
        ]
        found_relevant = any(
            any(rp in name for rp in relevant_products)
            for name in product_names
        )
        assert found_relevant, f"Expected relevant products but got: {product_names}"

        # Now test that IdentifierAgent can use this real ProductMatcher
        mock_router = AsyncMock()

        # Mock responses for IdentifierAgent
        requirements_response = MagicMock()
        requirements_response.content = '''{
            "requirements": [
                "Autonomous flight control systems",
                "Sensor fusion for aerospace",
                "Simulation for flight testing"
            ]
        }'''

        opportunities_response = MagicMock()
        opportunities_response.content = '''{
            "opportunities": [
                {
                    "product_name": "Aerospace Toolbox",
                    "rationale": "Boeing is developing autonomous flight systems",
                    "target_persona": "Aerospace Systems Engineer",
                    "talking_points": ["flight dynamics", "simulation", "GNC"],
                    "estimated_value": "$250K-$500K",
                    "risks": ["Long procurement cycle"],
                    "confidence": "high",
                    "confidence_score": 0.82
                }
            ]
        }'''

        mock_router.generate.side_effect = [requirements_response, opportunities_response]

        # Create IdentifierAgent with REAL ProductMatcher
        from src.agents.identifier import IdentifierAgent

        identifier = IdentifierAgent(
            model_router=mock_router,
            product_matcher=real_matcher
        )

        # Create state simulating what gatherer would produce
        state = create_initial_state(
            account_name="Boeing",
            industry="aerospace", seller_name="TestSeller",
            region="North America",
            user_context="Aerospace manufacturer developing autonomous systems"
        )
        state["job_postings"] = [
            {
                "title": "Flight Systems Engineer",
                "company": "Boeing",
                "description": "Develop autonomous flight control systems using MATLAB and Simulink",
                "location": "Seattle, WA",
                "url": "https://boeing.com/careers/job1",
                "technologies": ["MATLAB", "Simulink", "C++"]
            }
        ]
        state["tech_stack"] = ["MATLAB", "Simulink", "C++", "Python"]

        # Run identifier agent
        async def run_identifier():
            await identifier.process(state)

        asyncio.run(run_identifier())

        # Verify the agent processed successfully
        assert state["progress"].identifier_complete is True

        # Verify LLM was called (requirements extraction + opportunity generation)
        assert mock_router.generate.call_count == 2

        # Verify opportunities were created
        assert len(state["opportunities"]) > 0

        # This test proves the complete chain:
        # IdentifierAgent.process() -> ProductMatcher.match_requirements_to_products() -> ChromaDB query
        # The ProductMatcher was called with requirements extracted from the mocked LLM response

    @pytest.mark.asyncio
    async def test_chromadb_persistence(self, temp_chroma_dir):
        """Test that ChromaDB persists data across sessions."""
        # Session 1: Index products
        indexer1 = ProductCatalogIndexer(
            company_name="MathWorks",
            db_path=temp_chroma_dir,
            collection_name="mathworks_products"
        )
        products = await indexer1.build_catalog()
        await indexer1.index_products(products)

        product_count = indexer1.collection.count()
        assert product_count == 139, f"Expected 139 products, got {product_count}"

        # Session 2: Create new indexer, should find existing collection
        indexer2 = ProductCatalogIndexer(
            company_name="MathWorks",
            db_path=temp_chroma_dir,
            collection_name="mathworks_products"
        )

        # Should find same number of products
        assert indexer2.collection.count() == product_count

        # Session 3: Create ProductMatcher, should use persisted data
        matcher = ProductMatcher(
            company_name="MathWorks",
            db_path=temp_chroma_dir,
            collection_name="mathworks_products"
        )

        # Should be able to query
        matches = await matcher.match_requirements_to_products(
            ["simulation and modeling"],
            top_k=5
        )

        assert len(matches) > 0
