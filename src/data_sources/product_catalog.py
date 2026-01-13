"""
MathWorks product catalog scraping and semantic matching.
Builds searchable product index with ChromaDB for requirement matching.
"""
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from src.core.exceptions import DataSourceError
from src.data_sources.mcp_ddg_client import DuckDuckGoMCPClient
from src.data_sources.scraper import fetch_url, extract_text, extract_metadata
from src.models.domain import Product
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ProductCatalogIndexer:
    """Scrapes and indexes MathWorks products."""

    def __init__(self, db_path: str = "./data/chroma"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with sentence transformers
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name="mathworks_products",
            embedding_function=self.embedding_fn,
            metadata={"description": "MathWorks product catalog"}
        )

    async def build_catalog(self) -> list[Product]:
        """
        Build complete MathWorks product catalog.

        Strategy:
        1. Start with known core products (hardcoded fallback)
        2. Enrich with web scraping if available

        Returns:
            List of Product objects
        """
        # Core MathWorks products (fallback data)
        core_products = self._get_core_products()

        logger.info("catalog_build_started", product_count=len(core_products))

        # Try to enrich with web scraping
        try:
            enriched = await self._scrape_mathworks_products()
            if enriched:
                core_products = enriched
                logger.info("catalog_enriched_from_web", count=len(enriched))
        except Exception as e:
            logger.warning("web_scraping_failed_using_fallback", error=str(e))

        return core_products

    def _get_core_products(self) -> list[Product]:
        """Hardcoded core MathWorks products (fallback)."""
        products_data = [
            {
                "name": "MATLAB",
                "category": "Programming Platform",
                "description": "High-level language and interactive environment for numerical computation, visualization, and programming",
                "key_features": ["matrix operations", "data visualization", "algorithm development", "app building"],
                "use_cases": ["data analysis", "algorithm prototyping", "mathematical modeling"],
                "target_personas": ["data scientists", "engineers", "researchers"]
            },
            {
                "name": "Simulink",
                "category": "Model-Based Design",
                "description": "Block diagram environment for simulation and Model-Based Design of multidomain and embedded systems",
                "key_features": ["visual modeling", "simulation", "automatic code generation", "continuous testing"],
                "use_cases": ["control systems", "signal processing", "embedded systems", "communications"],
                "target_personas": ["control engineers", "embedded software engineers", "system architects"]
            },
            {
                "name": "Simulink Real-Time",
                "category": "Real-Time Simulation",
                "description": "Real-time simulation and testing environment for embedded systems",
                "key_features": ["real-time execution", "hardware-in-the-loop", "rapid prototyping"],
                "use_cases": ["HIL testing", "real-time control", "prototyping"],
                "target_personas": ["test engineers", "control engineers"]
            },
            {
                "name": "Embedded Coder",
                "category": "Code Generation",
                "description": "Generates readable, compact, fast C/C++ code for embedded systems",
                "key_features": ["code generation", "code optimization", "MISRA compliance", "traceability"],
                "use_cases": ["production code generation", "embedded software", "automotive systems"],
                "target_personas": ["embedded software engineers", "software architects"]
            },
            {
                "name": "Stateflow",
                "category": "State Machine Design",
                "description": "Design and develop state machines and control logic",
                "key_features": ["state machine design", "flow charts", "truth tables"],
                "use_cases": ["control logic", "supervisory control", "fault management"],
                "target_personas": ["control engineers", "software engineers"]
            },
            {
                "name": "Simscape",
                "category": "Physical Modeling",
                "description": "Model and simulate multidomain physical systems",
                "key_features": ["physical modeling", "multidomain simulation", "component libraries"],
                "use_cases": ["mechanical systems", "electrical systems", "hydraulic systems"],
                "target_personas": ["mechanical engineers", "electrical engineers"]
            },
            {
                "name": "Vehicle Dynamics Blockset",
                "category": "Automotive",
                "description": "Model and simulate vehicle dynamics and vehicle-level control systems",
                "key_features": ["vehicle modeling", "chassis simulation", "powertrain modeling"],
                "use_cases": ["autonomous vehicles", "ADAS", "vehicle testing"],
                "target_personas": ["automotive engineers", "vehicle dynamics engineers"]
            },
            {
                "name": "Automated Driving Toolbox",
                "category": "Autonomous Systems",
                "description": "Design, simulate, and test ADAS and autonomous driving systems",
                "key_features": ["sensor fusion", "path planning", "scenario generation"],
                "use_cases": ["ADAS development", "autonomous driving", "sensor testing"],
                "target_personas": ["ADAS engineers", "autonomy engineers"]
            },
            {
                "name": "Deep Learning Toolbox",
                "category": "AI/ML",
                "description": "Design, train, and analyze deep neural networks",
                "key_features": ["neural network design", "model training", "deployment"],
                "use_cases": ["image classification", "object detection", "AI deployment"],
                "target_personas": ["AI engineers", "data scientists", "ML engineers"]
            },
            {
                "name": "Reinforcement Learning Toolbox",
                "category": "AI/ML",
                "description": "Train policies for autonomous systems using reinforcement learning",
                "key_features": ["RL algorithms", "environment modeling", "policy training"],
                "use_cases": ["robotics control", "game AI", "autonomous systems"],
                "target_personas": ["AI engineers", "robotics engineers"]
            },
            {
                "name": "Signal Processing Toolbox",
                "category": "Signal Processing",
                "description": "Analyze, preprocess, and extract features from signals",
                "key_features": ["filtering", "spectral analysis", "signal transforms"],
                "use_cases": ["audio processing", "communications", "sensor data analysis"],
                "target_personas": ["signal processing engineers", "communications engineers"]
            },
            {
                "name": "Control System Toolbox",
                "category": "Controls",
                "description": "Design and analyze control systems",
                "key_features": ["PID tuning", "linear control design", "frequency analysis"],
                "use_cases": ["control system design", "stability analysis", "controller tuning"],
                "target_personas": ["control engineers", "systems engineers"]
            },
            {
                "name": "Optimization Toolbox",
                "category": "Optimization",
                "description": "Solve linear, quadratic, integer, and nonlinear optimization problems",
                "key_features": ["constrained optimization", "global optimization", "multiobjective optimization"],
                "use_cases": ["parameter optimization", "design optimization", "resource allocation"],
                "target_personas": ["engineers", "data scientists", "operations researchers"]
            },
            {
                "name": "Statistics and Machine Learning Toolbox",
                "category": "Data Science",
                "description": "Analyze and model data using statistics and machine learning",
                "key_features": ["classification", "regression", "clustering", "feature selection"],
                "use_cases": ["predictive modeling", "data analysis", "feature engineering"],
                "target_personas": ["data scientists", "statisticians", "analysts"]
            },
            {
                "name": "Computer Vision Toolbox",
                "category": "Computer Vision",
                "description": "Design and test computer vision and video processing systems",
                "key_features": ["image processing", "object detection", "tracking", "3D vision"],
                "use_cases": ["object recognition", "image analysis", "video surveillance"],
                "target_personas": ["computer vision engineers", "image processing engineers"]
            },
            {
                "name": "Image Processing Toolbox",
                "category": "Image Processing",
                "description": "Perform image processing, analysis, and algorithm development",
                "key_features": ["image enhancement", "segmentation", "morphology", "registration"],
                "use_cases": ["medical imaging", "quality inspection", "scientific imaging"],
                "target_personas": ["image processing engineers", "researchers"]
            },
            {
                "name": "Robotics System Toolbox",
                "category": "Robotics",
                "description": "Design, simulate, and test robotics applications",
                "key_features": ["motion planning", "kinematics", "ROS integration"],
                "use_cases": ["robot control", "path planning", "manipulation"],
                "target_personas": ["robotics engineers", "automation engineers"]
            },
            {
                "name": "Communications Toolbox",
                "category": "Communications",
                "description": "Design and simulate communications systems",
                "key_features": ["modulation", "channel modeling", "error correction"],
                "use_cases": ["wireless communications", "5G/6G", "satellite communications"],
                "target_personas": ["communications engineers", "RF engineers"]
            },
            {
                "name": "RF Toolbox",
                "category": "RF Design",
                "description": "Design and analyze RF systems and components",
                "key_features": ["RF modeling", "amplifier design", "matching networks"],
                "use_cases": ["RF circuit design", "antenna design", "wireless systems"],
                "target_personas": ["RF engineers", "hardware engineers"]
            },
            {
                "name": "Parallel Computing Toolbox",
                "category": "High Performance Computing",
                "description": "Speed up computations using parallel processing on multicore CPUs and GPUs",
                "key_features": ["parallel loops", "GPU computing", "distributed arrays"],
                "use_cases": ["large-scale computations", "GPU acceleration", "cloud computing"],
                "target_personas": ["computational scientists", "HPC engineers"]
            },
        ]

        products = [Product(**data) for data in products_data]
        logger.info("core_products_loaded", count=len(products))
        return products

    async def _scrape_mathworks_products(self) -> list[Product]:
        """
        Scrape MathWorks website for product information.

        Returns:
            List of enriched Product objects or empty list if scraping fails
        """
        try:
            # Search for MathWorks products page
            async with DuckDuckGoMCPClient() as client:
                results = await client.search("MathWorks products toolboxes", max_results=3)

                if not results:
                    return []

                # Try to fetch product listing page
                for result in results:
                    if "mathworks.com" in str(result.url) and "products" in str(result.url):
                        logger.info("fetching_products_page", url=result.url)
                        # Future: Parse product listing page
                        # For now, return empty to use fallback
                        break

            return []

        except Exception as e:
            logger.warning("scrape_mathworks_failed", error=str(e))
            return []

    async def index_products(self, products: list[Product]) -> None:
        """
        Index products in ChromaDB for semantic search.

        Args:
            products: List of Product objects to index
        """
        if not products:
            logger.warning("no_products_to_index")
            return

        # Prepare documents for indexing
        documents = []
        metadatas = []
        ids = []

        for i, product in enumerate(products):
            # Create searchable document
            doc = (
                f"{product.name}. {product.description}. "
                f"Features: {', '.join(product.key_features)}. "
                f"Use cases: {', '.join(product.use_cases)}. "
                f"Target users: {', '.join(product.target_personas)}."
            )

            documents.append(doc)
            metadatas.append({
                "name": product.name,
                "category": product.category,
            })
            ids.append(f"product_{i}")

        # Index in ChromaDB
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.info("products_indexed", count=len(products), collection="mathworks_products")


class ProductMatcher:
    """Matches requirements to products using semantic search."""

    def __init__(self, db_path: str = "./data/chroma"):
        self.db_path = Path(db_path)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        try:
            self.collection = self.client.get_collection(
                name="mathworks_products",
                embedding_function=self.embedding_fn
            )
        except Exception as e:
            logger.error("collection_not_found", error=str(e))
            raise DataSourceError("Product catalog not indexed. Run ProductCatalogIndexer first.")

    async def match_requirements_to_products(
        self,
        requirements: list[str],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Match requirements to products with confidence scores.

        Args:
            requirements: List of requirement strings
            top_k: Number of top matches to return per requirement

        Returns:
            List of (product_name, confidence_score) tuples
        """
        all_matches = []

        for req in requirements:
            try:
                results = self.collection.query(
                    query_texts=[req],
                    n_results=top_k
                )

                if results and results["metadatas"]:
                    for i, metadata in enumerate(results["metadatas"][0]):
                        product_name = metadata.get("name", "Unknown")
                        # ChromaDB returns distances, convert to similarity (1 - distance)
                        distance = results["distances"][0][i] if results["distances"] else 1.0
                        confidence = max(0.0, 1.0 - distance)

                        all_matches.append((product_name, confidence))

            except Exception as e:
                logger.warning("match_failed", requirement=req, error=str(e))
                continue

        # Aggregate and deduplicate
        product_scores: dict[str, float] = {}
        for product, score in all_matches:
            if product not in product_scores:
                product_scores[product] = score
            else:
                # Take max score for duplicate products
                product_scores[product] = max(product_scores[product], score)

        # Sort by confidence descending
        sorted_matches = sorted(
            product_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        logger.info(
            "requirements_matched",
            requirement_count=len(requirements),
            match_count=len(sorted_matches)
        )

        return sorted_matches[:top_k]

    async def explain_match(
        self,
        requirement: str,
        product_name: str
    ) -> str:
        """
        Generate explanation for why a product matches a requirement.

        Args:
            requirement: Requirement string
            product_name: Product name

        Returns:
            Explanation text
        """
        # Simple explanation based on semantic similarity
        return (
            f"{product_name} matches '{requirement}' based on its features "
            f"and use cases that align with this requirement."
        )
