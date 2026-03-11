"""
Generic product catalog scraping and semantic matching.
Builds searchable product index with ChromaDB for requirement matching.
Supports any company's product catalog via JSON configuration or dynamic scraping.
"""
import asyncio
import json
import math
from pathlib import Path
from typing import Any, Optional

import chromadb
from bs4 import BeautifulSoup
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import settings
from src.core.exceptions import DataSourceError
from src.data_sources.mcp_ddg_client import DuckDuckGoMCPClient
from src.data_sources.scraper import fetch_url, extract_text, extract_metadata, RateLimiter
from src.models.domain import Product
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ProductCatalogIndexer:
    """
    Generic product catalog indexer for any company.

    Supports multiple data sources:
    1. JSON file with product definitions
    2. Hardcoded products (for fallback)
    3. Web scraping (future enhancement)
    """

    def __init__(
        self,
        company_name: str,
        db_path: str = "./data/chroma",
        collection_name: Optional[str] = None,
        catalog_file: Optional[str] = None
    ):
        """
        Initialize product catalog indexer.

        Args:
            company_name: Name of the company (e.g., "MathWorks", "Salesforce")
            db_path: Path to ChromaDB storage
            collection_name: Optional custom collection name (defaults to "{company_name}_products")
            catalog_file: Optional path to JSON file with product catalog
        """
        self.company_name = company_name
        self.catalog_file = Path(catalog_file) if catalog_file else None
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

        # Use company-specific collection name
        self.collection_name = collection_name or f"{company_name.lower().replace(' ', '_')}_products"

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": f"{company_name} product catalog", "company": company_name}
        )

    async def build_catalog(self, fallback_products: Optional[list[dict]] = None) -> list[Product]:
        """
        Build product catalog from multiple sources.

        Priority order:
        1. JSON catalog file (if provided)
        2. Fallback products (if provided)
        3. Web scraping (if available)
        4. Built-in defaults for known companies

        Args:
            fallback_products: Optional list of product dictionaries for fallback

        Returns:
            List of Product objects
        """
        products = []

        # Try JSON file first
        if self.catalog_file and self.catalog_file.exists():
            try:
                products = self._load_from_json(self.catalog_file)
                logger.info(
                    "catalog_loaded_from_json",
                    file=str(self.catalog_file),
                    count=len(products)
                )
                return products
            except Exception as e:
                logger.warning("json_load_failed", file=str(self.catalog_file), error=str(e))

        # Try provided fallback products
        if fallback_products:
            try:
                products = [Product(**data) for data in fallback_products]
                logger.info("catalog_loaded_from_fallback", count=len(products))
                return products
            except Exception as e:
                logger.warning("fallback_products_invalid", error=str(e))

        # Try built-in defaults for known companies
        if self.company_name.lower() == "mathworks":
            products = self._get_mathworks_products()
            logger.info("catalog_loaded_from_builtin", company="MathWorks", count=len(products))
            return products

        # Try web scraping as last resort
        try:
            products = await self._scrape_company_products()
            if products:
                logger.info("catalog_loaded_from_web", count=len(products))
                return products
        except Exception as e:
            logger.warning("web_scraping_failed", error=str(e))

        # If all methods fail, return empty list
        logger.warning(
            "catalog_build_failed_no_products",
            company=self.company_name,
            message="Provide catalog_file or fallback_products"
        )
        return []

    def _load_from_json(self, json_path: Path) -> list[Product]:
        """
        Load product catalog from JSON file.

        Expected JSON format:
        [
            {
                "name": "Product Name",
                "category": "Category",
                "description": "Description",
                "key_features": ["feature1", "feature2"],
                "use_cases": ["use1", "use2"],
                "target_personas": ["persona1", "persona2"]
            },
            ...
        ]

        Args:
            json_path: Path to JSON catalog file

        Returns:
            List of Product objects
        """
        with open(json_path, "r", encoding="utf-8") as f:
            products_data = json.load(f)

        products = [Product(**data) for data in products_data]
        return products

    def _get_mathworks_products(self) -> list[Product]:
        """Complete MathWorks product catalog (147 products)."""
        products_data = [
            # MATLAB Product Family (28 products)
            {"name": "MATLAB", "category": "MATLAB Product Family", "description": "Core mathematical computing platform", "key_features": ["matrix operations", "data visualization", "algorithm development", "app building"], "use_cases": ["data analysis", "algorithm prototyping", "mathematical modeling"], "target_personas": ["data scientists", "engineers", "researchers"]},
            {"name": "MATLAB Copilot", "category": "MATLAB Product Family", "description": "AI assistant for MATLAB", "key_features": ["AI-powered assistance", "code suggestions", "documentation help"], "use_cases": ["code development", "learning MATLAB", "productivity enhancement"], "target_personas": ["MATLAB users", "developers", "students"]},
            {"name": "Parallel Computing Toolbox", "category": "MATLAB Product Family", "description": "Distributed computing capabilities", "key_features": ["parallel loops", "GPU computing", "distributed arrays"], "use_cases": ["large-scale computations", "GPU acceleration", "cloud computing"], "target_personas": ["computational scientists", "HPC engineers"]},
            {"name": "MATLAB Parallel Server", "category": "MATLAB Product Family", "description": "Enterprise parallel processing", "key_features": ["cluster computing", "batch processing", "scalability"], "use_cases": ["enterprise computing", "large simulations", "batch jobs"], "target_personas": ["HPC administrators", "computational scientists"]},
            {"name": "Deep Learning Toolbox", "category": "MATLAB Product Family", "description": "Neural network design and training", "key_features": ["neural network design", "model training", "deployment"], "use_cases": ["image classification", "object detection", "AI deployment"], "target_personas": ["AI engineers", "data scientists", "ML engineers"]},
            {"name": "Statistics and Machine Learning Toolbox", "category": "MATLAB Product Family", "description": "Statistical analysis tools", "key_features": ["classification", "regression", "clustering", "feature selection"], "use_cases": ["predictive modeling", "data analysis", "feature engineering"], "target_personas": ["data scientists", "statisticians", "analysts"]},
            {"name": "Curve Fitting Toolbox", "category": "MATLAB Product Family", "description": "Data fitting functionality", "key_features": ["curve fitting", "regression", "interpolation"], "use_cases": ["data modeling", "trend analysis", "calibration"], "target_personas": ["engineers", "scientists", "analysts"]},
            {"name": "Text Analytics Toolbox", "category": "MATLAB Product Family", "description": "Text processing and analysis", "key_features": ["text mining", "NLP", "sentiment analysis"], "use_cases": ["document analysis", "text classification", "entity extraction"], "target_personas": ["data scientists", "NLP engineers"]},
            {"name": "Optimization Toolbox", "category": "MATLAB Product Family", "description": "Constrained/unconstrained optimization", "key_features": ["constrained optimization", "global optimization", "multiobjective optimization"], "use_cases": ["parameter optimization", "design optimization", "resource allocation"], "target_personas": ["engineers", "data scientists", "operations researchers"]},
            {"name": "Global Optimization Toolbox", "category": "MATLAB Product Family", "description": "Multi-extremum search methods", "key_features": ["genetic algorithms", "particle swarm", "simulated annealing"], "use_cases": ["complex optimization", "multiple local minima", "design problems"], "target_personas": ["optimization engineers", "researchers"]},
            {"name": "Symbolic Math Toolbox", "category": "MATLAB Product Family", "description": "Symbolic computation engine", "key_features": ["symbolic math", "calculus", "equation solving"], "use_cases": ["analytical solutions", "mathematical proofs", "formula derivation"], "target_personas": ["mathematicians", "researchers", "educators"]},
            {"name": "Mapping Toolbox", "category": "MATLAB Product Family", "description": "Geographic data visualization", "key_features": ["map projections", "GIS data", "geospatial analysis"], "use_cases": ["mapping", "geographic analysis", "spatial data"], "target_personas": ["GIS analysts", "geographers"]},
            {"name": "Partial Differential Equation Toolbox", "category": "MATLAB Product Family", "description": "PDE solving capabilities", "key_features": ["PDE solving", "finite element", "boundary conditions"], "use_cases": ["physics simulation", "engineering analysis", "heat transfer"], "target_personas": ["physicists", "engineers", "researchers"]},
            {"name": "MATLAB Compiler", "category": "MATLAB Product Family", "description": "Desktop application creation", "key_features": ["standalone apps", "royalty-free deployment", "encryption"], "use_cases": ["application deployment", "IP protection", "distribution"], "target_personas": ["developers", "software architects"]},
            {"name": "MATLAB Compiler SDK", "category": "MATLAB Product Family", "description": "Component packaging tools", "key_features": ["library creation", "integration", "multi-language support"], "use_cases": ["component development", "enterprise integration", "APIs"], "target_personas": ["integration engineers", "developers"]},
            {"name": "MATLAB Production Server", "category": "MATLAB Product Family", "description": "Application server deployment", "key_features": ["scalable deployment", "web services", "enterprise integration"], "use_cases": ["production deployment", "web applications", "enterprise systems"], "target_personas": ["DevOps engineers", "IT administrators"]},
            {"name": "MATLAB Web App Server", "category": "MATLAB Product Family", "description": "Web interface hosting platform", "key_features": ["web hosting", "browser access", "user management"], "use_cases": ["web applications", "internal tools", "dashboards"], "target_personas": ["web developers", "IT teams"]},
            {"name": "Database Toolbox", "category": "MATLAB Product Family", "description": "Database connectivity", "key_features": ["database access", "SQL queries", "data import/export"], "use_cases": ["data integration", "database analysis", "ETL"], "target_personas": ["data engineers", "analysts"]},
            {"name": "MATLAB Report Generator", "category": "MATLAB Product Family", "description": "Document generation tools", "key_features": ["automated reports", "templates", "documentation"], "use_cases": ["report automation", "documentation", "compliance"], "target_personas": ["engineers", "analysts", "researchers"]},
            {"name": "Requirements Toolbox", "category": "MATLAB Product Family", "description": "Requirements management system", "key_features": ["requirements tracing", "linking", "verification"], "use_cases": ["requirements management", "traceability", "compliance"], "target_personas": ["systems engineers", "project managers"]},
            {"name": "MATLAB Test", "category": "MATLAB Product Family", "description": "Testing framework", "key_features": ["unit testing", "test automation", "coverage analysis"], "use_cases": ["code testing", "quality assurance", "CI/CD"], "target_personas": ["test engineers", "developers"]},
            {"name": "MATLAB Coder", "category": "MATLAB Product Family", "description": "C/C++ code generation", "key_features": ["C/C++ generation", "MEX files", "optimization"], "use_cases": ["code generation", "embedded deployment", "performance"], "target_personas": ["embedded engineers", "developers"]},
            {"name": "Embedded Coder", "category": "MATLAB Product Family", "description": "Embedded systems code generation", "key_features": ["code generation", "code optimization", "MISRA compliance", "traceability"], "use_cases": ["production code generation", "embedded software", "automotive systems"], "target_personas": ["embedded software engineers", "software architects"]},
            {"name": "HDL Coder", "category": "MATLAB Product Family", "description": "Generate Verilog, SystemVerilog, and VHDL code", "key_features": ["HDL generation", "FPGA/ASIC", "verification"], "use_cases": ["FPGA development", "hardware design", "prototyping"], "target_personas": ["FPGA engineers", "hardware designers"]},
            {"name": "HDL Verifier", "category": "MATLAB Product Family", "description": "RTL verification utilities", "key_features": ["cosimulation", "verification", "FPGA-in-the-loop"], "use_cases": ["HDL verification", "FPGA testing", "validation"], "target_personas": ["verification engineers", "FPGA designers"]},
            {"name": "Filter Design HDL Coder", "category": "MATLAB Product Family", "description": "DSP filter HDL implementation", "key_features": ["filter HDL", "DSP implementation", "FPGA deployment"], "use_cases": ["filter design", "DSP hardware", "signal processing"], "target_personas": ["DSP engineers", "FPGA developers"]},
            {"name": "Fixed-Point Designer", "category": "MATLAB Product Family", "description": "Fixed-point system design", "key_features": ["fixed-point math", "data type optimization", "precision analysis"], "use_cases": ["embedded systems", "hardware design", "optimization"], "target_personas": ["embedded engineers", "hardware designers"]},
            {"name": "GPU Coder", "category": "MATLAB Product Family", "description": "GPU accelerated code generation", "key_features": ["CUDA generation", "GPU optimization", "acceleration"], "use_cases": ["GPU computing", "deep learning deployment", "HPC"], "target_personas": ["GPU developers", "HPC engineers"]},

            # Simulink Product Family (35 products)
            {"name": "Simulink", "category": "Simulink Product Family", "description": "Dynamic system simulation platform", "key_features": ["visual modeling", "simulation", "automatic code generation", "continuous testing"], "use_cases": ["control systems", "signal processing", "embedded systems", "communications"], "target_personas": ["control engineers", "embedded software engineers", "system architects"]},
            {"name": "Stateflow", "category": "Simulink Product Family", "description": "State machine modeling", "key_features": ["state machine design", "flow charts", "truth tables"], "use_cases": ["control logic", "supervisory control", "fault management"], "target_personas": ["control engineers", "software engineers"]},
            {"name": "SimEvents", "category": "Simulink Product Family", "description": "Discrete event simulation", "key_features": ["discrete event", "queuing", "process modeling"], "use_cases": ["manufacturing", "logistics", "operations"], "target_personas": ["operations engineers", "industrial engineers"]},
            {"name": "Simscape", "category": "Simulink Product Family", "description": "Physical system modeling", "key_features": ["physical modeling", "multidomain simulation", "component libraries"], "use_cases": ["mechanical systems", "electrical systems", "hydraulic systems"], "target_personas": ["mechanical engineers", "electrical engineers"]},
            {"name": "Simscape Battery", "category": "Simulink Product Family", "description": "Battery system modeling", "key_features": ["battery modeling", "thermal management", "BMS design"], "use_cases": ["battery design", "EV development", "energy storage"], "target_personas": ["battery engineers", "EV engineers"]},
            {"name": "Simscape Driveline", "category": "Simulink Product Family", "description": "Powertrain component modeling", "key_features": ["powertrain", "transmission", "drivetrain"], "use_cases": ["vehicle powertrain", "transmission design", "driveline analysis"], "target_personas": ["powertrain engineers", "automotive engineers"]},
            {"name": "Simscape Electrical", "category": "Simulink Product Family", "description": "Electrical circuit modeling", "key_features": ["circuit simulation", "power electronics", "motor drives"], "use_cases": ["power systems", "motor control", "grid simulation"], "target_personas": ["electrical engineers", "power engineers"]},
            {"name": "Simscape Fluids", "category": "Simulink Product Family", "description": "Fluid system simulation", "key_features": ["hydraulic", "pneumatic", "thermal fluids"], "use_cases": ["hydraulic systems", "HVAC", "thermal management"], "target_personas": ["fluid engineers", "HVAC engineers"]},
            {"name": "Simscape Multibody", "category": "Simulink Product Family", "description": "Mechanical system modeling", "key_features": ["3D mechanics", "kinematics", "dynamics"], "use_cases": ["mechanical design", "robotics", "vehicles"], "target_personas": ["mechanical engineers", "robotics engineers"]},
            {"name": "Simulink Report Generator", "category": "Simulink Product Family", "description": "Simulation report creation", "key_features": ["automated reports", "documentation", "templates"], "use_cases": ["design documentation", "compliance", "reporting"], "target_personas": ["engineers", "documentation teams"]},
            {"name": "System Composer", "category": "Simulink Product Family", "description": "Architecture design framework", "key_features": ["architecture modeling", "interfaces", "requirements"], "use_cases": ["system architecture", "design planning", "interfaces"], "target_personas": ["system architects", "systems engineers"]},
            {"name": "Simulink Fault Analyzer", "category": "Simulink Product Family", "description": "Fault injection testing", "key_features": ["fault injection", "FMEA", "safety analysis"], "use_cases": ["safety testing", "fault analysis", "certification"], "target_personas": ["safety engineers", "test engineers"]},
            {"name": "Simulink Coder", "category": "Simulink Product Family", "description": "Production C code generation", "key_features": ["C/C++ generation", "embedded code", "optimization"], "use_cases": ["production code", "embedded systems", "real-time"], "target_personas": ["embedded engineers", "software developers"]},
            {"name": "DDS Blockset", "category": "Simulink Product Family", "description": "Data distribution service blocks", "key_features": ["DDS protocol", "middleware", "communication"], "use_cases": ["distributed systems", "IoT", "aerospace"], "target_personas": ["systems engineers", "middleware developers"]},
            {"name": "AUTOSAR Blockset", "category": "Simulink Product Family", "description": "Automotive standard blocks", "key_features": ["AUTOSAR", "automotive software", "ECU development"], "use_cases": ["automotive ECUs", "AUTOSAR compliance", "vehicle software"], "target_personas": ["automotive software engineers", "ECU developers"]},
            {"name": "C2000 Microcontroller Blockset", "category": "Simulink Product Family", "description": "TI microcontroller support", "key_features": ["C2000 support", "real-time control", "code generation"], "use_cases": ["motor control", "power electronics", "embedded control"], "target_personas": ["embedded engineers", "control engineers"]},
            {"name": "Simulink PLC Coder", "category": "Simulink Product Family", "description": "PLC code generation", "key_features": ["PLC code", "ladder logic", "structured text"], "use_cases": ["industrial automation", "PLC programming", "control systems"], "target_personas": ["automation engineers", "PLC programmers"]},
            {"name": "Simulink Code Inspector", "category": "Simulink Product Family", "description": "Code quality analysis", "key_features": ["code review", "traceability", "certification"], "use_cases": ["code verification", "DO-178C", "ISO 26262"], "target_personas": ["verification engineers", "quality engineers"]},
            {"name": "DO Qualification Kit", "category": "Simulink Product Family", "description": "DO-178 certification support", "key_features": ["DO-178C", "certification", "aerospace"], "use_cases": ["avionics certification", "safety-critical software", "compliance"], "target_personas": ["certification engineers", "aerospace engineers"]},
            {"name": "IEC Certification Kit", "category": "Simulink Product Family", "description": "ISO 26262 and IEC 61508 support", "key_features": ["ISO 26262", "IEC 61508", "functional safety"], "use_cases": ["automotive safety", "industrial safety", "certification"], "target_personas": ["safety engineers", "certification teams"]},
            {"name": "Simulink Real-Time", "category": "Simulink Product Family", "description": "Real-time simulation execution", "key_features": ["real-time execution", "hardware-in-the-loop", "rapid prototyping"], "use_cases": ["HIL testing", "real-time control", "prototyping"], "target_personas": ["test engineers", "control engineers"]},
            {"name": "Simulink Desktop Real-Time", "category": "Simulink Product Family", "description": "Desktop real-time testing", "key_features": ["desktop real-time", "I/O support", "testing"], "use_cases": ["desktop HIL", "I/O testing", "prototyping"], "target_personas": ["test engineers", "developers"]},
            {"name": "Simulink Check", "category": "Simulink Product Family", "description": "Model quality checking", "key_features": ["model checking", "standards compliance", "quality metrics"], "use_cases": ["model quality", "compliance checking", "best practices"], "target_personas": ["quality engineers", "model developers"]},
            {"name": "Simulink Coverage", "category": "Simulink Product Family", "description": "Test coverage analysis", "key_features": ["coverage metrics", "test analysis", "requirements coverage"], "use_cases": ["test verification", "certification", "quality assurance"], "target_personas": ["test engineers", "verification engineers"]},
            {"name": "Simulink Design Verifier", "category": "Simulink Product Family", "description": "Formal verification tools", "key_features": ["formal verification", "property proving", "test generation"], "use_cases": ["design verification", "safety analysis", "test automation"], "target_personas": ["verification engineers", "safety engineers"]},
            {"name": "Simulink Test", "category": "Simulink Product Family", "description": "Test management framework", "key_features": ["test management", "automation", "requirements-based testing"], "use_cases": ["test automation", "regression testing", "certification"], "target_personas": ["test engineers", "QA teams"]},
            {"name": "Polyspace Bug Finder", "category": "Simulink Product Family", "description": "Static defect detection", "key_features": ["static analysis", "bug detection", "code quality"], "use_cases": ["code review", "defect detection", "quality improvement"], "target_personas": ["developers", "quality engineers"]},
            {"name": "Polyspace Bug Finder Server", "category": "Simulink Product Family", "description": "Server-based defect analysis", "key_features": ["server analysis", "team collaboration", "CI/CD integration"], "use_cases": ["enterprise analysis", "continuous inspection", "quality gates"], "target_personas": ["DevOps teams", "quality engineers"]},
            {"name": "Polyspace Code Prover", "category": "Simulink Product Family", "description": "Runtime error verification", "key_features": ["runtime error proof", "formal verification", "safety"], "use_cases": ["safety-critical code", "certification", "quality assurance"], "target_personas": ["safety engineers", "verification engineers"]},
            {"name": "Polyspace Test", "category": "Simulink Product Family", "description": "C/C++ test development", "key_features": ["test generation", "coverage", "verification"], "use_cases": ["unit testing", "test automation", "coverage analysis"], "target_personas": ["test engineers", "developers"]},
            {"name": "Polyspace Access", "category": "Simulink Product Family", "description": "Metrics review platform", "key_features": ["metrics dashboard", "team collaboration", "reporting"], "use_cases": ["quality metrics", "team review", "management reporting"], "target_personas": ["quality managers", "team leads"]},
            {"name": "Polyspace Code Prover Server", "category": "Simulink Product Family", "description": "Cluster-based verification", "key_features": ["distributed verification", "scalability", "enterprise"], "use_cases": ["large codebases", "enterprise verification", "CI/CD"], "target_personas": ["enterprise teams", "DevOps engineers"]},
            {"name": "Polyspace Client for Ada", "category": "Simulink Product Family", "description": "Ada code verification", "key_features": ["Ada support", "static analysis", "safety"], "use_cases": ["Ada development", "aerospace", "defense"], "target_personas": ["Ada developers", "aerospace engineers"]},
            {"name": "Polyspace Server for Ada", "category": "Simulink Product Family", "description": "Ada server verification", "key_features": ["server Ada analysis", "team collaboration", "enterprise"], "use_cases": ["enterprise Ada", "team verification", "compliance"], "target_personas": ["enterprise Ada teams", "quality engineers"]},
            {"name": "Simulink Compiler", "category": "Simulink Product Family", "description": "Simulink application deployment", "key_features": ["standalone deployment", "simulation execution", "distribution"], "use_cases": ["simulation deployment", "customer delivery", "IP protection"], "target_personas": ["developers", "system integrators"]},

            # Signal Processing (5 products)
            {"name": "Signal Processing Toolbox", "category": "Signal Processing", "description": "Signal analysis and processing", "key_features": ["filtering", "spectral analysis", "signal transforms"], "use_cases": ["audio processing", "communications", "sensor data analysis"], "target_personas": ["signal processing engineers", "communications engineers"]},
            {"name": "DSP System Toolbox", "category": "Signal Processing", "description": "Digital signal processing", "key_features": ["DSP algorithms", "streaming", "real-time"], "use_cases": ["DSP development", "audio systems", "communications"], "target_personas": ["DSP engineers", "algorithm developers"]},
            {"name": "Audio Toolbox", "category": "Signal Processing", "description": "Audio signal processing", "key_features": ["audio processing", "spatial audio", "acoustic analysis"], "use_cases": ["audio applications", "acoustic design", "sound processing"], "target_personas": ["audio engineers", "acoustic engineers"]},
            {"name": "Wavelet Toolbox", "category": "Signal Processing", "description": "Wavelet analysis tools", "key_features": ["wavelet transforms", "time-frequency analysis", "compression"], "use_cases": ["signal denoising", "compression", "feature extraction"], "target_personas": ["signal processing engineers", "researchers"]},
            {"name": "DSP HDL Toolbox", "category": "Signal Processing", "description": "Design digital signal processing for FPGAs", "key_features": ["DSP IP", "FPGA deployment", "HDL generation"], "use_cases": ["FPGA DSP", "hardware acceleration", "real-time processing"], "target_personas": ["FPGA engineers", "DSP developers"]},

            # RF and Mixed Signal (7 products)
            {"name": "Antenna Toolbox", "category": "RF and Mixed Signal", "description": "Antenna design and analysis", "key_features": ["antenna modeling", "array design", "analysis"], "use_cases": ["antenna design", "wireless systems", "5G"], "target_personas": ["RF engineers", "antenna designers"]},
            {"name": "RF Toolbox", "category": "RF and Mixed Signal", "description": "RF circuit design tools", "key_features": ["RF modeling", "amplifier design", "matching networks"], "use_cases": ["RF circuit design", "antenna design", "wireless systems"], "target_personas": ["RF engineers", "hardware engineers"]},
            {"name": "RF PCB Toolbox", "category": "RF and Mixed Signal", "description": "RF/PCB layout analysis", "key_features": ["PCB analysis", "parasitic extraction", "EM simulation"], "use_cases": ["PCB design", "signal integrity", "EMI analysis"], "target_personas": ["PCB designers", "RF engineers"]},
            {"name": "RF Blockset", "category": "RF and Mixed Signal", "description": "RF system simulation blocks", "key_features": ["RF simulation", "system modeling", "behavioral"], "use_cases": ["RF system design", "wireless transceivers", "radar"], "target_personas": ["RF system engineers", "wireless designers"]},
            {"name": "Mixed-Signal Blockset", "category": "RF and Mixed Signal", "description": "Mixed-signal simulation", "key_features": ["mixed-signal", "ADC/DAC", "PLL"], "use_cases": ["mixed-signal design", "data converters", "clock systems"], "target_personas": ["mixed-signal engineers", "IC designers"]},
            {"name": "SerDes Toolbox", "category": "RF and Mixed Signal", "description": "Serializer/deserializer design", "key_features": ["SerDes modeling", "link analysis", "compliance"], "use_cases": ["high-speed interfaces", "PCIe", "USB"], "target_personas": ["SerDes engineers", "signal integrity engineers"]},
            {"name": "Signal Integrity Toolbox", "category": "RF and Mixed Signal", "description": "Signal integrity analysis", "key_features": ["SI analysis", "crosstalk", "reflections"], "use_cases": ["high-speed design", "PCB analysis", "compliance"], "target_personas": ["signal integrity engineers", "PCB designers"]},

            # Automotive (10 products)
            {"name": "Model-Based Calibration Toolbox", "category": "Automotive", "description": "Engine calibration tools", "key_features": ["calibration", "DoE", "optimization"], "use_cases": ["engine calibration", "powertrain tuning", "emissions"], "target_personas": ["calibration engineers", "powertrain engineers"]},
            {"name": "Powertrain Blockset", "category": "Automotive", "description": "Powertrain system simulation", "key_features": ["powertrain modeling", "thermal", "efficiency"], "use_cases": ["powertrain development", "hybrid vehicles", "emissions"], "target_personas": ["powertrain engineers", "automotive engineers"]},
            {"name": "Vehicle Dynamics Blockset", "category": "Automotive", "description": "Vehicle dynamics modeling", "key_features": ["vehicle modeling", "chassis simulation", "powertrain modeling"], "use_cases": ["autonomous vehicles", "ADAS", "vehicle testing"], "target_personas": ["automotive engineers", "vehicle dynamics engineers"]},
            {"name": "Automated Driving Toolbox", "category": "Automotive", "description": "ADAS system development", "key_features": ["sensor fusion", "path planning", "scenario generation"], "use_cases": ["ADAS development", "autonomous driving", "sensor testing"], "target_personas": ["ADAS engineers", "autonomy engineers"]},
            {"name": "Vehicle Network Toolbox", "category": "Automotive", "description": "Vehicle communication protocol support", "key_features": ["CAN", "LIN", "FlexRay", "XCP"], "use_cases": ["vehicle networks", "ECU testing", "diagnostics"], "target_personas": ["automotive engineers", "test engineers"]},
            {"name": "RoadRunner", "category": "Automotive", "description": "3D scene design environment", "key_features": ["3D scene design", "road modeling", "export"], "use_cases": ["ADAS testing", "autonomous driving", "simulation"], "target_personas": ["ADAS engineers", "simulation engineers"]},
            {"name": "RoadRunner Asset Library", "category": "Automotive", "description": "3D asset collection", "key_features": ["3D assets", "vehicles", "props"], "use_cases": ["scene creation", "simulation", "visualization"], "target_personas": ["simulation engineers", "3D designers"]},
            {"name": "RoadRunner Scenario", "category": "Automotive", "description": "Scenario creation and playback", "key_features": ["scenario design", "traffic", "testing"], "use_cases": ["ADAS testing", "scenario generation", "validation"], "target_personas": ["test engineers", "ADAS engineers"]},
            {"name": "RoadRunner Scene Builder", "category": "Automotive", "description": "Scene creation tools", "key_features": ["scene builder", "procedural", "library"], "use_cases": ["rapid scene creation", "batch generation", "testing"], "target_personas": ["test engineers", "simulation engineers"]},
            {"name": "Simulink 3D Animation", "category": "Automotive", "description": "3D visualization environment", "key_features": ["3D visualization", "animation", "VRML"], "use_cases": ["visualization", "animation", "presentations"], "target_personas": ["engineers", "researchers", "educators"]},

            # Test and Measurement (5 products)
            {"name": "Data Acquisition Toolbox", "category": "Test and Measurement", "description": "Hardware data collection", "key_features": ["DAQ support", "analog I/O", "digital I/O"], "use_cases": ["test automation", "data logging", "sensor interfacing"], "target_personas": ["test engineers", "lab engineers"]},
            {"name": "Instrument Control Toolbox", "category": "Test and Measurement", "description": "Test equipment communication", "key_features": ["instrument control", "VISA", "SCPI"], "use_cases": ["test automation", "lab automation", "measurement"], "target_personas": ["test engineers", "lab engineers"]},
            {"name": "Image Acquisition Toolbox", "category": "Test and Measurement", "description": "Camera and video capture", "key_features": ["camera interface", "video capture", "image acquisition"], "use_cases": ["vision systems", "inspection", "monitoring"], "target_personas": ["vision engineers", "test engineers"]},
            {"name": "Industrial Communication Toolbox", "category": "Test and Measurement", "description": "Industrial protocol support", "key_features": ["OPC", "Modbus", "MQTT"], "use_cases": ["industrial IoT", "SCADA", "automation"], "target_personas": ["automation engineers", "IoT developers"]},
            {"name": "ThingSpeak", "category": "Test and Measurement", "description": "IoT data collection platform", "key_features": ["cloud IoT", "analytics", "visualization"], "use_cases": ["IoT applications", "sensor networks", "monitoring"], "target_personas": ["IoT developers", "engineers"]},

            # Teaching and Learning (2 products)
            {"name": "MATLAB Grader", "category": "Teaching and Learning", "description": "Automated homework grading", "key_features": ["autograding", "feedback", "LMS integration"], "use_cases": ["education", "course management", "assessment"], "target_personas": ["educators", "instructors"]},
            {"name": "Online Training Suite", "category": "Teaching and Learning", "description": "Interactive learning platform", "key_features": ["online courses", "interactive", "certification"], "use_cases": ["training", "skill development", "learning"], "target_personas": ["learners", "professionals", "students"]},

            # Image Processing and Computer Vision (5 products)
            {"name": "Image Processing Toolbox", "category": "Image Processing and Computer Vision", "description": "Image analysis tools", "key_features": ["image enhancement", "segmentation", "morphology", "registration"], "use_cases": ["medical imaging", "quality inspection", "scientific imaging"], "target_personas": ["image processing engineers", "researchers"]},
            {"name": "Computer Vision Toolbox", "category": "Image Processing and Computer Vision", "description": "Vision algorithm development", "key_features": ["image processing", "object detection", "tracking", "3D vision"], "use_cases": ["object recognition", "image analysis", "video surveillance"], "target_personas": ["computer vision engineers", "image processing engineers"]},
            {"name": "Lidar Toolbox", "category": "Image Processing and Computer Vision", "description": "LiDAR data processing", "key_features": ["point cloud", "3D perception", "SLAM"], "use_cases": ["autonomous driving", "robotics", "mapping"], "target_personas": ["autonomy engineers", "robotics engineers"]},
            {"name": "Medical Imaging Toolbox", "category": "Image Processing and Computer Vision", "description": "Medical image analysis", "key_features": ["DICOM", "medical imaging", "visualization"], "use_cases": ["medical imaging", "diagnostics", "research"], "target_personas": ["medical engineers", "researchers"]},
            {"name": "Vision HDL Toolbox", "category": "Image Processing and Computer Vision", "description": "Design image processing for FPGAs", "key_features": ["vision IP", "FPGA deployment", "HDL generation"], "use_cases": ["FPGA vision", "real-time imaging", "hardware acceleration"], "target_personas": ["FPGA engineers", "vision developers"]},

            # Wireless Communications (8 products)
            {"name": "Communications Toolbox", "category": "Wireless Communications", "description": "Wireless system design", "key_features": ["modulation", "channel modeling", "error correction"], "use_cases": ["wireless communications", "5G/6G", "satellite communications"], "target_personas": ["communications engineers", "RF engineers"]},
            {"name": "5G Toolbox", "category": "Wireless Communications", "description": "5G system modeling", "key_features": ["5G NR", "mmWave", "beamforming"], "use_cases": ["5G development", "wireless research", "testing"], "target_personas": ["5G engineers", "wireless researchers"]},
            {"name": "LTE Toolbox", "category": "Wireless Communications", "description": "LTE/4G development", "key_features": ["LTE", "4G", "baseband"], "use_cases": ["LTE development", "wireless testing", "research"], "target_personas": ["LTE engineers", "wireless developers"]},
            {"name": "WLAN Toolbox", "category": "Wireless Communications", "description": "Wi-Fi system design", "key_features": ["Wi-Fi", "802.11", "OFDM"], "use_cases": ["Wi-Fi development", "wireless LANs", "IoT"], "target_personas": ["Wi-Fi engineers", "wireless developers"]},
            {"name": "Bluetooth Toolbox", "category": "Wireless Communications", "description": "Bluetooth protocol support", "key_features": ["Bluetooth", "BLE", "mesh"], "use_cases": ["Bluetooth development", "IoT", "wearables"], "target_personas": ["Bluetooth developers", "IoT engineers"]},
            {"name": "Satellite Communications Toolbox", "category": "Wireless Communications", "description": "Satellite system design", "key_features": ["satellite links", "orbits", "ground stations"], "use_cases": ["satellite communications", "link budgets", "mission design"], "target_personas": ["satellite engineers", "aerospace engineers"]},
            {"name": "Wireless HDL Toolbox", "category": "Wireless Communications", "description": "Design wireless communications for FPGAs", "key_features": ["wireless IP", "FPGA deployment", "HDL generation"], "use_cases": ["FPGA wireless", "hardware acceleration", "prototyping"], "target_personas": ["FPGA engineers", "wireless developers"]},
            {"name": "Wireless Testbench", "category": "Wireless Communications", "description": "Wireless testing platform", "key_features": ["over-the-air testing", "RF impairments", "validation"], "use_cases": ["wireless testing", "RF validation", "compliance"], "target_personas": ["test engineers", "RF engineers"]},

            # Aerospace (3 products)
            {"name": "Aerospace Blockset", "category": "Aerospace", "description": "Aircraft system simulation", "key_features": ["flight dynamics", "propulsion", "environment"], "use_cases": ["aircraft design", "flight simulation", "GNC"], "target_personas": ["aerospace engineers", "flight dynamics engineers"]},
            {"name": "Aerospace Toolbox", "category": "Aerospace", "description": "Aerospace engineering tools", "key_features": ["coordinate systems", "environment models", "standards"], "use_cases": ["aerospace analysis", "mission planning", "simulation"], "target_personas": ["aerospace engineers", "systems engineers"]},
            {"name": "UAV Toolbox", "category": "Aerospace", "description": "Unmanned vehicle development", "key_features": ["UAV simulation", "path planning", "GCS"], "use_cases": ["UAV development", "autonomous flight", "mission planning"], "target_personas": ["UAV engineers", "autonomy engineers"]},

            # FPGA, ASIC, and SoC Development (2 products)
            {"name": "Deep Learning HDL Toolbox", "category": "FPGA, ASIC, and SoC Development", "description": "FPGA neural network deployment", "key_features": ["DL to HDL", "FPGA deployment", "optimization"], "use_cases": ["edge AI", "FPGA inference", "hardware acceleration"], "target_personas": ["FPGA engineers", "AI engineers"]},
            {"name": "SoC Blockset", "category": "FPGA, ASIC, and SoC Development", "description": "Hardware/software co-design", "key_features": ["SoC modeling", "HW/SW codesign", "prototyping"], "use_cases": ["SoC development", "embedded systems", "FPGA prototyping"], "target_personas": ["SoC engineers", "embedded developers"]},

            # Control Systems (10 products)
            {"name": "Control System Toolbox", "category": "Control Systems", "description": "Control system design", "key_features": ["PID tuning", "linear control design", "frequency analysis"], "use_cases": ["control system design", "stability analysis", "controller tuning"], "target_personas": ["control engineers", "systems engineers"]},
            {"name": "System Identification Toolbox", "category": "Control Systems", "description": "Dynamic system modeling", "key_features": ["system ID", "parameter estimation", "model validation"], "use_cases": ["model identification", "parameter tuning", "validation"], "target_personas": ["control engineers", "modeling engineers"]},
            {"name": "Predictive Maintenance Toolbox", "category": "Control Systems", "description": "Condition monitoring", "key_features": ["condition monitoring", "fault detection", "RUL prediction"], "use_cases": ["predictive maintenance", "condition monitoring", "diagnostics"], "target_personas": ["reliability engineers", "maintenance engineers"]},
            {"name": "Robust Control Toolbox", "category": "Control Systems", "description": "Uncertain system design", "key_features": ["robust control", "uncertainty", "mu-synthesis"], "use_cases": ["robust design", "uncertain systems", "aerospace"], "target_personas": ["control engineers", "aerospace engineers"]},
            {"name": "Model Predictive Control Toolbox", "category": "Control Systems", "description": "MPC strategy design", "key_features": ["MPC", "constraints", "optimization"], "use_cases": ["advanced control", "process control", "automotive"], "target_personas": ["control engineers", "process engineers"]},
            {"name": "Fuzzy Logic Toolbox", "category": "Control Systems", "description": "Fuzzy logic systems", "key_features": ["fuzzy logic", "inference", "tuning"], "use_cases": ["fuzzy control", "decision making", "consumer products"], "target_personas": ["control engineers", "AI engineers"]},
            {"name": "Simulink Control Design", "category": "Control Systems", "description": "Control design tools", "key_features": ["control tuning", "linearization", "analysis"], "use_cases": ["control design", "system analysis", "tuning"], "target_personas": ["control engineers", "systems engineers"]},
            {"name": "Simulink Design Optimization", "category": "Control Systems", "description": "Parameter optimization", "key_features": ["parameter tuning", "optimization", "sensitivity"], "use_cases": ["design optimization", "parameter tuning", "calibration"], "target_personas": ["engineers", "optimization engineers"]},
            {"name": "Reinforcement Learning Toolbox", "category": "Control Systems", "description": "RL algorithm development", "key_features": ["RL algorithms", "environment modeling", "policy training"], "use_cases": ["robotics control", "game AI", "autonomous systems"], "target_personas": ["AI engineers", "robotics engineers"]},
            {"name": "Motor Control Blockset", "category": "Control Systems", "description": "Motor control simulation", "key_features": ["motor control", "FOC", "sensorless"], "use_cases": ["motor control", "inverter design", "e-mobility"], "target_personas": ["motor control engineers", "power electronics engineers"]},

            # Radar (3 products)
            {"name": "Radar Toolbox", "category": "Radar", "description": "Radar system design", "key_features": ["radar modeling", "waveform design", "processing"], "use_cases": ["radar development", "automotive radar", "defense"], "target_personas": ["radar engineers", "signal processing engineers"]},
            {"name": "Phased Array System Toolbox", "category": "Radar", "description": "Phased array design", "key_features": ["phased arrays", "beamforming", "direction finding"], "use_cases": ["phased arrays", "radar", "wireless"], "target_personas": ["antenna engineers", "radar engineers"]},
            {"name": "Sensor Fusion and Tracking Toolbox", "category": "Radar", "description": "Multi-sensor fusion", "key_features": ["sensor fusion", "tracking", "Kalman filters"], "use_cases": ["ADAS", "autonomous systems", "surveillance"], "target_personas": ["sensor fusion engineers", "autonomy engineers"]},

            # Computational Finance (6 products)
            {"name": "Datafeed Toolbox", "category": "Computational Finance", "description": "Financial data retrieval", "key_features": ["data feeds", "market data", "APIs"], "use_cases": ["trading", "analytics", "research"], "target_personas": ["quants", "traders", "analysts"]},
            {"name": "Econometrics Toolbox", "category": "Computational Finance", "description": "Time series econometrics", "key_features": ["econometrics", "time series", "forecasting"], "use_cases": ["forecasting", "economic modeling", "research"], "target_personas": ["economists", "quants", "researchers"]},
            {"name": "Financial Toolbox", "category": "Computational Finance", "description": "Quantitative finance tools", "key_features": ["pricing", "portfolio", "risk"], "use_cases": ["derivatives pricing", "portfolio management", "risk"], "target_personas": ["quants", "risk managers", "traders"]},
            {"name": "Financial Instruments Toolbox", "category": "Computational Finance", "description": "Fixed-income analytics", "key_features": ["bonds", "swaps", "credit"], "use_cases": ["fixed income", "derivatives", "valuation"], "target_personas": ["fixed income analysts", "quants"]},
            {"name": "Risk Management Toolbox", "category": "Computational Finance", "description": "Risk analysis tools", "key_features": ["VaR", "credit risk", "backtesting"], "use_cases": ["risk management", "compliance", "stress testing"], "target_personas": ["risk managers", "compliance officers"]},
            {"name": "Spreadsheet Link", "category": "Computational Finance", "description": "Excel integration tools", "key_features": ["Excel integration", "data exchange", "automation"], "use_cases": ["Excel automation", "reporting", "integration"], "target_personas": ["analysts", "engineers"]},

            # Computational Biology (2 products)
            {"name": "Bioinformatics Toolbox", "category": "Computational Biology", "description": "Sequence and genomics analysis", "key_features": ["genomics", "sequence analysis", "proteomics"], "use_cases": ["genomics research", "bioinformatics", "drug discovery"], "target_personas": ["bioinformaticians", "researchers"]},
            {"name": "SimBiology", "category": "Computational Biology", "description": "Systems biology simulation", "key_features": ["pharmacokinetics", "pathway modeling", "simulation"], "use_cases": ["drug development", "systems biology", "PKP modeling"], "target_personas": ["systems biologists", "pharmaceutical engineers"]},

            # Robotics and Autonomous Systems (3 products)
            {"name": "Robotics System Toolbox", "category": "Robotics and Autonomous Systems", "description": "Robot algorithm development", "key_features": ["motion planning", "kinematics", "ROS integration"], "use_cases": ["robot control", "path planning", "manipulation"], "target_personas": ["robotics engineers", "automation engineers"]},
            {"name": "Navigation Toolbox", "category": "Robotics and Autonomous Systems", "description": "Autonomous navigation design", "key_features": ["path planning", "localization", "SLAM"], "use_cases": ["autonomous navigation", "robotics", "vehicles"], "target_personas": ["autonomy engineers", "robotics engineers"]},
            {"name": "ROS Toolbox", "category": "Robotics and Autonomous Systems", "description": "Robot Operating System support", "key_features": ["ROS integration", "message passing", "simulation"], "use_cases": ["ROS development", "robotics", "research"], "target_personas": ["robotics engineers", "researchers"]},

            # Cloud Solutions (5 products)
            {"name": "MATLAB Online", "category": "Cloud Solutions", "description": "Browser-based MATLAB access", "key_features": ["cloud MATLAB", "browser access", "collaboration"], "use_cases": ["remote work", "education", "collaboration"], "target_personas": ["engineers", "students", "researchers"]},
            {"name": "MATLAB Online Server", "category": "Cloud Solutions", "description": "Enterprise cloud deployment", "key_features": ["private cloud", "enterprise", "security"], "use_cases": ["enterprise deployment", "secure access", "collaboration"], "target_personas": ["IT administrators", "enterprise users"]},
            {"name": "MATLAB Drive", "category": "Cloud Solutions", "description": "Cloud file storage", "key_features": ["cloud storage", "sync", "sharing"], "use_cases": ["file storage", "collaboration", "backup"], "target_personas": ["MATLAB users", "teams"]},
            {"name": "MATLAB Mobile", "category": "Cloud Solutions", "description": "Mobile app interface", "key_features": ["mobile access", "sensors", "cloud connection"], "use_cases": ["mobile computing", "sensor data", "monitoring"], "target_personas": ["engineers", "students", "field users"]},
            {"name": "Simulink Online", "category": "Cloud Solutions", "description": "Browser-based Simulink", "key_features": ["cloud Simulink", "browser access", "collaboration"], "use_cases": ["remote work", "education", "collaboration"], "target_personas": ["engineers", "students", "researchers"]},
        ]

        products = [Product(**data) for data in products_data]
        logger.info("complete_products_loaded", count=len(products))
        return products

    async def _scrape_company_products(self) -> list[Product]:
        """
        Scrape company website for product information.

        Generic implementation that can be extended for specific companies.

        Returns:
            List of enriched Product objects or empty list if scraping fails
        """
        try:
            # Search for company products page
            async with DuckDuckGoMCPClient() as client:
                query = f"{self.company_name} products catalog"
                results = await client.search(query, max_results=5)

                if not results:
                    return []

                # Try to fetch product listing page
                for result in results:
                    url_str = str(result.url).lower()
                    if "product" in url_str or "solution" in url_str or "offering" in url_str:
                        logger.info(
                            "fetching_products_page",
                            company=self.company_name,
                            url=result.url
                        )
                        # Future: Parse product listing page with LLM
                        # For now, return empty to use other methods
                        break

            return []

        except Exception as e:
            logger.warning(
                "scrape_company_failed",
                company=self.company_name,
                error=str(e)
            )
            return []

    async def build_catalog_from_url(self, url: str) -> list[Product]:
        """
        Build product catalog by scraping a URL and extracting products with LLM.

        Args:
            url: URL of the product catalog page

        Returns:
            List of Product objects extracted from the page
        """
        from src.core.model_router import ModelRouter

        try:
            # Fetch URL content
            logger.info("fetching_catalog_url", url=url)
            content = await fetch_url(url)

            if not content:
                logger.warning("url_fetch_empty", url=url)
                return []

            # Extract text from HTML
            text = extract_text(content)

            if not text or len(text) < 100:
                logger.warning("url_content_too_short", url=url, length=len(text) if text else 0)
                return []

            # Use LLM to extract product information
            return await self._extract_products_with_llm(text, source=url)

        except Exception as e:
            logger.error("url_catalog_extraction_failed", url=url, error=str(e))
            return []

    async def build_catalog_from_document(self, file_path: str) -> list[Product]:
        """
        Build product catalog by reading a document file and extracting products with LLM.

        Supports: .txt, .md, .json

        Args:
            file_path: Path to the document file

        Returns:
            List of Product objects extracted from the document
        """
        from pathlib import Path

        file_path = Path(file_path)

        if not file_path.exists():
            logger.error("document_not_found", path=str(file_path))
            return []

        try:
            # Read file content
            logger.info("reading_catalog_document", path=str(file_path))

            if file_path.suffix.lower() == '.json':
                # JSON files are handled by the regular build_catalog method
                return self._load_from_json(file_path)

            # Read text content
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            if not text or len(text) < 50:
                logger.warning("document_content_too_short", path=str(file_path), length=len(text) if text else 0)
                return []

            # Use LLM to extract product information
            return await self._extract_products_with_llm(text, source=str(file_path))

        except Exception as e:
            logger.error("document_catalog_extraction_failed", path=str(file_path), error=str(e))
            return []

    async def _extract_products_with_llm(self, text: str, source: str) -> list[Product]:
        """
        Use LLM to extract structured product information from unstructured text.

        Args:
            text: Raw text content containing product information
            source: Source of the text (URL or file path) for logging

        Returns:
            List of Product objects
        """
        from src.core.model_router import ModelRouter

        # Truncate very long content
        max_chars = 50000
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.info("text_truncated", source=source, max_chars=max_chars)

        prompt = f"""Extract all products/solutions from the following text.

For each product, provide:
- name: Product name
- category: Product category or family
- description: Brief description (1-2 sentences)
- key_features: List of 3-5 key features
- use_cases: List of 2-4 common use cases
- target_personas: List of 2-4 target user types

Return a JSON array of product objects. Only include products/solutions that are clearly described.
If no products can be extracted, return an empty array [].

Text:
{text}

JSON Array:"""

        try:
            model_router = ModelRouter()
            response = await model_router.generate(
                prompt=prompt,
                task_type="extraction",
                temperature=0.1,  # Low temperature for structured extraction
                max_tokens=8000
            )

            # Parse JSON response
            response_text = response.content.strip()

            # Find JSON array in response
            start = response_text.find('[')
            end = response_text.rfind(']') + 1

            if start == -1 or end == 0:
                logger.warning("no_json_array_in_response", source=source)
                return []

            json_str = response_text[start:end]
            products_data = json.loads(json_str)

            # Convert to Product objects
            products = []
            for data in products_data:
                try:
                    # Ensure required fields have defaults
                    product_data = {
                        "name": data.get("name", "Unknown"),
                        "category": data.get("category", "General"),
                        "description": data.get("description", ""),
                        "key_features": data.get("key_features", []),
                        "use_cases": data.get("use_cases", []),
                        "target_personas": data.get("target_personas", [])
                    }
                    products.append(Product(**product_data))
                except Exception as e:
                    logger.warning("product_parse_failed", data=data, error=str(e))
                    continue

            logger.info(
                "products_extracted_with_llm",
                source=source,
                count=len(products)
            )
            return products

        except json.JSONDecodeError as e:
            logger.error("json_parse_failed", source=source, error=str(e))
            return []
        except Exception as e:
            logger.error("llm_extraction_failed", source=source, error=str(e))
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

        logger.info("products_indexed", count=len(products), collection=self.collection_name)

    async def scrape_and_index_solutions(
        self,
        base_url: str = "https://www.mathworks.com",
    ) -> int | None:
        """Fetch MathWorks solution pages via Tavily Extract and index into ChromaDB.

        Each solution page is stored with metadata type='solution' so it can be
        distinguished from product docs. Solution docs enrich hybrid search recall
        but are filtered out of final product match output.

        Returns:
            None  — idempotency check passed (already ≥50 solution docs exist)
            int   — count of newly indexed docs from this run (may be 0 if all failed)

        Idempotent — safe to re-run. To force re-index, delete solution docs first.
        """
        from src.data_sources.tavily_client import TavilyClient

        # Idempotency check — skip if already indexed
        try:
            existing = self.collection.get(where={"type": {"$eq": "solution"}})
            if len(existing.get("ids", [])) >= 50:
                logger.info(
                    "solutions_already_indexed",
                    count=len(existing["ids"]),
                    message="Skipping — delete solution docs to re-index",
                )
                return None
        except Exception:
            pass  # collection may not support where-filter yet; proceed

        tavily = TavilyClient()
        documents, metadatas, ids = [], [], []
        indexed = 0

        # Build full URL list
        all_entries = [
            (i, entry, base_url + entry["url"])
            for i, entry in enumerate(MATHWORKS_SOLUTION_URLS)
        ]

        # Batch fetch — Tavily Extract supports max 20 URLs per call
        BATCH_SIZE = 20
        for batch_start in range(0, len(all_entries), BATCH_SIZE):
            batch = all_entries[batch_start : batch_start + BATCH_SIZE]
            batch_urls = [url for _, _, url in batch]

            content_by_url = await tavily.fetch_content_batch(batch_urls)

            for i, entry, url in batch:
                content = content_by_url.get(url, "")
                if not content:
                    logger.warning("solution_page_no_content", url=url)
                    continue

                category = entry["category"]
                # Derive title from URL slug (Tavily Extract doesn't return titles)
                slug = entry["url"].rstrip("/").split("/")[-1].replace(".html", "")
                title = slug.replace("-", " ").title()

                doc = (
                    f"Solution: {title}\n"
                    f"Category: {category}\n"
                    f"URL: {url}\n"
                    f"Content: {content[:3000]}"
                )

                documents.append(doc)
                metadatas.append({
                    "type": "solution",
                    "name": title,
                    "category": category,
                    "url": url,
                })
                ids.append(f"solution_{i}")
                indexed += 1
                logger.info("solution_page_fetched", title=title, url=url)

        if documents:
            self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(
                "solutions_indexed",
                count=indexed,
                collection=self.collection_name,
            )

        return indexed


# ---------------------------------------------------------------------------
# Solution URLs to scrape from mathworks.com/solutions.html
# ---------------------------------------------------------------------------

MATHWORKS_SOLUTION_URLS: list[dict] = [
    # Applications
    {"url": "/solutions/artificial-intelligence.html", "category": "Applications"},
    {"url": "/solutions/automated-driving.html", "category": "Applications"},
    {"url": "/solutions/computational-biology.html", "category": "Applications"},
    {"url": "/solutions/control-systems.html", "category": "Applications"},
    {"url": "/solutions/electrification.html", "category": "Applications"},
    {"url": "/solutions/embedded-ai.html", "category": "Applications"},
    {"url": "/solutions/embedded-systems.html", "category": "Applications"},
    {"url": "/solutions/enterprise-it-systems.html", "category": "Applications"},
    {"url": "/solutions/fpga-asic-soc-development.html", "category": "Applications"},
    {"url": "/solutions/image-processing-computer-vision.html", "category": "Applications"},
    {"url": "/solutions/internet-of-things.html", "category": "Applications"},
    {"url": "/solutions/mechatronics.html", "category": "Applications"},
    {"url": "/solutions/mixed-signal-systems.html", "category": "Applications"},
    {"url": "/solutions/predictive-maintenance.html", "category": "Applications"},
    {"url": "/solutions/aerospace-defense/radar-systems.html", "category": "Applications"},
    {"url": "/solutions/robotics.html", "category": "Applications"},
    {"url": "/solutions/signal-processing.html", "category": "Applications"},
    {"url": "/solutions/test-measurement.html", "category": "Applications"},
    {"url": "/solutions/wireless-communications.html", "category": "Applications"},
    # Industries
    {"url": "/solutions/aerospace-defense.html", "category": "Industries"},
    {"url": "/solutions/aerospace-defense/space-systems.html", "category": "Industries"},
    {"url": "/solutions/aerospace-defense/rf-systems.html", "category": "Industries"},
    {"url": "/solutions/aerospace-defense/maritime-systems.html", "category": "Industries"},
    {"url": "/solutions/aerospace-defense/digital-transformation.html", "category": "Industries"},
    {"url": "/solutions/agriculture.html", "category": "Industries"},
    {"url": "/solutions/automotive.html", "category": "Industries"},
    {"url": "/solutions/automotive/virtual-vehicle.html", "category": "Industries"},
    {"url": "/solutions/automotive/electric-vehicle.html", "category": "Industries"},
    {"url": "/solutions/automotive/software-defined-vehicle.html", "category": "Industries"},
    {"url": "/solutions/biotech-pharmaceutical.html", "category": "Industries"},
    {"url": "/solutions/communications.html", "category": "Industries"},
    {"url": "/solutions/electronics.html", "category": "Industries"},
    {"url": "/solutions/energy-production.html", "category": "Industries"},
    {"url": "/solutions/energy-production/utilities-energy.html", "category": "Industries"},
    {"url": "/solutions/energy-production/utilities-energy/power-system-studies.html", "category": "Industries"},
    {"url": "/solutions/energy-production/utilities-energy/grid-analytics.html", "category": "Industries"},
    {"url": "/solutions/industrial-automation-machinery.html", "category": "Industries"},
    {"url": "/solutions/industrial-automation-machinery/machine-builders.html", "category": "Industries"},
    {"url": "/solutions/industrial-automation-machinery/automation-components.html", "category": "Industries"},
    {"url": "/solutions/medical-devices.html", "category": "Industries"},
    {"url": "/solutions/medical-devices/therapeutic-devices.html", "category": "Industries"},
    {"url": "/solutions/medical-devices/medical-imaging.html", "category": "Industries"},
    {"url": "/solutions/medical-devices/fda-software-validation.html", "category": "Industries"},
    {"url": "/solutions/mining.html", "category": "Industries"},
    {"url": "/solutions/finance-and-risk-management.html", "category": "Industries"},
    {"url": "/solutions/finance-and-risk-management/machine-learning.html", "category": "Industries"},
    {"url": "/solutions/railway-systems.html", "category": "Industries"},
    {"url": "/solutions/semiconductors.html", "category": "Industries"},
    {"url": "/solutions/software-internet.html", "category": "Industries"},
    # Disciplines
    {"url": "/solutions/biological-sciences.html", "category": "Disciplines"},
    {"url": "/solutions/chemical-engineering.html", "category": "Disciplines"},
    {"url": "/solutions/chemistry.html", "category": "Disciplines"},
    {"url": "/solutions/electrical-computer-engineering.html", "category": "Disciplines"},
    {"url": "/solutions/geoscience.html", "category": "Disciplines"},
    {"url": "/solutions/mathematics.html", "category": "Disciplines"},
    {"url": "/solutions/mechanical-engineering.html", "category": "Disciplines"},
    {"url": "/solutions/neuroscience.html", "category": "Disciplines"},
    {"url": "/solutions/physics.html", "category": "Disciplines"},
    # Capabilities
    {"url": "/solutions/cloud.html", "category": "Capabilities"},
    {"url": "/solutions/deployment.html", "category": "Capabilities"},
    {"url": "/solutions/discrete-event-simulation.html", "category": "Capabilities"},
    {"url": "/solutions/embedded-code-generation.html", "category": "Capabilities"},
    {"url": "/solutions/embedded-security.html", "category": "Capabilities"},
    {"url": "/solutions/functional-safety.html", "category": "Capabilities"},
    {"url": "/solutions/gpu-computing.html", "category": "Capabilities"},
    {"url": "/solutions/model-based-design.html", "category": "Capabilities"},
    {"url": "/solutions/model-deployment.html", "category": "Capabilities"},
    {"url": "/solutions/parallel-computing.html", "category": "Capabilities"},
    {"url": "/solutions/parallel-simulation.html", "category": "Capabilities"},
    {"url": "/solutions/physical-modeling.html", "category": "Capabilities"},
    {"url": "/solutions/real-time-simulation-and-testing.html", "category": "Capabilities"},
    {"url": "/solutions/report-generation.html", "category": "Capabilities"},
    {"url": "/solutions/software-architectures.html", "category": "Capabilities"},
    {"url": "/solutions/model-based-systems-engineering.html", "category": "Capabilities"},
    {"url": "/solutions/verification-validation.html", "category": "Capabilities"},
]


class ProductMatcher:
    """Matches requirements to products using semantic search."""

    def __init__(
        self,
        company_name: str,
        db_path: str = "./data/chroma",
        collection_name: Optional[str] = None
    ):
        """
        Initialize product matcher.

        Args:
            company_name: Name of the company (must match ProductCatalogIndexer)
            db_path: Path to ChromaDB storage
            collection_name: Optional custom collection name (defaults to "{company_name}_products")
        """
        self.company_name = company_name
        self.db_path = Path(db_path)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Use company-specific collection name
        self.collection_name = collection_name or f"{company_name.lower().replace(' ', '_')}_products"

        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )
        except Exception as e:
            logger.error(
                "collection_not_found",
                company=company_name,
                collection=self.collection_name,
                error=str(e)
            )
            raise DataSourceError(
                f"Product catalog not indexed for {company_name}. "
                f"Run ProductCatalogIndexer.build_catalog() and index_products() first."
            )

        # Build BM25 index from all ChromaDB documents (products + solution enrichments)
        all_data = self.collection.get(include=["documents", "metadatas"])
        self._bm25_docs: list[str] = all_data.get("documents") or []
        self._bm25_names: list[str] = [
            m.get("name", "") for m in (all_data.get("metadatas") or [])
        ]
        self._bm25_index = BM25Okapi(
            [doc.lower().split() for doc in self._bm25_docs]
        ) if self._bm25_docs else None

        # Track product names only (exclude solution enrichment docs from output)
        self._product_names: set[str] = {
            m.get("name", "")
            for m in (all_data.get("metadatas") or [])
            if m.get("type", "product") != "solution"
        }

        logger.info(
            "bm25_index_built",
            total_docs=len(self._bm25_docs),
            product_count=len(self._product_names),
        )

        # Lazy cross-encoder slot — loaded on first call to _get_cross_encoder()
        self._cross_encoder: CrossEncoder | None = None
        self._cross_encoder_model = "mixedbread-ai/mxbai-rerank-xsmall-v1"

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """BM25 keyword search over all indexed documents."""
        if not self._bm25_index:
            return []
        tokens = query.lower().split()
        scores = self._bm25_index.get_scores(tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self._bm25_names[i], score) for i, score in indexed[:top_k]]

    def _get_cross_encoder(self) -> CrossEncoder:
        """Lazy-load cross-encoder model on first call. Cached for process lifetime.
        Raises on failure — re-ranking is required, not optional."""
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(self._cross_encoder_model)
            logger.info("cross_encoder_loaded", model=self._cross_encoder_model)
        return self._cross_encoder

    def _rerank_candidates(
        self,
        requirements: list[str],
        candidates: list[tuple[str, float]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Re-rank RRF candidates using cross-encoder for precision.

        Fetches full product documents from ChromaDB, scores all (requirement, document)
        pairs, aggregates by max score per product, and returns top_k re-sorted results.

        mxbai-rerank-xsmall-v1 outputs raw logits — sigmoid converts to [0, 1].
        Raises if cross-encoder fails — re-ranking is required, not optional.
        """
        if not candidates:
            return []
        ce = self._get_cross_encoder()

        candidate_names = [name for name, _ in candidates]
        stored = self.collection.get(
            where={"name": {"$in": candidate_names}},
            include=["documents", "metadatas"],
        )
        doc_by_name: dict[str, str] = {}
        for doc, meta in zip(stored["documents"], stored["metadatas"]):
            doc_by_name[meta.get("name", "")] = doc

        # Build (requirement, product_doc) pairs — one per (candidate, requirement) combo
        pairs: list[list[str]] = []
        pair_names: list[str] = []
        for name in candidate_names:
            doc = doc_by_name.get(name, name)
            for req in requirements:
                pairs.append([req, doc])
                pair_names.append(name)

        ce_scores_raw = ce.predict(pairs)

        # Aggregate per product: take max score across all requirements
        product_max_score: dict[str, float] = {}
        for name, score in zip(pair_names, ce_scores_raw):
            product_max_score[name] = max(
                product_max_score.get(name, float("-inf")), float(score)
            )

        reranked = sorted(product_max_score.items(), key=lambda x: x[1], reverse=True)

        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))

        result = [(name, round(sigmoid(score), 4)) for name, score in reranked[:top_k]]
        # Release ~220MB after each research run so subsequent Ollama calls have full RAM budget
        self._cross_encoder = None
        return result

    async def match_requirements_to_products(
        self,
        requirements: list[str],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Match requirements to products using hybrid BM25 + vector search with
        Reciprocal Rank Fusion (RRF).

        BM25 handles exact/keyword matches (e.g. "HDL Coder", "AUTOSAR").
        Vector search handles semantic similarity (e.g. "battery management" → Simscape Battery).
        RRF fuses both ranked lists without needing score normalization.

        Solution enrichment docs participate in retrieval to boost recall but are
        filtered from the final output — only product names are returned.

        Args:
            requirements: List of requirement strings
            top_k: Number of top product matches to return

        Returns:
            List of (product_name, confidence_score) tuples, confidence in [0, 1]
        """
        if not requirements:
            return []

        K = 60          # Standard RRF constant (robust to outliers)
        POOL = max(top_k * 4, 20)  # Wide retrieval pool — BM25 + vector recall breadth
        RERANK_POOL = settings.rerank_pool_size  # Cross-encoder input cap (tunable in config)

        rrf_scores: dict[str, float] = {}

        for req in requirements:
            # --- Vector retrieval (ChromaDB) ---
            try:
                vector_results = self.collection.query(
                    query_texts=[req],
                    n_results=POOL,
                )
                vector_ranked: list[str] = []
                if vector_results and vector_results["metadatas"]:
                    for meta in vector_results["metadatas"][0]:
                        vector_ranked.append(meta.get("name", ""))
            except Exception as e:
                logger.warning("vector_match_failed", requirement=req[:80], error=str(e))
                vector_ranked = []

            # --- BM25 retrieval (only include docs with score > 0) ---
            bm25_ranked = [
                name for name, score in self._bm25_search(req, POOL) if score > 0
            ]

            # --- RRF fusion ---
            for rank, name in enumerate(vector_ranked):
                rrf_scores[name] = rrf_scores.get(name, 0.0) + 1.0 / (K + rank + 1)
            for rank, name in enumerate(bm25_ranked):
                rrf_scores[name] = rrf_scores.get(name, 0.0) + 1.0 / (K + rank + 1)

        # Sort by RRF score, keep product docs only, normalize to [0, 1]
        # Use theoretical max (both retrievers return rank-0 for every requirement)
        # so that unrelated queries yield low confidence instead of always 1.0
        sorted_all = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        theoretical_max = len(requirements) * 2.0 / (K + 1)
        norm_denom = max(theoretical_max, sorted_all[0][1] if sorted_all else 1.0)

        results = [
            (name, round(score / norm_denom, 4))
            for name, score in sorted_all
            if name in self._product_names
        ]

        logger.info(
            "requirements_matched_hybrid",
            requirement_count=len(requirements),
            candidates=len(sorted_all),
            product_matches=len(results),
        )

        # Stage 2: cross-encoder re-ranking over top RERANK_POOL candidates
        # 2:1 ratio (default 20→10) preserves re-ranking value while halving pairs vs full POOL
        return self._rerank_candidates(requirements, results[:RERANK_POOL], top_k)

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
