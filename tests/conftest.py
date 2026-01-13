"""
Shared pytest fixtures for all tests.
Provides reusable test fixtures for data sources and common test data.
"""
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def sample_search_results():
    """Provide sample search results for testing."""
    from src.models.domain import SearchResult

    return [
        SearchResult(
            title="MathWorks Careers",
            url="https://www.mathworks.com/company/jobs.html",
            snippet="Join MathWorks and work on innovative products",
            source="duckduckgo"
        ),
        SearchResult(
            title="Software Engineer Jobs at MathWorks",
            url="https://www.mathworks.com/company/jobs/12345",
            snippet="Looking for talented software engineers",
            source="duckduckgo"
        ),
        SearchResult(
            title="MathWorks - Wikipedia",
            url="https://en.wikipedia.org/wiki/MathWorks",
            snippet="MathWorks is an American company that develops MATLAB",
            source="duckduckgo"
        ),
    ]


@pytest.fixture
def mock_html_greenhouse_page():
    """Provide mock Greenhouse HTML for testing job parsing."""
    return """
    <html>
        <head>
            <title>Careers at Example Corp</title>
        </head>
        <body>
            <div class="opening" data-qa="opening">
                <a href="/jobs/software-engineer-123" class="opening-title">
                    Software Engineer
                </a>
                <div class="location" data-qa="opening-location">
                    San Francisco, CA
                </div>
                <div class="department">Engineering</div>
            </div>
            <div class="opening">
                <a href="/jobs/senior-developer-456">
                    Senior Software Developer
                </a>
                <div class="location">Remote</div>
                <div class="department">Engineering</div>
            </div>
            <div class="opening">
                <a href="/jobs/data-scientist-789">
                    Data Scientist
                </a>
                <div class="location">Boston, MA</div>
                <div class="department">Data Science</div>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def mock_html_lever_page():
    """Provide mock Lever HTML for testing job parsing."""
    return """
    <html>
        <head>
            <title>Join Our Team | Example Corp</title>
        </head>
        <body>
            <div class="posting" data-qa="posting">
                <a href="https://jobs.lever.co/examplecorp/abc123">
                    <h5 class="posting-title">Product Manager</h5>
                </a>
                <div class="posting-categories">
                    <span class="location">New York, NY</span>
                    <span class="commitment">Full-time</span>
                </div>
            </div>
            <div class="posting">
                <a href="https://jobs.lever.co/examplecorp/def456">
                    <h5>UX Designer</h5>
                </a>
                <div class="posting-categories">
                    <span class="location">San Francisco, CA</span>
                </div>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def mock_html_generic_job_page():
    """Provide generic job page HTML for testing."""
    return """
    <html>
        <head>
            <title>Open Positions - Example Corp</title>
            <meta name="description" content="Current job openings at Example Corp">
        </head>
        <body>
            <h1>Current Openings</h1>
            <ul class="job-list">
                <li>
                    <a href="/careers/senior-software-engineer">
                        Senior Software Engineer
                    </a>
                </li>
                <li>
                    <a href="/careers/machine-learning-engineer">
                        Machine Learning Engineer
                    </a>
                </li>
                <li>
                    <a href="/careers/devops-engineer">
                        DevOps Engineer
                    </a>
                </li>
                <li>
                    <a href="/careers/backend-developer">
                        Backend Developer (Python/Java)
                    </a>
                </li>
                <li>
                    <a href="/about">About Us</a>
                </li>
                <li>
                    <a href="/contact">Contact</a>
                </li>
            </ul>
        </body>
    </html>
    """


@pytest.fixture
def mock_html_simple():
    """Provide simple HTML for basic parsing tests."""
    return """
    <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="A test page">
            <meta property="og:title" content="Test OG Title">
            <link rel="canonical" href="https://example.com/canonical">
        </head>
        <body>
            <div class="content">
                <h1>Welcome</h1>
                <p>This is a test page.</p>
                <a href="/page1">Link 1</a>
                <a href="https://example.com/page2">Link 2</a>
                <script>console.log('test');</script>
                <style>body { color: black; }</style>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def sample_products():
    """Provide sample Product objects for testing."""
    from src.models.domain import Product

    return [
        Product(
            name="MATLAB",
            category="Programming Platform",
            description="High-level language for numerical computation",
            key_features=["matrix operations", "data visualization", "algorithm development"],
            use_cases=["data analysis", "algorithm prototyping", "modeling"],
            target_personas=["data scientists", "engineers", "researchers"]
        ),
        Product(
            name="Simulink",
            category="Model-Based Design",
            description="Block diagram environment for simulation",
            key_features=["visual modeling", "simulation", "code generation"],
            use_cases=["control systems", "signal processing", "embedded systems"],
            target_personas=["control engineers", "embedded engineers", "system architects"]
        ),
        Product(
            name="Deep Learning Toolbox",
            category="AI/ML",
            description="Design and train deep neural networks",
            key_features=["neural network design", "model training", "deployment"],
            use_cases=["image classification", "object detection", "AI deployment"],
            target_personas=["AI engineers", "data scientists", "ML engineers"]
        ),
    ]


@pytest.fixture
def sample_job_postings():
    """Provide sample JobPosting objects for testing."""
    from src.models.domain import JobPosting

    return [
        JobPosting(
            title="Senior Software Engineer",
            company="MathWorks",
            description="Build cutting-edge engineering software",
            location="Natick, MA",
            url="https://www.mathworks.com/jobs/12345",
            confidence=0.9
        ),
        JobPosting(
            title="Machine Learning Engineer",
            company="MathWorks",
            description="Develop ML algorithms for engineering applications",
            location="Remote",
            url="https://www.mathworks.com/jobs/67890",
            confidence=0.85
        ),
        JobPosting(
            title="Embedded Systems Engineer",
            company="MathWorks",
            description="Work on real-time embedded systems software",
            location="Cambridge, UK",
            url="https://www.mathworks.com/jobs/11111",
            confidence=0.8
        ),
    ]


@pytest.fixture
def mock_company():
    """Provide sample Company object for testing."""
    from src.models.domain import Company

    return Company(
        name="MathWorks",
        domain="mathworks.com",
        industry="Software",
        description="Developer of MATLAB and Simulink",
        size="5000+"
    )


# Pytest markers for test categorization
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (slow, real connections)"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, mocked)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running tests"
    )
