"""
FastAPI application entry point.
Main API server for the Sales Research Agent system.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from api.routers import health_router, research_router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("api_starting", message="Sales Research Agent API starting up")
    yield
    logger.info("api_shutting_down", message="Sales Research Agent API shutting down")


# Create FastAPI app
app = FastAPI(
    title="Sales Research Agent API",
    description="""
    AI-powered sales intelligence system that researches target accounts
    and identifies sales opportunities by matching a seller's product catalog
    to the customer's needs.

    ## Features
    - Multi-agent research workflow (Gatherer, Identifier, Validator)
    - Real-time progress streaming via SSE
    - Human-in-the-loop feedback integration
    - LangGraph-based workflow orchestration

    ## Workflow
    1. **Start Research**: POST /api/research/start
    2. **Monitor Progress**: GET /api/research/{thread_id}/stream (SSE)
    3. **Get State**: GET /api/research/{thread_id}/state
    4. **Submit Feedback**: POST /api/research/{thread_id}/feedback
    5. **Get Report**: GET /api/research/{thread_id}/report
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative React port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api")
app.include_router(research_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Sales Research Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
