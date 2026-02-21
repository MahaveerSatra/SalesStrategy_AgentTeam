"""Health check endpoint."""
from fastapi import APIRouter

from api.schemas.api_models import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns the service status and version.
    """
    return HealthResponse(status="ok", version="1.0.0")
