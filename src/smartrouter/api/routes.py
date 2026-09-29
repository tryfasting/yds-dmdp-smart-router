from enum import Enum

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from smartrouter.core.logger import logger
from smartrouter.router.base import BaseRouter
from smartrouter.router.dynamic import IntensityRuleRouter

router = APIRouter()


class IntensityEnum(str, Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class FieldEnum(str, Enum):
    NONE = "NONE"
    EMAIL = "EMAIL"
    ARTICLE = "ARTICLE"
    THESIS = "THESIS"
    REPORT = "REPORT"
    MARKETING = "MARKETING"
    CUSTOMER_SERVICE = "CUSTOMER_SERVICE"


class RouteRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Sentence to be corrected")
    intensity: IntensityEnum = IntensityEnum.WEAK
    field: FieldEnum = FieldEnum.NONE


class RouteResponse(BaseModel):
    tier: str
    prob_hard: float
    threshold: float
    baseline_tier: str


def _get_router(request: Request) -> BaseRouter:
    model_router = getattr(request.app.state, "router", None)
    if model_router is None:
        raise HTTPException(status_code=503, detail="Classifier checkpoint is not loaded")
    return model_router


@router.get("/health")
async def health_check(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "router", None) is None:
        raise HTTPException(status_code=503, detail="Classifier checkpoint is not loaded")
    return {"status": "healthy"}


@router.post("/route", response_model=RouteResponse)
async def route_sentence(payload: RouteRequest, request: Request) -> RouteResponse:
    """Local routing decision only; no generation call is made."""
    model_router = _get_router(request)
    try:
        result = await run_in_threadpool(
            model_router.predict, payload.text, payload.intensity.value, payload.field.value
        )
    except Exception as exc:
        logger.error("Routing failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Routing failed") from exc

    baseline = IntensityRuleRouter().predict(payload.text, payload.intensity.value, payload.field.value)
    return RouteResponse(
        tier=result["tier"],
        prob_hard=result["prob_hard"],
        threshold=result["threshold"],
        baseline_tier=baseline["tier"],
    )
