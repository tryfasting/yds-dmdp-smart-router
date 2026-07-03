import time
from enum import Enum
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from smartrouter.core.logger import logger
from smartrouter.core import config
from smartrouter.router.dynamic import RoBERTaDynamicRouter
from smartrouter.services.base import BaseGenerator
from smartrouter.services.bedrock_generator import BedrockGenerator
from smartrouter.services.azure_generator import AzureOpenAIGenerator

router = APIRouter()

# --- Enums for strict validation ---
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

# --- Data Transfer Objects (DTOs) ---
class CorrectionRequest(BaseModel):
    text: str = Field(..., description="Original English sentence to correct", min_length=1)
    intensity: IntensityEnum = Field(IntensityEnum.WEAK, description="Correction intensity tag")
    field: FieldEnum = Field(FieldEnum.NONE, description="Target context domain field")

class CorrectionResponse(BaseModel):
    input_text: str
    final_output: str
    decision: str       # EASY, HARD, or FALLBACK
    target_model: str   # Model ID routed to
    risk_level: str     # low, medium, high, critical
    latency_ms: float   # End-to-end execution time in milliseconds
    generator_used: str # Class name of LLM client used

class EvaluationRequest(BaseModel):
    source_text: str = Field(..., description="Original English sentence")
    generated_text: str = Field(..., description="Corrected sentence to evaluate")
    intensity: IntensityEnum = Field(IntensityEnum.WEAK, description="Expected intensity")
    field: FieldEnum = Field(FieldEnum.NONE, description="Expected field")

class EvaluationResponse(BaseModel):
    score: float
    reasoning: str
    latency_ms: float
    evaluator_used: str

# --- Helper Dependency Injection ---
def get_router(request: Request) -> RoBERTaDynamicRouter:
    if not hasattr(request.app.state, "router") or request.app.state.router is None:
        raise HTTPException(status_code=503, detail="Classification model is degraded or starting up")
    return request.app.state.router

def get_generator(request: Request) -> BaseGenerator:
    if not hasattr(request.app.state, "generator") or request.app.state.generator is None:
        raise HTTPException(status_code=503, detail="LLM generator is degraded or starting up")
    return request.app.state.generator

# --- API Endpoints ---

@router.get("/health")
async def health_check(request: Request):
    """
    L7 health check endpoint used by orchestrators (AWS ECS, Docker) to confirm system availability.
    """
    has_router = hasattr(request.app.state, "router") and request.app.state.router is not None
    has_generator = hasattr(request.app.state, "generator") and request.app.state.generator is not None
    
    status = "healthy" if (has_router and has_generator) else "degraded"
    if status == "degraded":
        logger.warning("🚨 [Health Check] Health degraded: Model/Generator instances are missing.")
        raise HTTPException(status_code=503, detail="Models not fully loaded")
        
    return {"status": status}

@router.post("/correct", response_model=CorrectionResponse)
async def correct_sentence(
    request_data: CorrectionRequest,
    router_instance: RoBERTaDynamicRouter = Depends(get_router),
    generator_instance: BaseGenerator = Depends(get_generator)
):
    """
    Main Service Pipeline:
    1. Input Sentence -> Classifier evaluates difficulty and decides routing destination (Mini/Nano).
    2. Invokes the selected Bedrock Model (Claude Sonnet or Haiku) to get the correction.
    3. Calculates execution latency and returns metadata.
    """
    start_time = time.time()
    logger.info(f"📥 [API Request] Text received: '{request_data.text[:40]}...'")
    
    try:
        # Step 1. Routing classification (CPU Bound -> run in threadpool to avoid blocking event loop)
        router_result = await run_in_threadpool(
            router_instance.predict,
            text=request_data.text,
            intensity=request_data.intensity.value,
            field=request_data.field.value
        )
        
        target_model = router_result.get("target_model", config.MODEL_NANO)
        decision = router_result.get("decision", "EASY")
        risk_level = router_result.get("risk_level", "low")
        
        logger.info(f"🚦 [API Router] Decision: {decision} | Routed To: {target_model} | Risk: {risk_level}")
        
        # Step 2. Generate correction (Blocking Network I/O -> run in threadpool)
        final_output = await run_in_threadpool(
            generator_instance.get_correction,
            model_name=target_model,
            text=request_data.text,
            intensity=request_data.intensity.value,
            field=request_data.field.value
        )
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"📤 [API Success] Latency: {latency:.2f}ms | Output length: {len(final_output)} chars")
        
        return CorrectionResponse(
            input_text=request_data.text,
            final_output=final_output,
            decision=decision,
            target_model=target_model,
            risk_level=risk_level,
            latency_ms=round(latency, 2),
            generator_used=generator_instance.__class__.__name__
        )
        
    except Exception as e:
        logger.error(f"❌ [API Error] Failed to process request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_correction(
    request_data: EvaluationRequest,
    generator_instance: BaseGenerator = Depends(get_generator)
):
    """
    G-Eval based Quality Evaluation Endpoint:
    Uses the LLM-as-a-Judge to evaluate a given correction based on intent and context.
    """
    start_time = time.time()
    logger.info(f"📥 [API Request] Evaluation for text: '{request_data.source_text[:40]}...'")
    
    try:
        score, reasoning = await run_in_threadpool(
            generator_instance.get_evaluation,
            source_text=request_data.source_text,
            generated_text=request_data.generated_text,
            intensity=request_data.intensity.value,
            field=request_data.field.value
        )
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"📤 [API Success] Eval Score: {score} | Latency: {latency:.2f}ms")
        
        return EvaluationResponse(
            score=score,
            reasoning=reasoning,
            latency_ms=round(latency, 2),
            evaluator_used=generator_instance.__class__.__name__
        )
        
    except Exception as e:
        logger.error(f"❌ [API Error] Failed to evaluate: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

