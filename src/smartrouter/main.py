import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from smartrouter.core.logger import logger
from smartrouter.router.dynamic import RoBERTaDynamicRouter
from smartrouter.services.bedrock_generator import BedrockGenerator
from smartrouter.api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Manager. Loads heavy ML models and external API clients
    once during startup and unloads them gracefully during shutdown.
    """
    logger.info("⏳ [Lifespan] Initializing backend resources...")
    try:
        # Initialize heavy RoBERTa classifier and dynamic router
        app.state.router = RoBERTaDynamicRouter()
        logger.info("✅ [Lifespan] RoBERTa Classifier and Router initialized.")
        
        # Initialize AWS Bedrock Generator service (Default Provider)
        # To prove multi-cloud architecture, this could easily be swapped with AzureOpenAIGenerator()
        app.state.generator = BedrockGenerator()
        logger.info("✅ [Lifespan] Bedrock Generator initialized.")
        
    except Exception as e:
        logger.error(f"❌ [Lifespan] Failed to initialize backend resources: {e}", exc_info=True)
        # Note: In production we could raise the error to abort container startup
        app.state.router = None
        app.state.generator = None
        
    yield
    
    # Graceful shutdown/cleanup
    logger.info("🛑 [Lifespan] Tearing down backend resources...")
    app.state.router = None
    app.state.generator = None
    logger.info("👋 [Lifespan] Cleanup completed. Bye!")

# Create FastAPI application
app = FastAPI(
    title="SmartRouter AI Servicing",
    description="Cost-Quality Optimized Sentence Correction System using RoBERTa and AWS Bedrock",
    version="2.0.0",
    lifespan=lifespan
)

# Include core service APIs
app.include_router(api_router)

# Mount Static Files (Dashboard UI)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"🎨 [Static Files] Directory mounted at: {STATIC_DIR}")
    
    # Root route serves the dashboard UI
    @app.get("/")
    async def read_root():
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "SmartRouter Service is running. (Dashboard file not found)"}
else:
    logger.warning(f"⚠️ [Static Files] Static directory not found at: {STATIC_DIR}. Dashboard is disabled.")
    
    @app.get("/")
    async def read_root():
        return {"message": "SmartRouter API is running. (Dashboard is disabled)"}

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting local development server on http://localhost:8000")
    uvicorn.run("smartrouter.main:app", host="0.0.0.0", port=8000, reload=True)
