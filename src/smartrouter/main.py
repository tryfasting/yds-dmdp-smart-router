"""
Optional local API (install with `uv sync --extra api`).

    uv run --extra api uvicorn smartrouter.main:app --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from smartrouter.api.routes import router as api_router
from smartrouter.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from smartrouter.router.dynamic import RoBERTaDynamicRouter

        app.state.router = RoBERTaDynamicRouter()
    except Exception as exc:
        # Stay up and report 503 on /health instead of routing with an untrained model.
        logger.error("Classifier not loaded: %s", exc)
        app.state.router = None
    yield
    app.state.router = None


app = FastAPI(
    title="SmartRouter (local)",
    description="Difficulty routing with the submitted V8 RoBERTa checkpoint and an intensity-rule baseline.",
    version="3.0.0",
    lifespan=lifespan,
)
app.include_router(api_router)
