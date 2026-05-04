"""FastAPI application factory and lifespan model loader."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from ..config import get_settings
from ..observability.logging import configure_logging, get_logger
from ..observability.metrics import get_metrics
from .deps import get_model_bundle
from .routers import analyze, calibrate, health, predict


@asynccontextmanager
async def _lifespan(app: FastAPI):
    configure_logging(get_settings().log_level)
    log = get_logger("neurodrift.api")
    bundle = get_model_bundle()
    log.info(
        "api.startup",
        acceptance=bundle.acceptance_version,
        regressor=bundle.regressor_version,
        kmeans=bundle.kmeans_version,
    )
    yield
    log.info("api.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="NeuroDrift API",
        description=(
            "End-to-end MLOps service for motor-imagery EEG. "
            "Provides trial-level prediction, per-subject calibration, "
            "and session-level recovery analysis."
        ),
        version="0.1.0",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    allowed_origins = {
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    }

    @app.middleware("http")
    async def ensure_dev_cors(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method == "OPTIONS" and origin in allowed_origins:
            response: Response = JSONResponse({"ok": True})
        else:
            response = await call_next(request)

        if origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Vary"] = "Origin"
        return response

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(calibrate.router)
    app.include_router(analyze.router)

    @app.get("/metrics", response_class=PlainTextResponse, tags=["metrics"])
    def metrics() -> str:
        return get_metrics().render_prometheus()

    return app


app = create_app()
