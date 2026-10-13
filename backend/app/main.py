"""MCPGuard FastAPI application: REST API + static research dashboard.

Run locally:  ``uvicorn backend.app.main:app --port 7860``
Docs:         /docs (Swagger UI) and /redoc
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.api.routes import router
from backend.app.database.db import sync_artifacts
from backend.app.services.artifacts import artifacts
from src.inference.predictor import get_predictor
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_utils import get_logger, log_event

log = get_logger("app")
FRONTEND = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_event(log, "application startup", version=load_config()["project"]["version"])
    try:
        p = get_predictor()
        log_event(log, "model ready", model=p.metadata["model_name"])
    except FileNotFoundError:
        log_event(log, "model artefacts missing — API runs in degraded mode")
    try:
        counts = sync_artifacts(artifacts)
        log_event(log, "database synced", **counts)
    except Exception as exc:  # noqa: BLE001 - never block startup on the DB mirror
        log_event(log, "database sync skipped", error=type(exc).__name__)
    yield
    log_event(log, "application shutdown")


cfg = load_config()
app = FastAPI(
    title="MCPGuard API",
    description=f"**{cfg['project']['title']}** — {cfg['project']['subtitle']}.\n\n*{cfg['project']['tagline']}*",
    version=cfg["project"]["version"],
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=cfg["api"]["cors_origins"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def request_log(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    response = await call_next(request)
    if not request.url.path.startswith(("/static", "/figures-static")):
        log_event(log, "request", id=rid, method=request.method, path=request.url.path, status=response.status_code,
                  ms=round((time.perf_counter() - t0) * 1000, 1))
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    if request.url.path.startswith("/static"):
        # revalidate via ETag on every load so a redeploy never leaves browsers on stale CSS/JS
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": ".".join(str(x) for x in e["loc"] if x != "body"), "message": e["msg"].replace("Value error, ", "")}
              for e in exc.errors()]
    log_event(log, "validation failure", path=request.url.path, fields=[e["field"] for e in errors])
    return JSONResponse(status_code=422, content={"detail": "Invalid input", "errors": errors})


@app.exception_handler(StarletteHTTPException)
async def http_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):  # never leak internals / paths
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


app.include_router(router)
app.mount("/static", StaticFiles(directory=FRONTEND / "static"), name="static")
app.mount("/figures-static", StaticFiles(directory=cfg.path("figures")), name="figures_static")


@app.get("/", include_in_schema=False)
def index(request: Request):
    if "application/json" in request.headers.get("accept", "") and "text/html" not in request.headers.get("accept", ""):
        from backend.app.api.routes import api_index
        return api_index()
    return FileResponse(FRONTEND / "index.html", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("backend.app.main:app", host=cfg["api"]["host"], port=int(os.environ.get("PORT", cfg["api"]["port"])))
