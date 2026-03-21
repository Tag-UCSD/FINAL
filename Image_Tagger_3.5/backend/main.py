import os
from typing import Callable

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.services.storage import get_image_storage_root
from backend.api import (
    v1_annotation,
    v1_admin,
    v1_supervision,
    v1_discovery,
    v1_bn_export,
    v1_debug,
    v1_features,
    v1_vlm_health,
)
from backend.versioning import VERSION

# v3 Enterprise Application Entry Point
class PrefixStripMiddleware:
    """Strip known prefixes from incoming paths while preserving routing."""

    def __init__(self, app: Callable, prefixes: list[str]) -> None:
        self.app = app
        self.prefixes = [p.rstrip("/") for p in prefixes]

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] in {"http", "websocket"}:
            path = scope.get("path", "")
            for prefix in self.prefixes:
                if path == prefix or path.startswith(f"{prefix}/"):
                    scope = dict(scope)
                    scope["path"] = path[len(prefix):] or "/"
                    scope["root_path"] = f"{scope.get('root_path', '')}{prefix}"
                    break
        await self.app(scope, receive, send)


app = FastAPI(
    title=f"Image Tagger v3 (v{VERSION})",
    description="Unified API for Tagger Workbench, Supervisor, Admin, and Explorer.",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

if os.getenv("ENABLE_LEGACY_PREFIXES", "1").lower() not in {"0", "false"}:
    app.add_middleware(
        PrefixStripMiddleware,
        prefixes=[
            "/api/v1/tagger",
            "/api",
        ],
    )

# Router wiring
app.include_router(v1_annotation.router)
app.include_router(v1_admin.router)
app.include_router(v1_supervision.router)
app.include_router(v1_discovery.router)
app.include_router(v1_bn_export.router)
app.include_router(v1_debug.router)
app.include_router(v1_features.router)
app.include_router(v1_vlm_health.router)

# Static file mount for image assets
IMAGE_STORAGE_ROOT = get_image_storage_root()
app.mount("/static", StaticFiles(directory=str(IMAGE_STORAGE_ROOT)), name="static")


@app.get("/health")
def health_check():
    """Kubernetes/Docker Health Probe"""
    return {"status": "healthy", "version": VERSION}


@app.get("/")
def root():
    return {
        "message": "Image Tagger v3 API",
        "docs": "/docs",
        "workbench_api": "/v1/workbench/next",
    }
