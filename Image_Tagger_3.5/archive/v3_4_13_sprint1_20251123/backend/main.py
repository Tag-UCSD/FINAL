from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.api import v1_annotation, v1_admin, v1_supervision, v1_discovery, v1_bn_export, v1_debug
from backend.versioning import VERSION

# v3 Enterprise Application Entry Point
app = FastAPI(
    title=f"Image Tagger v3 (v{VERSION})",
    description="Unified API for Tagger Workbench, Supervisor, Admin, and Explorer.",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# API Routers
# All v1 routers are mounted here so that the smoketests and GUIs see a coherent API surface.
app.include_router(v1_annotation.router)
app.include_router(v1_admin.router)
app.include_router(v1_supervision.router)
app.include_router(v1_discovery.router)
app.include_router(v1_bn_export.router)
app.include_router(v1_debug.router)

# Frontend-facing mounts (ApiClient uses base '/api')
app.include_router(v1_annotation.router, prefix="/api")
app.include_router(v1_admin.router, prefix="/api")
app.include_router(v1_supervision.router, prefix="/api")
app.include_router(v1_discovery.router, prefix="/api")
app.include_router(v1_debug.router, prefix="/api")

# Static files (optional, for serving built frontends if desired)
# In Docker/Nginx this may be handled by the web server instead.
# app.mount("/static", StaticFiles(directory="static"), name="static")

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