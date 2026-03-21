from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.api import v1_annotation, v1_admin, v1_supervision, v1_discovery

# v3 Enterprise Application Entry Point
app = FastAPI(
    title="Image Tagger v3.2.33 (Enterprise)",
    description="Unified API for Tagger Workbench, Supervisor, Admin, and Explorer.",
    version="3.2.33",
    docs_url="/docs",
    redoc_url="/redoc",
)

# API Routers
# All v1 routers are mounted here so that the smoketests and GUIs see a coherent API surface.
app.include_router(v1_annotation.router)
app.include_router(v1_admin.router)
app.include_router(v1_supervision.router)
app.include_router(v1_discovery.router)

# Static files (optional, for serving built frontends if desired)
# In Docker/Nginx this may be handled by the web server instead.
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health")
def health_check():
    """Kubernetes/Docker Health Probe"""
    return {"status": "healthy", "version": "3.2.33"}

@app.get("/")
def root():
    return {
        "message": "Image Tagger v3 API",
        "docs": "/docs",
        "workbench_api": "/v1/workbench/next",
    }