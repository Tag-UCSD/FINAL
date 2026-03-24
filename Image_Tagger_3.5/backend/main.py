from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.services.storage import get_image_storage_root
from backend.api import v1_discovery, v1_debug, v1_features
from backend.versioning import VERSION

app = FastAPI(
    title=f"Image Tagger Explorer (v{VERSION})",
    version=VERSION,
    docs_url="/docs",
    redoc_url=None,
)

app.include_router(v1_discovery.router)
app.include_router(v1_debug.router)
app.include_router(v1_features.router)

IMAGE_STORAGE_ROOT = get_image_storage_root()
app.mount("/static", StaticFiles(directory=str(IMAGE_STORAGE_ROOT)), name="static")


@app.get("/health")
def health_check():
    return {"status": "healthy", "version": VERSION}


@app.get("/")
def root():
    return {"message": "Image Tagger Explorer API", "docs": "/docs"}
