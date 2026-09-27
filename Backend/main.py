from pathlib import Path
import urllib.request

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import config

# ---------------------------------------------------------
# Download model weights from Hugging Face if not present
# (both models live in the same repo: Tuhin-2003/paddy)
# ---------------------------------------------------------
MODEL_URLS = {
    config.PADDY_MODEL_PATH: "https://huggingface.co/Tuhin-2003/paddy/resolve/main/best_mobilenet_v3_small.pth",
    config.DISEASE_MODEL_PATH: "https://huggingface.co/Tuhin-2003/paddy/resolve/main/best_fresh_balanced_convnext.pth",
}

for path, url in MODEL_URLS.items():
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)

# Import AFTER the download loop — this import triggers model loading
from routers import predict

app = FastAPI(title="Paddy Check API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}


config.REFERENCE_GALLERY_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/reference",
    StaticFiles(directory=config.REFERENCE_GALLERY_DIR),
    name="reference-images",
)

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
