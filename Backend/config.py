"""
Central configuration for the paddy classifier backend.

Values below are taken directly from your two training notebooks
(paddy-or-not-paddy.ipynb and image-classification-paddy.ipynb) —
architecture, image size, normalization, class order, and the
confidence threshold all mirror what those notebooks actually did.
"""

from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

# ---------- model weights ----------
PADDY_MODEL_PATH = BASE_DIR / "models" / "paddy_or_not" / "best_mobilenet_v3_small.pth"
DISEASE_MODEL_PATH = BASE_DIR / "models" / "disease_classify" / "best_fresh_balanced_convnext.pth"

# ---------- paddy / not-paddy (MobileNetV3-Small) ----------
# Training used label 0 = non-paddy, label 1 = paddy (see df_non_paddy/df_paddy
# in paddy-or-not-paddy.ipynb).
PADDY_CLASSES = ["not_paddy", "paddy"]
PADDY_IMG_SIZE = 224

# The notebook's own inference function (predict_gatekeeper) only accepts a
# prediction if it clears this confidence bar; below it, it falls back to
# "Not Paddy [Uncertain]". Mirrored in ml/paddy_detector.py.
PADDY_CONFIDENCE_THRESHOLD = 0.85

# ---------- disease / pathogen type (ConvNeXt-Tiny) ----------
# best_fresh_balanced_convnext.pth classifies PATHOGEN TYPE, not individual
# diseases — see class_names in image-classification-paddy.ipynb (cell training
# "FRESH BALANCED FINE-TUNING (CONVNEXT-TINY @ 384x384)"). Order matches the
# notebook's CATEGORY_MAP: Bacterial=0, Fungal=1, Viral=2, Normal=3.
DISEASE_CLASSES = ["bacterial", "fungal", "viral", "normal"]
DISEASE_IMG_SIZE = 384

DISEASE_LABELS = {
    "bacterial": "Bacterial infection",
    "fungal": "Fungal infection",
    "viral": "Viral infection",
    "normal": "Healthy",
}

HEALTHY_CLASS_KEY = "normal"

# ---------- shared preprocessing ----------
# Same normalization stats used for both models in every notebook cell.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------- runtime ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CORS — add your frontend's actual origin(s) when you serve it from
# somewhere other than opening index.html directly as a file.
ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "null",
    "https://paddydisease.netlify.app",
]
# ---------- reference-image similarity / vectorization ----------
# Whole-image feature-vector search (ConvNeXt embeddings + cosine
# similarity), same approach as run_clean_balanced_diagnostic in
# image-classification-paddy.ipynb — no object detection involved.
SIMILARITY_IMG_SIZE = 384  # matches the ConvNeXt-Tiny @384 fine-tune
SIMILARITY_TOP_K = 5

# Folder of reference leaf photos + a manifest.csv describing them
# (columns: image_path,label).
REFERENCE_GALLERY_DIR = BASE_DIR / "data" / "reference_images"
SIMILARITY_MANIFEST_PATH = REFERENCE_GALLERY_DIR / "manifest.csv"

# Cached embeddings so the gallery isn't re-vectorized on every restart.
SIMILARITY_CACHE_PATH = BASE_DIR / "data" / "reference_embeddings.npz"

# URL prefix the frontend uses to load reference thumbnails — must match
# the StaticFiles mount in main.py.
REFERENCE_IMAGE_URL_PREFIX = "/static/reference"
