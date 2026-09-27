"""
Orchestrates the pipeline:
  1. Is this paddy at all?
  2. If yes, does it have a disease (or is it healthy)?
  3. If diseased, WHERE is the problem? (Grad-CAM + spot boxes)
  4. If yes, which reference images is it most visually similar to
     (vectorization + cosine similarity)?

No object/YOLO detection anywhere - step 3 draws boxes from the
classifier's own attention map, step 4 compares whole-image feature
vectors.

Models are loaded once, at import time, and reused across requests -
not reloaded per-request.
"""

from pathlib import Path

import config
from ml.paddy_detector import PaddyDetector
from ml.disease_classifier import DiseaseClassifier
from ml.feature_extractor import SimilarityEngine
from ml.spot_detector import localize_disease
from utils.image_utils import bytes_to_pil_image


_paddy_detector = PaddyDetector()
_disease_classifier = DiseaseClassifier()

# Reuses the already-loaded ConvNeXt weights from the disease classifier.
_similarity_engine = SimilarityEngine(_disease_classifier)


def _to_response_images(matches: list[dict]) -> list[dict]:
    return [
        {
            "image_url": (
                f"{config.REFERENCE_IMAGE_URL_PREFIX}/"
                f"{Path(m['image_path']).name}"
            ),
            "label": m["label"],
            "similarity": m["similarity"],
        }
        for m in matches
    ]


def run_pipeline(image_bytes: bytes) -> dict:
    # ---------------------------------------------------------
    # 1. Paddy detection
    # ---------------------------------------------------------
    is_paddy, paddy_confidence = _paddy_detector.predict(image_bytes)

    if not is_paddy:
        return {
            "is_paddy": False,
            "paddy_confidence": paddy_confidence,
            "disease": None,
            "disease_label": None,
            "disease_confidence": None,
            "similar_images": [],
            "localization": None,
        }

    # ---------------------------------------------------------
    # 2. Disease classification
    # ---------------------------------------------------------
    disease_key, disease_label, disease_confidence = (
        _disease_classifier.predict(image_bytes)
    )

    # ---------------------------------------------------------
    # 3. Similar-image search
    # ---------------------------------------------------------
    similar_images = _to_response_images(
        _similarity_engine.find_similar(image_bytes)
    )

    # ---------------------------------------------------------
    # 4. Healthy leaf -> no localization required
    # ---------------------------------------------------------
    if disease_key == config.HEALTHY_CLASS_KEY:
        return {
            "is_paddy": True,
            "paddy_confidence": paddy_confidence,
            "disease": None,
            "disease_label": None,
            "disease_confidence": disease_confidence,
            "similar_images": similar_images,
            "localization": None,
        }

    # ---------------------------------------------------------
    # 5. Diseased leaf -> Grad-CAM localization
    # ---------------------------------------------------------
    pred_idx = config.DISEASE_CLASSES.index(disease_key)

    image_pil = bytes_to_pil_image(image_bytes)

    localization = None

    # ---------------------------------------------------------
    # 6. Final response
    # ---------------------------------------------------------
    return {
        "is_paddy": True,
        "paddy_confidence": paddy_confidence,
        "disease": disease_key,
        "disease_label": disease_label,
        "disease_confidence": disease_confidence,
        "similar_images": similar_images,
        "localization": localization,
    }
