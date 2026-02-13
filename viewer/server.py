"""SHARP Viewer — Flask server for image upload and 3DGS prediction.

Loads the SHARP model once at startup, then serves a web UI for
uploading images and viewing predicted 3D Gaussian splats.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io as sharp_io
from sharp.utils.gaussians import save_ply

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

app = Flask(__name__, static_folder=str(STATIC_DIR))

# Global model reference
predictor = None
device = None


def load_model():
    """Load the SHARP predictor model."""
    global predictor, device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    LOGGER.info("Loading SHARP model on %s...", device)
    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    LOGGER.info("Model loaded successfully.")


@torch.no_grad()
def predict_image(image: np.ndarray, f_px: float) -> tuple:
    """Run SHARP inference on a single image. Returns (gaussians, f_px, height, width)."""
    from sharp.utils.gaussians import unproject_gaussians

    internal_shape = (1536, 1536)

    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    LOGGER.info("Running inference (%dx%d)...", width, height)
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians, f_px, height, width


# ── Routes ──────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(str(STATIC_DIR), filename)


@app.route("/upload", methods=["POST"])
def upload():
    """Accept an image upload, run SHARP prediction, return PLY path."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file
    ext = Path(file.filename).suffix.lower()
    stem = Path(file.filename).stem
    timestamp = int(time.time())
    safe_name = f"{stem}_{timestamp}{ext}"
    upload_path = UPLOAD_DIR / safe_name
    file.save(str(upload_path))

    LOGGER.info("Uploaded: %s", upload_path)

    try:
        # Load image using SHARP's own loader
        image, _, f_px = sharp_io.load_rgb(upload_path)
        height, width = image.shape[:2]

        # Run prediction
        t0 = time.time()
        gaussians, f_px, height, width = predict_image(image, f_px)
        elapsed = time.time() - t0
        LOGGER.info("Prediction took %.1fs", elapsed)

        # Save PLY
        ply_name = f"{stem}_{timestamp}.ply"
        ply_path = OUTPUT_DIR / ply_name
        save_ply(gaussians, f_px, (height, width), ply_path)
        LOGGER.info("Saved PLY: %s (%.1f MB)", ply_path, ply_path.stat().st_size / 1e6)

        return jsonify(
            {
                "success": True,
                "ply": ply_name,
                "elapsed": round(elapsed, 1),
                "size_mb": round(ply_path.stat().st_size / 1e6, 1),
                "image": safe_name,
            }
        )

    except Exception as e:
        LOGGER.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/results")
def list_results():
    """List all generated PLY files with metadata."""
    results = []
    for ply_file in sorted(OUTPUT_DIR.glob("*.ply"), key=os.path.getmtime, reverse=True):
        # Try to find matching upload image
        stem = ply_file.stem
        image_name = None
        for ext in [".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"]:
            candidate = UPLOAD_DIR / (stem + ext)
            if candidate.exists():
                image_name = candidate.name
                break

        results.append(
            {
                "ply": ply_file.name,
                "size_mb": round(ply_file.stat().st_size / 1e6, 1),
                "created": ply_file.stat().st_mtime,
                "image": image_name,
            }
        )

    return jsonify(results)


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5001, debug=False)
