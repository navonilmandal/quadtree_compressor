#!/usr/bin/env python3
"""
web/app.py - Complete web entrypoint for Quadtree Image Compressor.

Features:
- AJAX-friendly compress endpoint (returns JSON with base64 previews)
- Downscale-for-search (fast tolerance search) + parallel full-resolution build
- Edge-aware quadtree (uses core/quadtree_core.py)
- Saves outputs to ./output and returns download links

Usage (dev):
    python web/app.py
Or run inside your Docker container as configured.

Make sure core/quadtree_core.py exposes:
  pad_image_to_pow2, compute_edge_map, build_quadtree_edge_aware,
  downscale_image, parallel_build_full, serialize_quadtree,
  render_quadtree, count_nodes, estimate_serialized_bytes
"""

import os
import io
import math
import json
import tempfile
import sys
from pathlib import Path
from flask import Flask, request, render_template, send_file, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import optimized core (must exist)
from core.quadtree_core import (
    pad_image_to_pow2,
    compute_edge_map,
    build_quadtree_edge_aware,
    downscale_image,
    parallel_build_full,
    serialize_quadtree,
    render_quadtree,
    count_nodes,
    estimate_serialized_bytes
)

ALLOWED = {"png", "jpg", "jpeg"}
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(PROJECT_ROOT / "web" / "static"), template_folder=str(PROJECT_ROOT / "web" / "templates"))
app.secret_key = os.environ.get("FLASK_SECRET", "change_me_for_prod")

def allowed_filename(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

def pil_to_bytes_io(img: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf

def psnr_image(orig: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((orig.astype(np.float64) - recon.astype(np.float64))**2))
    if mse == 0.0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def _tree_json_bytes_from_qnode(qnode) -> int:
    """
    Compute JSON payload bytes for {"tree": <serialized tree>} for an in-memory qnode.
    This is used to estimate final compressed JSON size precisely.
    """
    try:
        d = serialize_quadtree(qnode)
        payload = {"tree": d}
        s = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        return len(s.encode("utf-8"))
    except Exception:
        return 10**9

@app.route("/", methods=["GET"])
def index():
    # default target and depth values shown in UI
    return render_template("index.html", result=None, default_target_kb=200, default_depth=8)

@app.route("/compress", methods=["POST"])
def compress():
    """
    Main compress endpoint.
    If request is AJAX (X-Requested-With: XMLHttpRequest) or Accept: application/json -> return JSON.
    Otherwise render template fallback.
    """
    # Detect if caller wants JSON (AJAX)
    prefer_json = (request.headers.get("X-Requested-With") == "XMLHttpRequest") or ("application/json" in (request.headers.get("Accept") or ""))

    # Basic checks
    if "image" not in request.files:
        msg = "No file uploaded"
        if prefer_json: return jsonify({"error": msg}), 400
        flash(msg); return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        msg = "No file selected"
        if prefer_json: return jsonify({"error": msg}), 400
        flash(msg); return redirect(url_for("index"))

    if not allowed_filename(file.filename):
        msg = "Unsupported file type (allowed: png, jpg, jpeg)"
        if prefer_json: return jsonify({"error": msg}), 400
        flash(msg); return redirect(url_for("index"))

    # Read params
    target_kb_str = (request.form.get("target_kb") or "").strip()
    max_depth_str = (request.form.get("max_depth") or "").strip()
    edge_threshold_str = (request.form.get("edge_threshold") or "24").strip()
    manual_tol_str = (request.form.get("tolerance") or "").strip()

    # Load image
    try:
        pil = Image.open(file.stream).convert("RGB")
    except Exception as e:
        msg = f"Cannot open image: {e}"
        if prefer_json: return jsonify({"error": msg}), 400
        flash(msg); return redirect(url_for("index"))

    arr = np.array(pil)
    padded, orig_w, orig_h = pad_image_to_pow2(arr)
    side = padded.shape[0]
    D = int(math.log2(side))

    # Parse numeric params
    try:
        max_depth = int(max_depth_str) if max_depth_str != "" else min(10, D)
    except:
        max_depth = min(10, D)
    max_depth = max(0, min(max_depth, D))

    try:
        edge_threshold = float(edge_threshold_str) if edge_threshold_str != "" else 24.0
    except:
        edge_threshold = 24.0

    # Compute edge_map once (used by full builds)
    try:
        edge_map = compute_edge_map(padded, threshold=edge_threshold)
    except Exception:
        # fallback: no edge map if compute fails
        edge_map = None

    qt = None
    used_tol = None
    used_msg = ""
    est_bytes = None

    # Helper: return error
    def respond_error(msg, http_code=400):
        if prefer_json:
            return jsonify({"error": msg}), http_code
        flash(msg)
        return redirect(url_for("index"))

    # Path A: If target_kb specified -> auto search (fast downscale search -> parallel full build)
    if target_kb_str != "":
        try:
            target_kb = float(target_kb_str)
            if target_kb <= 0:
                return respond_error("Target KB must be positive", 400)
            target_bytes = int(target_kb * 1024)
        except:
            return respond_error("Invalid target KB value", 400)

        # Decide downscale factor (bigger images -> larger factor). Keep factor >=1
        # Aim: small side ~ 128..256 for search
        desired_search_side = 256
        factor = max(1, side // desired_search_side)
        small = downscale_image(padded, factor) if factor > 1 else padded.copy()
        small_side = small.shape[0]

        # Fast coarse binary-like search on tolerance using estimate_serialized_bytes on small image
        # This is cheap and guides the final exact build.
        low, high = 0.0, max(1.0, 1000.0)
        best_tol = None
        best_score = float("inf")

        # We'll run a limited number of iterations
        for i in range(12):
            mid = (low + high) / 2.0
            try:
                # run a small-tree build without edge forcing in search (faster)
                qt_s = build_quadtree_edge_aware(small, 0, 0, small_side, tol=mid, max_depth=min(max_depth, int(math.log2(small_side))), edge_map=None)
                est = estimate_serialized_bytes(qt_s)
            except Exception:
                est = 10**9
            diff = abs(est - target_bytes)
            if diff < best_score:
                best_score = diff
                best_tol = mid
            # move direction
            if est > target_bytes:
                low = mid
            else:
                high = mid

        used_tol = best_tol if best_tol is not None else 800.0
        used_msg = f"Auto-search (factor={factor})"

        # Final exact full-resolution build (parallel)
        try:
            qt = parallel_build_full(padded, tol=used_tol, max_depth=max_depth, edge_map=edge_map)
            est_bytes = _tree_json_bytes_from_qnode(qt)
        except Exception as e:
            return respond_error(f"Final build failed: {e}", 500)

    else:
        # Path B: manual tolerance provided? else use default
        if manual_tol_str != "":
            try:
                used_tol = float(manual_tol_str)
            except:
                return respond_error("Invalid manual tolerance value", 400)
        else:
            used_tol = 800.0
        used_msg = "Manual tolerance"
        try:
            qt = parallel_build_full(padded, tol=used_tol, max_depth=max_depth, edge_map=edge_map)
            est_bytes = _tree_json_bytes_from_qnode(qt)
        except Exception as e:
            return respond_error(f"Build failed: {e}", 500)

    # Render reconstruction and compute PSNR
    try:
        canvas = np.zeros_like(padded, dtype=np.uint8)
        render_quadtree(qt, canvas, 0, 0, side)
        recon = canvas[:orig_h, :orig_w]
        pil_recon = Image.fromarray(recon)
    except Exception as e:
        return respond_error(f"Render failed: {e}", 500)

    # Compute PSNR
    try:
        pval = psnr_image(arr, recon)
        p_str = "inf" if math.isinf(pval) else f"{pval:.2f}"
    except Exception:
        p_str = "N/A"

    # Save outputs
    uid = next(tempfile._get_candidate_names())
    safe_recon_name = f"recon_{uid}.png"
    safe_json_name = f"tree_{uid}.json"
    recon_path = OUTPUT_DIR / safe_recon_name
    json_path = OUTPUT_DIR / safe_json_name

    pil_recon.save(recon_path, format="PNG")
    # Save serialized tree object (small metadata + serialized tree)
    try:
        payload = {"orig_w": int(orig_w), "orig_h": int(orig_h), "side": int(side), "used_depth": int(max_depth), "tree": serialize_quadtree(qt)}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
    except Exception as e:
        # still continue, but warn
        app.logger.warning("Failed to write json: %s", e)

    est_kb = (est_bytes / 1024.0) if est_bytes is not None else None

    # Prepare result object
    result = {
        "psnr": p_str,
        "nodes": count_nodes(qt),
        "estimated": f"{est_kb:.2f} KB ({est_bytes} bytes)" if est_kb is not None else "N/A",
        "recon_name": safe_recon_name,
        "json_name": safe_json_name,
        "used_tol": f"{used_tol:.2f}" if used_tol is not None else "N/A",
        "used_depth": int(max_depth),
        "msg": used_msg
    }

    # Inline previews for AJAX
    if prefer_json:
        try:
            import base64
            ob = pil_to_bytes_io(pil).getvalue()
            rb = pil_to_bytes_io(pil_recon).getvalue()
            result["orig_b64"] = base64.b64encode(ob).decode("ascii")
            result["recon_b64"] = base64.b64encode(rb).decode("ascii")
        except Exception:
            result["orig_b64"] = ""
            result["recon_b64"] = ""
        return jsonify(result)

    # HTML fallback: render template with result
    return render_template("index.html", result=result, default_target_kb=(round(est_kb) if est_kb else 200), default_depth=max_depth)

@app.route("/download/recon/<fname>")
def download_recon(fname):
    p = OUTPUT_DIR / secure_filename(fname)
    if not p.exists():
        flash("File not found")
        return redirect(url_for("index"))
    return send_file(str(p), as_attachment=True)

@app.route("/download/json/<fname>")
def download_json(fname):
    p = OUTPUT_DIR / secure_filename(fname)
    if not p.exists():
        flash("File not found")
        return redirect(url_for("index"))
    return send_file(str(p), as_attachment=True)

if __name__ == "__main__":
    # When run directly we use the Flask dev server (helpful for debugging)
    print("Starting quadtree web app on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
