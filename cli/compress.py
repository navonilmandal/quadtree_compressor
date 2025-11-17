#!/usr/bin/env python3
"""
cli/compress.py
Command-line wrapper using core/quadtree_core.py

Usage examples:
  # compress an image (writes JSON + reconstructed PNG to ./output)
  python cli/compress.py compress path/to/image.jpg --tolerance 1000 --max-depth 7

  # decompress a previously-saved JSON
  python cli/compress.py decompress output/tree_image.json --out recon.png

  # roundtrip (compress in memory, write reconstruction and stats)
  python cli/compress.py roundtrip path/to/image.jpg --tolerance 800 --max-depth 7 --out recon.png
"""

import os
import sys
import argparse
import json
import math
from pathlib import Path

# Make sure core package can be imported when running this script directly
this_dir = Path(__file__).resolve().parent
project_root = this_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.quadtree_core import (
    pad_image_to_pow2,
    build_quadtree,
    render_quadtree,
    serialize_quadtree,
    deserialize_quadtree,
    count_nodes,
)
from PIL import Image
import numpy as np

OUTPUT_DIR = project_root / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def psnr(orig_np: np.ndarray, recon_np: np.ndarray) -> float:
    mse = float(np.mean((orig_np.astype(np.float64) - recon_np.astype(np.float64))**2))
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20.0 * math.log10(PIXEL_MAX / math.sqrt(mse))

def compress_image(in_path: str, tolerance: float, max_depth: int, out_prefix: str = None):
    img = Image.open(in_path).convert("RGB")
    arr = np.array(img)
    padded, orig_w, orig_h = pad_image_to_pow2(arr)
    side = padded.shape[0]
    D = int(math.log2(side))
    if max_depth is None:
        max_depth = D
    else:
        max_depth = max(0, min(max_depth, D))

    print(f"[+] Input: {in_path}")
    print(f"[+] Original size: {orig_w}x{orig_h}, padded side: {side}, depth: {D}, using max_depth={max_depth}")
    qt = build_quadtree(padded, 0, 0, side, tol=tolerance, max_depth=max_depth)
    # render
    canvas = np.zeros_like(padded, dtype=np.uint8)
    render_quadtree(qt, canvas, 0, 0, side)
    recon = canvas[:orig_h, :orig_w]

    # output filenames
    base = Path(in_path).stem
    if out_prefix:
        base = out_prefix
    json_name = OUTPUT_DIR / f"{base}_qtree.json"
    recon_name = OUTPUT_DIR / f"{base}_recon.png"

    # write JSON
    data = {
        "orig_w": int(orig_w), "orig_h": int(orig_h), "side": int(side),
        "depth": int(D), "used_depth": int(max_depth),
        "tree": serialize_quadtree(qt)
    }
    with open(json_name, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)

    # write recon PNG
    Image.fromarray(recon).save(recon_name, format="PNG")

    # stats
    try:
        p = psnr(arr, recon)
    except Exception:
        p = float('nan')
    nodes = count_nodes(qt)
    est_size = estimate_serialized_size_estimate(qt=qt)  # local helper below

    print(f"[+] Wrote: {json_name}")
    print(f"[+] Wrote: {recon_name}")
    print(f"[+] PSNR: {'inf' if math.isinf(p) else f'{p:.2f}'} dB")
    print(f"[+] Nodes: {nodes}, estimated serialized bytes: {est_size}")
    return {"json": str(json_name), "recon": str(recon_name), "psnr": p, "nodes": nodes, "est_bytes": est_size}

def decompress_json(json_path: str, out_path: str = None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tree = data["tree"]
    side = int(data["side"])
    orig_w = int(data["orig_w"])
    orig_h = int(data["orig_h"])
    qt = deserialize_quadtree(tree)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    render_quadtree(qt, canvas, 0, 0, side)
    recon = canvas[:orig_h, :orig_w]
    if out_path is None:
        out_path = OUTPUT_DIR / (Path(json_path).stem + "_decompressed.png")
    Image.fromarray(recon).save(out_path, format="PNG")
    print(f"[+] Decompressed saved to {out_path}")
    return str(out_path)

def roundtrip(in_path: str, tolerance: float, max_depth: int, out_path: str = None):
    info = compress_image(in_path, tolerance, max_depth)
    # compute PSNR already printed by compress_image; recon path available
    recon_path = info["recon"]
    # load original and recon for PSNR (already computed, but we recompute to be robust)
    orig = np.array(Image.open(in_path).convert("RGB"))
    recon = np.array(Image.open(recon_path).convert("RGB"))
    p = psnr(orig, recon)
    print(f"[+] Roundtrip PSNR: {'inf' if math.isinf(p) else f'{p:.2f}'} dB")
    return info

# small helper for estimating bytes (similar to core's estimate)
def estimate_serialized_size_estimate(qt):
    # naive estimator: leaf -> 4 bytes (tag+3color), internal -> 1 + sum(children)
    def rec(node):
        if node["leaf"] if isinstance(node, dict) else False:
            return 1 + 3
        # if it's QNode-like (in-memory), adapt:
        if hasattr(node, "is_leaf"):
            if node.is_leaf:
                return 1 + 3
            s = 1
            for c in node.children:
                s += rec(c)
            return s
        # else assume dict form with children
        if isinstance(node, dict):
            if node.get("leaf", False):
                return 1 + 3
            s = 1
            for c in node.get("children", []):
                s += rec(c)
            return s
        return 1
    # allow passing either serialized dict or QNode
    return rec(qt)

def parse_args():
    p = argparse.ArgumentParser(prog="compress.py", description="Quadtree image compressor CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress", help="Compress image -> JSON + reconstructed PNG")
    c.add_argument("input", help="Input image path (PNG/JPG)")
    c.add_argument("--tolerance", type=float, default=1000.0, help="variance tolerance (lower = higher quality)")
    c.add_argument("--max-depth", type=int, default=None, help="max quadtree depth (<= log2(padded side))")
    c.add_argument("--out-prefix", type=str, default=None, help="optional prefix for output filenames")

    d = sub.add_parser("decompress", help="Decompress JSON -> PNG")
    d.add_argument("json", help="Compressed JSON file (produced by compress)")
    d.add_argument("--out", help="Output PNG path (default -> ./output/<json>_decompressed.png)")

    r = sub.add_parser("roundtrip", help="Compress in-memory and reconstruct (prints PSNR)")
    r.add_argument("input", help="Input image path (PNG/JPG)")
    r.add_argument("--tolerance", type=float, default=1000.0)
    r.add_argument("--max-depth", type=int, default=None)
    r.add_argument("--out", help="Optional output recon PNG path")

    return p.parse_args()

def main():
    args = parse_args()
    if args.cmd == "compress":
        compress_image(args.input, tolerance=args.tolerance, max_depth=args.max_depth, out_prefix=args.out_prefix)
    elif args.cmd == "decompress":
        decompress_json(args.json, out_path=args.out)
    elif args.cmd == "roundtrip":
        roundtrip(args.input, tolerance=args.tolerance, max_depth=args.max_depth, out_path=args.out)
    else:
        print("Unknown command")

if __name__ == "__main__":
    main()
