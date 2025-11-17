# core/quadtree_core.py (optimized)
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import json, math, multiprocessing, os

@dataclass
class QNode:
    is_leaf: bool
    color: Optional[Tuple[int,int,int]] = None
    children: Optional[List["QNode"]] = None

def next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def pad_image_to_pow2(img: np.ndarray) -> Tuple[np.ndarray, int, int]:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be HxW x 3 RGB uint8")
    h, w = img.shape[:2]
    side = max(next_power_of_two(w), next_power_of_two(h))
    if side == w and side == h:
        return img.copy(), w, h
    out = np.zeros((side, side, 3), dtype=img.dtype)
    out[:h, :w] = img
    return out, w, h

# ---------------- Edge detection (fast) ----------------
def compute_edge_map(img: np.ndarray, threshold: float = 40.0) -> np.ndarray:
    arr = img.astype(np.float32)
    Y = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
    Kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    padY = np.pad(Y, 1, mode='reflect')
    gx = (Kx[0,0]*padY[0:-2,0:-2] + Kx[0,1]*padY[0:-2,1:-1] + Kx[0,2]*padY[0:-2,2:] +
          Kx[1,0]*padY[1:-1,0:-2] + Kx[1,1]*padY[1:-1,1:-1] + Kx[1,2]*padY[1:-1,2:] +
          Kx[2,0]*padY[2:,0:-2] + Kx[2,1]*padY[2:,1:-1] + Kx[2,2]*padY[2:,2:])
    gy = (Ky[0,0]*padY[0:-2,0:-2] + Ky[0,1]*padY[0:-2,1:-1] + Ky[0,2]*padY[0:-2,2:] +
          Ky[1,0]*padY[1:-1,0:-2] + Ky[1,1]*padY[1:-1,1:-1] + Ky[1,2]*padY[1:-1,2:] +
          Ky[2,0]*padY[2:,0:-2] + Ky[2,1]*padY[2:,1:-1] + Ky[2,2]*padY[2:,2:])
    grad = np.hypot(gx, gy)
    return grad > threshold

# ---------------- luminance stats ----------------
def _block_luminance_stats(img: np.ndarray, x0:int, y0:int, size:int):
    block = img[y0:y0+size, x0:x0+size].astype(np.float64)
    Y = 0.299*block[...,0] + 0.587*block[...,1] + 0.114*block[...,2]
    return float(Y.reshape(-1).mean()), float(Y.reshape(-1).var())

# ---------------- basic (single-process) edge-aware builder ----------------
def build_quadtree_edge_aware(img: np.ndarray, x0:int, y0:int, size:int,
                   tol: float, max_depth: int, cur_depth: int = 0,
                   edge_map: Optional[np.ndarray] = None) -> QNode:
    if edge_map is not None and cur_depth < max_depth and size > 1:
        if edge_map[y0:y0+size, x0:x0+size].any():
            hs = size // 2
            children = [
                build_quadtree_edge_aware(img, x0,      y0+hs, hs, tol, max_depth, cur_depth+1, edge_map),
                build_quadtree_edge_aware(img, x0+hs,   y0+hs, hs, tol, max_depth, cur_depth+1, edge_map),
                build_quadtree_edge_aware(img, x0,      y0,    hs, tol, max_depth, cur_depth+1, edge_map),
                build_quadtree_edge_aware(img, x0+hs,   y0,    hs, tol, max_depth, cur_depth+1, edge_map),
            ]
            if all(c.is_leaf for c in children):
                cols = [c.color for c in children]
                if cols.count(cols[0]) == 4:
                    return QNode(is_leaf=True, color=cols[0])
            return QNode(is_leaf=False, children=children)
    meanY, varY = _block_luminance_stats(img, x0, y0, size)
    if varY <= tol or cur_depth >= max_depth or size == 1:
        block = img[y0:y0+size, x0:x0+size].astype(np.float64)
        meanRGB = tuple(int(round(c)) for c in block.reshape(-1,3).mean(axis=0))
        return QNode(is_leaf=True, color=meanRGB)
    hs = size // 2
    children = [
        build_quadtree_edge_aware(img, x0,      y0+hs, hs, tol, max_depth, cur_depth+1, edge_map),
        build_quadtree_edge_aware(img, x0+hs,   y0+hs, hs, tol, max_depth, cur_depth+1, edge_map),
        build_quadtree_edge_aware(img, x0,      y0,    hs, tol, max_depth, cur_depth+1, edge_map),
        build_quadtree_edge_aware(img, x0+hs,   y0,    hs, tol, max_depth, cur_depth+1, edge_map),
    ]
    if all(c.is_leaf for c in children):
        cols = [c.color for c in children]
        if cols.count(cols[0]) == 4:
            return QNode(is_leaf=True, color=cols[0])
    return QNode(is_leaf=False, children=children)

# ---------------- multiprocessed final build ----------------
def _build_subtree_worker(args):
    img_segment, x0, y0, size, tol, max_depth, edge_map = args
    # img_segment is the full padded image reference (we keep it as global by send picklable args)
    return build_quadtree_edge_aware(img_segment, x0, y0, size, tol, max_depth, 0, edge_map)

def parallel_build_full(img: np.ndarray, tol: float, max_depth: int, edge_map: Optional[np.ndarray]=None):
    """Split top-level quad into four and build in parallel; merge into root."""
    side = img.shape[0]
    if side == 1:
        return build_quadtree_edge_aware(img, 0, 0, 1, tol, max_depth, 0, edge_map)
    hs = side // 2
    # tasks: (img, x0, y0, size, tol, max_depth, edge_map)
    tasks = [
        (img, 0,      hs, hs, tol, max_depth-1, edge_map),  # SW
        (img, hs,     hs, hs, tol, max_depth-1, edge_map),  # SE
        (img, 0,      0,  hs, tol, max_depth-1, edge_map),  # NW
        (img, hs,     0,  hs, tol, max_depth-1, edge_map),  # NE
    ]
    # choose pool size
    pool_size = min(4, max(1, multiprocessing.cpu_count()//1))
    with multiprocessing.Pool(processes=pool_size) as pool:
        results = pool.map(_build_subtree_worker, tasks)
    root = QNode(is_leaf=False, children=list(results))
    # attempt collapse
    if all(c.is_leaf for c in root.children):
        cols = [c.color for c in root.children]
        if cols.count(cols[0]) == 4:
            return QNode(is_leaf=True, color=cols[0])
    return root

# ---------------- downscale-for-search helper ----------------
def downscale_image(img: np.ndarray, factor: int):
    if factor <= 1:
        return img
    pil = Image.fromarray(img)
    w,h = pil.size
    neww = max(1, w//factor)
    newh = max(1, h//factor)
    small = pil.resize((neww,newh), Image.BILINEAR)
    return np.array(small)

# ---------------- JSON helpers & render ----------------
def serialize_quadtree(node: QNode) -> Dict[str,Any]:
    if node.is_leaf:
        return {"leaf": True, "color": list(node.color)}
    return {"leaf": False, "children": [serialize_quadtree(c) for c in node.children]}

def deserialize_quadtree(d: Dict[str,Any]) -> QNode:
    if d["leaf"]:
        return QNode(is_leaf=True, color=tuple(d["color"]))
    return QNode(is_leaf=False, children=[deserialize_quadtree(c) for c in d["children"]])

def render_quadtree(node: QNode, canvas: np.ndarray, x0:int, y0:int, size:int):
    if node.is_leaf:
        canvas[y0:y0+size, x0:x0+size] = node.color
        return
    hs = size // 2
    render_quadtree(node.children[0], canvas, x0,      y0+hs, hs)
    render_quadtree(node.children[1], canvas, x0+hs,   y0+hs, hs)
    render_quadtree(node.children[2], canvas, x0,      y0,    hs)
    render_quadtree(node.children[3], canvas, x0+hs,   y0,    hs)

def count_nodes(node: QNode) -> int:
    if node.is_leaf: return 1
    return 1 + sum(count_nodes(c) for c in node.children)

def estimate_serialized_bytes(node: QNode) -> int:
    if node.is_leaf: return 1 + 3
    return 1 + sum(estimate_serialized_bytes(c) for c in node.children)

# ---------------- small self-test if run directly ----------------
if __name__ == "__main__":
    print("Optimized quadtree_core quick test")
    W,H = 512, 320
    img = np.full((H,W,3), 240, dtype=np.uint8)
    img[40:44, 20:480] = [30,30,30]
    padded, ow, oh = pad_image_to_pow2(img)
    side = padded.shape[0]
    edge_map = compute_edge_map(padded, threshold=30.0)
    # quick search uses small downscale for speed; final build uses parallel_build_full
    qt_small = build_quadtree_edge_aware(downscale_image(padded, 8), 0,0,downscale_image(padded,8).shape[0], tol=50.0, max_depth=6, edge_map=None)
    qt = parallel_build_full(padded, tol=50.0, max_depth=8, edge_map=edge_map)
    print("nodes:", count_nodes(qt))
    canvas = np.zeros_like(padded)
    render_quadtree(qt, canvas, 0,0,side)
    Image.fromarray(canvas[:oh,:ow]).save("perf_test_recon.png")
    Image.fromarray(img).save("perf_test_orig.png")
    print("wrote test images")
