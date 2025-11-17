🌳 Quadtree Image Compressor
High-Performance, Edge-Aware Image Compression using Adaptive Quadtrees

Modern Web UI | Flask API | Dockerized | GHCR Deployment | Parallel Processing

🚀 Overview

This project is a full production-ready quadtree image compressor built with:

Adaptive Quadtree compression

Edge-aware subdivision

Parallel full-resolution build

Fast downscale-guided tolerance search

Modern drag-and-drop web interface

Dockerized runtime

GitHub Actions CI/CD

Container hosting via GHCR

Designed to compress images without losing structural detail while offering high-speed performance and a clean browser-based UI.

✨ Features
🖼 Modern Web Interface

Drag & drop image upload

Live preview

Auto-target compression size (KB)

Manual tolerance mode

Max depth control

Real-time PSNR

Download compressed PNG & JSON quadtree structure

Fully responsive (mobile-ready)

⚡ Performance Optimizations

Edge-aware quadtree splitting

Adaptive tolerance based on downscaled preview

Parallelized final build

Smart variance-based subdivision

Optional subtree collapse for identical children

Constant-time deserialization for reconstructing images

🐳 Full Docker Support

Lightweight Python 3.10-slim image

Pillow, NumPy, Flask preinstalled

Production server with gunicorn

docker-compose.yml for local dev

Published image via GitHub Container Registry (GHCR)

☁️ CI/CD with GitHub Actions

Automatic pipelines:

Build Docker image (multi-platform ready)

Push to GHCR on every main branch push

Zero configuration needed on your side

Installation (Local)
1. Clone the repo
git clone https://github.com/navonilmandal/quadtree_compressor.git
cd <repo>

2. Install dependencies
pip install -r requirements.txt

3. Run the app
python web/app.py


Open:
👉 http://localhost:5000

🐳 Docker Usage (Recommended)
Build & run locally
docker build -t quadtree .
docker run -p 5000:5000 quadtree

Or using docker-compose
docker compose up --build


Open:
👉 http://localhost:5000
