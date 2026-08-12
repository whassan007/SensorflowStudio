# Sensorflow Studio — Hugging Face Spaces (Docker SDK) image.
# Multi-stage: build the React frontend, then assemble a slim Python runtime.
# The frontend is built INSIDE the image so deploys only need repo files.

# ---------- Stage 1: frontend build ----------
FROM node:22-alpine AS frontend
WORKDIR /build

# Lockfile may or may not exist / be in sync mid-refactor; fall back gracefully.
COPY package.json package-lock.json* ./
RUN npm ci || npm install

COPY index.html vite.config.ts tsconfig.json ./
COPY src ./src
# The React app is served at /dashboard/ (root is taken by the legacy static UI),
# so asset URLs must be built with that base. vite.config.ts must not be edited,
# hence the --base flag here.
RUN npx vite build --base=/dashboard/

# ---------- Stage 2: Python runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HF_HOME=/app/.cache

WORKDIR /app

# Slim dependency set: heavy GPU/vision deps (torch, ultralytics, opencv) are
# lazily imported by app_backend routes and are NOT needed at import time.
COPY deploy/huggingface/requirements-space.txt ./requirements-space.txt
RUN pip install --no-cache-dir -r requirements-space.txt

# Backend code + assets (filtered by .dockerignore: no node_modules, .venv,
# runs/, *.pt, models/, dist/, .git).
COPY . .

# Built React app from stage 1.
COPY --from=frontend /build/dist ./dist

# HF Spaces runs the container as uid 1000; the app writes runtime files
# (runs/studio_config.json, ssam_streets.json, ...) relative to the workdir,
# so /app must be writable by that user.
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860

# app_backend.py's own __main__ block uses 127.0.0.1:8000 and must not be
# edited; space_app wraps it and adds the /dashboard mount for the React build.
CMD ["uvicorn", "space_app:app", "--host", "0.0.0.0", "--port", "7860"]
