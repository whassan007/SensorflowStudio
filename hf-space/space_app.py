"""Hugging Face Space entrypoint for Sensorflow Studio.

Wraps the existing FastAPI app from app_backend.py WITHOUT modifying it, and
serves the built React frontend (dist/) at /dashboard.

app_backend.py mounts the legacy static UI as a catch-all at "/" during module
import, so by the time this module runs, that mount is already the last route.
A mount added via app.mount() here would land *after* the catch-all and never
match; instead the /dashboard mount is inserted just before it in the route
table.

Run with: uvicorn space_app:app --host 0.0.0.0 --port 7860
"""
from pathlib import Path

from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from app_backend import app

_dist_dir = Path(__file__).parent / "dist"

if _dist_dir.exists():
    _dashboard = Mount(
        "/dashboard",
        app=StaticFiles(directory=str(_dist_dir), html=True),
        name="dashboard",
    )
    _routes = app.router.routes
    _root_idx = next(
        (
            i
            for i, r in enumerate(_routes)
            if isinstance(r, Mount) and r.path in ("", "/")
        ),
        len(_routes),
    )
    _routes.insert(_root_idx, _dashboard)
