"""FastAPI routes for product version / About (GET /api/about, GET /api/version)."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from sensorflow.about.catalog import get_about, get_version

router = APIRouter(prefix="/api", tags=["about"])


@router.get("/about")
def about_payload() -> Dict[str, Any]:
    """Product name, current version, links, and chronological release notes."""
    return get_about()


@router.get("/version")
def version_payload() -> Dict[str, Any]:
    """Current version plus the release list (for help chat / “what’s new”)."""
    return get_version()
