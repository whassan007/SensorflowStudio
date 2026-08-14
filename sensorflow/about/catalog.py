"""Product version and release-notes catalog (shared with the About UI)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List


def _releases_json_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "src" / "content" / "releases.json",  # hf-space/src/content
        here.parent / "releases.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("releases.json not found next to the About catalog")


@lru_cache(maxsize=1)
def get_about() -> Dict[str, Any]:
    """Return {name, version, description, links, releases} from the JSON catalog."""
    raw = json.loads(_releases_json_path().read_text(encoding="utf-8"))
    releases: List[Dict[str, Any]] = list(raw.get("releases") or [])
    version = str(raw.get("version") or (releases[0]["version"] if releases else "0.0.0"))
    links = raw.get("links") or {}
    return {
        "name": str(raw.get("name") or "Sensorflow Studio"),
        "version": version,
        "description": str(raw.get("description") or ""),
        "links": {
            "github": str(links.get("github") or "https://github.com/whassan007/SensorflowStudio"),
            "hf_space": str(links.get("hf_space") or "https://huggingface.co/spaces/whassan/sensorflow-studio"),
        },
        "releases": releases,
    }


def get_version() -> Dict[str, Any]:
    about = get_about()
    return {"version": about["version"], "name": about["name"], "releases": about["releases"]}
