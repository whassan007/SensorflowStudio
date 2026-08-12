"""Six-axis failure taxonomy for stratification."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TaxonomyAxes(BaseModel):
    """Six-axis failure taxonomy: Mode, Polarity, Actor, Vulnerable, Legality, Infra."""

    mode: str = "urban"
    polarity: str = "benign"
    actor: str = "unknown"
    vulnerable: bool = False
    legality: str = "legal"
    infra: str = "standard"


VULNERABLE_ACTORS = {
    "pedestrian", "cyclist", "motorcyclist", "child", "wheelchair",
    "person", "skater", "scooter",
}

INFRA_CLASSES = {
    "construction", "barrier", "traffic_cone", "traffic_light",
    "stop_sign", "temporary",
}

ILLEGAL_CONTEXTS = {"jaywalking", "wrong_way", "red_light_run"}


def assign_taxonomy_axes(
    class_name: str,
    speed_kmh: float = 0.0,
    context: Optional[str] = None,
) -> TaxonomyAxes:
    """Derive six-axis taxonomy tags from class name and scene context."""
    actor = class_name.lower().replace(" ", "_")
    vulnerable = actor in VULNERABLE_ACTORS
    infra = "temporary" if actor in INFRA_CLASSES else "standard"

    if speed_kmh > 80:
        mode = "highway"
    elif speed_kmh > 40:
        mode = "arterial"
    else:
        mode = "urban"

    polarity = "adversarial" if context in {"occluded", "edge_case", "rare"} else "benign"
    legality = "illegal" if context in ILLEGAL_CONTEXTS else "legal"

    return TaxonomyAxes(
        mode=mode,
        polarity=polarity,
        actor=actor,
        vulnerable=vulnerable,
        legality=legality,
        infra=infra,
    )
