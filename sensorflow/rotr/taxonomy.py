"""Six-axis violation taxonomy, structured query, and clustering.

Axes: Actor / Vulnerability / Legality / Environment / Interaction /
Behavior, plus extensions RoadGeometry / TrafficControl / Visibility.

The structured query turns a natural-language request such as "failed to
yield to pedestrian at uncontrolled intersection during low visibility"
into a filterable query object via a DETERMINISTIC keyword map (no LLM in
the retrieval path), then filters violation signatures exactly.

Clustering mirrors raremine's mining semantics conceptually (signature ->
dedup -> diversity-aware exemplar) over ROTR-native structured signatures —
own implementation, raremine's store is untouched. Exact structured
matching over ~10^3 records needs a dict, not a vector DB (see the
DO-NOT-BUILD decision in the architecture doc).
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from pydantic import BaseModel

from sensorflow.rotr.models import ROTRScenario, ROTRViolation

TAXONOMY_VERSION = "rotr-taxonomy-1.0.0"

AXES = ["actor", "vulnerability", "legality", "environment", "interaction",
        "behavior", "road_geometry", "traffic_control", "visibility"]

RULE_TO_LEGALITY = {
    "R-YIELD-PED-01": "YIELD",
    "R-PATH-RESTRICT-02": "RESTRICTED_PATH",
    "R-LANE-MANEUVER-03": "LANE_ASSOCIATION",
    "R-INT-CONFLICT-04": "SIGNAL",
    "R-MERGE-GAP-05": "MERGE",
    "R-STOP-06": "STOP",
}

RULE_TO_BEHAVIOR = {
    "R-YIELD-PED-01": "proceed_without_yield",
    "R-PATH-RESTRICT-02": "enter_restricted_path",
    "R-LANE-MANEUVER-03": "unpermitted_maneuver",
    "R-INT-CONFLICT-04": "run_red_signal",
    "R-MERGE-GAP-05": "insufficient_gap_merge",
    "R-STOP-06": "rolling_stop",
}


def signature(violation: ROTRViolation, scenario: ROTRScenario) -> Dict[str, str]:
    material = [a for a in scenario.actors
                if a.actor_id in set(violation.actor_ids)]
    actor_cls = material[0].class_name if material else "none"
    vru = any(a.class_name in ("pedestrian", "cyclist") for a in material)
    interaction = "NONE"
    if material:
        intent = material[0].intent
        interaction = {"CROSSING": "CROSSING", "MERGING": "MERGING",
                       "PROCEEDING": "FOLLOWING"}.get(intent, "OTHER")
    env = scenario.environment
    ctx = scenario.actual_context
    return {
        "actor": actor_cls,
        "vulnerability": "VRU" if vru else "NON_VRU",
        "legality": RULE_TO_LEGALITY.get(violation.rule_id, "OTHER"),
        "environment": f"{env.lighting}/{env.weather}",
        "interaction": interaction,
        "behavior": RULE_TO_BEHAVIOR.get(violation.rule_id, "other"),
        "road_geometry": ctx.intersection_type,
        "traffic_control": ctx.control,
        "visibility": env.visibility,
    }


# ------------------------------------------------------------ structured query


class ROTRQuery(BaseModel):
    """Filterable query object over the taxonomy axes (all optional)."""

    actor: Optional[str] = None
    vulnerability: Optional[str] = None
    legality: Optional[str] = None
    environment: Optional[str] = None
    interaction: Optional[str] = None
    behavior: Optional[str] = None
    road_geometry: Optional[str] = None
    traffic_control: Optional[str] = None
    visibility: Optional[str] = None
    lighting: Optional[str] = None
    weather: Optional[str] = None
    consequence_class: Optional[str] = None
    primary_layer: Optional[str] = None
    text: Optional[str] = None            # original text, for provenance


# Deterministic keyword -> axis map (retrieval path is LLM-free).
_KEYWORDS = [
    ("pedestrian", "actor", "pedestrian"),
    ("cyclist", "actor", "cyclist"),
    ("vulnerable", "vulnerability", "VRU"),
    ("yield", "legality", "YIELD"),
    ("restricted", "legality", "RESTRICTED_PATH"),
    ("bus lane", "legality", "RESTRICTED_PATH"),
    ("wrong lane", "legality", "LANE_ASSOCIATION"),
    ("lane association", "legality", "LANE_ASSOCIATION"),
    ("red light", "legality", "SIGNAL"),
    ("signal", "legality", "SIGNAL"),
    ("merge", "legality", "MERGE"),
    ("stop sign", "legality", "STOP"),
    ("rolling stop", "legality", "STOP"),
    ("uncontrolled intersection", "road_geometry", "uncontrolled"),
    ("controlled intersection", "road_geometry", "controlled"),
    ("intersection", "road_geometry", None),   # weak; only if none set
    ("low visibility", "visibility", "low"),
    ("night", "lighting", "night"),
    ("dusk", "lighting", "dusk"),
    ("rain", "weather", "rain"),
    ("crossing", "interaction", "CROSSING"),
    ("safety critical", "consequence_class", "SAFETY_CRITICAL"),
]


def parse_query(text: str) -> ROTRQuery:
    low = text.lower()
    fields: Dict[str, str] = {}
    for kw, axis, value in _KEYWORDS:
        if kw in low and value is not None and axis not in fields:
            fields[axis] = value
    return ROTRQuery(text=text, **fields)


def matches(query: ROTRQuery, sig: Dict[str, str],
            consequence_class: Optional[str],
            primary_layer: Optional[str],
            environment: Dict[str, str]) -> bool:
    for axis in AXES:
        want = getattr(query, axis)
        if want is not None and sig.get(axis) != want:
            return False
    if query.lighting is not None and environment.get("lighting") != query.lighting:
        return False
    if query.weather is not None and environment.get("weather") != query.weather:
        return False
    if query.consequence_class is not None and \
            consequence_class != query.consequence_class:
        return False
    if query.primary_layer is not None and primary_layer != query.primary_layer:
        return False
    return True


# ------------------------------------------------------------ clustering


CLUSTER_KEY_AXES = ["legality", "actor", "traffic_control", "behavior"]


def cluster_key(sig: Dict[str, str], primary_layer: Optional[str]) -> str:
    parts = [sig.get(a, "?") for a in CLUSTER_KEY_AXES] + [primary_layer or "?"]
    return "|".join(parts)


def cluster_id_for(key: str) -> str:
    return "cl-" + hashlib.sha256(
        (TAXONOMY_VERSION + "|" + key).encode()).hexdigest()[:8]


def build_clusters(items: List[Dict]) -> List[Dict]:
    """items: [{violation_id, signature, primary_layer, consequence_class,
    environment}] -> recurring-pattern clusters with a diversity-aware
    exemplar (max environment spread, raremine-style selection concept)."""
    groups: Dict[str, List[Dict]] = {}
    for it in items:
        key = cluster_key(it["signature"], it.get("primary_layer"))
        groups.setdefault(key, []).append(it)
    clusters = []
    for key, members in sorted(groups.items()):
        envs = sorted({m["signature"].get("environment", "") +
                       "/" + m["signature"].get("visibility", "")
                       for m in members})
        cons: Dict[str, int] = {}
        for m in members:
            c = m.get("consequence_class") or "UNCLASSIFIED"
            cons[c] = cons.get(c, 0) + 1
        # Exemplar: the member in the least-represented environment cell
        # (diversity-aware pick, not just the first).
        env_counts: Dict[str, int] = {}
        for m in members:
            e = m["signature"].get("environment", "")
            env_counts[e] = env_counts.get(e, 0) + 1
        exemplar = min(members,
                       key=lambda m: (env_counts[m["signature"].get("environment", "")],
                                      m["violation_id"]))
        clusters.append({
            "cluster_id": cluster_id_for(key),
            "key": key,
            "count": len(members),
            "member_violation_ids": [m["violation_id"] for m in members],
            "exemplar_violation_id": exemplar["violation_id"],
            "environment_spread": envs,
            "consequence_distribution": cons,
            "taxonomy_version": TAXONOMY_VERSION,
        })
    clusters.sort(key=lambda c: -c["count"])
    return clusters
