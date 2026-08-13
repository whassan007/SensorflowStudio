"""Versioned safety-threshold registry with per-value provenance.

Audit F-007: the same "severity" concept is computed with TWO different
magic weightings in app_backend.py (line ~1101 vs ~1139), and TTC/PET
scenario cutoffs are buried as literals in evaluation/rare_events.py. None
of them carries provenance. This registry centralizes every audited value,
tags each with where it came from and whether it is defensible policy or an
illustrative placeholder, and versions the whole set so a SafetyAssessment
can record exactly which config produced it.

Provenance labels:
- FHWA_SSAM_DEFAULT: consistent with FHWA Surrogate Safety Assessment Model
  guidance (e.g. TTC < 1.5 s as critical-conflict threshold).
- ILLUSTRATIVE_THRESHOLD: a prototype constant with NO documented safety
  basis. It must not gate a real launch; replace with policy-derived values.

Wiring status: sensorflow/safety/ssam_ext.py already documents its defaults;
app_backend.py wiring is a follow-up (one-additive-edit budget, see audit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

CONFIG_VERSION = "safety-config-v1"

FHWA_SSAM_DEFAULT = "FHWA_SSAM_DEFAULT"
ILLUSTRATIVE_THRESHOLD = "ILLUSTRATIVE_THRESHOLD"


@dataclass(frozen=True)
class ThresholdSpec:
    name: str
    value: float
    unit: str
    provenance: str                 # FHWA_SSAM_DEFAULT | ILLUSTRATIVE_THRESHOLD
    source: str                     # FILE:LINE or document reference
    description: str = ""


# Every audited literal, in one place. Names are namespaced by the component
# that used them so the two conflicting severity weightings are both visible.
THRESHOLDS: Dict[str, ThresholdSpec] = {spec.name: spec for spec in [
    # --- SSAM conflict thresholds (defensible) -------------------------------
    ThresholdSpec("ssam.ttc_critical_s", 1.5, "s", FHWA_SSAM_DEFAULT,
                  "FHWA-HRT-08-051; safety/ssam_ext.py defaults",
                  "TTC below this marks a critical conflict."),
    ThresholdSpec("ssam.pet_critical_s", 5.0, "s", FHWA_SSAM_DEFAULT,
                  "FHWA SSAM guidance; app_backend.py:1101",
                  "PET cap used in severity normalization."),
    # --- Legacy severity weighting #1 (app_backend._compute_severity) --------
    ThresholdSpec("legacy.severity.ttc_weight", 0.5, "unitless", ILLUSTRATIVE_THRESHOLD,
                  "app_backend.py:1101", "Weight of normalized TTC in severity_index."),
    ThresholdSpec("legacy.severity.pet_weight", 0.2, "unitless", ILLUSTRATIVE_THRESHOLD,
                  "app_backend.py:1101", "Weight of normalized PET in severity_index."),
    ThresholdSpec("legacy.severity.speed_weight", 0.3, "unitless", ILLUSTRATIVE_THRESHOLD,
                  "app_backend.py:1101", "Weight of normalized max_speed in severity_index."),
    ThresholdSpec("legacy.severity.speed_cap_mps", 18.0, "m/s", ILLUSTRATIVE_THRESHOLD,
                  "app_backend.py:1101", "Speed normalization cap."),
    # --- Legacy severity weighting #2 (streets endpoint) — CONFLICTS with #1 -
    ThresholdSpec("legacy.streets.ttc_weight", 0.7, "unitless", ILLUSTRATIVE_THRESHOLD,
                  "app_backend.py:1139",
                  "DIFFERENT weighting for the same severity concept; the two "
                  "endpoints disagree about identical inputs (audit F-007)."),
    ThresholdSpec("legacy.streets.pet_weight", 0.3, "unitless", ILLUSTRATIVE_THRESHOLD,
                  "app_backend.py:1139", "See above."),
    # --- Severity band cutoffs ------------------------------------------------
    ThresholdSpec("legacy.severity.band_critical", 0.7, "unitless", ILLUSTRATIVE_THRESHOLD,
                  "app_backend.py:1103", "severity_index >= this -> Critical."),
    ThresholdSpec("legacy.severity.band_high", 0.45, "unitless", ILLUSTRATIVE_THRESHOLD,
                  "app_backend.py:1105", "severity_index >= this -> High."),
    ThresholdSpec("legacy.severity.band_medium", 0.25, "unitless", ILLUSTRATIVE_THRESHOLD,
                  "app_backend.py:1107", "severity_index >= this -> Medium."),
    # --- Rare-event scenario cutoffs ------------------------------------------
    ThresholdSpec("rare_events.ttc_extreme_s", 3.0, "s", ILLUSTRATIVE_THRESHOLD,
                  "evaluation/rare_events.py:84", "extreme_ttc scenario cutoff."),
    ThresholdSpec("rare_events.ttc_near_collision_s", 1.8, "s", ILLUSTRATIVE_THRESHOLD,
                  "evaluation/rare_events.py:86", "near_collision scenario cutoff."),
    ThresholdSpec("rare_events.pet_extreme_s", 1.5, "s", ILLUSTRATIVE_THRESHOLD,
                  "evaluation/rare_events.py:89", "extreme_pet scenario cutoff."),
    # --- Legacy quality gate ---------------------------------------------------
    ThresholdSpec("quality_gate.map_3d_min", 0.65, "unitless", ILLUSTRATIVE_THRESHOLD,
                  "sensorflow/quality_gate.py:24", "Launch-gate mAP floor."),
    ThresholdSpec("quality_gate.orientation_error_max_deg", 5.0, "deg", ILLUSTRATIVE_THRESHOLD,
                  "sensorflow/quality_gate.py:25", "Launch-gate orientation ceiling."),
    ThresholdSpec("quality_gate.id_swap_rate_max", 0.02, "unitless", ILLUSTRATIVE_THRESHOLD,
                  "sensorflow/quality_gate.py:26", "Launch-gate ID-swap ceiling."),
    ThresholdSpec("quality_gate.fragmentation_rate_max", 0.05, "unitless", ILLUSTRATIVE_THRESHOLD,
                  "sensorflow/quality_gate.py:27", "Launch-gate fragmentation ceiling."),
    ThresholdSpec("quality_gate.position_error_max_m", 2.0, "m", ILLUSTRATIVE_THRESHOLD,
                  "sensorflow/quality_gate.py:28", "Launch-gate position-error ceiling."),
]}


def get_threshold(name: str) -> ThresholdSpec:
    return THRESHOLDS[name]


def compute_severity(ttc_s: Optional[float], pet_s: Optional[float],
                     max_speed_mps: Optional[float]) -> Dict:
    """Canonical severity computation replacing the two conflicting legacy
    weightings (F-007). Same functional form as app_backend._compute_severity,
    but every constant is sourced from the registry, missing inputs degrade
    the confidence rather than defaulting silently, and the config version is
    stamped on the result.
    """
    used: List[str] = []
    score = 1.0
    if ttc_s is not None:
        cap = THRESHOLDS["ssam.ttc_critical_s"].value
        w = THRESHOLDS["legacy.severity.ttc_weight"].value
        score -= (min(ttc_s, cap) / cap) * w
        used.append("ttc")
    if pet_s is not None:
        cap = THRESHOLDS["ssam.pet_critical_s"].value
        w = THRESHOLDS["legacy.severity.pet_weight"].value
        score -= (min(pet_s, cap) / cap) * w
        used.append("pet")
    if max_speed_mps is not None:
        cap = THRESHOLDS["legacy.severity.speed_cap_mps"].value
        w = THRESHOLDS["legacy.severity.speed_weight"].value
        score -= (1.0 - min(max_speed_mps, cap) / cap) * w
        used.append("speed")

    score = round(max(0.0, min(1.0, score)), 3)
    if score >= THRESHOLDS["legacy.severity.band_critical"].value:
        label = "Critical"
    elif score >= THRESHOLDS["legacy.severity.band_high"].value:
        label = "High"
    elif score >= THRESHOLDS["legacy.severity.band_medium"].value:
        label = "Medium"
    else:
        label = "Low"
    return {
        "severity_index": score,
        "severity_label": label,
        "inputs_used": used,
        "inputs_missing": [k for k in ("ttc", "pet", "speed") if k not in used],
        "config_version": CONFIG_VERSION,
        "weights_provenance": ILLUSTRATIVE_THRESHOLD,
    }


def registry_summary() -> List[Dict]:
    return [
        {"name": s.name, "value": s.value, "unit": s.unit,
         "provenance": s.provenance, "source": s.source, "description": s.description}
        for s in THRESHOLDS.values()
    ]
