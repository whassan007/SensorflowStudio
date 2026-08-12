"""Population-scale evaluation foundation — extends MegaEval + LabelEval.

Phase 1: schemas, aggregate levels, metrics, container quality, model compare,
multi-gate skeleton, evidence packages. Phases 2–7 leave TODO hooks only.
"""

from sensorflow.platform.levels import AggregateLevel, EvaluationScope
from sensorflow.platform.entities import Provenance, EvidencePackage

__all__ = [
    "AggregateLevel",
    "EvaluationScope",
    "Provenance",
    "EvidencePackage",
]
