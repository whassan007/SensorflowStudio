"""Production-readiness scorecard computed from audit.json.

Rule enforced in code: NO "production ready" status can be reported while
any Critical finding remains open. The scorecard is derived, never
hand-written, so it cannot drift from the audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

AUDIT_JSON = Path(__file__).resolve().parents[2] / "docs" / "hardening" / "audit.json"

#: Category -> (prototype description, production requirement). Findings are
#: assigned to categories via keyword match against their free-text audit
#: `area` field (first matching category wins, in this order).
CATEGORY_SPECS: Dict[str, Dict] = {
    "Data honesty & provenance": {
        "prototype": "Several legacy endpoints fabricate data or serve mocks unlabeled",
        "production": "Every payload carries provenance + simulated markers; "
                      "LabelProvenance enforced at gates",
        "keywords": ["mock", "leakage", "provenance", "determinism / mock"],
    },
    "Security": {
        "prototype": "Arbitrary path read/write, CORS wildcard with credentials, "
                     "no authn",
        "production": "AuthN/AuthZ on all endpoints, path allowlists, strict CORS",
        "keywords": ["security"],
    },
    "Safety configuration": {
        "prototype": "Magic severity weights, conflicting formulas, buried literals",
        "production": "Versioned threshold registry with per-value provenance; "
                      "one severity definition",
        "keywords": ["safety thresholds", "scoring provenance"],
    },
    "Caching & reproducibility": {
        "prototype": "Incomplete cache keys, no integrity checks, unseeded fallbacks",
        "production": "Manifest-keyed caches with checksums; all randomness seeded",
        "keywords": ["cache", "determinism", "configuration hygiene"],
    },
    "HITL & triage": {
        "prototype": "FIFO review queues, arbitrary evidence snippets",
        "production": "Information-gain prioritization; router gated on precision "
                      "AND critical-miss-rate",
        "keywords": ["hitl"],
    },
    "Statistical validity": {
        "prototype": "Point-delta tolerances, rolling baselines, no power analysis "
                     "outside seqeval",
        "production": "CI/e-process decisions, pinned baselines, derived sample "
                      "sizes, multiplicity control",
        "keywords": ["regression statistics", "sampling", "grader", "novelty",
                     "statistics", "anomaly ensemble", "meta-classifier"],
    },
    "Metric correctness": {
        "prototype": "Cross-frame matching, ungated ID-swap matching, mAP misnomer",
        "production": "Frame-scoped matching, gated association, literature-"
                      "consistent naming",
        "keywords": ["metric", "temporal", "geometric validation"],
    },
    "Scalability & infrastructure": {
        "prototype": "In-process dicts and JSON files under runs/; single node",
        "production": "Storage/compute seams (interfaces.py) backed by managed "
                      "services",
        "keywords": ["scalability"],
    },
}

_FIXED_DISPOSITIONS = {"fix_now", "fix_now_partial", "fix_now_layered"}


def load_audit(path: Optional[Path] = None) -> Dict:
    p = path or AUDIT_JSON
    return json.loads(p.read_text())


def _categorize(findings: List[Dict]) -> Dict[str, List[Dict]]:
    """Assign each finding to the first category whose keyword matches its area."""
    buckets: Dict[str, List[Dict]] = {name: [] for name in CATEGORY_SPECS}
    for f in findings:
        area = str(f.get("area", "")).lower()
        for name, spec in CATEGORY_SPECS.items():
            if any(kw in area for kw in spec["keywords"]):
                buckets[name].append(f)
                break
    return buckets


def scorecard(audit: Optional[Dict] = None) -> Dict:
    """Category | Prototype | Production Requirement | Gap, from audit.json."""
    audit = audit or load_audit()
    findings = audit.get("findings", [])
    buckets = _categorize(findings)

    categories: List[Dict] = []
    for name, spec in CATEGORY_SPECS.items():
        related = buckets[name]
        open_findings = [f for f in related
                         if f.get("disposition") not in _FIXED_DISPOSITIONS]
        open_critical = [f for f in open_findings
                         if str(f.get("severity", "")).lower() == "critical"]
        partial = [f for f in related if f.get("disposition") in
                   ("fix_now_partial", "fix_now_layered")]

        if related and not open_findings and not partial:
            status = "closed"
        elif open_critical:
            status = "blocked_critical"
        elif open_findings:
            status = "gaps_open"
        elif partial:
            status = "partially_hardened"
        else:
            status = "no_findings"

        categories.append({
            "category": name,
            "prototype": spec["prototype"],
            "production_requirement": spec["production"],
            "gap_count": len(open_findings),
            "open_finding_ids": [f["id"] for f in open_findings],
            "open_critical_ids": [f["id"] for f in open_critical],
            "partially_fixed_ids": [f["id"] for f in partial],
            "status": status,
        })

    any_open_critical = any(
        str(f.get("severity", "")).lower() == "critical" and
        f.get("disposition") not in _FIXED_DISPOSITIONS
        for f in findings)

    return {
        "categories": categories,
        "overall_status": "NOT_PRODUCTION_READY" if any_open_critical
        else ("HARDENING_IN_PROGRESS"
              if any(c["gap_count"] for c in categories) else "READY_CANDIDATE"),
        "rule": "No production-ready status while any Critical finding is open.",
        "summary": audit.get("summary", {}),
    }
