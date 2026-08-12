"""Launch gate evaluator: block export until quality thresholds pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_THRESHOLDS_PATH = Path("runs/pipeline/gate_thresholds.json")

DEFAULT_THRESHOLDS = {
    "map_3d": 0.65,
    "orientation_error_deg": 5.0,
    "id_swap_rate": 0.02,
    "track_fragmentation_rate": 0.05,
    "position_error_m": 2.0,
}


class LaunchGateEvaluator:
    """Validate benchmark metrics against safety thresholds before export."""

    def __init__(self, thresholds_path: Optional[Path] = None):
        self.thresholds_path = thresholds_path or DEFAULT_THRESHOLDS_PATH
        self.thresholds = self._load_thresholds()

    def _load_thresholds(self) -> Dict[str, float]:
        if self.thresholds_path.exists():
            with open(self.thresholds_path) as f:
                return json.load(f)
        self.thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.thresholds_path, "w") as f:
            json.dump(DEFAULT_THRESHOLDS, f, indent=2)
        return DEFAULT_THRESHOLDS.copy()

    def evaluate(self, sequence_id: str) -> Dict[str, Any]:
        metric_path = Path("runs/pipeline") / sequence_id / "benchmark" / "metric_card.json"
        if not metric_path.exists():
            return {
                "passed": False,
                "failures": [{"metric": "benchmark", "message": "No benchmark results found. Run quality gate first."}],
                "blocked_stages": ["export"],
            }

        with open(metric_path) as f:
            metrics = json.load(f)

        failures: List[Dict] = []
        checks = [
            ("map_3d", metrics.get("map_3d", 0), self.thresholds["map_3d"], ">="),
            ("orientation_error_deg", metrics.get("orientation_error_deg", 999), self.thresholds["orientation_error_deg"], "<="),
            ("id_swap_rate", metrics.get("id_swap_rate", 1), self.thresholds["id_swap_rate"], "<="),
            ("track_fragmentation_rate", metrics.get("track_fragmentation_rate", 1), self.thresholds["track_fragmentation_rate"], "<="),
        ]
        for name, value, threshold, op in checks:
            failed = (value < threshold) if op == ">=" else (value > threshold)
            if failed:
                failures.append({
                    "metric": name,
                    "value": value,
                    "threshold": threshold,
                    "operator": op,
                })

        passed = len(failures) == 0
        result = {
            "passed": passed,
            "failures": failures,
            "blocked_stages": [] if passed else ["export"],
            "metrics": metrics,
        }

        gate_path = Path("runs/pipeline") / sequence_id / "launch_gate.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gate_path, "w") as f:
            json.dump(result, f, indent=2)

        state_path = Path("runs/pipeline/state.json")
        state = {}
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
        state.setdefault(sequence_id, {})["launch_gate_passed"] = passed
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        return result

    def is_export_allowed(self, sequence_id: str) -> bool:
        gate_path = Path("runs/pipeline") / sequence_id / "launch_gate.json"
        if gate_path.exists():
            with open(gate_path) as f:
                return json.load(f).get("passed", False)
        return False
