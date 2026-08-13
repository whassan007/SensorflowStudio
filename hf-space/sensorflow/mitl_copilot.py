"""MITL copilot: LLM failure-mode critique and edge-case routing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


def _llm_endpoints() -> List[Dict[str, str]]:
    """LLM endpoints in fallback order, configurable via environment.

    SENSORFLOW_LLM_URL / SENSORFLOW_LLM_MODEL prepend a primary endpoint
    (e.g. a remote GPU box); local Ollama is always the last fallback.
    Deployment-specific hostnames must never be hard-coded in source.
    """
    endpoints: List[Dict[str, str]] = []
    url = os.environ.get("SENSORFLOW_LLM_URL")
    if url:
        endpoints.append({
            "url": url,
            "model": os.environ.get("SENSORFLOW_LLM_MODEL", "gemma4:latest"),
        })
    endpoints.append({"url": "http://localhost:11434/api/chat", "model": "gemma4:latest"})
    return endpoints


OLLAMA_ENDPOINTS = _llm_endpoints()

MITL_QUEUE_PATH = Path("runs/pipeline/mitl_queue.json")


class MitlCopilot:
    """Route edge cases to human review with LLM-generated critiques."""

    def __init__(self, queue_path: Optional[Path] = None):
        self.queue_path = queue_path or MITL_QUEUE_PATH

    def route_edge_cases(
        self,
        sequence_id: str,
        metric_card: Dict[str, float],
        pred_tracks: List[Dict],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        thresholds = thresholds or {
            "id_swap_rate": 0.02,
            "position_error_m": 2.0,
            "track_fragmentation_rate": 0.05,
        }
        queue = self._load_queue()
        edge_cases = []

        if metric_card.get("id_swap_rate", 0) > thresholds["id_swap_rate"]:
            edge_cases.append({
                "type": "id_swap",
                "severity": "high",
                "metric_value": metric_card["id_swap_rate"],
                "tracks": pred_tracks[:3],
            })

        if metric_card.get("position_error_m", 0) > thresholds["position_error_m"]:
            edge_cases.append({
                "type": "position_error",
                "severity": "high",
                "metric_value": metric_card["position_error_m"],
                "tracks": pred_tracks[:3],
            })

        if metric_card.get("track_fragmentation_rate", 0) > thresholds["track_fragmentation_rate"]:
            edge_cases.append({
                "type": "fragmentation",
                "severity": "medium",
                "metric_value": metric_card["track_fragmentation_rate"],
                "tracks": pred_tracks[:3],
            })

        for case in edge_cases:
            case["sequence_id"] = sequence_id
            case["critique"] = self.generate_critique(case, metric_card)
            queue.append(case)

        self._save_queue(queue)
        return edge_cases

    def generate_critique(
        self,
        edge_case: Dict[str, Any],
        metric_card: Dict[str, float],
    ) -> str:
        prompt = f"""You are a Lead QA Auditor for an autonomous driving 3D perception pipeline.
Analyze this failure mode and provide a concise structural critique.

FAILURE TYPE: {edge_case['type']}
SEVERITY: {edge_case['severity']}
METRIC VALUE: {edge_case['metric_value']}

FULL METRIC CARD:
{json.dumps(metric_card, indent=2)}

AFFECTED TRACKS (sample):
{json.dumps(edge_case.get('tracks', [])[:2], indent=2)}

Provide a professional markdown critique covering:
1. Root cause hypothesis
2. Impact on downstream safety
3. Recommended remediation steps

Be concise and actionable."""

        for ep in OLLAMA_ENDPOINTS:
            try:
                payload = {
                    "model": ep["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                res = httpx.post(ep["url"], json=payload, timeout=25.0)
                if res.status_code == 200:
                    text = res.json().get("message", {}).get("content", "")
                    if text:
                        return text
            except Exception:
                continue

        return self._mock_critique(edge_case)

    def _mock_critique(self, edge_case: Dict) -> str:
        return (
            f"### Edge Case: {edge_case['type']}\n\n"
            f"Metric value **{edge_case['metric_value']:.4f}** exceeds threshold. "
            f"Review affected tracks manually. LLM critique unavailable (Ollama offline)."
        )

    def _load_queue(self) -> List[Dict]:
        if self.queue_path.exists():
            with open(self.queue_path) as f:
                return json.load(f)
        return []

    def _save_queue(self, queue: List[Dict]):
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.queue_path, "w") as f:
            json.dump(queue, f, indent=2)

    def get_queue(self, sequence_id: Optional[str] = None) -> List[Dict]:
        queue = self._load_queue()
        if sequence_id:
            return [q for q in queue if q.get("sequence_id") == sequence_id]
        return queue
