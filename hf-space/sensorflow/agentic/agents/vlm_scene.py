"""VLMSceneAnalysisAgent — scene-context hypotheses.

Produces hypotheses about WHY the scene context may explain the failure,
from scene metadata via a deterministic rule lexicon (with an optional Ollama
rationale on top). Its entire output is explicitly labeled HYPOTHESIS and is
never treated as ground truth by any downstream component: the policy engine
only consumes it as narrative context, never as evidence of cause.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from sensorflow.agentic.agents.base import BaseAgent, compact, no_escalation
from sensorflow.agentic.models import AgentEscalation, FailureEvent

# Deterministic rule lexicon: (predicate over instance context) -> hypothesis.
LEXICON = [
    ("construction_zone", lambda i: i.construction_zone,
     "Construction-zone visual context (cones, barriers, hi-vis colors) may "
     "bias the classifier toward construction furniture classes."),
    ("night_lighting", lambda i: i.time_of_day == "night",
     "Low illumination reduces appearance cues; silhouette-based features "
     "dominate and small upright objects become confusable."),
    ("rain_weather", lambda i: i.weather == "rain",
     "Rain scatter degrades both camera contrast and LiDAR returns, "
     "increasing shape ambiguity."),
    ("long_range", lambda i: i.distance_m > 40.0,
     "At long range the object subtends few pixels/points; class evidence "
     "is weak and priors dominate."),
    ("occlusion", lambda i: i.occluded,
     "Partial occlusion truncates the silhouette, which can resemble a "
     "static vertical object."),
    ("cone_confusion", lambda i: i.predicted_class == "construction_cone",
     "Pedestrian-to-cone confusion is consistent with hi-vis clothing or a "
     "stationary pedestrian adopting a cone-like compact silhouette."),
]


class VLMSceneAnalysisAgent(BaseAgent):
    name = "vlm_scene_analysis"
    version = "1.0.0"
    epistemic_status = "HYPOTHESIS"   # everything here is a hypothesis

    def _analyze(self, failure_id: str, failure: FailureEvent = None,
                 **inputs) -> Tuple[Dict[str, Any], float, str, AgentEscalation]:
        assert failure is not None
        hypotheses: List[Dict] = []
        counts: Dict[str, int] = {}
        for inst in failure.instances:
            for key, pred, text in LEXICON:
                if pred(inst):
                    counts[key] = counts.get(key, 0) + 1
        n = max(len(failure.instances), 1)
        for key, pred, text in LEXICON:
            if key in counts:
                hypotheses.append({
                    "label": "HYPOTHESIS",
                    "factor": key,
                    "text": text,
                    "instance_support": counts[key],
                    "instance_share": round(counts[key] / n, 3),
                })
        hypotheses.sort(key=lambda h: -h["instance_support"])

        output = {
            "epistemic_label": ("HYPOTHESIS — scene-context conjectures from "
                                "metadata rules; NOT ground truth, NOT "
                                "evidence of cause"),
            "hypotheses": hypotheses,
            "instances_examined": len(failure.instances),
            "lexicon_version": self.version,
        }
        confidence = min(0.6, 0.2 + 0.1 * len(hypotheses))
        return (output, confidence,
                "rule-lexicon coverage of instance contexts; capped at 0.6 "
                "because hypotheses are unverified",
                no_escalation())

    def _llm_prompt(self, failure_id, output, **inputs):
        return ("You are a scene-analysis assistant. Given these rule-based "
                "hypotheses about a perception failure, write a short "
                "clearly-hedged paragraph of additional scene-context "
                "hypotheses. Never state a cause as fact. "
                + compact(output))
