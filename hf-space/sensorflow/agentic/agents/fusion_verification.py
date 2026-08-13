"""SensorFusionVerificationAgent — cross-modality coherence check.

Re-simulates the bevfusion camera and LiDAR sensor models on the failure
frames (deterministic seeded reuse of sensorflow.bevfusion.sensors) and
cross-checks camera / LiDAR / tracking / ground-truth coherence for each
failure instance.

Verdicts:
  multi_modal_supported  both modalities independently corroborate the GT
                         object (the misclassification is well-evidenced)
  single_modality_only   only one modality carries evidence — weaker claim
  modality_conflict      camera and LiDAR genuinely disagree -> MANDATORY
                         human review (enforced via escalation triggers)
  verification_failed    the check itself could not run -> INDETERMINATE-ward
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from sensorflow.agentic import data as data_mod
from sensorflow.agentic.agents.base import BaseAgent, compact, escalate, no_escalation
from sensorflow.agentic.models import AgentEscalation, FailureEvent
from sensorflow.bevfusion import scenes as scenes_mod
from sensorflow.bevfusion import sensors as sensors_mod

POSITION_AGREEMENT_M = 2.0
CONE_DIMS = (0.4, 0.4, 0.75)  # reference template for the cone hypothesis
VRU = {"pedestrian", "cyclist", "motorcycle"}


def _dims_log_error(dims: List[float], template: Tuple[float, float, float]) -> float:
    return sum(math.log(max(d, 0.05) / t) ** 2 for d, t in zip(dims, template))


class SensorFusionVerificationAgent(BaseAgent):
    name = "sensor_fusion_verification"
    version = "1.0.0"
    epistemic_status = "DERIVED"

    def _analyze(self, failure_id: str, failure: FailureEvent = None,
                 seed: int = data_mod.DEFAULT_SEED,
                 **inputs) -> Tuple[Dict[str, Any], float, str, AgentEscalation]:
        assert failure is not None
        sequences = {s.sequence_id: (qi, s) for qi, s in
                     enumerate(scenes_mod.generate_sequences(
                         n_sequences=data_mod.DEFAULT_N_SEQUENCES,
                         frames_per_sequence=data_mod.DEFAULT_FRAMES,
                         seed=seed))}
        per_instance: List[Dict] = []
        for inst in failure.instances[:12]:  # bounded, deterministic subset
            entry = sequences.get(inst.sequence_id)
            if entry is None:
                per_instance.append({"instance_id": inst.instance_id,
                                     "verdict": "verification_failed",
                                     "reason": "sequence not reproducible"})
                continue
            qi, seq = entry
            frame = next((f for f in seq.frames if f.frame_id == inst.frame_id), None)
            if frame is None:
                per_instance.append({"instance_id": inst.instance_id,
                                     "verdict": "verification_failed",
                                     "reason": "frame not reproducible"})
                continue

            cam = sensors_mod.simulate_camera(frame, seq,
                                              sensors_mod.camera_rng(seed, qi))
            lid = sensors_mod.simulate_lidar(frame, seq,
                                             sensors_mod.lidar_rng(seed, qi))
            cam_det = next((d for d in cam
                            if d.source_instance_id == inst.object_instance_id), None)
            lid_det = next((d for d in lid
                            if d.source_instance_id == inst.object_instance_id), None)

            gt_dims = scenes_mod.CLASS_DIMS.get(inst.gt_class, (0.8, 0.8, 1.8))
            checks: Dict[str, Any] = {}
            camera_supports = lidar_supports = False
            conflict = False

            if lid_det is not None:
                err_gt = _dims_log_error(lid_det.dims, gt_dims)
                err_cone = _dims_log_error(lid_det.dims, CONE_DIMS)
                lidar_supports = err_gt < err_cone
                checks["lidar_shape"] = {
                    "dims": [round(d, 2) for d in lid_det.dims],
                    "log_err_vs_gt_template": round(err_gt, 3),
                    "log_err_vs_cone_template": round(err_cone, 3),
                    "supports_gt_class": lidar_supports,
                }
            if cam_det is not None:
                camera_supports = cam_det.class_name in VRU
                checks["camera_class"] = {
                    "class": cam_det.class_name,
                    "confidence": round(cam_det.confidence, 3),
                    "supports_gt_class": camera_supports,
                }
            if cam_det is not None and lid_det is not None:
                dist = math.hypot(cam_det.x - lid_det.x, cam_det.y - lid_det.y)
                checks["position_agreement_m"] = round(dist, 2)
                if dist > POSITION_AGREEMENT_M or (camera_supports != lidar_supports):
                    conflict = True

            if conflict:
                verdict = "modality_conflict"
            elif cam_det is not None and lid_det is not None:
                verdict = ("multi_modal_supported"
                           if camera_supports and lidar_supports
                           else "modality_conflict")
            elif cam_det is not None or lid_det is not None:
                verdict = "single_modality_only"
            else:
                verdict = "single_modality_only"
                checks["note"] = ("neither re-simulated modality re-detected "
                                  "the object this frame; tracking continuity "
                                  "is the only supporting evidence")
            per_instance.append({"instance_id": inst.instance_id,
                                 "frame_id": inst.frame_id,
                                 "verdict": verdict, "checks": checks})

        counts: Dict[str, int] = {}
        for r in per_instance:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        n = max(len(per_instance), 1)
        if counts.get("verification_failed", 0) == n:
            overall = "verification_failed"
        elif counts.get("modality_conflict", 0) > 0:
            overall = "modality_conflict"
        elif counts.get("multi_modal_supported", 0) / n >= 0.5:
            overall = "multi_modal_supported"
        else:
            overall = "single_modality_only"

        output = {
            "overall_verdict": overall,
            "verdict_counts": counts,
            "instances_verified": len(per_instance),
            "per_instance": per_instance,
            "method": ("deterministic re-simulation of bevfusion camera/LiDAR "
                       "sensor models; LiDAR shape template check vs GT class "
                       "and cone hypothesis; camera class check; BEV position "
                       "agreement"),
        }
        confidence = round(min(0.9, 0.4 + 0.5 * counts.get("multi_modal_supported", 0) / n), 3)
        if overall == "modality_conflict":
            return (output, confidence,
                    "modality conflict detected — verdict requires human eyes",
                    escalate(["camera and LiDAR evidence disagree"],
                             ["modality_disagreement"]))
        if overall == "verification_failed":
            return (output, 0.0, "verification could not run",
                    escalate(["fusion verification failed"], ["agent_failure"]))
        return (output, confidence,
                "share of instances with independent two-modality corroboration",
                no_escalation())

    def _llm_prompt(self, failure_id, output, **inputs):
        return ("Explain this multi-sensor verification result to a triage "
                "engineer in three sentences: " + compact(output))
