"""FailureDetectionAgent — layer 1.

Scans candidate-vs-ground-truth predictions on the synthetic populations
(bevfusion scene campaign + vectorized rate population) for:

  * classification flips (pedestrian->construction_cone, pedestrian->vehicle,
    bicycle->vehicle, ..., anything->background),
  * localization regressions (mean center error delta),
  * detection regressions (per-class recall delta),
  * confidence-calibration regressions (mean confidence on correct
    predictions),
  * temporal regressions (per-track prediction churn).

Every emitted FailureEvent carries its deterministic detection basis (counts,
denominators, metric deltas) — the agent flags, it does not judge.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from sensorflow.agentic import data as data_mod
from sensorflow.agentic.agents.base import BaseAgent, compact, no_escalation
from sensorflow.agentic.models import (AgentEscalation, DetectionBasis,
                                       FailureEvent, FailureInstance, new_id)

SAFETY_CRITICAL = {"pedestrian", "cyclist", "motorcycle"}

LOCALIZATION_DELTA_THRESHOLD_M = 0.02
CONFIDENCE_DELTA_THRESHOLD = 0.03
CHURN_DELTA_THRESHOLD = 0.02


class FailureDetectionAgent(BaseAgent):
    name = "failure_detection"
    version = "1.0.0"
    epistemic_status = "DERIVED"

    def detect(self, seed: int = data_mod.DEFAULT_SEED) -> Tuple[List[FailureEvent], Dict]:
        """Full campaign scan. Returns (events, scan_summary)."""
        campaign = data_mod.get_campaign(seed)
        rates = data_mod.get_rate_arrays(seed)
        events: List[FailureEvent] = []

        # ---- classification flips (candidate wrong, baseline right) ------
        flips: Dict[Tuple[str, str], List] = defaultdict(list)
        for obs in campaign.observations:
            if (obs.candidate.predicted_class != obs.gt_class
                    and obs.baseline.predicted_class == obs.gt_class):
                flips[(obs.gt_class, obs.candidate.predicted_class)].append(obs)

        for (gt_cls, pred_cls), members in sorted(flips.items()):
            safety = gt_cls in SAFETY_CRITICAL
            if not safety and len(members) < 3:
                continue  # noise floor for non-safety patterns
            if gt_cls == "pedestrian" and pred_cls == "construction_cone":
                basis = DetectionBasis(
                    method="paired candidate-vs-GT scan + rate-population "
                           "telemetry denominator",
                    candidate_events=int(rates.candidate_flip.sum()),
                    baseline_events=int(rates.baseline_flip.sum()),
                    denominator=rates.n,
                    candidate_rate=float(rates.candidate_flip.mean()),
                    baseline_rate=float(rates.baseline_flip.mean()),
                    notes=[f"{len(members)} concrete scene instances captured "
                           "in the campaign for snippet/replay evidence",
                           "rate telemetry provides the population denominator"])
            else:
                n_cls = sum(1 for o in campaign.observations if o.gt_class == gt_cls)
                b_ev = sum(1 for o in campaign.observations
                           if o.gt_class == gt_cls
                           and o.baseline.predicted_class == pred_cls)
                basis = DetectionBasis(
                    method="paired candidate-vs-GT scan (campaign only)",
                    candidate_events=len(members), baseline_events=b_ev,
                    denominator=n_cls,
                    candidate_rate=len(members) / max(n_cls, 1),
                    baseline_rate=b_ev / max(n_cls, 1))

            instances = [FailureInstance(
                instance_id=new_id("inst"),
                sequence_id=o.sequence_id, frame_id=o.frame_id,
                frame_index=o.frame_index,
                object_instance_id=o.object_instance_id,
                gt_class=o.gt_class,
                predicted_class=o.candidate.predicted_class,
                confidence=o.candidate.confidence,
                distance_m=o.distance_m,
                construction_zone=o.construction_zone,
                time_of_day=o.time_of_day, weather=o.weather,
                geo_bucket=o.geo_bucket, occluded=o.occluded,
                has_planner_trace=o.has_planner_trace,
            ) for o in members]

            events.append(FailureEvent(
                failure_id=new_id("fail"),
                kind="classification_flip",
                title=f"{gt_cls} misclassified as {pred_cls}",
                description=(f"Candidate {campaign.candidate_model} predicts "
                             f"{pred_cls} where GT and baseline say {gt_cls}."),
                gt_class=gt_cls, predicted_class=pred_cls,
                detection_basis=basis,
                instances=instances,
                baseline_model=campaign.baseline_model,
                candidate_model=campaign.candidate_model,
                dataset_fingerprint=data_mod.campaign_fingerprint(campaign),
                population_id=f"rate-pop-{seed}-n{rates.n}",
            ))

        # ---- metric regressions ------------------------------------------
        summary = self._metric_scan(campaign)
        for kind, title, delta, threshold, extra in summary["checks"]:
            if abs(delta) <= threshold:
                continue
            events.append(FailureEvent(
                failure_id=new_id("fail"),
                kind=kind, title=title,
                description=f"metric delta {delta:+.4f} exceeded threshold "
                            f"{threshold} on the paired campaign scan",
                detection_basis=DetectionBasis(
                    method="paired campaign metric scan",
                    candidate_events=0, baseline_events=0,
                    denominator=len(campaign.observations),
                    candidate_rate=0.0, baseline_rate=0.0,
                    metric_deltas={title: round(delta, 5), **extra}),
                baseline_model=campaign.baseline_model,
                candidate_model=campaign.candidate_model,
                dataset_fingerprint=data_mod.campaign_fingerprint(campaign),
                population_id=f"campaign-{seed}",
            ))
        return events, summary

    def _metric_scan(self, campaign: data_mod.Campaign) -> Dict:
        obs = campaign.observations

        def center_err(pred, gt):
            return float(np.hypot(pred[0] - gt[0], pred[1] - gt[1]))

        cand_loc = float(np.mean([center_err(o.candidate.bbox_3d, o.gt_bbox_3d)
                                  for o in obs]))
        base_loc = float(np.mean([center_err(o.baseline.bbox_3d, o.gt_bbox_3d)
                                  for o in obs]))

        cand_conf = float(np.mean([o.candidate.confidence for o in obs
                                   if o.candidate.predicted_class == o.gt_class]))
        base_conf = float(np.mean([o.baseline.confidence for o in obs
                                   if o.baseline.predicted_class == o.gt_class]))

        cand_acc = float(np.mean([o.candidate.predicted_class == o.gt_class
                                  for o in obs]))
        base_acc = float(np.mean([o.baseline.predicted_class == o.gt_class
                                  for o in obs]))

        def churn(model: str) -> float:
            by_track: Dict[str, List[str]] = defaultdict(list)
            for o in sorted(obs, key=lambda x: x.frame_index):
                pred = o.candidate if model == "candidate" else o.baseline
                by_track[f"{o.sequence_id}:{o.object_instance_id}"].append(
                    pred.predicted_class)
            switches = total = 0
            for classes in by_track.values():
                for a, b in zip(classes, classes[1:]):
                    total += 1
                    switches += a != b
            return switches / max(total, 1)

        cand_churn, base_churn = churn("candidate"), churn("baseline")

        return {
            "observations": len(obs),
            "candidate_accuracy": round(cand_acc, 5),
            "baseline_accuracy": round(base_acc, 5),
            "checks": [
                ("localization_regression", "mean_center_error_delta_m",
                 cand_loc - base_loc, LOCALIZATION_DELTA_THRESHOLD_M,
                 {"candidate_m": round(cand_loc, 4), "baseline_m": round(base_loc, 4)}),
                ("confidence_calibration_regression",
                 "mean_correct_confidence_delta",
                 cand_conf - base_conf, CONFIDENCE_DELTA_THRESHOLD,
                 {"candidate": round(cand_conf, 4), "baseline": round(base_conf, 4)}),
                ("detection_regression", "classification_accuracy_delta",
                 base_acc - cand_acc, 0.01,
                 {"candidate": round(cand_acc, 4), "baseline": round(base_acc, 4)}),
                ("temporal_regression", "track_class_churn_delta",
                 cand_churn - base_churn, CHURN_DELTA_THRESHOLD,
                 {"candidate": round(cand_churn, 4), "baseline": round(base_churn, 4)}),
            ],
        }

    # BaseAgent plumbing (detect() is the natural API; _analyze wraps it).
    def _analyze(self, failure_id: str, **inputs):
        events, summary = self.detect(inputs.get("seed", data_mod.DEFAULT_SEED))
        output = {
            "events_detected": len(events),
            "titles": [e.title for e in events],
            "scan_summary": {k: v for k, v in summary.items() if k != "checks"},
        }
        conf = 0.9 if events else 0.6
        return (output, conf,
                "deterministic paired scan; confidence reflects scan coverage",
                no_escalation())

    def _llm_prompt(self, failure_id, output, **inputs):
        return ("Summarize this perception failure detection scan for an "
                "engineering audience: " + compact(output))
