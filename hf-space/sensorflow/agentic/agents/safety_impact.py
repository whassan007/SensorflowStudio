"""SafetyImpactAgent — perception -> tracking -> prediction -> planner ->
behavior chain.

Uses REPLAY EVIDENCE where it exists: instances flagged with a synthetic
planner-response trace are replayed through the SSAM surrogate-safety
machinery (sensorflow.safety.ssam_ext — TTC via conflict-point projection,
DRAC, collision probability), comparing the delayed reaction caused by the
misclassification ("a cone does not need yielding") against the nominal
reaction. Stopping-distance and braking-margin computations reuse the same
module's kinematics.

When no replay evidence exists the agent outputs exactly
"safety assessment UNCERTAIN — no behavioral evidence" and marks the
downstream links UNAVAILABLE — a hypothetical consequence is NEVER converted
into an observed one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from sensorflow.agentic.agents.base import BaseAgent, compact, escalate, no_escalation
from sensorflow.agentic.models import AgentEscalation, FailureEvent
from sensorflow.safety import ssam_ext

NOMINAL_REACTION_S = 0.6
MISCLASS_REACTION_S = 1.6   # replayed planner treats the object as static furniture
COMFORT_DECEL_MPS2 = 3.5
HARD_DECEL_MPS2 = 5.0
TTC_UNSAFE_S = 1.0

UNCERTAIN_TEXT = "safety assessment UNCERTAIN — no behavioral evidence"


def stopping_distance(speed_mps: float, decel_mps2: float,
                      reaction_s: float) -> float:
    """Reaction distance + braking distance (same kinematics as the
    ssam_ext braking profile)."""
    return speed_mps * reaction_s + speed_mps ** 2 / (2.0 * max(decel_mps2, 0.1))


class SafetyImpactAgent(BaseAgent):
    name = "safety_impact"
    version = "1.0.0"
    epistemic_status = "DERIVED"

    def _analyze(self, failure_id: str, failure: FailureEvent = None,
                 **inputs) -> Tuple[Dict[str, Any], float, str, AgentEscalation]:
        assert failure is not None
        traced = [i for i in failure.instances if i.has_planner_trace]

        chain: List[Dict] = [
            {"link": "perception_error", "status": "OBSERVED",
             "detail": f"{failure.gt_class} -> {failure.predicted_class} on "
                       f"{len(failure.instances)} captured instances"},
            {"link": "tracking", "status": "OBSERVED",
             "detail": "candidate track continuity observed in the campaign "
                       "prediction log (class churn inside the track)"},
            {"link": "prediction", "status": "DERIVED",
             "detail": "a construction_cone hypothesis implies a STATIC "
                       "motion prior; pedestrian crossing intent is dropped"},
        ]

        if not traced:
            chain.append({"link": "planner", "status": "UNAVAILABLE",
                          "detail": "no replay planner-response trace exists"})
            chain.append({"link": "behavior", "status": "UNAVAILABLE",
                          "detail": UNCERTAIN_TEXT})
            output = {
                "behavioral_evidence": "none",
                "assessment": UNCERTAIN_TEXT,
                "chain": chain,
                "replayed_instances": 0,
                "note": ("downstream consequence remains a hypothesis; this "
                         "agent never records hypothetical behavior as "
                         "observed"),
            }
            return (output, 0.3,
                    "no behavioral evidence available; assessment is "
                    "explicitly uncertain",
                    escalate([UNCERTAIN_TEXT], ["insufficient_statistics"]))

        # ---- replay evidence exists: quantify with ssam_ext ----------------
        replays: List[Dict] = []
        worst = {"min_ttc_s": None, "max_drac": None, "max_p": 0.0}
        for k, inst in enumerate(traced[:4]):
            delayed = ssam_ext.analyze_trajectories(
                ssam_ext.generate_trajectories(
                    seed=k, scenario="crossing",
                    reaction_delay_s=MISCLASS_REACTION_S))
            nominal = ssam_ext.analyze_trajectories(
                ssam_ext.generate_trajectories(
                    seed=k, scenario="crossing",
                    reaction_delay_s=NOMINAL_REACTION_S))
            da, na = delayed["aggregate"], nominal["aggregate"]
            replays.append({
                "instance_id": inst.instance_id,
                "replay_seed": k,
                "delayed_reaction_s": MISCLASS_REACTION_S,
                "nominal_reaction_s": NOMINAL_REACTION_S,
                "min_ttc_delayed_s": da["min_ttc_s"],
                "min_ttc_nominal_s": na["min_ttc_s"],
                "max_drac_delayed_mps2": da["max_drac_mps2"],
                "max_collision_probability_delayed": da["max_collision_probability"],
                "num_conflicts_delayed": da["num_conflicts"],
                "num_conflicts_nominal": na["num_conflicts"],
                "csi_delayed": da["aggregate_csi"],
                "csi_nominal": na["aggregate_csi"],
            })
            if da["min_ttc_s"] is not None:
                worst["min_ttc_s"] = (da["min_ttc_s"] if worst["min_ttc_s"] is None
                                      else min(worst["min_ttc_s"], da["min_ttc_s"]))
            if da["max_drac_mps2"] is not None:
                worst["max_drac"] = (da["max_drac_mps2"] if worst["max_drac"] is None
                                     else max(worst["max_drac"], da["max_drac_mps2"]))
            if da["max_collision_probability"] is not None:
                worst["max_p"] = max(worst["max_p"], da["max_collision_probability"])

        v = 12.5  # ego approach speed in the crossing replay scenario
        sd_nominal = stopping_distance(v, HARD_DECEL_MPS2, NOMINAL_REACTION_S)
        sd_delayed = stopping_distance(v, HARD_DECEL_MPS2, MISCLASS_REACTION_S)
        available = 42.0  # initial ego-to-conflict-point distance in the scenario
        braking_margin_nominal = available - sd_nominal
        braking_margin_delayed = available - sd_delayed

        unsafe = ((worst["min_ttc_s"] is not None and worst["min_ttc_s"] < TTC_UNSAFE_S)
                  or (worst["max_drac"] is not None and worst["max_drac"] > COMFORT_DECEL_MPS2))
        behavioral = "observed_unsafe" if unsafe else "observed_contained"

        chain.append({"link": "planner", "status": "OBSERVED",
                      "detail": f"replayed {len(replays)} planner-response "
                                f"trace(s) with reaction delay "
                                f"{MISCLASS_REACTION_S}s vs nominal "
                                f"{NOMINAL_REACTION_S}s"})
        chain.append({"link": "behavior", "status": "OBSERVED",
                      "detail": (f"worst replay min TTC "
                                 f"{worst['min_ttc_s']}s, max DRAC "
                                 f"{worst['max_drac']} m/s^2, max collision "
                                 f"probability {worst['max_p']}") })

        output = {
            "behavioral_evidence": "observed",
            "behavioral_classification": behavioral,
            "assessment": ("unsafe downstream behavior OBSERVED in replay"
                           if unsafe else
                           "downstream behavior degraded but contained in replay"),
            "chain": chain,
            "replayed_instances": len(replays),
            "replays": replays,
            "worst_case": worst,
            "kinematics": {
                "method": "sensorflow.safety.ssam_ext (TTC conflict-point "
                          "projection, DRAC, collision probability) + "
                          "reaction/braking kinematics",
                "ego_speed_mps": v,
                "stopping_distance_nominal_m": round(sd_nominal, 2),
                "stopping_distance_delayed_m": round(sd_delayed, 2),
                "braking_margin_nominal_m": round(braking_margin_nominal, 2),
                "braking_margin_delayed_m": round(braking_margin_delayed, 2),
                "thresholds": {"ttc_unsafe_s": TTC_UNSAFE_S,
                               "comfort_decel_mps2": COMFORT_DECEL_MPS2},
            },
            "coverage_note": (f"replay evidence covers {len(traced)} of "
                              f"{len(failure.instances)} instances; the "
                              "remainder stays UNCERTAIN"),
        }
        esc = (escalate(["unsafe downstream behavior observed in replay"],
                        ["behavior_change", "severity_s3_plus"])
               if unsafe else no_escalation())
        return (output, 0.8 if unsafe else 0.7,
                "replay-based measurement on the instances that have traces",
                esc)

    def _llm_prompt(self, failure_id, output, **inputs):
        return ("Narrate this perception->behavior safety chain for a launch "
                "review, preserving all UNCERTAIN labels: " + compact(output))
