"""Per-violation causal-layer attribution.

THE invariant of this module: a downstream failure NEVER auto-implies an
upstream cause. Every causal layer gets an INDEPENDENT evidence test that
returns SUPPORTED, RULED_OUT, or UNKNOWN; a layer is implicated only by its
own positive evidence, never by another layer's silence. Missing evidence
is UNKNOWN — an honest triage outcome, not a defect.

The planning test is the keystone of perception/planning separation: the
plan is re-judged by the SAME rule engine, but against the stack's OWN
world view (mapped context, detections it actually made, intents it
actually predicted, in its believed localization frame). A compliant plan
rules planning out; a non-compliant plan given the stack's own information
implicates planning regardless of what perception did.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from sensorflow.rotr import SOFTWARE_VERSION
from sensorflow.rotr.models import (
    CAUSAL_LAYERS, Actor, FailureAttribution, LayerEvidence, Provenance,
    ROTRScenario, ROTRViolation,
)
from sensorflow.rotr.rules import plan_violations

ATTRIBUTION_VERSION = "rotr-attr-1.0.0"

# ILLUSTRATIVE evidence thresholds (synthetic substrate).
DETECTION_RATE_RULED_OUT = 0.8
DETECTION_RATE_SUPPORTED = 0.5
POSITION_ERROR_SUPPORTED_M = 1.0
LOCALIZATION_ERROR_SUPPORTED_M = 1.0
CONTROL_POS_DEV_SUPPORTED_M = 1.5
CONTROL_SPEED_DEV_SUPPORTED_MPS = 2.0


def _material_actors(scenario: ROTRScenario,
                     violation: ROTRViolation) -> List[Actor]:
    ids = set(violation.actor_ids)
    return [a for a in scenario.actors if a.actor_id in ids]


# ------------------------------------------------------------ layer tests


def _test_perception(scenario: ROTRScenario,
                     violation: ROTRViolation) -> LayerEvidence:
    actors = _material_actors(scenario, violation)
    if not actors:
        return LayerEvidence(
            layer="perception", status="RULED_OUT", confidence=0.7,
            evidence="rule trigger is ego/map-geometric; no actor whose "
                     "perception could have caused it")
    # Judge detection quality in the window that mattered: from the actor
    # becoming relevant (within 60 m) up to the violation instant.
    worst: Optional[LayerEvidence] = None
    for actor in actors:
        n_frames = n_detected = n_absent = 0
        pos_err_sum = 0.0
        for i, s in enumerate(actor.states):
            e = scenario.ego[min(i, len(scenario.ego) - 1)]
            if s.t > violation.t_end + 0.5:
                break
            if abs(s.x - e.x) > 60 or abs(s.y - e.y) > 30:
                continue
            n_frames += 1
            if s.observation is None:
                n_absent += 1
            elif s.observation.detected:
                n_detected += 1
                if s.observation.x is not None:
                    pos_err_sum += abs(s.observation.x - s.x) + \
                        abs((s.observation.y or s.y) - s.y)
        if n_frames == 0:
            continue
        if n_absent / n_frames > 0.5:
            cand = LayerEvidence(
                layer="perception", status="UNKNOWN", confidence=0.0,
                evidence=f"{actor.actor_id}: perception telemetry missing for "
                         f"{n_absent}/{n_frames} relevant frames")
        else:
            assessable = n_frames - n_absent
            rate = n_detected / assessable if assessable else 0.0
            mean_err = pos_err_sum / n_detected if n_detected else 0.0
            if rate < DETECTION_RATE_SUPPORTED:
                cand = LayerEvidence(
                    layer="perception", status="SUPPORTED", confidence=0.9,
                    evidence=f"{actor.actor_id}: missed detection — GT visible "
                             f"{assessable} frames, detected {n_detected} "
                             f"(rate {rate:.2f})")
            elif mean_err > POSITION_ERROR_SUPPORTED_M:
                cand = LayerEvidence(
                    layer="perception", status="SUPPORTED", confidence=0.75,
                    evidence=f"{actor.actor_id}: detected but mislocalized "
                             f"(mean |err| {mean_err:.2f} m)")
            elif rate >= DETECTION_RATE_RULED_OUT:
                cand = LayerEvidence(
                    layer="perception", status="RULED_OUT", confidence=0.9,
                    evidence=f"{actor.actor_id}: detected {n_detected}/"
                             f"{assessable} relevant frames (rate {rate:.2f}, "
                             f"mean |err| {mean_err:.2f} m) — GT-vs-detected "
                             f"diff is clean")
            else:
                cand = LayerEvidence(
                    layer="perception", status="SUPPORTED", confidence=0.6,
                    evidence=f"{actor.actor_id}: degraded detection "
                             f"(rate {rate:.2f})")
        worst = _worse(worst, cand)
    return worst or LayerEvidence(layer="perception", status="UNKNOWN",
                                  evidence="no relevant frames to assess")


def _worse(a: Optional[LayerEvidence], b: LayerEvidence) -> LayerEvidence:
    """Prefer SUPPORTED > UNKNOWN > RULED_OUT when merging actor evidence."""
    if a is None:
        return b
    order = {"SUPPORTED": 2, "UNKNOWN": 1, "RULED_OUT": 0}
    if order[b.status] > order[a.status]:
        return b
    if order[b.status] == order[a.status] and b.confidence > a.confidence:
        return b
    return a


def _test_prediction(scenario: ROTRScenario,
                     violation: ROTRViolation) -> LayerEvidence:
    actors = _material_actors(scenario, violation)
    if not actors:
        return LayerEvidence(layer="prediction", status="RULED_OUT",
                             confidence=0.7,
                             evidence="no actor whose intent needed predicting")
    worst: Optional[LayerEvidence] = None
    for actor in actors:
        if actor.predicted_intent is None:
            cand = LayerEvidence(layer="prediction", status="UNKNOWN",
                                 evidence=f"{actor.actor_id}: no recorded "
                                          "intent prediction")
        elif actor.predicted_intent != actor.intent:
            cand = LayerEvidence(
                layer="prediction", status="SUPPORTED", confidence=0.85,
                evidence=f"{actor.actor_id}: GT intent {actor.intent} but "
                         f"stack predicted {actor.predicted_intent}")
        else:
            cand = LayerEvidence(
                layer="prediction", status="RULED_OUT", confidence=0.85,
                evidence=f"{actor.actor_id}: predicted intent matches GT "
                         f"({actor.intent})")
        worst = _worse(worst, cand)
    return worst


def _test_planning(scenario: ROTRScenario,
                   violation: ROTRViolation) -> LayerEvidence:
    if not scenario.planned:
        return LayerEvidence(layer="planning", status="UNKNOWN",
                             evidence="no planned trajectory recorded")
    hits = plan_violations(scenario)
    same_rule = [h for h in hits if h["rule_id"] == violation.rule_id]
    if same_rule:
        return LayerEvidence(
            layer="planning", status="SUPPORTED", confidence=0.85,
            evidence=f"plan violates {violation.rule_id} given the stack's "
                     f"OWN world view (detections + predicted intents + "
                     f"believed frame): {same_rule[0]['evidence']}")
    if hits:
        return LayerEvidence(
            layer="planning", status="SUPPORTED", confidence=0.6,
            evidence="plan violates a different rule given the stack's own "
                     f"view: {hits[0]['rule_id']}")
    return LayerEvidence(
        layer="planning", status="RULED_OUT", confidence=0.85,
        evidence="plan is rule-compliant given the stack's own world view — "
                 "the plan was reasonable for what the stack knew")


def _test_localization(scenario: ROTRScenario,
                       violation: ROTRViolation) -> LayerEvidence:
    errs = []
    for e in scenario.ego:
        if e.believed_x is None or e.believed_y is None:
            return LayerEvidence(layer="localization", status="UNKNOWN",
                                 evidence="no believed-pose telemetry")
        errs.append(abs(e.believed_x - e.x) + abs(e.believed_y - e.y))
    max_err = max(errs) if errs else 0.0
    lane_mismatch = any(e.believed_lane_id != e.lane_id for e in scenario.ego
                        if e.lane_id and e.believed_lane_id)
    if max_err > LOCALIZATION_ERROR_SUPPORTED_M or lane_mismatch:
        return LayerEvidence(
            layer="localization", status="SUPPORTED", confidence=0.9,
            evidence=f"believed-vs-true pose error up to {max_err:.2f} m"
                     + ("; believed lane differs from true lane"
                        if lane_mismatch else ""))
    return LayerEvidence(layer="localization", status="RULED_OUT",
                         confidence=0.9,
                         evidence=f"pose error <= {max_err:.2f} m; lane "
                                  "association consistent")


def _test_map(scenario: ROTRScenario,
              violation: ROTRViolation) -> LayerEvidence:
    m, a = scenario.map_context, scenario.actual_context
    diffs = []
    if m.control != a.control:
        diffs.append(f"control mapped={m.control!r} actual={a.control!r}")
    if m.signal_state_for_ego != a.signal_state_for_ego:
        diffs.append(f"signal mapped={m.signal_state_for_ego!r} "
                     f"actual={a.signal_state_for_ego!r}")
    a_lanes = {l.lane_id: l for l in a.lanes}
    for lane in m.lanes:
        al = a_lanes.get(lane.lane_id)
        if al is None:
            diffs.append(f"lane {lane.lane_id} not in actual geometry")
            continue
        if lane.restricted_to != al.restricted_to:
            diffs.append(f"lane {lane.lane_id} restriction mapped="
                         f"{lane.restricted_to!r} actual={al.restricted_to!r}")
        if set(lane.permitted_maneuvers) != set(al.permitted_maneuvers):
            diffs.append(f"lane {lane.lane_id} permitted maneuvers differ")
    if diffs:
        return LayerEvidence(layer="map", status="SUPPORTED", confidence=0.9,
                             evidence="map-vs-actual mismatch: " + "; ".join(diffs))
    return LayerEvidence(layer="map", status="RULED_OUT", confidence=0.9,
                         evidence="mapped context matches as-built geometry")


def _test_control(scenario: ROTRScenario,
                  violation: ROTRViolation) -> LayerEvidence:
    if not scenario.planned:
        return LayerEvidence(layer="control", status="UNKNOWN",
                             evidence="no planned trajectory to compare "
                                      "execution against")
    n = min(len(scenario.planned), len(scenario.ego))
    max_pos = max_spd = 0.0
    for k in range(n):
        p, e = scenario.planned[k], scenario.ego[k]
        max_pos = max(max_pos, abs(p.x - e.x) + abs(p.y - e.y))
        max_spd = max(max_spd, abs(p.v - e.speed))
    if max_pos > CONTROL_POS_DEV_SUPPORTED_M or \
            max_spd > CONTROL_SPEED_DEV_SUPPORTED_MPS:
        return LayerEvidence(
            layer="control", status="SUPPORTED", confidence=0.85,
            evidence=f"executed trajectory deviates from plan: max position "
                     f"dev {max_pos:.2f} m, max speed dev {max_spd:.2f} m/s")
    return LayerEvidence(
        layer="control", status="RULED_OUT", confidence=0.85,
        evidence=f"execution tracks the plan (max pos dev {max_pos:.2f} m, "
                 f"max speed dev {max_spd:.2f} m/s)")


def _test_data_label(scenario: ROTRScenario,
                     violation: ROTRViolation) -> LayerEvidence:
    problems = []
    ts = [e.t for e in scenario.ego]
    if any(t2 <= t1 for t1, t2 in zip(ts, ts[1:])):
        problems.append("ego timestamps not strictly increasing")
    for a in scenario.actors:
        if len(a.states) != len(scenario.ego):
            problems.append(f"{a.actor_id}: state array length mismatch")
    if problems:
        return LayerEvidence(layer="data_label", status="SUPPORTED",
                             confidence=0.95,
                             evidence="record integrity failed: "
                                      + "; ".join(problems))
    return LayerEvidence(layer="data_label", status="RULED_OUT",
                         confidence=0.9,
                         evidence="record integrity checks pass "
                                  "(monotone timestamps, consistent arrays)")


def _test_policy_rule(others: Dict[str, LayerEvidence]) -> LayerEvidence:
    statuses = {le.status for le in others.values()}
    if statuses == {"RULED_OUT"}:
        return LayerEvidence(
            layer="policy_rule", status="SUPPORTED", confidence=0.4,
            evidence="every stack layer independently RULED_OUT — the rule "
                     "itself needs HITL review (possible rule misfire)")
    if "SUPPORTED" in statuses:
        return LayerEvidence(layer="policy_rule", status="RULED_OUT",
                             confidence=0.7,
                             evidence="a stack layer has positive evidence; "
                                      "no indication the rule misfired")
    return LayerEvidence(layer="policy_rule", status="UNKNOWN",
                         evidence="stack evidence incomplete; rule quality "
                                  "cannot be judged")


# ------------------------------------------------------------ entry point


def attribute(scenario: ROTRScenario,
              violation: ROTRViolation) -> FailureAttribution:
    layers: Dict[str, LayerEvidence] = {
        "perception": _test_perception(scenario, violation),
        "prediction": _test_prediction(scenario, violation),
        "planning": _test_planning(scenario, violation),
        "localization": _test_localization(scenario, violation),
        "map": _test_map(scenario, violation),
        "control": _test_control(scenario, violation),
        "data_label": _test_data_label(scenario, violation),
    }
    layers["policy_rule"] = _test_policy_rule(layers)

    supported = [(name, le) for name, le in layers.items()
                 if le.status == "SUPPORTED"]
    primary = None
    if supported:
        supported.sort(key=lambda kv: (-kv[1].confidence,
                                       CAUSAL_LAYERS.index(kv[0])))
        primary = supported[0][0]
    note = ("no layer has positive evidence — triaged to HITL, NOT "
            "defaulted to perception" if primary is None else
            f"primary={primary} by highest-confidence positive evidence; "
            f"{len(supported)} layer(s) SUPPORTED")

    return FailureAttribution(
        violation_id=violation.violation_id,
        scenario_id=scenario.scenario_id,
        layers=layers, primary_layer=primary, note=note,
        provenance=Provenance(
            scenario_id=scenario.scenario_id,
            dataset_version=scenario.provenance.dataset_version,
            model_version=scenario.provenance.model_version,
            software_version=f"{SOFTWARE_VERSION}/{ATTRIBUTION_VERSION}",
            source=scenario.provenance.source))
