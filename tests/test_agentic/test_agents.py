"""Agents: seqeval delegation, small-sample flags, fusion-conflict escalation,
hypothesis labeling, safety UNCERTAIN handling, advisory-only authority."""

from __future__ import annotations

from sensorflow.agentic import data as data_mod
from sensorflow.agentic import review as review_mod
from sensorflow.agentic.agents import (SafetyImpactAgent,
                                       SensorFusionVerificationAgent,
                                       StatisticalRegressionAgent,
                                       VLMSceneAnalysisAgent)
from sensorflow.agentic.agents import statistical as stat_mod
from sensorflow.agentic.models import (DetectionBasis, FailureEvent,
                                       FailureInstance, new_id)


def _instance(**overrides) -> FailureInstance:
    kwargs = dict(
        instance_id=new_id("inst"), sequence_id="bev-seq-101-0",
        frame_id="bev-seq-101-0-f0006", frame_index=6,
        object_instance_id="bev-seq-101-0-obj-1",
        gt_class="pedestrian", predicted_class="construction_cone",
        confidence=0.62, distance_m=22.0, construction_zone=True,
        time_of_day="day", weather="clear", geo_bucket="bay_area_urban",
        occluded=False, has_planner_trace=False)
    kwargs.update(overrides)
    return FailureInstance(**kwargs)


def _failure(instances) -> FailureEvent:
    return FailureEvent(
        failure_id=new_id("fail"), kind="classification_flip",
        title="pedestrian misclassified as construction_cone",
        gt_class="pedestrian", predicted_class="construction_cone",
        detection_basis=DetectionBasis(
            method="test", candidate_events=25, baseline_events=5,
            denominator=240_000, candidate_rate=25 / 240_000,
            baseline_rate=5 / 240_000),
        instances=instances, dataset_fingerprint="fp-test",
        population_id="rate-pop-test")


# ------------------------------------------------------------------ statistical


def test_statistical_agent_delegates_to_seqeval_and_flags_small_samples(monkeypatch):
    constructed = []
    real_cls = stat_mod.PairedSequentialTest

    class SpyTest(real_cls):
        def __init__(self, *a, **kw):
            constructed.append((a, kw))
            super().__init__(*a, **kw)

    monkeypatch.setattr(stat_mod, "PairedSequentialTest", SpyTest)
    agent = StatisticalRegressionAgent()
    assessment = agent.assess(seed=data_mod.DEFAULT_SEED)

    # the anytime-valid machinery really is seqeval's, not a reimplementation
    assert constructed, "PairedSequentialTest (seqeval) was not instantiated"
    assert assessment.seqeval["delegated_to"] == \
        "sensorflow.seqeval.sequential.PairedSequentialTest"
    assert assessment.seqeval["decision"] in (
        "REGRESSION", "PASS", "INSUFFICIENT_EVIDENCE")
    assert assessment.power_mde["method"] == \
        "sensorflow.seqeval.sequential.approx_mde"

    # rare-event handling: exact binomial drives the significance claim
    assert "binomial" in assessment.significance_method
    assert assessment.exact_binomial_p is not None
    assert assessment.significant is True

    # small-sample instability is flagged (few dozen events on 240k denom)
    assert assessment.small_sample_flags
    assert any("event count" in f for f in assessment.small_sample_flags)

    # Wilson CIs present and contain the point estimates
    lo, hi = assessment.candidate.wilson_ci
    assert lo <= assessment.candidate.rate <= hi


def test_statistical_agent_reports_measurements_not_hypotheses():
    result = StatisticalRegressionAgent().run("fail-x", seed=data_mod.DEFAULT_SEED)
    assert result.authority == "ADVISORY_ONLY"
    assert result.epistemic_status == "OBSERVED"
    assert result.status == "ok"


# ------------------------------------------------------------------ fusion


class _FakeDet:
    def __init__(self, inst_id, cls, dims, x, y):
        self.source_instance_id = inst_id
        self.class_name = cls
        self.confidence = 0.9
        self.dims = list(dims)
        self.x, self.y = x, y


def test_fusion_conflict_triggers_mandatory_human_review(ped_cone_failure,
                                                         monkeypatch):
    """Force camera/LiDAR disagreement: camera says 'car' (non-VRU), LiDAR
    shape matches the GT pedestrian template -> modality_conflict, which must
    escalate to mandatory human review."""
    from sensorflow.agentic.agents import fusion_verification as fv

    inst = ped_cone_failure.instances[0]
    gt_dims = fv.scenes_mod.CLASS_DIMS.get(inst.gt_class, (0.8, 0.8, 1.8))

    monkeypatch.setattr(
        fv.sensors_mod, "simulate_camera",
        lambda frame, seq, rng: [_FakeDet(inst.object_instance_id, "car",
                                          gt_dims, 10.0, 5.0)])
    monkeypatch.setattr(
        fv.sensors_mod, "simulate_lidar",
        lambda frame, seq, rng: [_FakeDet(inst.object_instance_id,
                                          inst.gt_class, gt_dims, 10.0, 5.0)])

    result = SensorFusionVerificationAgent().run(
        ped_cone_failure.failure_id, failure=ped_cone_failure)
    assert result.output["overall_verdict"] == "modality_conflict"
    assert result.escalation.required
    assert "modality_disagreement" in result.escalation.human_review_triggers

    # and the governance layer fires the mandatory trigger on the verdict
    triggers = review_mod.mandatory_review_triggers(
        {"severity": "S1", "outcome": "CONTINUE_INVESTIGATION",
         "recommended_option": "EXPAND_EVALUATION"},
        fusion_verdict="modality_conflict", gt_available=True,
        small_sample=False, behavioral_evidence="none")
    fired = {t["trigger"] for t in triggers if t["fired"]}
    assert "modality_disagreement" in fired


def test_fusion_agent_runs_deterministically(ped_cone_failure):
    r1 = SensorFusionVerificationAgent().run(
        ped_cone_failure.failure_id, failure=ped_cone_failure)
    r2 = SensorFusionVerificationAgent().run(
        ped_cone_failure.failure_id, failure=ped_cone_failure)
    assert r1.output["overall_verdict"] == r2.output["overall_verdict"]
    assert r1.output["per_instance"] == r2.output["per_instance"]
    assert r1.output["overall_verdict"] != "verification_failed"


# ------------------------------------------------------------------ VLM + safety


def test_vlm_output_is_always_labeled_hypothesis():
    failure = _failure([_instance(), _instance(time_of_day="night")])
    result = VLMSceneAnalysisAgent().run(failure.failure_id, failure=failure)
    assert result.epistemic_status == "HYPOTHESIS"
    assert "HYPOTHESIS" in result.output["epistemic_label"]
    assert all(h["label"] == "HYPOTHESIS" for h in result.output["hypotheses"])
    # confidence capped: hypotheses can never look like confirmed evidence
    assert result.confidence <= 0.6


def test_safety_agent_uncertain_without_replay_evidence():
    failure = _failure([_instance(has_planner_trace=False)])
    result = SafetyImpactAgent().run(failure.failure_id, failure=failure)
    assert result.output["behavioral_evidence"] == "none"
    assert result.output["assessment"] == \
        "safety assessment UNCERTAIN — no behavioral evidence"
    chain = {c["link"]: c["status"] for c in result.output["chain"]}
    assert chain["planner"] == "UNAVAILABLE"
    assert chain["behavior"] == "UNAVAILABLE"
    assert result.escalation.required  # uncertainty escalates, never passes


def test_safety_agent_uses_replay_evidence_and_ssam_kinematics():
    failure = _failure([_instance(has_planner_trace=True)])
    result = SafetyImpactAgent().run(failure.failure_id, failure=failure)
    assert result.output["behavioral_evidence"] == "observed"
    assert result.output["replayed_instances"] == 1
    kin = result.output["kinematics"]
    assert "ssam_ext" in kin["method"]
    # delayed reaction must cost stopping distance
    assert kin["stopping_distance_delayed_m"] > kin["stopping_distance_nominal_m"]
    chain = {c["link"]: c["status"] for c in result.output["chain"]}
    assert chain["planner"] == "OBSERVED"
    assert chain["behavior"] == "OBSERVED"


def test_agent_failure_is_captured_not_raised():
    result = SafetyImpactAgent().run("fail-x", failure=None)  # assert trips
    assert result.status == "failed"
    assert result.escalation.required
    assert "agent_failure" in result.escalation.human_review_triggers
