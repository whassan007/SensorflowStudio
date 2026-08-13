"""Counterfactual engine: provenance, determinism, transformation semantics."""

from __future__ import annotations

import math

import pytest

from sensorflow.nextgen import counterfactual as cf
from sensorflow.nextgen.models import DataLabel, TransformationStep
from sensorflow.nextgen.worldmodel import ExternalWorldModelAdapter


def _recipe(kind, **params):
    return [TransformationStep(kind=kind, params=params)]


def test_provenance_recorded_and_labeled():
    s = cf.generate_counterfactuals(_recipe("environment.day_to_night"),
                                    seed=5, n_scenarios=1)[0]
    assert s.provenance.data_label == DataLabel.COUNTERFACTUAL
    assert s.provenance.source_scene_id.startswith("bev-seq-5")
    assert s.provenance.recipe[0].kind == "environment.day_to_night"
    assert s.provenance.seed != 0
    assert "DeterministicSceneTransformer" in s.provenance.generator


def test_deterministic_same_seed_same_scenario():
    r = _recipe("actors.sudden_brake")
    b1 = cf.load_bundle(cf.generate_counterfactuals(r, seed=7, n_scenarios=1)[0].scenario_id)
    b2 = cf.load_bundle(cf.generate_counterfactuals(r, seed=7, n_scenarios=1)[0].scenario_id)
    f1 = b1["sequence"].frames[-1]
    f2 = b2["sequence"].frames[-1]
    assert [g.bbox_3d for g in f1.gt] == [g.bbox_3d for g in f2.gt]


def test_environment_transform_changes_sequence_condition():
    s = cf.generate_counterfactuals(
        [TransformationStep(kind="environment.day_to_night"),
         TransformationStep(kind="environment.clear_to_rain")],
        seed=7, n_scenarios=1)[0]
    b = cf.load_bundle(s.scenario_id)
    assert b["sequence"].time_of_day == "night"
    assert b["sequence"].weather == "rain"
    assert s.environment["time_of_day"] == "night"


def test_sudden_brake_actually_slows_target():
    s = cf.generate_counterfactuals(_recipe("actors.sudden_brake",
                                            decel_mps2=7.0, t_start_s=1.0),
                                    seed=11, n_scenarios=1,
                                    frames_per_sequence=40)[0]
    b = cf.load_bundle(s.scenario_id)
    note = next(n for n in b["notes"] if n.startswith("sudden_brake"))
    target_id = note.split()[1]
    target = next(a for a in b["actors"] if a.instance_id == target_id)
    v0 = math.hypot(target.states[0]["vx"], target.states[0]["vy"])
    v_end = math.hypot(target.states[-1]["vx"], target.states[-1]["vy"])
    assert v_end < v0 * 0.5


def test_occluded_emergence_adds_occluded_then_visible_pedestrian():
    s = cf.generate_counterfactuals(_recipe("actors.occluded_emergence",
                                            t_emerge_s=1.5),
                                    seed=7, n_scenarios=1,
                                    frames_per_sequence=40)[0]
    b = cf.load_bundle(s.scenario_id)
    ped = next(a for a in b["actors"] if "emergent-ped" in a.instance_id)
    assert ped.states[5]["occluded"] is True
    assert ped.states[-1]["occluded"] is False
    assert ped.states[-1]["y"] < ped.states[0]["y"]  # crossing toward the lane


def test_unknown_transformation_rejected():
    with pytest.raises(ValueError, match="unknown transformation"):
        cf.generate_counterfactuals(_recipe("actors.does_not_exist"),
                                    seed=7, n_scenarios=1)


def test_external_worldmodel_stub_documents_but_does_not_pretend():
    from sensorflow.bevfusion.scenes import generate_sequences
    seq = generate_sequences(n_sequences=1, frames_per_sequence=10, seed=7)[0]
    with pytest.raises(NotImplementedError):
        ExternalWorldModelAdapter().transform(seq, [], seed=0)


def test_catalogue_covers_all_transformations():
    kinds = {c["kind"] for c in cf.transformation_catalogue()}
    assert kinds == set(cf.TRANSFORMATIONS)
