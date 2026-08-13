"""Regression tests pinning the surgical fixes in existing packages.

Each test names the audit finding it pins. If any of these breaks, the
original bug has been reintroduced.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ F-003 yaw penalty


class TestYawPenaltyFix:
    def _tracker_with_track(self, yaw: float, vx: float):
        pytest.importorskip("scipy")
        from sensorflow.temporal_tracker import KalmanTrack, TemporalTracker

        tracker = TemporalTracker(distance_gate=3.0, velocity_penalty=2.0)
        track = KalmanTrack(
            track_id=1,
            state=np.array([0.0, 0.0, vx, 0.0]),
            covariance=np.eye(4),
        )
        track.history.append({"frame_id": "f0",
                              "bbox_3d": [0, 0, 0, 4, 2, 1.5, yaw],
                              "confidence": 0.9})
        tracker.tracks[1] = track
        return tracker

    def test_fast_track_with_matching_yaw_associates(self):
        """The old bug computed the 'yaw penalty' as abs(vx): a track moving
        at 5 m/s got penalty 2*5=10 > gate and could never associate. With
        the fix, matching yaw means zero penalty regardless of speed."""
        from sensorflow.schemas.unified_frame import Object3D

        tracker = self._tracker_with_track(yaw=1.0, vx=5.0)
        prop = Object3D(bbox_3d=[0.5, 0.0, 0.0, 4, 2, 1.5, 1.0], class_name="car",
                        confidence=0.9)
        assignments = tracker._associate([prop])
        assert assignments == [(1, 0)]

    def test_yaw_difference_is_wrapped(self):
        """pi-0.05 vs -pi+0.05 is a 0.1 rad disagreement, not ~2*pi."""
        from sensorflow.schemas.unified_frame import Object3D

        tracker = self._tracker_with_track(yaw=math.pi - 0.05, vx=0.0)
        prop = Object3D(bbox_3d=[0.5, 0.0, 0.0, 4, 2, 1.5, -math.pi + 0.05],
                        class_name="car", confidence=0.9)
        assignments = tracker._associate([prop])
        assert assignments == [(1, 0)]  # unwrapped diff would cost ~12 > gate

    def test_opposite_yaw_still_penalized(self):
        from sensorflow.schemas.unified_frame import Object3D

        tracker = self._tracker_with_track(yaw=0.0, vx=0.0)
        prop = Object3D(bbox_3d=[0.5, 0.0, 0.0, 4, 2, 1.5, math.pi],
                        class_name="car", confidence=0.9)
        assert tracker._associate([prop]) == []  # 2*pi penalty > gate


# ------------------------------------------------------------------ F-008 seeded fallback


class TestSeededLidarFallback:
    def test_missing_file_fallback_is_deterministic(self):
        from sensorflow.perception_automator import PerceptionAutomator

        auto = PerceptionAutomator(use_sam=False)
        a = auto._load_lidar("does/not/exist.bin")
        b = auto._load_lidar("does/not/exist.bin")
        np.testing.assert_array_equal(a, b)

    def test_different_paths_different_clouds(self):
        from sensorflow.perception_automator import PerceptionAutomator

        auto = PerceptionAutomator(use_sam=False)
        a = auto._load_lidar("path/one.bin")
        b = auto._load_lidar("path/two.bin")
        assert not np.array_equal(a, b)


# ------------------------------------------------------------------ F-006 cache key


class TestMegaevalCacheKey:
    def test_run_id_differentiates_keys(self):
        from sensorflow.megaeval.cube import QueryCache

        common = dict(dataset_version="pop-1", model_version="m-1",
                      filters={}, group_by=[], metrics=["precision"])
        k_base = QueryCache.key(**common, run_id="eval-baseline")
        k_inj = QueryCache.key(**common, run_id="eval-injected")
        assert k_base != k_inj
        assert QueryCache.key(**common, run_id="eval-baseline") == k_base

    def test_router_does_not_serve_other_runs_rows(self):
        from sensorflow.megaeval.cube import STAT_COLS, QueryRouter
        from sensorflow.megaeval.population import DIM_NAMES

        class Run:
            def __init__(self, run_id):
                self.run_id = run_id
                self.population_id = "pop-1"
                self.model_version = "m-1"

        def cube_with(tp):
            row = {d: 0 for d in DIM_NAMES}
            row.update({c: 0 for c in STAT_COLS})
            row.update({"n": 100, "tp": tp, "fp": 100 - tp})
            return pd.DataFrame([row])

        router = QueryRouter()
        res_a = router.query(Run("eval-a"), cube_with(90), {}, None, ["precision"], None)
        res_b = router.query(Run("eval-b"), cube_with(10), {}, None, ["precision"], None)
        # Same population+model, different run: MUST NOT be a cache hit.
        assert res_b["meta"]["cache_hit"] is False
        assert res_a["rows"][0]["precision"] != res_b["rows"][0]["precision"]


# ------------------------------------------------------------------ F-009 per-frame gate


class TestQualityGatePerFrameMatching:
    def _sequence(self, gt_frame: str):
        from sensorflow.schemas.unified_frame import (
            FusedFrame, GroundTruthObject, UnifiedSequence)

        frames = []
        for fid in ("f1", "f2"):
            gts = []
            if fid == gt_frame:
                gts = [GroundTruthObject(bbox_3d=[0, 0, 0, 4, 2, 1.5, 0],
                                         class_name="car", confidence=1.0,
                                         instance_id="i1")]
            frames.append(FusedFrame(frame_id=fid, ground_truth=gts))
        return UnifiedSequence(sequence_id="s", frames=frames)

    def _tracks(self, pred_frame: str):
        return [{"track_id": 1, "class_name": "car", "frames": [
            {"frame_id": pred_frame, "bbox_3d": [0, 0, 0, 4, 2, 1.5, 0],
             "confidence": 0.9}]}]

    def test_cross_frame_boxes_no_longer_match(self):
        """Identical box, but prediction in f1 and GT in f2: physically
        impossible match. The old pooled matching scored map_3d=1.0 here."""
        from sensorflow.quality_gate import QualityGate

        result = QualityGate().evaluate(self._sequence("f2"), self._tracks("f1"))
        assert result["metric_card"]["map_3d"] == 0.0
        assert result["metric_card"]["mar_3d"] == 0.0

    def test_same_frame_boxes_still_match(self):
        from sensorflow.quality_gate import QualityGate

        result = QualityGate().evaluate(self._sequence("f1"), self._tracks("f1"))
        assert result["metric_card"]["map_3d"] == 1.0
        assert result["metric_card"]["position_error_m"] == 0.0


# ------------------------------------------------------------------ F-010 CI regression


class TestRegressionCI:
    def _store(self, tmp_path):
        from sensorflow.evaluation.records import EvalStore
        return EvalStore(base_dir=tmp_path)

    def _compare(self, store, sample_sizes=None):
        from sensorflow.evaluation.regression import compare_runs
        return compare_runs(
            store,
            current_metrics={"precision": 0.85},
            baseline_metrics={"precision": 0.90},
            model_version="m2", dataset_version="d1", run_id="r1",
            baseline_version="m1", sample_sizes=sample_sizes)

    def test_small_n_noise_not_flagged(self, tmp_path):
        """delta=-0.05 on n=50: the CI spans the tolerance, so no regression
        is declared (the legacy point rule would have flagged it)."""
        result = self._compare(self._store(tmp_path),
                               sample_sizes={"precision": (50, 50)})
        assert result.regression_detected is False

    def test_large_n_real_regression_flagged(self, tmp_path):
        result = self._compare(self._store(tmp_path),
                               sample_sizes={"precision": (20000, 20000)})
        assert result.regression_detected is True

    def test_legacy_behavior_preserved_without_sample_sizes(self, tmp_path):
        result = self._compare(self._store(tmp_path))
        assert result.regression_detected is True  # point rule unchanged

    def test_regression_output_completeness(self, tmp_path):
        from sensorflow.evaluation.regression import METRIC_SPECS, compare_runs
        current = {m: 0.5 for m in METRIC_SPECS}
        baseline = {m: 0.5 for m in METRIC_SPECS}
        result = compare_runs(self._store(tmp_path), current,
                              model_version="m2", dataset_version="d1",
                              run_id="r1", baseline_metrics=baseline,
                              baseline_version="m1")
        assert len(result.deltas) == len(METRIC_SPECS)
        for d in result.deltas:
            assert d.metric in METRIC_SPECS
            for field in ("baseline", "current", "delta", "tolerance", "regressed"):
                assert getattr(d, field) is not None

    def test_wilson_interval_sane(self):
        from sensorflow.evaluation.regression import wilson_interval
        lo, hi = wilson_interval(0.9, 100)
        assert 0 <= lo < 0.9 < hi <= 1
        lo2, hi2 = wilson_interval(0.9, 10000)
        assert (hi2 - lo2) < (hi - lo)  # tighter with more data
        assert wilson_interval(1.0, 10)[1] <= 1.0  # never degenerate


# ------------------------------------------------------------------ F-019 ID-swap gate


class TestIdSwapDistanceGate:
    def test_distant_track_cannot_manufacture_swap(self):
        from sensorflow.metrics.temporal_mot import compute_id_swap_rate

        gt = [{"instance_id": "i1", "frames": [
            {"frame_id": "f1", "bbox_3d": [0, 0, 0, 4, 2, 1.5, 0]},
            {"frame_id": "f2", "bbox_3d": [0, 0, 0, 4, 2, 1.5, 0]},
        ]}]
        pred = [
            {"track_id": 1, "frames": [
                {"frame_id": "f1", "bbox_3d": [0.2, 0, 0, 4, 2, 1.5, 0]}]},
            # 100 m away: without the gate this "matched" at f2 -> phantom swap.
            {"track_id": 2, "frames": [
                {"frame_id": "f1", "bbox_3d": [100, 100, 0, 4, 2, 1.5, 0]},
                {"frame_id": "f2", "bbox_3d": [100, 100, 0, 4, 2, 1.5, 0]}]},
        ]
        assert compute_id_swap_rate(pred, gt) == 0.0

    def test_real_swap_still_detected(self):
        from sensorflow.metrics.temporal_mot import compute_id_swap_rate

        gt = [{"instance_id": "i1", "frames": [
            {"frame_id": "f1", "bbox_3d": [0, 0, 0, 4, 2, 1.5, 0]},
            {"frame_id": "f2", "bbox_3d": [0, 0, 0, 4, 2, 1.5, 0]},
        ]}]
        pred = [
            {"track_id": 1, "frames": [
                {"frame_id": "f1", "bbox_3d": [0.2, 0, 0, 4, 2, 1.5, 0]}]},
            {"track_id": 2, "frames": [
                {"frame_id": "f2", "bbox_3d": [0.2, 0, 0, 4, 2, 1.5, 0]}]},
        ]
        assert compute_id_swap_rate(pred, gt) == 0.5


# ------------------------------------------------------------------ F-021 ensemble


class TestEnsembleFailureExclusion:
    def test_failed_detector_excluded_and_recorded(self):
        from sensorflow.evaluation.detectors import AnomalyEnsemble

        engine = AnomalyEnsemble(seed=7)

        class ExplodingDetector:
            name = "exploding"

            def fit(self, X):
                raise RuntimeError("synthetic failure")

            def score(self, X):
                return np.zeros(len(X))

        engine.detectors = [engine.detectors[0], ExplodingDetector()]
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (60, 4))
        ensemble, raw, norm = engine.run(X)
        assert "exploding" not in raw and "exploding" not in norm
        assert "exploding" in engine.last_failures
        assert "RuntimeError" in engine.last_failures["exploding"]
        assert len(ensemble) == 60
        assert not np.allclose(ensemble, 0)  # surviving detector still speaks


# ------------------------------------------------------------------ F-023 endpoints


class TestLlmEndpointConfig:
    def test_no_hardcoded_tailnet_host_in_source(self):
        src = (REPO_ROOT / "sensorflow" / "mitl_copilot.py").read_text()
        assert "ts.net" not in src
        assert "dgx-spark" not in src

    def test_env_override_prepends_primary(self, monkeypatch):
        from sensorflow.mitl_copilot import _llm_endpoints

        monkeypatch.setenv("SENSORFLOW_LLM_URL", "http://gpu-box:11434/api/chat")
        monkeypatch.setenv("SENSORFLOW_LLM_MODEL", "mymodel:7b")
        eps = _llm_endpoints()
        assert eps[0] == {"url": "http://gpu-box:11434/api/chat", "model": "mymodel:7b"}
        assert eps[-1]["url"].startswith("http://localhost")

    def test_default_is_localhost_only(self, monkeypatch):
        from sensorflow.mitl_copilot import _llm_endpoints

        monkeypatch.delenv("SENSORFLOW_LLM_URL", raising=False)
        eps = _llm_endpoints()
        assert len(eps) == 1
        assert eps[0]["url"].startswith("http://localhost")


# ------------------------------------------------------------------ F-014 main.py markers


class TestMainSimulationMarkers:
    def test_analyze_response_labeled_simulated(self):
        from fastapi.testclient import TestClient

        import main as main_module

        client = TestClient(main_module.app)
        csv = "a,b\n1,2\n3,4\n"
        resp = client.post("/api/analyze",
                           files={"file": ("data.csv", csv, "text/csv")})
        assert resp.status_code == 200
        body = resp.json()
        assert body["simulated"] is True
        assert body["analysis_provenance"] == "MOCK_ENGINE"
        # Real elapsed seconds, not the old row-count-as-seconds bug.
        assert 0 <= body["processing_time_seconds"] < 30
        assert body["processing_time_seconds"] != body["records_processed"]

        insights = client.get("/api/insights").json()
        assert insights["simulated"] is True
