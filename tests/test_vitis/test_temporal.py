"""Temporal profiling: flow recovery, engine ranking, stereo geometry."""

import numpy as np

from sensorflow.bevfusion.scenes import generate_sequences
from sensorflow.vitis.backend import get_backend
from sensorflow.vitis.render import (
    BASELINE_M, FOCAL_PX, render_stereo_pair,
)
from sensorflow.vitis.temporal import (
    _aggregate_engine, _engine_track_metrics, _match_engine_to_gt,
    compute_flow_baseline, run_temporal_profile,
)


class TestOpticalFlowRecovery:
    def test_recovers_planted_constant_velocity(self):
        """A textured square moving at constant (dx, dy) px/frame must be
        recovered by the pyramidal LK flow within tolerance."""
        rng = np.random.default_rng(2)
        h, w = 96, 128
        tex = rng.uniform(0.4, 1.0, (18, 18)).astype(np.float32)
        dx, dy = 3.0, 1.0

        def frame(t):
            img = np.full((h, w), 0.05, dtype=np.float32)
            y0, x0 = int(30 + dy * t), int(20 + dx * t)
            img[y0:y0 + 18, x0:x0 + 18] = tex
            return img

        for name in ("reference", "vitis_emulated"):
            be = get_backend(name)
            flow = be.optical_flow(frame(0), frame(1), levels=3, window=9,
                                   iterations=3)
            patch = flow[32:46, 22:36]
            assert abs(float(np.median(patch[..., 0])) - dx) < 0.5, name
            assert abs(float(np.median(patch[..., 1])) - dy) < 0.5, name

    def test_flow_baseline_marks_tracks_continuous(self):
        seqs = generate_sequences(1, 10, seed=7)
        be = get_backend("reference")
        records = compute_flow_baseline(seqs[0], be, seed=7)
        continuous = [r["continuous"] for per in records.values()
                      for r in per.values()]
        assert sum(continuous) > 0.5 * len(continuous)


class TestEngineRanking:
    def test_unstable_engine_ranks_below_stable(self, vitis_root):
        """The naive camera-only engine (frame-to-frame NN tracker, misses
        under occlusion/night) must score below the fused engine."""
        run = run_temporal_profile(n_sequences=2, frames_per_sequence=12,
                                   seed=7)
        for be_name in ("reference", "vitis_emulated"):
            eng = run["results"][be_name]["engines"]
            cam = eng["perception-v1-camera"]
            fused = eng["perception-v3-bevfusion"]
            assert fused["stability_score"] > cam["stability_score"], be_name
            assert cam["flicker_rate"] >= fused["flicker_rate"], be_name
            assert cam["fragmentation_per_track"] > \
                fused["fragmentation_per_track"], be_name

    def test_synthetic_flicker_engine_penalized(self):
        """Deliberately dropping every third detection of a stable engine
        must raise flicker and lower the aggregate score."""
        seqs = generate_sequences(1, 14, seed=9)
        seq = seqs[0]
        be = get_backend("reference")
        flow_records = compute_flow_baseline(seq, be, seed=9)

        stable_out = {f.frame_id: [{"bbox_3d": g.bbox_3d,
                                    "class_name": g.class_name,
                                    "confidence": 0.9,
                                    "track_id": g.instance_id}
                                   for g in f.gt]
                      for f in seq.frames}
        flicker_out = {fid: (dets if i % 3 else dets[:max(0, len(dets) - 3)])
                       for i, (fid, dets) in enumerate(stable_out.items())}

        m_stable = _aggregate_engine([_engine_track_metrics(
            seq, flow_records, _match_engine_to_gt(seq, stable_out))])
        m_flicker = _aggregate_engine([_engine_track_metrics(
            seq, flow_records, _match_engine_to_gt(seq, flicker_out))])
        assert m_flicker["flicker_rate"] > m_stable["flicker_rate"]
        assert m_flicker["stability_score"] < m_stable["stability_score"]

    def test_backend_agreement_metacheck(self, vitis_root):
        run = run_temporal_profile(n_sequences=2, frames_per_sequence=12,
                                   seed=7, width_bits=12)
        meta = run["backend_agreement"]
        assert meta["ranking_agrees"] is True
        assert meta["max_abs_score_delta"] < 5.0

    def test_cohort_breakdown_present(self, vitis_root):
        run = run_temporal_profile(n_sequences=3, frames_per_sequence=12,
                                   seed=7)
        cohorts = run["results"]["reference"]["engines"][
            "perception-v3-bevfusion"]["cohorts"]
        assert "night/clear" in cohorts and "occluded" in cohorts


class TestStereoConsistency:
    def test_disparity_depth_within_tolerance_of_geometry(self):
        seqs = generate_sequences(1, 6, seed=7)
        seq = seqs[0]
        be = get_backend("reference")
        rel_errors = []
        for frame in seq.frames[:3]:
            left, right, objs = render_stereo_pair(frame, seq, seed=7)
            disp = be.stereo_block_match(left, right, max_disparity=48, block=9)
            for o in objs:
                u, v = int(round(o["u"])), int(round(o["v"]))
                if not (4 <= v < disp.shape[0] - 4 and 4 <= u < disp.shape[1] - 4):
                    continue
                d = float(np.median(disp[v - 2:v + 3, u - 2:u + 3]))
                if d < 1.0 or o["depth_m"] > 30.0:
                    continue  # near field is the geometry-limited regime
                depth = FOCAL_PX * BASELINE_M / d
                rel_errors.append(abs(depth - o["depth_m"]) / o["depth_m"])
        # Median over objects: individual matches can be corrupted by
        # object overlap, but the population must track scene geometry.
        assert len(rel_errors) >= 3
        assert float(np.median(rel_errors)) < 0.15, rel_errors

    def test_stereo_report_in_profile(self, vitis_root):
        run = run_temporal_profile(n_sequences=1, frames_per_sequence=10,
                                   seed=7)
        stereo = run["results"]["reference"]["stereo"]
        assert stereo["objects_checked"] > 0
        assert stereo["median_abs_disparity_error_px"] < 2.0
