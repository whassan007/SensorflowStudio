"""HIL feature: ablation attribution, sequential/fallback verdicts, sweep."""

import numpy as np

from sensorflow.vitis.hil import (
    _fallback_verdict, run_bitwidth_sweep, run_hil, sequential_verdict,
)


class TestAblationAttribution:
    def test_planted_precision_regression_attributed_to_precision(self, vitis_root):
        # W=8 with generous depth and LUTs on: precision is the only real
        # constraint that bites at this image size.
        run = run_hil(n_sequences=2, frames_per_sequence=8, seed=7,
                      width_bits=8, int_bits=4,
                      max_line_buffer_depth=2048, use_lut_approx=True)
        attr = run["ablation"]["attribution"]
        assert attr["precision_only"] > 0.8
        assert attr["precision_only"] > attr["streaming_only"]
        assert attr["precision_only"] > attr["hls_approx_only"]

    def test_high_precision_run_has_negligible_gap(self, vitis_root):
        run = run_hil(n_sequences=2, frames_per_sequence=8, seed=7,
                      width_bits=16, int_bits=4, run_ablation=False)
        assert run["comparison"]["gap_score"] < 0.01
        assert run["comparison"]["totals"]["dropped_by_vitis"] == 0


class TestSequentialVerdicts:
    def test_fallback_regression_on_planted_regression(self):
        v = _fallback_verdict([-0.25 + 0.01 * (i % 3) for i in range(40)],
                              delta=0.02, alpha=0.05)
        assert v["decision"] == "REGRESSION"

    def test_fallback_pass_on_no_regression(self):
        v = _fallback_verdict([0.002 - 0.001 * (i % 2) for i in range(40)],
                              delta=0.02, alpha=0.05)
        assert v["decision"] == "PASS"

    def test_fallback_insufficient_on_tiny_sample(self):
        assert _fallback_verdict([0.0], 0.02, 0.05)["decision"] == \
            "INSUFFICIENT_EVIDENCE"

    def test_sequential_verdict_detects_planted_regression(self):
        # Strong planted regression: vitis path loses ~half its objects.
        rng = np.random.default_rng(5)
        deltas = list(np.clip(rng.normal(-0.5, 0.05, 120), -1, 1))
        pairs = [(True, bool(rng.random() > 0.5)) for _ in range(600)]
        v = sequential_verdict(deltas, pairs, delta=0.02, alpha=0.05)
        assert v["decision"] == "REGRESSION"
        assert v["method"] in ("seqeval_anytime_valid", "paired_t_fallback")

    def test_sequential_verdict_no_false_regression(self):
        deltas = [0.0] * 120
        pairs = [(True, True)] * 600
        v = sequential_verdict(deltas, pairs, delta=0.02, alpha=0.05)
        assert v["decision"] != "REGRESSION"


class TestSweep:
    def test_sweep_monotone_gap_and_minimal_config(self, vitis_root):
        sw = run_bitwidth_sweep(n_sequences=2, frames_per_sequence=8, seed=7,
                                widths=[6, 8, 12, 16])
        pts = sw["points"]  # sorted ascending by width
        gaps = [p["gap_score"] for p in pts]
        assert gaps == sorted(gaps, reverse=True), f"gap not monotone: {gaps}"
        assert pts[0]["decision"] == "REGRESSION"  # W=6 is a planted disaster
        mp = sw["minimal_passing_config"]
        assert mp is not None and mp["width_bits"] > 6

    def test_run_persisted(self, vitis_root):
        run = run_hil(n_sequences=1, frames_per_sequence=6, width_bits=12,
                      run_ablation=False)
        assert (vitis_root / "hil" / f"{run['run_id']}.json").exists()
