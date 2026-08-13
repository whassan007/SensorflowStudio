"""Backend abstraction: parity, quantization monotonicity, XFCVDEPTH,
LUT approximations, latency model, registry."""

import numpy as np
import pytest

from sensorflow.vitis.backend import (
    BACKENDS, DeviceConfig, PipelineConfig, backend_status, get_backend,
    make_bayer_mosaic,
)


@pytest.fixture(scope="module")
def img():
    rng = np.random.default_rng(3)
    base = rng.random((96, 128, 3)).astype(np.float32)
    return base


@pytest.fixture(scope="module")
def gray(img):
    return img.mean(axis=2).astype(np.float32)


def _cfg(w, i=6, **kw):
    return PipelineConfig(precision={"default": (w, i)}, **kw)


class TestParityAndMonotonicity:
    def test_high_bitwidth_parity_within_quantization_bound(self, img):
        ref = get_backend("reference")
        vit = get_backend("vitis_emulated", _cfg(16))
        r = ref.gaussian_filter(img, 1.5)
        v = vit.gaussian_filter(img, 1.5)
        # ap_fixed<16,6> has 10 fractional bits -> lsb ~ 1e-3; allow a few lsb
        # of accumulated error through the filter.
        assert float(np.abs(r - v).mean()) < 4 * 2.0 ** -10

    def test_divergence_grows_monotonically_as_bits_shrink(self, img):
        ref = get_backend("reference")
        target = ref.gaussian_filter(img, 1.5)
        errs = []
        for w in (16, 12, 10, 8):
            vit = get_backend("vitis_emulated", _cfg(w))
            errs.append(float(np.abs(vit.gaussian_filter(img, 1.5) - target).mean()))
        assert errs == sorted(errs), f"not monotone: {errs}"
        assert errs[-1] > 10 * errs[0]

    def test_ops_run_on_both_backends(self, img, gray):
        raw = make_bayer_mosaic(img)
        for name in ("reference", "vitis_emulated"):
            be = get_backend(name, _cfg(16))
            assert be.demosaic(raw).shape == img.shape
            assert be.resize(img, 48, 64).shape == (48, 64, 3)
            assert be.resize(gray, 48, 64, "area").shape == (48, 64)
            assert be.crop(img, 10, 10, 20, 30).shape == (20, 30, 3)
            yuv = be.rgb_to_yuv(img)
            back = be.yuv_to_rgb(yuv)
            assert float(np.abs(back - img).mean()) < 0.05
            assert be.bad_pixel_correction(gray).shape == gray.shape
            assert be.hdr_tone_map(img).max() <= 1.0
            assert be.gain_exposure(img, 2.0).max() <= 1.0
            assert be.lens_distortion(gray, k1=0.2).shape == gray.shape
            assert be.median_filter(gray, 3).shape == gray.shape
            merged = be.hdr_merge([img * 0.3, img], [0.5, 1.0])
            assert merged.shape == img.shape


class TestStreamingDepthConstraint:
    def test_no_artifacts_when_depth_sufficient(self, gray):
        wide = get_backend("vitis_emulated", _cfg(16, max_line_buffer_depth=4096))
        exact = get_backend("vitis_emulated",
                            _cfg(16, max_line_buffer_depth=gray.shape[1]))
        a = wide.gaussian_filter(gray, 2.0)
        b = exact.gaussian_filter(gray, 2.0)
        assert np.array_equal(a, b)

    def test_localized_boundary_artifacts_only_when_exceeded(self, gray):
        depth = 48
        full = get_backend("vitis_emulated", _cfg(16, max_line_buffer_depth=4096))
        narrow = get_backend("vitis_emulated",
                             _cfg(16, max_line_buffer_depth=depth))
        a = full.gaussian_filter(gray, 2.0)
        b = narrow.gaussian_filter(gray, 2.0)
        col_err = np.abs(a - b).mean(axis=0)
        radius = 8  # scipy gaussian kernel truncates at 4*sigma = 8 px
        seam_cols = [c for s in range(depth, gray.shape[1], depth)
                     for c in range(max(0, s - radius),
                                    min(gray.shape[1], s + radius))]
        interior = [c for c in range(gray.shape[1]) if c not in set(seam_cols)]
        assert col_err[seam_cols].max() > 1e-3, "expected seam artifacts"
        assert col_err[interior].max() < 1e-6, "artifacts leaked beyond seams"


class TestHlsApproximations:
    def test_lut_recip_bounded_relative_error(self):
        cfg = _cfg(24, i=8, lut_bits=8)
        from sensorflow.vitis.backend import _FixedNumerics
        num = _FixedNumerics(cfg)
        x = np.linspace(0.01, 50.0, 5000).astype(np.float32)
        rel = np.abs(num.recip(x) - 1.0 / x) * x
        assert float(rel.max()) < 2.0 ** -7  # ~2x lsb of a 256-entry LUT

    def test_lut_sqrt_bounded_relative_error(self):
        cfg = _cfg(24, i=8, lut_bits=8)
        from sensorflow.vitis.backend import _FixedNumerics
        num = _FixedNumerics(cfg)
        x = np.linspace(0.01, 50.0, 5000).astype(np.float32)
        rel = np.abs(num.sqrt(x) - np.sqrt(x)) / np.sqrt(x)
        assert float(rel.max()) < 2.0 ** -8

    def test_lut_toggle_changes_output(self, img):
        on = get_backend("vitis_emulated", _cfg(20, i=6, use_lut_approx=True))
        off = get_backend("vitis_emulated", _cfg(20, i=6, use_lut_approx=False))
        a = on.hdr_tone_map(img)
        b = off.hdr_tone_map(img)
        assert not np.array_equal(a, b)


class TestLatencyModelAndRegistry:
    def test_modeled_latency_deterministic_and_labeled(self, gray):
        be = get_backend("vitis_emulated", _cfg(12))
        be.gaussian_filter(gray, 1.0)
        rep = be.profile_report()[-1]
        assert rep["modeled"]["modeled_not_measured"] is True
        assert rep["modeled"]["latency_ms"] > 0
        be2 = get_backend("vitis_emulated", _cfg(12))
        be2.gaussian_filter(gray, 1.0)
        assert be2.profile_report()[-1]["modeled"]["latency_ms"] == \
            rep["modeled"]["latency_ms"]

    def test_aie_placement_depends_on_device(self, gray):
        versal = get_backend("vitis_emulated", PipelineConfig(
            device=DeviceConfig(name="versal-ai-edge")))
        zynq = get_backend("vitis_emulated", PipelineConfig(
            device=DeviceConfig(name="zynq-ultrascale")))
        versal.gaussian_filter(gray, 1.0)
        zynq.gaussian_filter(gray, 1.0)
        assert versal.profile_report()[-1]["modeled"]["placement"] == "AIE"
        assert zynq.profile_report()[-1]["modeled"]["placement"] == "PL"

    def test_reference_backend_reports_no_modeled_numbers(self, gray):
        ref = get_backend("reference")
        ref.gaussian_filter(gray, 1.0)
        assert "modeled" not in ref.profile_report()[-1]

    def test_registry_and_hw_stub(self):
        assert set(BACKENDS) == {"reference", "vitis_emulated", "vitis_hw"}
        with pytest.raises(NotImplementedError):
            get_backend("vitis_hw")
        with pytest.raises(ValueError):
            get_backend("nonsense")
        status = backend_status()
        assert status["hardware_present"] is False
        emu = next(b for b in status["backends"] if b["name"] == "vitis_emulated")
        assert emu["emulated"] is True and "modeled" in emu["description"]
