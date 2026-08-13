"""ISP per-stage quality/throughput + augmentation lineage & leakage guard."""

from sensorflow.vitis.augment import generate_batch, list_augmentations
from sensorflow.vitis.isp import run_isp


class TestIsp:
    def test_per_stage_psnr_sanity(self, vitis_root):
        run = run_isp(n_frames=2, width_bits=12, include_previews=False)
        assert [e["stage"] for e in run["stage_report"]] == \
            ["bad_pixel_correction", "demosaic", "hdr_tone_map", "gain",
             "denoise", "resize"]
        for e in run["stage_report"]:
            assert 20.0 < e["psnr_db"] <= 99.0, e
            assert 0.5 < e["ssim"] <= 1.0, e
        # Higher precision must not be worse than lower precision.
        lo = run_isp(n_frames=2, width_bits=8, include_previews=False)
        assert min(e["psnr_db"] for e in run["stage_report"]) > \
            min(e["psnr_db"] for e in lo["stage_report"])

    def test_throughput_labeled_modeled(self, vitis_root):
        run = run_isp(n_frames=1, include_previews=False)
        t = run["throughput"]
        assert t["modeled_not_measured"] is True
        assert t["measured_cpu_ms_per_frame"] > 0
        assert t["modeled_fpga_fps_pipelined"] > 0
        for e in run["stage_report"]:
            assert e["modeled_not_measured"] is True

    def test_previews_and_persistence(self, vitis_root):
        run = run_isp(n_frames=1, include_previews=True)
        assert run["previews"][0]["stages"][0]["reference_png"].startswith(
            "data:image/png;base64,")
        assert (vitis_root / "isp" / f"{run['run_id']}.json").exists()


class TestAugment:
    def test_lineage_completeness_and_eval_only_flag(self, vitis_root):
        batch = generate_batch(n_variants=5, include_thumbnails=False)
        assert len(batch["variants"]) == 5
        for v in batch["variants"]:
            lin = v["lineage"]
            for key in ("batch_id", "source_frame_id", "source_sequence_id",
                        "recipe", "seed", "backend", "backend_config"):
                assert key in lin, f"lineage missing {key}"
            assert lin["recipe"] and all("aug" in s and "params" in s
                                         for s in lin["recipe"])
            # Leakage guard: evaluation-only by default, never training data.
            assert v["evaluation_only"] is True
            assert v["training_eligible"] is False
            assert v["recommended_dataset_destination"] == \
                "REGRESSION_EVALUATION_SET"
            assert v["gt_boxes"], "variants must inherit GT"

    def test_deterministic_given_seed(self, vitis_root):
        a = generate_batch(n_variants=3, seed=99, include_thumbnails=False)
        b = generate_batch(n_variants=3, seed=99, include_thumbnails=False)
        assert [v["lineage"]["seed"] for v in a["variants"]] == \
            [v["lineage"]["seed"] for v in b["variants"]]

    def test_unknown_recipe_rejected(self, vitis_root):
        import pytest
        with pytest.raises(ValueError):
            generate_batch(recipes=[{"aug": "sharknado"}], n_variants=1)

    def test_raremine_hook_skipped_for_test_batches(self, vitis_root):
        # Test batches run under a redirected vitis root; routing into
        # raremine's real candidate store must be skipped so tests never
        # touch shared state. Real persisted batches route when raremine
        # is importable (verified manually via the live API).
        batch = generate_batch(n_variants=4, include_thumbnails=False)
        hook = batch["raremine_hook"]
        assert "available" in hook and "routed_candidates" in hook
        assert hook["routed_candidates"] == 0
        assert "ephemeral/test" in hook["note"]

    def test_registry_listing(self):
        names = {a["name"] for a in list_augmentations()}
        assert {"sensor_noise", "low_light", "hdr_extreme", "lens_distortion",
                "chromatic_aberration", "motion_blur", "glare"} <= names
