"""API lifecycle across every endpoint group."""


class TestBackendsApi:
    def test_status(self, vitis_client):
        r = vitis_client.get("/api/vitis/backends/status")
        assert r.status_code == 200
        body = r.json()
        assert body["hardware_present"] is False
        names = {b["name"]: b for b in body["backends"]}
        assert names["vitis_emulated"]["emulated"] is True
        assert names["vitis_hw"]["available"] is False

    def test_devices(self, vitis_client):
        r = vitis_client.get("/api/vitis/backends/devices")
        assert {d["name"] for d in r.json()["devices"]} == \
            {"versal-ai-edge", "zynq-ultrascale"}


class TestHilApi:
    def test_run_report_lifecycle(self, vitis_client):
        r = vitis_client.post("/api/vitis/hil/run", json={
            "n_sequences": 1, "frames_per_sequence": 6, "width_bits": 10,
            "run_ablation": True})
        assert r.status_code == 200
        run = r.json()
        assert run["verdict"]["decision"] in (
            "REGRESSION", "PASS", "INSUFFICIENT_EVIDENCE")
        assert run["ablation"]["attribution"]

        rid = run["run_id"]
        assert vitis_client.get(f"/api/vitis/hil/runs/{rid}").status_code == 200
        listing = vitis_client.get("/api/vitis/hil/runs").json()["runs"]
        assert any(x["run_id"] == rid for x in listing)
        assert vitis_client.get("/api/vitis/hil/runs/nope").status_code == 404

    def test_sweep(self, vitis_client):
        r = vitis_client.post("/api/vitis/hil/sweep", json={
            "n_sequences": 1, "frames_per_sequence": 6, "widths": [8, 12]})
        assert r.status_code == 200
        assert len(r.json()["points"]) == 2

    def test_validation(self, vitis_client):
        assert vitis_client.post("/api/vitis/hil/run", json={
            "width_bits": 4, "int_bits": 4}).status_code == 400
        assert vitis_client.post("/api/vitis/hil/run", json={
            "device": "cray-1"}).status_code == 400


class TestIspAugmentApi:
    def test_isp_lifecycle(self, vitis_client):
        r = vitis_client.post("/api/vitis/isp/run", json={
            "n_frames": 1, "include_previews": False})
        assert r.status_code == 200
        run = r.json()
        assert run["throughput"]["modeled_not_measured"] is True
        rid = run["run_id"]
        assert vitis_client.get(f"/api/vitis/isp/runs/{rid}").status_code == 200
        assert vitis_client.post("/api/vitis/isp/run", json={
            "stages": ["defrag"]}).status_code == 400
        assert vitis_client.get("/api/vitis/isp/stages").json()["stages"]

    def test_augment_lifecycle(self, vitis_client):
        r = vitis_client.get("/api/vitis/augment/recipes")
        assert len(r.json()["augmentations"]) == 7
        r = vitis_client.post("/api/vitis/augment/generate", json={
            "n_variants": 2, "include_thumbnails": False})
        assert r.status_code == 200
        batch = r.json()
        assert all(v["evaluation_only"] for v in batch["variants"])
        bid = batch["run_id"]
        assert vitis_client.get(
            f"/api/vitis/augment/batches/{bid}").status_code == 200
        variants = vitis_client.get("/api/vitis/augment/variants").json()
        assert len(variants["variants"]) >= 2
        assert vitis_client.post("/api/vitis/augment/generate", json={
            "backend": "vitis_hw"}).status_code == 400


class TestTemporalApi:
    def test_temporal_lifecycle(self, vitis_client):
        engines = vitis_client.get("/api/vitis/temporal/engines").json()["engines"]
        assert len(engines) == 2
        r = vitis_client.post("/api/vitis/temporal/run", json={
            "n_sequences": 1, "frames_per_sequence": 10})
        assert r.status_code == 200
        run = r.json()
        assert set(run["results"]) == {"reference", "vitis_emulated"}
        assert "ranking_agrees" in run["backend_agreement"]
        rid = run["run_id"]
        assert vitis_client.get(
            f"/api/vitis/temporal/runs/{rid}").status_code == 200
        assert vitis_client.post("/api/vitis/temporal/run", json={
            "engines": ["skynet"]}).status_code == 400


class TestPrdApi:
    def test_prd_listing_and_content(self, vitis_client):
        listing = vitis_client.get("/api/vitis/prd").json()["prds"]
        assert {p["id"] for p in listing} == {
            "vitis-hil-regression", "vitis-isp-preprocessing",
            "vitis-temporal-stability"}
        assert all(p["available"] for p in listing)
        doc = vitis_client.get("/api/vitis/prd/vitis-hil-regression").json()
        assert "Value Proposition" in doc["markdown"]
        assert vitis_client.get("/api/vitis/prd/nope").status_code == 404
