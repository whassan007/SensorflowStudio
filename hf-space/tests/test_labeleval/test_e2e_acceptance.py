"""End-to-end acceptance test through the UI-backed HTTP APIs (spec §49).

Dataset -> precheck -> ingest synthetic sensor data -> generate labels ->
queue -> anomaly detection -> rare events -> regression -> grader comparison ->
strict geometric validation -> precision/recall -> tracking -> consensus ->
triage -> auto-release valid -> flag invalid -> HITL tasks -> correct a label ->
re-validate -> verify -> training dataset -> train -> stream logs -> evaluate
new model -> regression/improvement -> complete flywheel.
"""

import time

import pytest
from fastapi.testclient import TestClient

from sensorflow.evaluation.pipeline import reset_pipeline
from sensorflow.evaluation.records import reset_store


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    store = reset_store(tmp_path_factory.mktemp("e2e"))
    reset_pipeline(store)
    from app_backend import app
    with TestClient(app) as c:
        yield c


def _wait_pipeline(client, timeout=90):
    for _ in range(timeout * 4):
        state = client.get("/api/labeleval/pipeline").json()
        if not state["running"] and state["stage"] in ("complete", "failed", "idle"):
            return state
        time.sleep(0.25)
    raise AssertionError("pipeline did not finish in time")


def test_full_flywheel_through_apis(client):
    # 1. generate synthetic dataset (raw sensor data + reference GT)
    ds = client.post("/api/labeleval/datasets/generate",
                     json={"name": "e2e", "num_sequences": 3, "frames_per_sequence": 20, "seed": 42}).json()
    dataset_id = ds["dataset_id"]
    assert ds["num_frames"] == 60
    assert ds["gt_availability"]["gt_type"] == "VENDOR_GROUND_TRUTH"

    # 2. precheck
    pre = client.post("/api/dataset/precheck", json={"dataset_id": dataset_id}).json()
    assert pre["status"] == "success", pre
    assert all(c["passed"] for c in pre["checks"] if c["applicable"])

    # 3. run the evaluation pipeline (queue -> workers -> triage)
    run = client.post("/api/labeleval/run", json={"dataset_id": dataset_id}).json()
    assert run["status"] == "started"
    state = _wait_pipeline(client)
    assert state["stage"] == "complete", state["stage"]

    # queue drained and services healthy
    q = client.get("/api/queue/status").json()
    assert q["pending"] == 0 and q["completed"] > 0 and q["failed"] == 0
    services = {s["service"]: s for s in state["services"]}
    for name in ("label-generator", "anomaly-detector", "grader", "quality-validator",
                 "rare-event-detector", "triage-engine", "review-service", "metrics-service"):
        assert services[name]["state"] in ("HEALTHY", "DEGRADED"), name

    # 4. overview counters populated
    ov = client.get("/api/labeleval/overview").json()
    c = ov["counters"]
    assert c["auto_labeled"] > 100
    assert c["verified"] > 0 and c["flagged"] > 0
    assert c["rare_events"] > 0
    assert ov["process_units_total"] > 0

    # 5. quality metrics: precision/recall only because reference GT exists
    m = client.get("/api/quality/metrics", params={"dataset_id": dataset_id}).json()
    assert m["gt_available"] and m["gt_type"] == "VENDOR_GROUND_TRUTH"
    g = m["global"]
    assert 0.5 < g["precision"] <= 1.0 and 0.5 < g["recall"] <= 1.0
    assert g["safety_critical_recall"] is not None
    assert g["grader_consensus"] is not None
    assert len(m["per_class"]) >= 4 and len(m["per_scenario"]) >= 3

    # 6. verified vs non-verified groups
    groups = client.get("/api/quality/groups", params={"dataset_id": dataset_id}).json()
    names = {x["name"]: x for x in groups["groups"]}
    assert set(names) == {"verified", "non_verified", "hitl", "rejected"}
    assert groups["total"] == c["auto_labeled"]
    gd = client.get(f"/api/quality/groups/{dataset_id}:hitl").json()
    assert gd["count"] == names["hitl"]["count"]
    assert gd["failure_reason_counts"]

    # 7. haystack renders all categories and points open evidence
    hs = client.get("/api/labeleval/haystack", params={"dataset_id": dataset_id}).json()["points"]
    cats = {p["category"] for p in hs}
    assert {"normal", "anomaly", "rare_event", "false_negative"} <= cats
    ann_point = next(p for p in hs if p["kind"] == "annotation")
    ev = client.get(f"/api/labeleval/evaluations/{ann_point['id']}").json()
    assert ev["anomaly"]["detector_scores"] and ev["decision"] is not None

    # 8. rare events with evidence
    events = client.get("/api/labeleval/rare-events").json()["events"]
    assert events and all(e["evidence_frames"] for e in events)

    # 9. regression endpoint has an entry for this run
    reg = client.get("/api/regression").json()
    assert reg["entries"]

    # 10. HITL: find a position-error task, correct it with the reference box
    tasks = client.get("/api/review/tasks").json()["tasks"]
    assert tasks
    target = None
    for t in tasks:
        if t["status"] != "open":
            continue
        full = client.get(f"/api/review/tasks/{t['task_id']}").json()
        rec = full["evidence"]
        if rec and "WRONG_POSITION" in rec["injected_errors"]:
            target = full
            break
    assert target is not None, "expected an open WRONG_POSITION review task"

    frame = client.get(f"/api/labeleval/frames/{target['frame_id']}").json()["frame"]
    gt = next(g for g in frame["gt_boxes"] if g["gt_id"] == target["evidence"]["ground_truth_id"])
    res = client.post(f"/api/review/tasks/{target['task_id']}",
                      json={"action": "correct",
                            "corrected_bbox_3d": gt["bbox_3d"],
                            "corrected_class": gt["class_name"]}).json()
    assert res["task"]["status"] == "resolved"
    assert res["task"]["resolution"]["revalidation_passed"] is True, res["message"]
    assert res["task"]["resolution"]["final_status"] == "VERIFIED"
    # Corrected label was re-validated against a human-verified reference.
    reval = res["revalidation"]
    assert reval["ground_truth_type"] == "HUMAN_VERIFIED_GROUND_TRUTH"
    assert reval["geometry"]["iou_3d"] >= 0.99
    assert reval["decision"]["status"] == "AUTO_GRADED"

    # funnel now shows the HITL side branch flowing back to verified
    funnel = client.get("/api/labeleval/funnel").json()
    side = {s["stage"]: s["count"] for s in funnel["side_path"]}
    assert side["RE-LABELING"] >= 1 and side["VERIFIED (post-HITL)"] >= 1

    # 11. training on the verified pool; logs must stream (grow over time)
    tr = client.post("/api/train", json={"dataset_version": dataset_id,
                                         "training_parameters": {"epochs": 3}}).json()
    job_id = tr["job_id"]
    assert tr["model_version"].startswith("model-v")

    seen_lens = []
    status = None
    for _ in range(120):
        status = client.get(f"/api/train/jobs/{job_id}").json()
        seen_lens.append(len(status["logs"]))
        if status["status"] == "completed":
            break
        time.sleep(0.3)
    assert status["status"] == "completed", status
    assert status["epoch"] == 3
    assert status["rare_recall"] > 0 and status["safety_recall"] > 0
    assert status["process_units"] > 0
    assert max(seen_lens) > min(seen_lens), "training logs should stream incrementally"
    assert status["lineage"]["parent_dataset"] == dataset_id
    assert status["lineage"]["validated_by_policy"]

    # 12. new model evaluated and registered with regression status
    models = client.get("/api/models").json()["models"]
    trained = [mm for mm in models if mm["model_id"] == tr["model_id"]]
    assert trained and trained[0]["metrics"]["f1"] is not None
    assert trained[0]["regression_status"] in ("baseline", "improved", "regressed")

    # 13. process units accounted at every stage with unit economics
    pu = client.get("/api/labeleval/process-units").json()
    for stage in ("ingestion", "label_generation", "anomaly_detection", "validation",
                  "grading", "rare_event_detection", "hitl", "training"):
        assert pu["by_stage"].get(stage, 0) > 0, stage
    assert pu["unit_economics"]["per_verified_event"] is not None

    # 14. audit trail captured the flywheel
    audit = client.get("/api/labeleval/audit").json()["events"]
    actions = {e["action"] for e in audit}
    assert {"dataset_generated", "labels_generated", "pipeline_completed",
            "review_correct", "training_started", "training_completed"} <= actions


def test_benchmark_and_copilot_through_apis(client):
    bench = client.post("/api/benchmark/techniques", json={}).json()
    assert len(bench["rows"]) >= 10
    techniques = {r["technique"] for r in bench["rows"]}
    assert {"knn", "lof", "isolation_forest", "ocsvm", "dbscan",
            "autoencoder", "vae", "gan", "few_shot"} <= techniques
    assert {"best_rare_recall", "best_safety_recall", "lowest_fp_rate",
            "lowest_process_units", "lowest_tracking_error"} <= set(bench["highlights"])

    # Copilot on a flagged annotation: advisory, graceful without Ollama.
    tasks = client.get("/api/review/tasks").json()["tasks"]
    open_tasks = [t for t in tasks if t["status"] == "open"]
    assert open_tasks
    cp = client.post("/api/copilot/explain",
                     json={"context_type": "anomaly",
                           "annotation_id": open_tasks[0]["annotation_id"]}).json()
    assert cp["status"] == "ok"
    assert cp["analysis"]
    assert cp["structured"]["hypothesis"].startswith("HYPOTHESIS")
    assert cp["structured"]["observed_evidence"]
    assert 0 <= cp["structured"]["confidence"] <= 1


def test_sse_stream_emits_state(client):
    # Bounded tick count: TestClient buffers the response, so the stream must
    # terminate; real clients consume the long-lived stream incrementally.
    res = client.get("/api/events/stream", params={"ticks": 2})
    assert res.headers["content-type"].startswith("text/event-stream")
    lines = [l for l in res.text.splitlines() if l.startswith("data: ")]
    assert len(lines) == 2
    import json
    payload = json.loads(lines[0][6:])
    assert "pipeline" in payload and "training" in payload and "ts" in payload
