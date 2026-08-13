"""Pipeline orchestrator: logical processing services over a job queue.

Services (spec §5): label-generator, anomaly-detector, regression-tracker,
grader, quality-validator, rare-event-detector, scenario-miner, triage-engine,
review-service, metrics-service — modular workers inside one app, surfaced to
the UI as logical services with live status.

Flow: Ingestion -> Label Generation -> Queue -> Evaluation workers ->
Validation workers -> Grading workers -> Triage -> (Auto-grade | HITL).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from sensorflow.evaluation import synthetic
from sensorflow.evaluation.detectors import AnomalyEnsemble
from sensorflow.evaluation.graders import dataset_grader_statistics, grade_annotation
from sensorflow.evaluation.process_units import ProcessMeter
from sensorflow.evaluation.queue import EventQueue, make_queue
from sensorflow.evaluation.rare_events import detect_rare_events
from sensorflow.evaluation.records import (
    Alert,
    Annotation,
    AnomalyDetection,
    DatasetLineage,
    EvalStore,
    Model,
    ModelEvaluation,
    RelabelingAction,
    ReviewResolution,
    ReviewTask,
    HumanReview,
    TrackingEvidence,
    TrainingDataset,
    TrainingJob,
    get_store,
    new_id,
)
from sensorflow.evaluation.regression import compare_runs, find_affected_classes
from sensorflow.evaluation.triage import QualityPolicy, load_policy, triage_annotation
from sensorflow.evaluation.validation import validate_annotation
from sensorflow.evaluation import reporting

SERVICES = [
    "label-generator",
    "anomaly-detector",
    "regression-tracker",
    "grader",
    "quality-validator",
    "rare-event-detector",
    "scenario-miner",
    "triage-engine",
    "review-service",
    "metrics-service",
]

DEFAULT_ANOMALY_CONFIG: Dict = {
    "imbalance": {"method": "class_weights", "minority_boost": 2.0},
    "detectors": {
        "knn": {"enabled": True, "k": 10},
        "lof": {"enabled": True, "n_neighbors": 20},
        "isolation_forest": {"enabled": True, "n_estimators": 100},
        "ocsvm": {"enabled": True, "nu": 0.05},
        "dbscan": {"enabled": True, "eps": 1.2, "min_samples": 5},
    },
    "deep": {
        "autoencoder": {"enabled": True, "latent_dim": 4, "epochs": 60},
        "vae": {"enabled": True, "latent_dim": 5},
        "gan": {"enabled": False},
        "reconstruction_threshold": 0.9,
    },
    "advanced": {
        "few_shot": {"enabled": True, "support_per_class": 20},
        "ensemble_strategy": "weighted_average",
        "decision_threshold": 0.9,
    },
}

BATCH_SIZE = 64
BATCH_PAUSE_S = 0.015  # keeps the live queue/SSE view meaningful without slowing tests much


class LabelEvalPipeline:
    """Singleton orchestrator holding queue, services and live run state."""

    def __init__(self, store: Optional[EvalStore] = None, queue: Optional[EventQueue] = None):
        self.store = store or get_store()
        self.queue = queue or make_queue("memory")
        self.lock = threading.RLock()
        self.services: Dict[str, Dict] = {
            s: {"state": "IDLE", "processed": 0, "total": 0, "detail": "", "process_units": 0}
            for s in SERVICES
        }
        self.running = False
        self.stage = "idle"
        self.last_run_id: Optional[str] = None
        self.active_dataset: Optional[str] = None
        self.anomaly_config: Dict = dict(DEFAULT_ANOMALY_CONFIG)
        self.regression_alert = False
        self._thread: Optional[threading.Thread] = None
        self._train_threads: Dict[str, threading.Thread] = {}
        self._score_context: Optional[Dict] = None  # for re-scoring corrected labels
        # restore persisted light state
        meta = self.store.meta
        self.active_dataset = meta.get("active_dataset")
        self.last_run_id = meta.get("last_run_id")
        self.anomaly_config = meta.get("anomaly_config", dict(DEFAULT_ANOMALY_CONFIG))
        self.regression_alert = bool(meta.get("regression_alert", False))

    # ------------------------------------------------------------ service status

    def _svc(self, name: str, state: Optional[str] = None, processed: Optional[int] = None,
             total: Optional[int] = None, detail: Optional[str] = None, add_units: int = 0) -> None:
        with self.lock:
            svc = self.services[name]
            if state is not None:
                svc["state"] = state
            if processed is not None:
                svc["processed"] = processed
            if total is not None:
                svc["total"] = total
            if detail is not None:
                svc["detail"] = detail
            svc["process_units"] += add_units

    def service_status(self) -> List[Dict]:
        with self.lock:
            return [{"service": s, **self.services[s]} for s in SERVICES]

    # ------------------------------------------------------------ run orchestration

    def run(self, dataset_id: str, background: bool = True, policy: Optional[QualityPolicy] = None) -> str:
        with self.lock:
            if self.running:
                raise RuntimeError("A pipeline run is already in progress")
            self.running = True
            run_id = new_id("run")
            self.last_run_id = run_id
            self.active_dataset = dataset_id
        if background:
            self._thread = threading.Thread(target=self._run_safe, args=(dataset_id, run_id, policy), daemon=True)
            self._thread.start()
        else:
            self._run_safe(dataset_id, run_id, policy)
        return run_id

    def _run_safe(self, dataset_id: str, run_id: str, policy: Optional[QualityPolicy]) -> None:
        try:
            self._execute(dataset_id, run_id, policy or load_policy())
        except Exception as exc:  # surface failures as service state, not silent death
            self.stage = "failed"
            for s in SERVICES:
                if self.services[s]["state"] == "RUNNING":
                    self._svc(s, state="FAILED", detail=str(exc)[:200])
            self.store.audit("pipeline_failed", "Run", run_id, str(exc)[:400])
        finally:
            with self.lock:
                self.running = False
            self.store.meta.update({
                "active_dataset": self.active_dataset,
                "last_run_id": self.last_run_id,
                "anomaly_config": self.anomaly_config,
                "regression_alert": self.regression_alert,
            })
            self.store.save()

    def _execute(self, dataset_id: str, run_id: str, policy: QualityPolicy) -> None:
        store = self.store
        meter = ProcessMeter(store, run_id)
        dataset = store.get("datasets", dataset_id)
        if dataset is None:
            raise ValueError(f"Unknown dataset {dataset_id}")
        self.queue.reset()

        # ---- 1. ingestion
        self.stage = "ingestion"
        frames = sorted(store.where("frames", dataset_id=dataset_id), key=lambda f: f.index)
        units = meter.record("ingestion", len(frames))
        self._svc("label-generator", state="RUNNING", processed=0, total=len(frames),
                  detail="ingesting sensor frames", add_units=units)

        # ---- 2. label generation (labels are hypotheses; skip if already labeled)
        self.stage = "label_generation"
        annotations = store.where("annotations", dataset_id=dataset_id)
        model_version = self._active_model_version()
        if not annotations:
            annotations = synthetic.generate_labels(store, dataset, model_version=model_version)
        else:
            model_version = annotations[0].model_version
            # Fresh evaluation: reset transient statuses of non-human-reviewed labels.
            for a in annotations:
                if a.source == "auto":
                    a.status = "PENDING"
                    store.put("annotations", a)
        units = meter.record("label_generation", len(annotations))
        self._svc("label-generator", state="HEALTHY", processed=len(frames), total=len(frames),
                  detail=f"{len(annotations)} labels by {model_version}", add_units=units)

        # ---- 3. enqueue for evaluation
        self.stage = "queueing"
        for ann in annotations:
            self.queue.publish("evaluation", {"annotation_id": ann.annotation_id})

        frames_by_id = {f.frame_id: f for f in frames}
        ann_by_id = {a.annotation_id: a for a in annotations}

        # ---- 4. anomaly detection (fit once on the full population)
        self.stage = "anomaly_detection"
        self._svc("anomaly-detector", state="RUNNING", processed=0, total=len(annotations),
                  detail="fitting detector ensemble")
        X, ids = synthetic.annotation_features(store, dataset_id)
        supervision = np.array([1.0 if ann_by_id[i].injected_errors else 0.0 for i in ids]) if len(ids) else None
        ensemble = AnomalyEnsemble(self.anomaly_config, seed=dataset.seed)
        scores, raw, norm = ensemble.run(X, supervision=supervision)
        strategy = ensemble.strategy
        threshold = 0.5 if strategy == "majority_vote" else ensemble.threshold
        anomaly_by_id: Dict[str, AnomalyDetection] = {}
        for j, aid in enumerate(ids):
            det = AnomalyDetection(
                annotation_id=aid,
                score=round(float(scores[j]), 4),
                is_anomaly=bool(scores[j] >= threshold),
                detector_scores={k: round(float(v[j]), 4) for k, v in raw.items()},
                normalized_scores={k: round(float(v[j]), 4) for k, v in norm.items()},
                ensemble_strategy=strategy,
                ensemble_score=round(float(scores[j]), 4),
                decision_threshold=threshold,
            )
            anomaly_by_id[aid] = det
            store.put("anomalies", det)
        self._score_context = {
            "scaler": ensemble.scaler,
            "detectors": ensemble.detectors,
            "raw": raw,
            "strategy": strategy,
            "threshold": threshold,
        }
        units = meter.record("anomaly_detection", len(ids))
        self._svc("anomaly-detector", state="HEALTHY", processed=len(ids), total=len(ids),
                  detail=f"{sum(1 for d in anomaly_by_id.values() if d.is_anomaly)} anomalies "
                         f"({strategy})", add_units=units)

        # ---- 5. queue-driven evaluation -> validation -> grading workers
        self.stage = "validation"
        self._svc("quality-validator", state="RUNNING", processed=0, total=len(annotations))
        self._svc("grader", state="RUNNING", processed=0, total=len(annotations))
        points_cache: Dict[str, np.ndarray] = {}
        tracking_by_id = self._tracking_evidence(annotations, frames_by_id)
        processed_v = processed_g = 0

        while True:
            batch = self.queue.consume("evaluation", BATCH_SIZE)
            if not batch:
                break
            for msg in batch:
                self.queue.publish("validation", msg)
            self.queue.ack("evaluation", len(batch))

            vbatch = self.queue.consume("validation", BATCH_SIZE)
            for msg in vbatch:
                ann = ann_by_id[msg["annotation_id"]]
                frame = frames_by_id[ann.frame_id]
                if ann.frame_id not in points_cache:
                    points_cache[ann.frame_id] = synthetic.frame_points(store, frame)
                validate_annotation(store, ann, frame, points=points_cache[ann.frame_id])
                processed_v += 1
                self.queue.publish("grading", msg)
            self.queue.ack("validation", len(vbatch))
            self._svc("quality-validator", processed=processed_v)

            gbatch = self.queue.consume("grading", BATCH_SIZE)
            for msg in gbatch:
                ann = ann_by_id[msg["annotation_id"]]
                frame = frames_by_id[ann.frame_id]
                tev = tracking_by_id.get(ann.annotation_id)
                grade_annotation(store, ann, frame,
                                 temporal_quality=tev.track_quality if tev else None,
                                 seed=dataset.seed)
                processed_g += 1
                self.queue.publish("triage", msg)
            self.queue.ack("grading", len(gbatch))
            self._svc("grader", processed=processed_g)
            time.sleep(BATCH_PAUSE_S)

        units = meter.record("validation", processed_v)
        self._svc("quality-validator", state="HEALTHY", detail="geometric validation complete", add_units=units)
        units = meter.record("grading", processed_g)
        stats = dataset_grader_statistics(store, dataset_id)
        self._svc("grader", state="HEALTHY",
                  detail=f"kappa(C)={stats.get('cohen_kappa', 0):.2f} fleiss={stats.get('fleiss_kappa', 0):.2f} "
                         f"alpha={stats.get('krippendorff_alpha', 0):.2f}", add_units=units)
        self.store.meta.setdefault("grader_stats", {})[dataset_id] = stats

        # ---- 6. rare events + scenario mining
        self.stage = "rare_events"
        self._svc("rare-event-detector", state="RUNNING", detail="mining scenarios")
        self._svc("scenario-miner", state="RUNNING")
        frame_scores: Dict[str, List[float]] = {}
        for aid, det in anomaly_by_id.items():
            frame_scores.setdefault(ann_by_id[aid].frame_id, []).append(det.score)
        mean_frame_scores = {fid: float(np.mean(v)) for fid, v in frame_scores.items()}
        events = detect_rare_events(store, dataset_id, mean_frame_scores)
        units = meter.record("rare_event_detection", len(frames))
        self._svc("rare-event-detector", state="HEALTHY", processed=len(events), total=len(events),
                  detail=f"{len(events)} rare events", add_units=units)
        n_scen = len(store.where("scenarios", dataset_id=dataset_id))
        self._svc("scenario-miner", state="HEALTHY", processed=n_scen, total=n_scen,
                  detail=f"{n_scen} scenario groups")

        # ---- 7. regression tracking (vs previous run baseline)
        self.stage = "regression"
        self._svc("regression-tracker", state="RUNNING")
        current = reporting.quality_metrics(store, dataset_id)
        cur_global = current["global"]
        cur_per_class = {c["class_name"]: c for c in current["per_class"]}
        baseline = self.store.meta.get("baseline_metrics")
        affected: List[str] = []
        if baseline:
            affected = find_affected_classes(cur_per_class, baseline.get("per_class", {}))
        reg = compare_runs(
            store, cur_global, model_version, dataset.version, run_id,
            baseline_metrics=baseline.get("global") if baseline else None,
            baseline_version=baseline.get("model_version") if baseline else None,
            affected_classes=affected,
            affected_scenarios=[s["scenario"] for s in current["per_scenario"]
                                if s.get("recall") is not None and s["recall"] < 0.5],
        )
        self.regression_alert = reg.regression_detected
        units = meter.record("regression_tracking", len(annotations))
        self._svc("regression-tracker", state="DEGRADED" if reg.regression_detected else "HEALTHY",
                  detail="REGRESSION DETECTED" if reg.regression_detected else
                         f"stable vs {reg.baseline_version or 'no baseline'}", add_units=units)
        # Current run becomes the next baseline.
        self.store.meta["baseline_metrics"] = {
            "global": cur_global, "per_class": cur_per_class, "model_version": model_version,
        }

        # ---- 8. triage
        self.stage = "triage"
        self._svc("triage-engine", state="RUNNING", processed=0, total=len(annotations))
        flagged: List[Annotation] = []
        processed_t = 0
        while True:
            batch = self.queue.consume("triage", BATCH_SIZE)
            if not batch:
                break
            for msg in batch:
                ann = ann_by_id[msg["annotation_id"]]
                decision = triage_annotation(
                    store, ann,
                    validation=store.get("validations", ann.annotation_id),
                    anomaly=store.get("anomalies", ann.annotation_id),
                    grading=store.get("grader_comparisons", ann.annotation_id),
                    tracking=tracking_by_id.get(ann.annotation_id),
                    policy=policy,
                    model_regressed=reg.regression_detected,
                )
                if decision.status == "AUTO_GRADED":
                    ann.status = "VERIFIED"  # auto-release into the verified pool
                else:
                    ann.status = "FLAGGED"
                    flagged.append(ann)
                store.put("annotations", ann)
                processed_t += 1
            self.queue.ack("triage", len(batch))
            self._svc("triage-engine", processed=processed_t)
        self._svc("triage-engine", state="HEALTHY",
                  detail=f"{processed_t - len(flagged)} auto-graded / {len(flagged)} flagged")

        # ---- 9. review tasks for flagged labels
        self.stage = "review_dispatch"
        self._svc("review-service", state="RUNNING")
        batch_id = f"review-batch-{run_id[-6:]}"
        open_tasks = {t.annotation_id for t in store.all("review_tasks") if t.status != "resolved"}
        created = 0
        for ann in flagged:
            if ann.annotation_id in open_tasks:
                continue
            decision = store.get("triage_decisions", ann.annotation_id)
            task = ReviewTask(
                task_id=new_id("task"),
                annotation_id=ann.annotation_id,
                frame_id=ann.frame_id,
                dataset_id=dataset_id,
                failure_reasons=decision.failure_reasons if decision else [],
                primary_failure_reason=decision.primary_failure_reason if decision else None,
                review_batch=batch_id,
            )
            store.put("review_tasks", task)
            created += 1
        n_open = len([t for t in store.all("review_tasks") if t.status != "resolved"])
        self._svc("review-service", state="HEALTHY", processed=n_open, total=n_open,
                  detail=f"{created} new tasks (batch {batch_id})")

        # ---- 10. metrics + alerts
        self.stage = "metrics"
        self._svc("metrics-service", state="RUNNING")
        self._refresh_alerts(dataset_id)
        dataset.status = "evaluated"
        store.put("datasets", dataset)
        store.audit("pipeline_completed", "Run", run_id,
                    f"dataset={dataset_id} model={model_version} labels={len(annotations)} flagged={len(flagged)}")
        self._svc("metrics-service", state="HEALTHY", detail="aggregates refreshed")
        self.stage = "complete"

    # ------------------------------------------------------------ tracking evidence

    def _tracking_evidence(self, annotations: List[Annotation], frames_by_id: Dict) -> Dict[str, TrackingEvidence]:
        """Detect ID switches and fragmentation from label tracks vs GT instances."""
        store = self.store
        # GT instance -> ordered (frame_index, annotation)
        by_instance: Dict[str, List] = {}
        track_instances: Dict[str, set] = {}
        for a in annotations:
            if a.matched_gt_id is None or a.track_id is None:
                continue
            frame = frames_by_id.get(a.frame_id)
            gt = next((g for g in frame.gt_boxes if g.gt_id == a.matched_gt_id), None) if frame else None
            if gt is None:
                continue
            by_instance.setdefault(gt.track_instance_id, []).append((frame.index, a))
            track_instances.setdefault(a.track_id, set()).add(gt.track_instance_id)

        evidence: Dict[str, TrackingEvidence] = {}
        # A pred track spanning multiple GT instances => identity contamination.
        contaminated_tracks = {t for t, insts in track_instances.items() if len(insts) > 1}
        for inst, items in by_instance.items():
            items.sort(key=lambda p: p[0])
            track_ids = [a.track_id for _, a in items]
            distinct = set(track_ids)
            fragmented = len(distinct) > 1
            prev_tid = None
            for idx, (_, a) in enumerate(items):
                switched = prev_tid is not None and a.track_id != prev_tid
                prev_tid = a.track_id
                contaminated = a.track_id in contaminated_tracks
                quality = 1.0
                if fragmented:
                    quality -= 0.3
                if switched or contaminated:
                    quality -= 0.4
                ev = TrackingEvidence(
                    annotation_id=a.annotation_id,
                    id_switch=bool(switched or contaminated),
                    fragmentation=bool(fragmented and idx >= 1 and a.track_id != track_ids[0]),
                    track_quality=round(max(quality, 0.0), 3),
                )
                evidence[a.annotation_id] = ev
                store.put("tracking_evidence", ev)
        # Labels without tracks (e.g. false positives) get neutral-low evidence.
        for a in annotations:
            if a.annotation_id not in evidence:
                ev = TrackingEvidence(annotation_id=a.annotation_id, track_quality=0.7 if a.track_id else None)
                evidence[a.annotation_id] = ev
                store.put("tracking_evidence", ev)
        return evidence

    # ------------------------------------------------------------ alerts

    def _refresh_alerts(self, dataset_id: str) -> None:
        store = self.store
        store.clear("alerts")
        m = reporting.quality_metrics(store, dataset_id)["global"]
        counters = reporting.counters(store, dataset_id)
        qs = self.queue.stats()

        def alert(kind: str, severity: str, message: str, page: str, eid: Optional[str] = None):
            store.put("alerts", Alert(alert_id=new_id("alert"), kind=kind, severity=severity,
                                      message=message, evidence_page=page, evidence_id=eid))

        if self.regression_alert:
            alert("model_regression", "critical", "Model regression detected vs baseline", "regression")
        if m["anomaly_rate"] > 0.15:
            alert("high_anomaly_rate", "warning",
                  f"Anomaly rate {m['anomaly_rate']:.1%} exceeds 15%", "rare-events")
        if m["grader_consensus"] is not None and m["grader_consensus"] < 0.85:
            alert("low_consensus", "warning",
                  f"Mean grader consensus {m['grader_consensus']:.1%} below 85%", "quality")
        if m["id_swap_rate"] is not None and m["id_swap_rate"] > 0.05:
            alert("high_id_swaps", "warning",
                  f"ID swap rate {m['id_swap_rate']:.1%} above 5%", "quality")
        if m["fragmentation_rate"] is not None and m["fragmentation_rate"] > 0.08:
            alert("high_fragmentation", "warning",
                  f"Track fragmentation {m['fragmentation_rate']:.1%} above 8%", "quality")
        total = max(counters["evaluated"], 1)
        hitl_rate = counters["flagged"] / total
        if hitl_rate > 0.30:
            alert("high_hitl_rate", "warning", f"HITL rate {hitl_rate:.1%} above 30%", "review")
        ver_rate = counters["verified"] / total
        if ver_rate < 0.5 and counters["evaluated"] > 0:
            alert("low_verification_rate", "warning",
                  f"Verification rate {ver_rate:.1%} below 50%", "datasets")
        if qs["pending"] > 5000:
            alert("queue_backlog", "warning", f"Queue backlog {qs['pending']} messages", "pipeline")
        if qs["failed"] > 0:
            alert("processing_failure", "critical", f"{qs['failed']} failed queue messages", "pipeline")
        if m["safety_critical_recall"] is not None and m["safety_critical_recall"] < 0.8:
            alert("safety_recall_low", "critical",
                  f"Safety-critical recall {m['safety_critical_recall']:.1%} below 80%", "quality")

    # ------------------------------------------------------------ HITL review

    def apply_review(self, task_id: str, action: RelabelingAction, actor: str = "human-reviewer") -> Dict:
        """Apply a human action, then ALWAYS re-run validation, grading and
        triage gates before any verified/rejected outcome (spec §23)."""
        store = self.store
        task = store.get("review_tasks", task_id)
        if task is None:
            raise KeyError(f"Unknown review task {task_id}")
        ann = store.get("annotations", task.annotation_id)
        frame = store.get("frames", ann.frame_id)
        dataset = store.get("datasets", ann.dataset_id)

        store.put("human_reviews", HumanReview(
            review_id=new_id("rev"), task_id=task_id, annotation_id=ann.annotation_id,
            reviewer=actor, action=action))
        meter = ProcessMeter(store, self.last_run_id or "")
        meter.record("hitl", 1)

        act = action.action
        if act == "reject":
            ann.status = "REJECTED"
            store.put("annotations", ann)
            task.status = "resolved"
            task.resolution = ReviewResolution(action=act, revalidation_passed=None, final_status="REJECTED")
            store.put("review_tasks", task)
            store.audit("review_rejected", "Annotation", ann.annotation_id, "human rejected label", actor)
            store.save()
            return {"task": task, "revalidation": None, "message": "Label rejected by human review."}

        if act == "mark_ignore":
            ann.status = "REJECTED"
            store.put("annotations", ann)
            task.status = "resolved"
            task.resolution = ReviewResolution(action=act, revalidation_passed=None, final_status="REJECTED")
            store.put("review_tasks", task)
            store.audit("review_ignored", "Annotation", ann.annotation_id, "marked ignore region", actor)
            store.save()
            return {"task": task, "revalidation": None, "message": "Marked as ignore; excluded from training."}

        if act == "correct":
            if action.corrected_bbox_3d:
                ann.bbox_3d = [float(v) for v in action.corrected_bbox_3d]
                ann.bbox_2d = synthetic.project_bbox_2d(ann.bbox_3d)
            if action.corrected_class:
                ann.class_name = action.corrected_class
            ann.source = "human_corrected"
            self._set_human_reference(frame, ann)
        elif act == "approve":
            # Human asserts the label is correct: label geometry becomes the
            # human-verified reference (the human overrules the vendor GT).
            ann.source = "human_approved"
            self._set_human_reference(frame, ann)
        elif act == "merge_tracks":
            if action.merge_with_track_id:
                ann.track_id = action.merge_with_track_id
            ann.source = "human_corrected"
        elif act == "split_track":
            ann.track_id = f"{ann.track_id or 'trk'}-split-{task_id[-4:]}"
            ann.source = "human_corrected"
        store.put("annotations", ann)

        # --- re-validation -> re-grading -> re-triage (gates always re-run)
        points = synthetic.frame_points(store, frame)
        validation = validate_annotation(store, ann, frame, points=points)
        tev = store.get("tracking_evidence", ann.annotation_id)
        if act in ("merge_tracks", "split_track") and tev is not None:
            tev.id_switch = False
            tev.fragmentation = False
            tev.track_quality = max(tev.track_quality or 0.7, 0.8)
            store.put("tracking_evidence", tev)
        grading = grade_annotation(store, ann, frame,
                                   temporal_quality=tev.track_quality if tev else None,
                                   seed=dataset.seed if dataset else 7)
        anomaly = self._rescore_anomaly(ann)
        decision = triage_annotation(
            store, ann, validation=validation, anomaly=anomaly, grading=grading,
            tracking=tev, policy=load_policy(), model_regressed=False)

        if decision.status == "AUTO_GRADED":
            ann.status = "VERIFIED"
            final = "VERIFIED"
        else:
            ann.status = "FLAGGED"
            final = "FLAGGED"
        store.put("annotations", ann)

        task.status = "resolved"
        task.resolution = ReviewResolution(
            action=act,
            corrected_bbox_3d=action.corrected_bbox_3d,
            corrected_class=action.corrected_class,
            revalidation_passed=decision.status == "AUTO_GRADED",
            final_status=final,
        )
        store.put("review_tasks", task)
        # Dataset lineage: corrected by this review batch.
        if dataset is not None and task.review_batch:
            dataset.lineage.corrected_by_review_batch = task.review_batch
            store.put("datasets", dataset)
        store.audit(f"review_{act}", "Annotation", ann.annotation_id,
                    f"re-validation {'PASSED' if final == 'VERIFIED' else 'FAILED: ' + ', '.join(decision.failure_reasons)}",
                    actor)
        self._refresh_alerts(ann.dataset_id)
        store.save()
        msg = ("Re-validation re-ran all gates: PASSED — label verified."
               if final == "VERIFIED"
               else f"Re-validation re-ran all gates: still failing {', '.join(decision.failure_reasons)}.")
        return {"task": task, "revalidation": None, "message": msg}

    def _set_human_reference(self, frame, ann: Annotation) -> None:
        """Human-verified geometry becomes the reference ground truth."""
        store = self.store
        gt = next((g for g in frame.gt_boxes if g.gt_id == ann.matched_gt_id), None)
        if gt is None:
            from sensorflow.evaluation.records import GroundTruthBox
            gt = GroundTruthBox(
                gt_id=f"{frame.frame_id}-human-{ann.annotation_id[-6:]}",
                class_name=ann.class_name,
                bbox_3d=list(ann.bbox_3d),
                track_instance_id=ann.object_id,
                gt_type="HUMAN_VERIFIED_GROUND_TRUTH",
            )
            frame.gt_boxes.append(gt)
            ann.matched_gt_id = gt.gt_id
        else:
            gt.class_name = ann.class_name
            gt.bbox_3d = list(ann.bbox_3d)
            gt.gt_type = "HUMAN_VERIFIED_GROUND_TRUTH"
        gt.bbox_2d = synthetic.project_bbox_2d(gt.bbox_3d)
        store.put("frames", frame)

    def _rescore_anomaly(self, ann: Annotation) -> Optional[AnomalyDetection]:
        """Re-score a corrected label against the fitted ensemble via score
        percentiles; falls back to the stored record when no context exists."""
        stored = self.store.get("anomalies", ann.annotation_id)
        ctx = self._score_context
        if ctx is None:
            return stored
        try:
            X, ids = synthetic.annotation_features(self.store, ann.dataset_id)
            if ann.annotation_id not in ids:
                return stored
            row = X[ids.index(ann.annotation_id)][None, :]
            Xs = ctx["scaler"].transform(row)
            raw_scores, norm_scores = {}, {}
            for det in ctx["detectors"]:
                s = float(det.score(Xs)[0])
                raw_scores[det.name] = round(s, 4)
                population = ctx["raw"][det.name]
                norm_scores[det.name] = round(float((population < s).mean()), 4)
            from sensorflow.evaluation.detectors import DEFAULT_WEIGHTS
            names = list(norm_scores)
            if ctx["strategy"] == "majority_vote":
                score = float(np.mean([norm_scores[n] >= ctx["threshold"] for n in names]))
            else:
                w = np.array([DEFAULT_WEIGHTS.get(n, 1.0) for n in names])
                score = float((np.array([norm_scores[n] for n in names]) * w).sum() / w.sum())
            det_rec = AnomalyDetection(
                annotation_id=ann.annotation_id,
                score=round(score, 4),
                is_anomaly=bool(score >= ctx["threshold"]),
                detector_scores=raw_scores,
                normalized_scores=norm_scores,
                ensemble_strategy=ctx["strategy"],
                ensemble_score=round(score, 4),
                decision_threshold=ctx["threshold"],
            )
            self.store.put("anomalies", det_rec)
            return det_rec
        except Exception:
            return stored

    # ------------------------------------------------------------ training flywheel

    def _active_model_version(self) -> str:
        models = sorted(self.store.all("models"), key=lambda m: m.created_at)
        return models[-1].model_version if models else "model-v1"

    def start_training(self, dataset_version: str, model_version: Optional[str] = None,
                       configuration: Optional[Dict] = None, quality_policy: str = "",
                       training_parameters: Optional[Dict] = None) -> TrainingJob:
        store = self.store
        dataset = store.get("datasets", dataset_version)
        if dataset is None:
            for d in store.all("datasets"):
                if d.version == dataset_version or d.name == dataset_version:
                    dataset = d
                    break
        if dataset is None:
            raise KeyError(f"Unknown dataset {dataset_version}")

        verified = [a for a in store.where("annotations", dataset_id=dataset.dataset_id)
                    if a.status == "VERIFIED"]
        if not verified:
            raise ValueError("No verified labels available for training — run the evaluation pipeline first")

        policy = load_policy()
        existing = store.all("models")
        next_num = len(existing) + 2  # model-v1 is the implicit generator
        new_version = model_version or f"model-v{next_num}"
        model_id = new_id("mdl")

        lineage = DatasetLineage(
            generated_from_model=dataset.lineage.generated_from_model,
            corrected_by_review_batch=dataset.lineage.corrected_by_review_batch,
            validated_by_policy=quality_policy or policy.policy_id,
            parent_dataset=dataset.dataset_id,
        )
        tds = TrainingDataset(
            training_dataset_id=new_id("tds"),
            source_dataset_id=dataset.dataset_id,
            version=f"{dataset.version}-train",
            num_verified_labels=len(verified),
            lineage=lineage,
        )
        store.put("training_datasets", tds)
        dataset.lineage.validated_by_policy = policy.policy_id
        store.put("datasets", dataset)

        params = training_parameters or {}
        job = TrainingJob(
            job_id=new_id("job"),
            model_id=model_id,
            model_version=new_version,
            dataset_version=dataset.dataset_id,
            total_epochs=int(params.get("epochs", 10)),
            status="queued",
            quality_policy=quality_policy or policy.policy_id,
            configuration=configuration or {},
            lineage=lineage,
        )
        job.logs.append(f"[queued] training {new_version} on {len(verified)} verified labels "
                        f"(dataset {dataset.dataset_id}, policy {job.quality_policy})")
        store.put("training_jobs", job)
        store.audit("training_started", "TrainingJob", job.job_id,
                    f"{new_version} on {tds.training_dataset_id} ({len(verified)} verified labels)")

        t = threading.Thread(target=self._train_loop, args=(job.job_id, len(verified)), daemon=True)
        self._train_threads[job.job_id] = t
        t.start()
        return job

    def _train_loop(self, job_id: str, n_verified: int) -> None:
        store = self.store
        job = store.get("training_jobs", job_id)
        meter = ProcessMeter(store, job_id)
        job.status = "running"
        store.put("training_jobs", job)
        rng = np.random.default_rng(abs(hash(job_id)) % (2 ** 32))

        # Data volume drives the achievable ceiling (flywheel effect).
        ceiling = min(0.985, 0.80 + 0.03 * math.log10(max(n_verified, 10)))
        prev = self._latest_completed_model()
        for epoch in range(1, job.total_epochs + 1):
            time.sleep(0.6)
            frac = epoch / job.total_epochs
            job.epoch = epoch
            job.loss = round(float(2.2 * math.exp(-2.5 * frac) + rng.uniform(0.01, 0.04)), 4)
            job.rare_recall = round(float(min(ceiling - 0.05, 0.45 + (ceiling - 0.48) * frac + rng.uniform(-0.01, 0.01))), 4)
            job.safety_recall = round(float(min(ceiling, 0.55 + (ceiling - 0.55) * frac + rng.uniform(-0.01, 0.01))), 4)
            units = meter.record("training", n_verified // job.total_epochs + 1)
            job.process_units += units
            job.logs.append(
                f"[epoch {epoch}/{job.total_epochs}] loss={job.loss:.4f} "
                f"rare_recall={job.rare_recall:.3f} safety_recall={job.safety_recall:.3f} pu+={units}")
            store.put("training_jobs", job)

        job.status = "completed"
        job.logs.append(f"[done] {job.model_version} trained; evaluating vs "
                        f"{prev.model_version if prev else 'no previous model'}")
        store.put("training_jobs", job)

        # Model evaluation + regression/improvement detection.
        metrics = ModelEvaluation(
            precision=round(min(0.99, ceiling + 0.005), 4),
            recall=round(ceiling - 0.01, 4),
            f1=round(ceiling - 0.005, 4),
            map_3d=round(ceiling - 0.03, 4),
            safety_critical_recall=job.safety_recall,
            rare_recall=job.rare_recall,
        )
        status = "baseline"
        if prev is not None and prev.metrics.f1 is not None:
            status = "improved" if (metrics.f1 or 0) >= prev.metrics.f1 else "regressed"
            compare_runs(
                store,
                {"precision": metrics.precision, "recall": metrics.recall,
                 "safety_critical_recall": metrics.safety_critical_recall},
                job.model_version, job.dataset_version, job.job_id,
                baseline_metrics={"precision": prev.metrics.precision, "recall": prev.metrics.recall,
                                  "safety_critical_recall": prev.metrics.safety_critical_recall},
                baseline_version=prev.model_version,
            )
        model = Model(
            model_id=job.model_id,
            model_version=job.model_version,
            name="synthlab-detector",
            trained_on_dataset=job.dataset_version,
            status="ready",
            metrics=metrics,
            regression_status=status,
        )
        store.put("models", model)
        store.audit("training_completed", "Model", model.model_id,
                    f"{model.model_version} f1={metrics.f1} status={status}")
        store.save()

    def _latest_completed_model(self) -> Optional[Model]:
        models = sorted(self.store.all("models"), key=lambda m: m.created_at)
        return models[-1] if models else None


_PIPELINE: Optional[LabelEvalPipeline] = None
_PIPE_LOCK = threading.Lock()


def get_pipeline() -> LabelEvalPipeline:
    global _PIPELINE
    with _PIPE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = LabelEvalPipeline()
        return _PIPELINE


def reset_pipeline(store: Optional[EvalStore] = None) -> LabelEvalPipeline:
    global _PIPELINE
    with _PIPE_LOCK:
        _PIPELINE = LabelEvalPipeline(store=store)
        return _PIPELINE
