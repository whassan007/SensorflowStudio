"""Evaluation runs: first-class async jobs producing cubes, error indexes and sketches.

Lifecycle:  create -> queued -> running (workers over partitions, partial stats)
            -> reducing -> materializing -> published   (or failed)

Each worker processes one population partition (Spark-executor stand-in):
  1. vectorized model simulation (deterministic given lineage seed)
  2. partial sufficient statistics grouped by the 9 cube dims
  3. per-object outputs persisted for forensic drill-down / honest scan fallback
  4. error rows + sketch updates + container partial aggregates

Reduce = concat partials + groupby-sum (cube.reduce_partials).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sensorflow.megaeval import cube as cube_mod
from sensorflow.megaeval import population as pop_mod
from sensorflow.megaeval.population import DIMENSIONS, DIM_NAMES
from sensorflow.megaeval.sketches import HyperLogLog, QuantileHistogram, Reservoir

ERROR_TYPES = ["FN", "FP", "LOCALIZATION", "ANOMALY", "LOW_CONF"]
CONTAINER_STATUS = ["ok", "warn", "critical"]
IOU_MATCH_THRESHOLD = 0.5
EVALUATOR_CODE_VERSION = "megaeval-1.0.0"
METRIC_VERSION = "metrics-1.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_dir(run_id: str) -> str:
    return os.path.join(pop_mod.MEGA_ROOT, "runs", run_id)


# ------------------------------------------------------------------ model profiles


def model_params(model_version: str, overrides: Optional[Dict] = None) -> Dict:
    """Deterministic model skill profile derived from the version string.

    Overrides let tests/demos inject regressions (e.g. worse night detection).
    """
    h = int(hashlib.sha256(model_version.encode()).hexdigest()[:8], 16)
    r = np.random.default_rng(h)
    params = {
        "base_detect": 0.972 + float(r.uniform(-0.01, 0.015)),
        "difficulty_weight": 0.18,
        "night_penalty": 0.030 + float(r.uniform(0, 0.02)),
        "rain_penalty": 0.020 + float(r.uniform(0, 0.015)),
        "occlusion_penalty": 0.050,
        "vru_penalty": 0.020 + float(r.uniform(0, 0.015)),
        "far_penalty": 0.045,
        "iou_base": 0.875 + float(r.uniform(-0.01, 0.015)),
        "iou_difficulty": 0.28,
        "fp_per_container": 0.30 + float(r.uniform(0, 0.08)),
        "anomaly_base": 0.018,
        "low_conf_threshold": 0.45,
    }
    params.update(overrides or {})
    return params


# ------------------------------------------------------------------ run entity


class EvaluationRun:
    def __init__(self, population_id: str, model_version: str,
                 overrides: Optional[Dict] = None, seed: Optional[int] = None,
                 worker_delay_s: float = 0.0, workers: int = 4,
                 label_version: str = "labels-v1",
                 threshold_config: Optional[Dict] = None,
                 sampling_config: Optional[Dict] = None):
        self.run_id = f"eval-{uuid.uuid4().hex[:10]}"
        self.population_id = population_id
        self.model_version = model_version
        self.overrides = overrides or {}
        self.seed = seed if seed is not None else int(
            hashlib.sha256(f"{population_id}|{model_version}|{label_version}".encode()
                           ).hexdigest()[:8], 16)
        self.worker_delay_s = worker_delay_s
        self.workers = workers
        self.status = "created"
        self.error: Optional[str] = None
        self.created_at = _now()
        self.started_at: Optional[str] = None
        self.published_at: Optional[str] = None
        self.partitions_total = 0
        self.partitions_done = 0
        self.objects_total = 0
        self.objects_processed = 0
        self.throughput_objs_per_s = 0.0
        self.eta_s: Optional[float] = None
        self.headline: Dict = {}
        self.per_class: Dict = {}
        self.lineage = {
            "evaluation_id": self.run_id,
            "dataset_version": population_id,
            "model_version": model_version,
            "model_checkpoint": f"ckpt-{hashlib.sha256(model_version.encode()).hexdigest()[:10]}",
            "label_version": label_version,
            "evaluator_code_version": EVALUATOR_CODE_VERSION,
            "metric_version": METRIC_VERSION,
            "threshold_config": {"iou_match": IOU_MATCH_THRESHOLD,
                                 **(threshold_config or {})},
            "sampling_config": sampling_config or {"method": "stratified_risk_weighted",
                                                   "target_n": 1500},
            "seed": self.seed,
            "hardware": f"{platform.machine()}/{platform.system()} single-node",
            "timestamp": self.created_at,
        }

    def progress_dict(self) -> Dict:
        pct = (self.partitions_done / self.partitions_total * 100) if self.partitions_total else 0.0
        return {
            "run_id": self.run_id,
            "population_id": self.population_id,
            "model_version": self.model_version,
            "status": self.status,
            "percent": round(pct, 2),
            "partitions_done": self.partitions_done,
            "partitions_total": self.partitions_total,
            "objects_processed": int(self.objects_processed),
            "objects_total": int(self.objects_total),
            "workers": self.workers,
            "throughput_objs_per_s": round(self.throughput_objs_per_s, 1),
            "eta_s": round(self.eta_s, 1) if self.eta_s is not None else None,
            "error": self.error,
        }

    def to_dict(self) -> Dict:
        return {
            **self.progress_dict(),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "published_at": self.published_at,
            "lineage": self.lineage,
            "headline": self.headline,
            "per_class": self.per_class,
            "overrides": self.overrides,
        }


# ------------------------------------------------------------------ partition evaluation


def evaluate_partition(pop_id: str, partition: int, params: Dict, seed: int) -> Dict:
    """Simulate the model on one partition; return partials + per-object outputs.

    Fully vectorized and deterministic for (lineage seed, partition).
    """
    cols = pop_mod.load_partition(pop_id, partition)
    n = cols["object_id"].shape[0]
    rng = np.random.default_rng((seed * 1_000_003 + partition) % (2**63))

    difficulty = cols["difficulty"].astype(np.float64)
    p_det = params["base_detect"] - difficulty * params["difficulty_weight"]
    p_det -= params["night_penalty"] * (cols["lighting"] == 1)
    p_det -= params["rain_penalty"] * (cols["weather"] >= 1)
    p_det -= params["occlusion_penalty"] * (cols["occlusion"] == 2)
    p_det -= params["vru_penalty"] * np.isin(cols["class"], [1, 2, 3])
    p_det -= params["far_penalty"] * (cols["distance_band"] == 3)
    p_det = np.clip(p_det, 0.02, 0.995)
    detected = rng.random(n) < p_det

    iou = np.zeros(n, dtype=np.float64)
    iou[detected] = np.clip(
        rng.normal(params["iou_base"] - difficulty[detected] * params["iou_difficulty"], 0.07),
        0.02, 0.985)
    tp = detected & (iou >= IOU_MATCH_THRESHOLD)
    loc = detected & ~tp

    conf = np.zeros(n, dtype=np.float64)
    conf[detected] = np.clip(
        0.55 * iou[detected] + 0.25 * (1 - difficulty[detected])
        + rng.normal(0.05, 0.08, size=int(detected.sum())), 0.01, 0.999)

    anomaly_p = params["anomaly_base"] * (1 + 2.0 * difficulty
                                          + 0.8 * (cols["scenario"] >= 4))
    anomaly = rng.random(n) < anomaly_p
    sensor_disagree = rng.random(n) < (0.04 + 0.25 * (cols["scenario"] == 5))

    safety = cols["safety_critical"].astype(bool)

    # ---- spurious FPs per container
    uniq_containers, first_idx = np.unique(cols["container_id"], return_index=True)
    ncont = uniq_containers.size
    wx_c = cols["weather"][first_idx]
    lt_c = cols["lighting"][first_idx]
    lam = params["fp_per_container"] * (1 + 0.5 * (wx_c >= 1) + 0.6 * (lt_c == 1))
    fp_counts = rng.poisson(lam)
    total_fp = int(fp_counts.sum())
    fp_container = np.repeat(uniq_containers, fp_counts)
    fp_src = np.repeat(first_idx, fp_counts)
    fp_cols = {k: cols[k][fp_src] for k in pop_mod.CONTAINER_DIMS}
    fp_cols["class"] = rng.choice(6, size=total_fp,
                                  p=[0.42, 0.18, 0.10, 0.08, 0.14, 0.08]).astype(np.int8)
    fp_cols["sensor"] = rng.choice(3, size=total_fp).astype(np.int8)
    fp_cols["distance_band"] = rng.choice(4, size=total_fp, p=[0.15, 0.35, 0.30, 0.20]).astype(np.int8)
    fp_cols["speed_band"] = rng.choice(4, size=total_fp).astype(np.int8)
    fp_cols["occlusion"] = rng.choice(3, size=total_fp, p=[0.5, 0.3, 0.2]).astype(np.int8)
    fp_conf = np.clip(rng.beta(2.0, 4.5, size=total_fp) + 0.05, 0.01, 0.99)

    # ---- contribution rows -> partial stats
    gt = pd.DataFrame({k: cols[k] for k in DIM_NAMES})
    gt["n"] = 1
    gt["tp"] = tp.astype(np.int64)
    gt["fp"] = loc.astype(np.int64)          # badly localized box counts as FP too
    gt["fn"] = (~tp).astype(np.int64)
    gt["loc_err"] = loc.astype(np.int64)
    gt["anomalies"] = anomaly.astype(np.int64)
    gt["sum_iou"] = np.where(tp, iou, 0.0)
    gt["sum_conf"] = np.where(detected, conf, 0.0)
    gt["sum_conf2"] = np.where(detected, conf**2, 0.0)
    gt["safety_n"] = safety.astype(np.int64)
    gt["safety_tp"] = (safety & tp).astype(np.int64)
    gt["reviewed"] = 0
    gt["verified"] = 0

    fpf = pd.DataFrame({k: fp_cols[k] for k in DIM_NAMES})
    fpf["n"] = 0
    fpf["tp"] = 0
    fpf["fp"] = 1
    fpf["fn"] = 0
    fpf["loc_err"] = 0
    fpf["anomalies"] = 0
    fpf["sum_iou"] = 0.0
    fpf["sum_conf"] = fp_conf
    fpf["sum_conf2"] = fp_conf**2
    fpf["safety_n"] = 0
    fpf["safety_tp"] = 0
    fpf["reviewed"] = 0
    fpf["verified"] = 0

    partial = cube_mod.partial_stats(pd.concat([gt, fpf], ignore_index=True))

    # ---- error rows (error index)
    err_frames = []
    fn_mask = ~detected
    for etype, mask, sev_base in (("FN", fn_mask, 0.75), ("LOCALIZATION", loc, 0.55)):
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        code = ERROR_TYPES.index(etype)
        err_frames.append(pd.DataFrame({
            "object_id": cols["object_id"][idx],
            "container_id": cols["container_id"][idx],
            "error_type": np.full(idx.size, code, dtype=np.int8),
            "severity": np.clip(sev_base + 0.25 * safety[idx] + 0.15 * difficulty[idx], 0, 1),
            "confidence": conf[idx],
            "safety_critical": safety[idx].astype(np.uint8),
            "sensor_disagree": sensor_disagree[idx].astype(np.uint8),
            **{k: cols[k][idx] for k in DIM_NAMES},
        }))
    an_idx = np.where(anomaly & tp)[0]
    if an_idx.size:
        err_frames.append(pd.DataFrame({
            "object_id": cols["object_id"][an_idx],
            "container_id": cols["container_id"][an_idx],
            "error_type": np.full(an_idx.size, ERROR_TYPES.index("ANOMALY"), dtype=np.int8),
            "severity": np.clip(0.4 + 0.3 * difficulty[an_idx], 0, 1),
            "confidence": conf[an_idx],
            "safety_critical": safety[an_idx].astype(np.uint8),
            "sensor_disagree": sensor_disagree[an_idx].astype(np.uint8),
            **{k: cols[k][an_idx] for k in DIM_NAMES},
        }))
    lc_idx = np.where(tp & (conf < params["low_conf_threshold"]))[0]
    if lc_idx.size:
        err_frames.append(pd.DataFrame({
            "object_id": cols["object_id"][lc_idx],
            "container_id": cols["container_id"][lc_idx],
            "error_type": np.full(lc_idx.size, ERROR_TYPES.index("LOW_CONF"), dtype=np.int8),
            "severity": np.clip(0.3 + 0.2 * difficulty[lc_idx], 0, 1),
            "confidence": conf[lc_idx],
            "safety_critical": safety[lc_idx].astype(np.uint8),
            "sensor_disagree": sensor_disagree[lc_idx].astype(np.uint8),
            **{k: cols[k][lc_idx] for k in DIM_NAMES},
        }))
    if total_fp:
        err_frames.append(pd.DataFrame({
            "object_id": np.full(total_fp, -1, dtype=np.int64),
            "container_id": fp_container,
            "error_type": np.full(total_fp, ERROR_TYPES.index("FP"), dtype=np.int8),
            "severity": np.clip(0.45 + 0.4 * fp_conf, 0, 1),
            "confidence": fp_conf,
            "safety_critical": np.zeros(total_fp, dtype=np.uint8),
            "sensor_disagree": (rng.random(total_fp) < 0.08).astype(np.uint8),
            **{k: fp_cols[k] for k in DIM_NAMES},
        }))
    errors = (pd.concat(err_frames, ignore_index=True) if err_frames
              else pd.DataFrame())

    # ---- container partial aggregates
    cdf = pd.DataFrame({
        "container_id": cols["container_id"],
        **{k: cols[k] for k in pop_mod.CONTAINER_DIMS},
        "n_objects": 1,
        "tp": tp.astype(np.int64),
        "fp": loc.astype(np.int64),
        "fn": (~tp).astype(np.int64),
        "anomalies": anomaly.astype(np.int64),
        "sum_iou": np.where(tp, iou, 0.0),
        "safety_n": safety.astype(np.int64),
    })
    containers = cdf.groupby(["container_id"] + pop_mod.CONTAINER_DIMS,
                             as_index=False, observed=True).sum()
    if total_fp:
        fp_per = pd.Series(fp_container).value_counts()
        containers["fp"] = containers["fp"] + containers["container_id"].map(fp_per).fillna(0).astype(np.int64)

    # ---- sketches (per partition, merged at reduce)
    conf_hist = QuantileHistogram(0, 1, 64)
    conf_hist.add(np.concatenate([conf[detected], fp_conf]))
    iou_hist = QuantileHistogram(0, 1, 64)
    iou_hist.add(iou[tp])
    hll = HyperLogLog(12)
    hll.add_ids(uniq_containers)
    reservoir = Reservoir(k=100, seed=partition)
    if len(errors):
        reservoir.add_ids(errors["object_id"].to_numpy()[errors["object_id"].to_numpy() >= 0][:2000])

    # ---- per-object outputs (forensic view + scan fallback + brute-force tests)
    obj_out = {
        "object_id": cols["object_id"],
        "container_id": cols["container_id"],
        "detected": detected.astype(np.uint8),
        "tp": tp.astype(np.uint8),
        "iou": iou.astype(np.float32),
        "confidence": conf.astype(np.float32),
        "anomaly": anomaly.astype(np.uint8),
        "sensor_disagree": sensor_disagree.astype(np.uint8),
    }
    fp_out = {
        "container_id": fp_container,
        "confidence": fp_conf.astype(np.float32),
        **{k: fp_cols[k] for k in DIM_NAMES},
    }

    return {"partial": partial, "errors": errors, "containers": containers,
            "conf_hist": conf_hist, "iou_hist": iou_hist, "hll": hll,
            "reservoir": reservoir, "objects": obj_out, "fps": fp_out,
            "rows": n}


# ------------------------------------------------------------------ store


class MegaStore:
    """Registry of populations + runs, with lazily-loaded run artifacts."""

    def __init__(self):
        self.lock = threading.RLock()
        self.runs: Dict[str, EvaluationRun] = {}
        self._artifacts: Dict[str, Dict] = {}
        self.router = cube_mod.QueryRouter()
        self._load_persisted_runs()

    def _load_persisted_runs(self) -> None:
        base = os.path.join(pop_mod.MEGA_ROOT, "runs")
        if not os.path.isdir(base):
            return
        for rid in sorted(os.listdir(base)):
            meta_path = os.path.join(base, rid, "run.json")
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path) as f:
                    d = json.load(f)
                run = EvaluationRun.__new__(EvaluationRun)
                run.__dict__.update(d["state"])
                self.runs[run.run_id] = run
            except Exception:
                continue

    def save_run(self, run: EvaluationRun) -> None:
        d = run_dir(run.run_id)
        os.makedirs(d, exist_ok=True)
        state = {k: v for k, v in run.__dict__.items() if not k.startswith("_")}
        with open(os.path.join(d, "run.json"), "w") as f:
            json.dump({"state": state}, f)

    # ---- artifacts

    def artifacts(self, run_id: str) -> Dict:
        with self.lock:
            if run_id in self._artifacts:
                return self._artifacts[run_id]
        d = run_dir(run_id)
        art: Dict = {}
        cube_path = os.path.join(d, "cube.npz")
        if os.path.exists(cube_path):
            art["cube"] = cube_mod.load_cube(cube_path)
        err_path = os.path.join(d, "errors.npz")
        if os.path.exists(err_path):
            with np.load(err_path) as z:
                art["errors"] = pd.DataFrame({k: z[k] for k in z.files})
        cont_path = os.path.join(d, "containers.npz")
        if os.path.exists(cont_path):
            with np.load(cont_path) as z:
                art["containers"] = pd.DataFrame({k: z[k] for k in z.files})
        emb_path = os.path.join(d, "embeddings.npz")
        if os.path.exists(emb_path):
            with np.load(emb_path) as z:
                art["emb_ids"] = z["container_id"]
                art["emb"] = z["emb"]
        sk_path = os.path.join(d, "sketches.npz")
        if os.path.exists(sk_path):
            with np.load(sk_path) as z:
                art["sketches"] = {
                    "confidence": QuantileHistogram.from_arrays(0, 1, z["conf_hist"]),
                    "iou": QuantileHistogram.from_arrays(0, 1, z["iou_hist"]),
                    "container_hll_estimate": float(z["hll_estimate"]),
                    "exemplar_ids": z["exemplars"].tolist(),
                }
        rev_path = os.path.join(d, "review.json")
        if os.path.exists(rev_path):
            with open(rev_path) as f:
                art["review"] = json.load(f)
        with self.lock:
            if len(self._artifacts) > 6:
                self._artifacts.pop(next(iter(self._artifacts)))
            self._artifacts[run_id] = art
        return art

    def invalidate_artifacts(self, run_id: str) -> None:
        with self.lock:
            self._artifacts.pop(run_id, None)
        self.router.cache.invalidate()

    # ---- lifecycle

    def create_run(self, **kwargs) -> EvaluationRun:
        run = EvaluationRun(**kwargs)
        meta = pop_mod.load_meta(run.population_id)
        if meta is None:
            raise KeyError(f"Unknown population {run.population_id}")
        run.partitions_total = meta["num_partitions"]
        run.objects_total = meta["num_objects"]
        run.status = "queued"
        with self.lock:
            self.runs[run.run_id] = run
        self.save_run(run)
        return run

    def start_async(self, run: EvaluationRun) -> None:
        t = threading.Thread(target=self._execute, args=(run,), daemon=True)
        t.start()

    def execute_sync(self, run: EvaluationRun) -> None:
        self._execute(run)

    def _execute(self, run: EvaluationRun) -> None:
        try:
            run.status = "running"
            run.started_at = _now()
            t_start = time.perf_counter()
            params = model_params(run.model_version, run.overrides)
            d = run_dir(run.run_id)
            os.makedirs(d, exist_ok=True)

            partials: List[pd.DataFrame] = []
            error_frames: List[pd.DataFrame] = []
            container_frames: List[pd.DataFrame] = []
            conf_hist = QuantileHistogram(0, 1, 64)
            iou_hist = QuantileHistogram(0, 1, 64)
            hll = HyperLogLog(12)
            reservoir = Reservoir(k=200, seed=run.seed)
            agg_lock = threading.Lock()

            def work(p: int) -> None:
                res = evaluate_partition(run.population_id, p, params, run.seed)
                if run.worker_delay_s:
                    time.sleep(run.worker_delay_s)  # simulated distributed I/O latency
                # uncompressed npz: these are hot-path worker writes (few MB total)
                np.savez(os.path.join(d, f"objects-part-{p:04d}.npz"), **res["objects"])
                np.savez(os.path.join(d, f"fp-part-{p:04d}.npz"), **res["fps"])
                nonlocal conf_hist, iou_hist, hll, reservoir
                with agg_lock:
                    partials.append(res["partial"])
                    if len(res["errors"]):
                        error_frames.append(res["errors"])
                    container_frames.append(res["containers"])
                    conf_hist = conf_hist.merge(res["conf_hist"])
                    iou_hist = iou_hist.merge(res["iou_hist"])
                    hll = hll.merge(res["hll"])
                    reservoir = reservoir.merge(res["reservoir"])
                    run.partitions_done += 1
                    run.objects_processed += res["rows"]
                    elapsed = time.perf_counter() - t_start
                    run.throughput_objs_per_s = run.objects_processed / max(elapsed, 1e-6)
                    remaining = run.objects_total - run.objects_processed
                    run.eta_s = remaining / max(run.throughput_objs_per_s, 1e-6)

            with ThreadPoolExecutor(max_workers=run.workers) as ex:
                list(ex.map(work, range(run.partitions_total)))

            run.status = "reducing"
            cube_df = cube_mod.reduce_partials(partials)

            run.status = "materializing"
            cube_mod.save_cube(os.path.join(d, "cube.npz"), cube_df)

            errors = (pd.concat(error_frames, ignore_index=True)
                      if error_frames else pd.DataFrame())
            if len(errors):
                errors = errors.reset_index(drop=True)
                errors["error_id"] = np.arange(len(errors), dtype=np.int64)
                risk = (errors["severity"] * 0.5
                        + errors["safety_critical"] * 0.3
                        + (1 - errors["confidence"]) * 0.15
                        + errors["sensor_disagree"] * 0.05)
                errors["risk_score"] = np.clip(risk, 0, 1)
                np.savez_compressed(os.path.join(d, "errors.npz"),
                                    **{c: errors[c].to_numpy() for c in errors.columns})

            containers = (pd.concat(container_frames, ignore_index=True)
                          .groupby(["container_id"] + pop_mod.CONTAINER_DIMS,
                                   as_index=False, observed=True).sum())
            containers["reviewed"] = 0
            containers["verified"] = 0
            err_share = (containers["fn"] + containers["fp"]) / containers["n_objects"].clip(lower=1)
            containers["risk_score"] = np.clip(
                0.55 * err_share + 0.25 * (containers["anomalies"] / containers["n_objects"].clip(lower=1))
                + 0.20 * (containers["safety_n"] > 0), 0, 1).round(4)
            containers["status"] = np.select(
                [containers["risk_score"] >= 0.45, containers["risk_score"] >= 0.2],
                [2, 1], default=0).astype(np.int8)
            np.savez_compressed(os.path.join(d, "containers.npz"),
                                **{c: containers[c].to_numpy() for c in containers.columns})

            self._build_embeddings(d, containers)

            np.savez_compressed(os.path.join(d, "sketches.npz"),
                                conf_hist=conf_hist.counts, iou_hist=iou_hist.counts,
                                hll_estimate=np.array(hll.estimate()),
                                exemplars=np.array(reservoir.items, dtype=np.int64))

            headline_rows, _ = cube_mod.aggregate(
                cube_df, None, None,
                ["n", "tp", "fp", "fn", "loc_err", "anomalies", "precision", "recall", "f1",
                 "mean_iou", "anomaly_rate", "safety_recall", "conf_mean", "error_rate"], 1)
            run.headline = headline_rows[0]
            run.headline["containers"] = int(containers.shape[0])
            run.headline["containers_hll_estimate"] = round(hll.estimate(), 1)

            per_class_rows, _ = cube_mod.aggregate(
                cube_df, None, ["class"],
                ["n", "precision", "recall", "f1", "mean_iou", "safety_recall", "anomaly_rate"], 50)
            run.per_class = {r["class"]: r for r in per_class_rows}

            run.status = "published"
            run.published_at = _now()
            run.eta_s = 0.0
            self.save_run(run)
            self.invalidate_artifacts(run.run_id)
        except Exception as e:  # pragma: no cover - defensive
            run.status = "failed"
            run.error = str(e)
            self.save_run(run)

    @staticmethod
    def _build_embeddings(d: str, containers: pd.DataFrame) -> None:
        """32-d container embeddings: structural features + seeded random projection."""
        feats = []
        nobj = containers["n_objects"].clip(lower=1).to_numpy(dtype=np.float64)
        for col in ("tp", "fp", "fn", "anomalies", "safety_n"):
            feats.append(containers[col].to_numpy(dtype=np.float64) / nobj)
        feats.append(containers["sum_iou"].to_numpy(dtype=np.float64)
                     / containers["tp"].clip(lower=1).to_numpy(dtype=np.float64))
        feats.append(np.log1p(nobj) / 4.0)
        feats.append(containers["risk_score"].to_numpy(dtype=np.float64))
        for dim in pop_mod.CONTAINER_DIMS:
            k = len(DIMENSIONS[dim])
            onehot = np.eye(k)[containers[dim].to_numpy(dtype=np.int64)]
            feats.extend(onehot.T)
        X = np.vstack(feats).T  # (n, ~25)
        rng = np.random.default_rng(1234)
        proj = rng.normal(size=(X.shape[1], 32)) / np.sqrt(X.shape[1])
        emb = X @ proj
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
        np.savez_compressed(os.path.join(d, "embeddings.npz"),
                            container_id=containers["container_id"].to_numpy(),
                            emb=emb.astype(np.float32))

    # ---- forensic drill-down (deepest level; loads exactly one partition)

    def container_objects(self, run: EvaluationRun, container_id: int) -> List[Dict]:
        p = pop_mod.partition_of_container(container_id)
        pop_cols = pop_mod.load_partition(run.population_id, p)
        d = run_dir(run.run_id)
        with np.load(os.path.join(d, f"objects-part-{p:04d}.npz")) as z:
            out_cols = {k: z[k] for k in z.files}
        mask = pop_cols["container_id"] == container_id
        idx = np.where(mask)[0]
        rows = []
        for i in idx.tolist():
            det = bool(out_cols["detected"][i])
            tp = bool(out_cols["tp"][i])
            rows.append({
                "annotation_id": f"obj-{int(pop_cols['object_id'][i])}",
                "container_id": int(container_id),
                **{k: DIMENSIONS[k][int(pop_cols[k][i])] for k in DIM_NAMES},
                "safety_critical": bool(pop_cols["safety_critical"][i]),
                "difficulty": round(float(pop_cols["difficulty"][i]), 4),
                "detected": det,
                "outcome": "TP" if tp else ("LOCALIZATION" if det else "FN"),
                "iou": round(float(out_cols["iou"][i]), 4) if det else None,
                "confidence": round(float(out_cols["confidence"][i]), 4) if det else None,
                "anomaly": bool(out_cols["anomaly"][i]),
                "sensor_disagree": bool(out_cols["sensor_disagree"][i]),
            })
        with np.load(os.path.join(d, f"fp-part-{p:04d}.npz")) as z:
            fp_cols = {k: z[k] for k in z.files}
        fp_mask = fp_cols["container_id"] == container_id
        for j in np.where(fp_mask)[0].tolist():
            rows.append({
                "annotation_id": f"fp-{run.run_id[-4:]}-{p}-{j}",
                "container_id": int(container_id),
                **{k: DIMENSIONS[k][int(fp_cols[k][j])] for k in DIM_NAMES},
                "safety_critical": False,
                "difficulty": None,
                "detected": True,
                "outcome": "FP",
                "iou": 0.0,
                "confidence": round(float(fp_cols["confidence"][j]), 4),
                "anomaly": False,
                "sensor_disagree": False,
            })
        return rows

    # ---- honest record-scan fallback for the query router

    def scan_records(self, run: EvaluationRun, filters: Optional[Dict],
                     group_by: Optional[List[str]], metrics: Optional[List[str]],
                     limit: int) -> tuple:
        frames = []
        d = run_dir(run.run_id)
        for p in range(run.partitions_total):
            pop_cols = pop_mod.load_partition(run.population_id, p)
            with np.load(os.path.join(d, f"objects-part-{p:04d}.npz")) as z:
                df = pd.DataFrame({
                    **{k: pop_cols[k] for k in DIM_NAMES},
                    "container_id": pop_cols["container_id"],
                    "tp": z["tp"].astype(np.int64),
                    "fn": (1 - z["tp"]).astype(np.int64),
                    "fp": (z["detected"] & ~z["tp"].astype(bool)).astype(np.int64),
                    "anomalies": z["anomaly"].astype(np.int64),
                    "iou": z["iou"],
                })
            frames.append(df)
        full = pd.concat(frames, ignore_index=True)
        full = cube_mod.apply_filters(full, filters)
        gb = [g for g in (group_by or []) if g in list(full.columns)]
        if not gb:
            gb = ["container_id"]
        agg = full.groupby(gb, as_index=False, observed=True).agg(
            n=("tp", "size"), tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
            anomalies=("anomalies", "sum"), mean_iou=("iou", "mean"))
        agg["recall"] = agg["tp"] / (agg["tp"] + agg["fn"]).clip(lower=1)
        agg = agg.sort_values("n", ascending=False).head(limit)
        rows = []
        for _, r in agg.iterrows():
            row = {}
            for g in gb:
                row[g] = DIMENSIONS[g][int(r[g])] if g in DIM_NAMES else int(r[g])
            row.update({"n": int(r["n"]), "tp": int(r["tp"]), "fp": int(r["fp"]),
                        "fn": int(r["fn"]), "anomalies": int(r["anomalies"]),
                        "mean_iou": round(float(r["mean_iou"]), 6),
                        "recall": round(float(r["recall"]), 6)})
            rows.append(row)
        return rows, len(full)


_STORE: Optional[MegaStore] = None
_STORE_LOCK = threading.Lock()


def get_mega_store() -> MegaStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = MegaStore()
        return _STORE


def reset_mega_store() -> None:
    global _STORE
    with _STORE_LOCK:
        _STORE = None
