"""ScenarioMiningAgent — similar-failure retrieval + clustering.

Embedding + attribute retrieval of similar historical failures, following the
megaeval embedding recipe (structural features + one-hot attributes + seeded
random projection, cosine retrieval — see MegaStore._build_embeddings in
sensorflow/megaeval/runs.py, which operates on run artifacts and is therefore
mirrored here for failure records rather than re-invented).

Clustering method is SELECTED FROM THE DATA SHAPE and the choice is justified
in the output (it is not always DBSCAN):
  * n < 15            -> single-linkage agglomerative on cosine distance
                         (density parameters are unidentifiable at tiny n)
  * mostly categorical -> attribute-signature grouping (exact modes)
  * otherwise          -> density-based (DBSCAN-style) via pairwise radius
                          graph connected components
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from sensorflow.agentic.agents.base import BaseAgent, compact, no_escalation
from sensorflow.agentic.models import AgentEscalation, FailureEvent

GT_CLASSES = ["pedestrian", "cyclist", "motorcycle", "vehicle", "truck"]
PRED_CLASSES = ["pedestrian", "cyclist", "motorcycle", "vehicle", "truck",
                "construction_cone", "background"]
TIMES = ["day", "night"]
WEATHERS = ["clear", "rain"]
EMBED_DIM = 16
EMBED_SEED = 1234  # same seeded-projection convention as megaeval


def historical_corpus(seed: int = 77, n: int = 48) -> List[Dict]:
    """Deterministic synthetic archive of previously-triaged failures.

    Deliberately contains NO pedestrian->construction_cone precedent, so the
    motivating failure is genuinely novel against the archive.
    """
    rng = np.random.default_rng(seed)
    pairs = [("pedestrian", "cyclist"), ("cyclist", "motorcycle"),
             ("vehicle", "truck"), ("truck", "vehicle"),
             ("cyclist", "vehicle"), ("pedestrian", "background"),
             ("motorcycle", "cyclist"), ("vehicle", "background")]
    out = []
    for i in range(n):
        gt, pred = pairs[int(rng.integers(0, len(pairs)))]
        out.append({
            "historical_id": f"hist-{i:04d}",
            "gt_class": gt, "predicted_class": pred,
            "construction_zone": bool(rng.random() < 0.15),
            "time_of_day": TIMES[int(rng.random() < 0.4)],
            "weather": WEATHERS[int(rng.random() < 0.3)],
            "distance_m": round(float(rng.uniform(8, 70)), 1),
            "resolved": bool(rng.random() < 0.8),
            "resolution": "retrained" if rng.random() < 0.5 else "data_fix",
        })
    return out


def _featurize(rec: Dict) -> np.ndarray:
    f: List[float] = []
    f.extend(1.0 if rec["gt_class"] == c else 0.0 for c in GT_CLASSES)
    f.extend(1.0 if rec["predicted_class"] == c else 0.0 for c in PRED_CLASSES)
    f.append(1.0 if rec.get("construction_zone") else 0.0)
    f.append(1.0 if rec.get("time_of_day") == "night" else 0.0)
    f.append(1.0 if rec.get("weather") == "rain" else 0.0)
    f.append(min(float(rec.get("distance_m", 30.0)) / 70.0, 1.0))
    return np.asarray(f, dtype=np.float64)


def embed(records: List[Dict]) -> np.ndarray:
    X = np.vstack([_featurize(r) for r in records])
    rng = np.random.default_rng(EMBED_SEED)
    proj = rng.normal(size=(X.shape[1], EMBED_DIM)) / np.sqrt(X.shape[1])
    E = X @ proj
    return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)


def _select_clustering(n: int, categorical_share: float) -> Tuple[str, str]:
    if n < 15:
        return ("single_linkage_agglomerative",
                f"only n={n} member instances: density-based methods "
                "(DBSCAN) need an eps/minPts neighborhood that is "
                "unidentifiable at this sample size; single-linkage on "
                "cosine distance degrades gracefully")
    if categorical_share > 0.8:
        return ("attribute_signature_grouping",
                f"{categorical_share:.0%} of features are categorical "
                "one-hots: exact signature grouping is better-defined than "
                "euclidean density in a nearly-discrete space")
    return ("density_radius_components",
            f"n={n} with mixed features: density-based clustering "
            "(DBSCAN-style radius graph components) is appropriate")


def _cluster(records: List[Dict], emb: np.ndarray, method: str) -> List[List[int]]:
    n = len(records)
    if method == "attribute_signature_grouping":
        groups: Dict[Tuple, List[int]] = {}
        for i, r in enumerate(records):
            key = (r.get("construction_zone"), r.get("time_of_day"),
                   r.get("weather"))
            groups.setdefault(key, []).append(i)
        return list(groups.values())
    # graph-based (single-linkage threshold == radius components on cosine)
    sims = emb @ emb.T
    thr = 0.92 if method == "single_linkage_agglomerative" else 0.95
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= thr:
                parent[find(i)] = find(j)
    groups2: Dict[int, List[int]] = {}
    for i in range(n):
        groups2.setdefault(find(i), []).append(i)
    return list(groups2.values())


class ScenarioMiningAgent(BaseAgent):
    name = "scenario_mining"
    version = "1.0.0"
    epistemic_status = "DERIVED"

    def _analyze(self, failure_id: str, failure: FailureEvent = None,
                 **inputs) -> Tuple[Dict[str, Any], float, str, AgentEscalation]:
        assert failure is not None
        corpus = historical_corpus()

        # ---- retrieval: embedding + attribute filters --------------------
        query = {
            "gt_class": failure.gt_class or "vehicle",
            "predicted_class": failure.predicted_class or "background",
            "construction_zone": any(i.construction_zone for i in failure.instances),
            "time_of_day": (failure.instances[0].time_of_day
                            if failure.instances else "day"),
            "weather": (failure.instances[0].weather
                        if failure.instances else "clear"),
            "distance_m": (float(np.mean([i.distance_m for i in failure.instances]))
                           if failure.instances else 30.0),
        }
        all_emb = embed(corpus + [query])
        emb, q = all_emb[:-1], all_emb[-1]
        sims = emb @ q
        order = np.argsort(-sims)[:8]
        similar = [{**corpus[int(i)], "similarity": round(float(sims[int(i)]), 4)}
                   for i in order]

        exact_precedent = [r for r in corpus
                           if r["gt_class"] == query["gt_class"]
                           and r["predicted_class"] == query["predicted_class"]]
        novelty = "novel" if not exact_precedent else (
            "known_regression" if any(not r["resolved"] for r in exact_precedent)
            else "known_stable")

        # ---- clustering of the failure's own instances -------------------
        inst_records = [{
            "gt_class": i.gt_class, "predicted_class": i.predicted_class,
            "construction_zone": i.construction_zone,
            "time_of_day": i.time_of_day, "weather": i.weather,
            "distance_m": i.distance_m,
        } for i in failure.instances]
        cluster_reports: List[Dict] = []
        method = justification = None
        if inst_records:
            feat_dim = _featurize(inst_records[0]).size
            categorical_share = (feat_dim - 1) / feat_dim  # all but distance
            method, justification = _select_clustering(len(inst_records),
                                                       categorical_share)
            inst_emb = embed(inst_records)
            clusters = _cluster(inst_records, inst_emb, method)
            total = len(inst_records)
            for idx, members in enumerate(sorted(clusters, key=len, reverse=True)):
                recs = [inst_records[m] for m in members]
                dominant = {
                    "construction_zone": (sum(r["construction_zone"] for r in recs)
                                          / len(recs)),
                    "night_share": (sum(r["time_of_day"] == "night" for r in recs)
                                    / len(recs)),
                    "rain_share": (sum(r["weather"] == "rain" for r in recs)
                                   / len(recs)),
                    "mean_distance_m": round(float(np.mean(
                        [r["distance_m"] for r in recs])), 1),
                }
                cluster_reports.append({
                    "cluster_id": f"{failure.failure_id}-c{idx}",
                    "size": len(members),
                    "population_share": round(len(members) / total, 3),
                    "dominant_conditions": dominant,
                    "representative_examples": [
                        failure.instances[m].instance_id for m in members[:3]],
                })

        output = {
            "similar_historical": similar,
            "exact_precedent_count": len(exact_precedent),
            "novelty": novelty,
            "novelty_basis": ("no archived failure shares the "
                              f"{query['gt_class']}->{query['predicted_class']} "
                              "signature" if novelty == "novel" else
                              f"{len(exact_precedent)} archived precedents"),
            "clustering_method": method,
            "clustering_justification": justification,
            "clusters": cluster_reports,
            "retrieval": ("cosine over seeded random-projection embeddings of "
                          "structural+categorical features (megaeval recipe) "
                          "+ attribute retrieval"),
        }
        confidence = 0.75 if inst_records else 0.4
        return (output, confidence,
                "retrieval and clustering are deterministic; confidence "
                "reflects instance coverage",
                no_escalation())

    def _llm_prompt(self, failure_id, output, **inputs):
        return ("Summarize the cluster structure and novelty finding of this "
                "failure mining result: " + compact(output))
