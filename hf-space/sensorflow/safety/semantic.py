"""Neuro-symbolic semantic scenario mining (lightweight, two-stage).

Industry concept: natural-language scenario search over fleet data (Voxel51 /
Scale Nucleus semantic search, Applied Intuition scenario mining). Two stages:

1. SYMBOLIC filter — structured attributes (container ODD dimensions, error
   rates, dynamics) restrict the candidate set exactly.
2. REASONING scorer — if a local Ollama endpoint is reachable (reusing the
   copilot plumbing in sensorflow/evaluation/copilot.py) it contributes a
   natural-language rationale for the top candidates; otherwise (and always,
   for the actual ranking) a DETERMINISTIC rule-based scorer maps concept
   keywords to structured evidence via an explicit lexicon. Every result
   carries per-stage explanations of why it matched.

Hybrid retrieval: the rule score is blended with cosine similarity over the
run's existing 32-d structural container embeddings (centroid of the top rule
matches acts as the query anchor), combining symbolic and dense signals.

Honest marker: the "reasoning" stage is a keyword lexicon over structured
evidence unless Ollama is reachable; rankings are deterministic either way
(the LLM adds rationale text, it never reorders results).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from sensorflow.megaeval import population as pop_mod
from sensorflow.megaeval.population import DIMENSIONS

# Each lexicon entry: (regex over the concept text, human-readable meaning,
# scorer(row) -> (score in [0,1], evidence string)).
_LEXICON: List[Tuple[str, str, object]] = []


def _rule(pattern: str, meaning: str):
    def deco(fn):
        _LEXICON.append((pattern, meaning, fn))
        return fn
    return deco


@_rule(r"pedestrian|vru|vulnerable|person|cyclist|bike", "vulnerable road users present")
def _vru(row):
    share = row["safety_n"] / max(row["n_objects"], 1)
    return min(1.0, share * 4.0), f"safety-critical VRU share {share:.0%}"


@_rule(r"night|dark|glare|low light", "night / low-illumination context")
def _night(row):
    if row["lighting"] == "night":
        return 1.0, "lighting=night"
    if row["lighting"] == "dusk":
        return 0.5, "lighting=dusk"
    return 0.0, ""


@_rule(r"rain|wet|fog|snow|weather|storm", "adverse weather context")
def _weather(row):
    return (1.0, f"weather={row['weather']}") if row["weather"] != "clear" else (0.0, "")


@_rule(r"hesitant|erratic|unusual|anomal|unpredict|jerky", "anomalous / erratic behavior evidence")
def _anom(row):
    rate = row["anomalies"] / max(row["n_objects"], 1)
    return min(1.0, rate * 8.0), f"anomaly rate {rate:.0%}"


@_rule(r"occlu|hidden|blocked|obstruct", "heavy occlusion context")
def _occl(row):
    return (1.0, "scenario=occlusion_heavy") if row["scenario"] == "occlusion_heavy" else (0.0, "")


@_rule(r"near[ -]?miss|close call|conflict|collision|ttc", "near-miss / extreme-TTC scenario")
def _nearmiss(row):
    if row["scenario"] in ("near_miss", "extreme_ttc"):
        return 1.0, f"scenario={row['scenario']}"
    return 0.0, ""


@_rule(r"intersection|junction|crossing", "intersection scenery")
def _intersection(row):
    return (1.0, "road_type=intersection") if row["road_type"] == "intersection" else (0.0, "")


@_rule(r"urban|city|street", "urban scenery")
def _urban(row):
    return (1.0, "road_type=urban") if row["road_type"] == "urban" else (0.0, "")


@_rule(r"highway|freeway|motorway", "highway scenery")
def _highway(row):
    return (1.0, "road_type=highway") if row["road_type"] == "highway" else (0.0, "")


@_rule(r"rural|country", "rural scenery")
def _rural(row):
    return (1.0, "road_type=rural") if row["road_type"] == "rural" else (0.0, "")


@_rule(r"lane edge|lane|curb|edge|merge", "lane-boundary interaction (proxy)")
def _lane(row):
    # Proxy: the population has no lane geometry; urban/intersection scenes are
    # where lane-edge interactions concentrate. Explained as a proxy.
    if row["road_type"] in ("urban", "intersection"):
        return 0.5, f"road_type={row['road_type']} (lane-edge proxy — no lane geometry instrumented)"
    return 0.0, ""


@_rule(r"miss|fail|undetected|false negative|fn\b", "model misses (FN) evidence")
def _fn(row):
    rate = row["fn"] / max(row["n_objects"], 1)
    return min(1.0, rate * 5.0), f"FN rate {rate:.0%}"


@_rule(r"false positive|ghost|hallucinat|fp\b", "false-positive evidence")
def _fp(row):
    rate = row["fp"] / max(row["n_objects"], 1)
    return min(1.0, rate * 5.0), f"FP rate {rate:.0%}"


@_rule(r"sensor|degraded|dropout|noise", "sensor-degraded scenario")
def _sensor(row):
    return (1.0, "scenario=sensor_degraded") if row["scenario"] == "sensor_degraded" else (0.0, "")


@_rule(r"risk|critical|danger|safety|severe", "high composite risk")
def _risk(row):
    return float(row["risk_score"]), f"risk_score {row['risk_score']:.2f}"


def _container_rows(store, run) -> List[Dict]:
    art = store.artifacts(run.run_id)
    df = art["containers"]
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "container_id": int(r["container_id"]),
            **{dim: DIMENSIONS[dim][int(r[dim])] for dim in pop_mod.CONTAINER_DIMS},
            "n_objects": int(r["n_objects"]),
            "fn": int(r["fn"]), "fp": int(r["fp"]),
            "anomalies": int(r["anomalies"]),
            "safety_n": int(r["safety_n"]),
            "risk_score": float(r["risk_score"]),
        })
    return rows


def _symbolic_filter(rows: List[Dict], filters: Optional[Dict]) -> Tuple[List[Dict], Dict]:
    before = len(rows)
    applied = {}
    for dim, values in (filters or {}).items():
        if dim not in pop_mod.CONTAINER_DIMS:
            continue
        vals = values if isinstance(values, list) else [values]
        rows = [r for r in rows if r[dim] in vals]
        applied[dim] = vals
    return rows, {"stage": "symbolic_filter", "applied_filters": applied,
                  "candidates_before": before, "candidates_after": len(rows)}


def _rule_score(concept: str, row: Dict) -> Tuple[float, List[Dict]]:
    matched = []
    total = 0.0
    weight_sum = 0.0
    for pattern, meaning, fn in _LEXICON:
        if not re.search(pattern, concept, flags=re.IGNORECASE):
            continue
        score, evidence = fn(row)
        weight_sum += 1.0
        total += score
        if score > 0:
            matched.append({"term": pattern, "meaning": meaning,
                            "evidence": evidence, "contribution": round(score, 4)})
    if weight_sum == 0:
        return float(row["risk_score"]), [{
            "term": "(fallback)", "meaning": "no lexicon term matched the concept",
            "evidence": f"ranked by composite risk_score {row['risk_score']:.2f}",
            "contribution": round(float(row["risk_score"]), 4)}]
    return total / weight_sum, matched


def _llm_rationale(concept: str, top_rows: List[Dict], timeout_s: float = 2.0) -> Optional[Dict]:
    """Optional Ollama pass (copilot plumbing); never reorders results."""
    try:
        import json as _json

        import httpx

        from sensorflow.evaluation.copilot import OLLAMA_ENDPOINTS
        prompt = (
            "You are a scenario-mining assistant. Concept query: "
            f"{concept!r}. Candidate scenes (structured evidence):\n"
            f"{_json.dumps(top_rows[:6], indent=1)[:2500]}\n"
            "In 3 sentences, explain which candidates best match the concept and why."
        )
        for ep in OLLAMA_ENDPOINTS:
            try:
                res = httpx.post(ep["url"], json={
                    "model": ep["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }, timeout=timeout_s)
                if res.status_code == 200:
                    text = res.json().get("message", {}).get("content", "")
                    if text:
                        return {"provider": ep["url"], "rationale": text}
            except Exception:
                continue
    except Exception:
        pass
    return None


def search_containers(store, run, concept: str, filters: Optional[Dict] = None,
                      k: int = 12, use_llm: Optional[bool] = None) -> Dict:
    """Two-stage neuro-symbolic search over a run's containers.

    Ranking = 0.65 * deterministic rule score + 0.35 * embedding similarity to
    the centroid of the top rule matches (hybrid retrieval). LLM (if enabled
    and reachable) adds rationale text only.
    """
    rows = _container_rows(store, run)
    rows, stage1 = _symbolic_filter(rows, filters)

    scored = []
    for row in rows:
        score, matched = _rule_score(concept, row)
        scored.append((score, row, matched))
    scored.sort(key=lambda t: (-t[0], t[1]["container_id"]))

    # Hybrid: embedding similarity to the centroid of the top rule matches.
    art = store.artifacts(run.run_id)
    emb_ids, emb = art.get("emb_ids"), art.get("emb")
    emb_sim: Dict[int, float] = {}
    if emb is not None and scored:
        anchor_ids = [t[1]["container_id"] for t in scored[:10] if t[0] > 0]
        pos = np.where(np.isin(emb_ids, anchor_ids))[0]
        if pos.size:
            centroid = emb[pos].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-9:
                sims = emb @ (centroid / norm)
                emb_sim = {int(cid): float(s) for cid, s in zip(emb_ids, sims)}

    results = []
    for rule_score, row, matched in scored:
        sim = emb_sim.get(row["container_id"], 0.0)
        sim01 = (sim + 1.0) / 2.0
        final = 0.65 * rule_score + 0.35 * sim01 if emb_sim else rule_score
        results.append({
            **row,
            "score": round(final, 4),
            "rule_score": round(rule_score, 4),
            "embedding_similarity": round(sim, 4) if emb_sim else None,
            "explanations": {
                "stage1_symbolic": ("passed structured filters"
                                    if stage1["applied_filters"] else "no filters applied"),
                "stage2_reasoning": matched,
            },
        })
    results.sort(key=lambda r: (-r["score"], r["container_id"]))
    results = results[:max(1, min(k, 100))]

    llm = None
    if use_llm or use_llm is None:
        # auto mode: short-timeout attempt; deterministic ranking is unaffected
        if use_llm:  # explicit request gets a slightly longer timeout
            llm = _llm_rationale(concept, results, timeout_s=6.0)
        else:
            llm = _llm_rationale(concept, results, timeout_s=1.5)

    return {
        "concept": concept,
        "run_id": run.run_id,
        "target": "containers",
        "stage1": stage1,
        "stage2": {
            "stage": "reasoning_scorer",
            "provider": (llm or {}).get("provider", "offline_deterministic"),
            "note": "deterministic lexicon over structured evidence; LLM (when "
                    "reachable) adds rationale text and never reorders results",
        },
        "hybrid": {"rule_weight": 0.65, "embedding_weight": 0.35,
                   "embedding_source": "32-d structural container embeddings"},
        "llm_rationale": (llm or {}).get("rationale"),
        "results": results,
    }


def search_scenarios(concept: str, filters: Optional[Dict] = None, k: int = 12) -> Dict:
    """Same two-stage idea over the scenario database (odd_tags as attributes)."""
    from sensorflow.safety.scenario_db import get_db
    records = get_db().search(limit=1000, odd_tags=None)
    stage1_before = len(records)
    filters = filters or {}
    for key, val in filters.items():
        vals = val if isinstance(val, list) else [val]
        records = [r for r in records
                   if r.odd_tags.get(key) in vals or getattr(r, key, None) in vals]

    results = []
    for r in records:
        row = {  # adapt scenario record to the lexicon's row shape
            "lighting": r.odd_tags.get("lighting", ""),
            "weather": r.odd_tags.get("weather", ""),
            "road_type": r.odd_tags.get("road_type", ""),
            "scenario": r.scenario_type,
            "n_objects": 1, "fn": 0, "fp": 0, "anomalies": 0,
            "safety_n": 1 if r.odd_tags.get("actor_class") in
                             ("pedestrian", "cyclist", "motorcycle") else 0,
            "risk_score": {"low": 0.2, "medium": 0.45, "high": 0.7,
                           "critical": 0.95}.get(r.severity, 0.4),
        }
        score, matched = _rule_score(concept, row)
        text_bonus = 0.15 if re.search(r"\w", concept) and any(
            w in r.description.lower() for w in re.findall(r"[a-z]{4,}", concept.lower())) else 0.0
        results.append({
            "scenario_id": r.scenario_id,
            "scenario_type": r.scenario_type,
            "source": r.source,
            "severity": r.severity,
            "odd_tags": r.odd_tags,
            "description": r.description,
            "score": round(min(1.0, score + text_bonus), 4),
            "explanations": {"stage1_symbolic": filters or "no filters applied",
                             "stage2_reasoning": matched},
        })
    results.sort(key=lambda x: (-x["score"], x["scenario_id"]))
    return {
        "concept": concept,
        "target": "scenarios",
        "stage1": {"stage": "symbolic_filter", "applied_filters": filters,
                   "candidates_before": stage1_before, "candidates_after": len(results)},
        "stage2": {"stage": "reasoning_scorer", "provider": "offline_deterministic"},
        "results": results[:max(1, min(k, 100))],
    }
