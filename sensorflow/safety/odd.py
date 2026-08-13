"""ODD coverage analysis (ISO 34503 / ASAM OpenODD-inspired).

Industry concept mapping
------------------------
ISO 34503 defines an Operational Design Domain taxonomy in three top-level
categories — scenery, environmental conditions, dynamic elements. ASAM OpenODD
standardizes machine-readable ODD definitions. This module:

1. exposes an ODD taxonomy built on the megaeval population dimensions (so
   coverage is computed against the same vocabulary the metric cube uses),
2. computes *combinatorial coverage* of the ODD space for an evaluation run:
   which cells have data, how much, and with what quality (recall + Wilson CI),
3. identifies statistically under-covered gaps (too few samples, or a recall
   CI too wide to support any claim),
4. ranks gaps by risk = production-like frequency x performance deficit,
5. emits synthetic gap-filling scenario REQUESTS and can actually fulfil them
   through the existing synthetic generator (sensorflow.evaluation.synthetic),
   folding the generated objects back into coverage as a clearly-marked
   synthetic supplement.

Honest markers: the "production-like distribution" is the product of the
population's per-dimension marginals (independence assumption, documented in
the response); gap filling uses the deterministic synthetic generator, and the
generator does not model road geometry — road_type on a supplement is a
recorded request attribute, not simulated physics.
"""

from __future__ import annotations

import itertools
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sensorflow.megaeval import cube as cube_mod
from sensorflow.megaeval import population as pop_mod
from sensorflow.megaeval.population import DIMENSIONS, DIM_NAMES
from sensorflow.megaeval.sampling import wilson_interval
from sensorflow.safety.store import read_json, write_json

DEFAULT_COVERAGE_DIMS = ["weather", "lighting", "road_type", "class"]
DEFAULT_MIN_SAMPLES = 50
DEFAULT_MAX_CI_WIDTH = 0.25

# ISO 34503-style categorization of the dimensions we can instrument.
ISO34503_CATEGORY = {
    "road_type": "scenery",
    "weather": "environmental_conditions",
    "lighting": "environmental_conditions",
    "occlusion": "environmental_conditions",
    "class": "dynamic_elements",
    "speed_band": "dynamic_elements",
    "scenario": "dynamic_elements",
    "distance_band": "dynamic_elements",
    "sensor": "sensor_configuration",
}

# megaeval class vocabulary -> labeleval synthetic-generator class vocabulary.
_CLASS_TO_GENERATOR = {
    "vehicle": "vehicle",
    "pedestrian": "pedestrian",
    "cyclist": "cyclist",
    "motorcycle": "motorcycle",
    "truck": "truck",
    "bus": "truck",  # generator has no bus prior; truck is the closest proxy
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def taxonomy() -> Dict:
    """Machine-readable ODD taxonomy (ISO 34503 categories, ASAM OpenODD-ish)."""
    dims = {}
    for name in DIM_NAMES:
        dims[name] = {
            "values": DIMENSIONS[name],
            "iso34503_category": ISO34503_CATEGORY.get(name, "dynamic_elements"),
            "instrumented": True,
            "source": "megaeval population dimension",
        }
    # Declared but not instrumented in the current synthetic population.
    dims["geography"] = {
        "values": ["urban_california", "suburban_california", "highway_california",
                   "rural_california"],
        "iso34503_category": "scenery",
        "instrumented": False,
        "source": "placeholder — the current synthetic population does not carry "
                  "a geography attribute; excluded from coverage computation",
    }
    return {
        "standard_basis": "ISO 34503 ODD taxonomy structure (scenery / "
                          "environmental conditions / dynamic elements), "
                          "ASAM OpenODD-style machine-readable attributes",
        "dimensions": dims,
        "default_coverage_dims": DEFAULT_COVERAGE_DIMS,
        "notes": [
            "Coverage cells are the cartesian product of the selected dimensions.",
            "Only instrumented dimensions may be used for coverage computation.",
        ],
    }


# ------------------------------------------------------------------ core math


def _marginals(dim_counts: Optional[Dict], cube_df, dims: List[str]) -> Dict[str, Dict[str, float]]:
    """Per-dimension marginal probabilities of the production-like distribution.

    Prefers the population dataset card (exact counts); falls back to the
    cube's own n if no card is available (pure-cube unit tests).
    """
    out: Dict[str, Dict[str, float]] = {}
    for dim in dims:
        if dim_counts and dim in dim_counts:
            counts = {v: float(c) for v, c in dim_counts[dim].items()}
        else:
            grp = cube_df.groupby(dim, observed=True)["n"].sum()
            counts = {DIMENSIONS[dim][int(code)]: float(n) for code, n in grp.items()}
        total = sum(counts.values()) or 1.0
        out[dim] = {v: counts.get(v, 0.0) / total for v in DIMENSIONS[dim]}
    return out


def cell_id(cell: Dict[str, str], dims: List[str]) -> str:
    return "|".join(f"{d}={cell[d]}" for d in dims)


def compute_coverage(
    cube_df,
    dims: Optional[List[str]] = None,
    dim_counts: Optional[Dict] = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    max_ci_width: float = DEFAULT_MAX_CI_WIDTH,
    target_recall: Optional[float] = None,
    supplements: Optional[List[Dict]] = None,
    max_gaps: int = 50,
) -> Dict:
    """Combinatorial ODD coverage from a metric cube (pure function, exact math).

    Per cell: n (ground-truth objects), tp/fn, recall with Wilson CI,
    adequacy (n >= min_samples AND CI width <= max_ci_width), production share
    (product of marginals — independence assumption), performance deficit
    (target_recall - Wilson lower bound, floored at 0) and
    risk = production_share x deficit.
    """
    dims = [d for d in (dims or DEFAULT_COVERAGE_DIMS) if d in DIM_NAMES]
    if not dims:
        raise ValueError("no valid ODD dimensions selected")
    n_cells = math.prod(len(DIMENSIONS[d]) for d in dims)
    if n_cells > 2000:
        raise ValueError(f"selected dimensions produce {n_cells} cells (max 2000); "
                         "choose fewer dimensions")

    rows, _ = cube_mod.aggregate(cube_df, None, dims, ["n", "tp", "fn", "recall", "mean_iou"],
                                 limit=2000)
    observed = {tuple(r[d] for d in dims): r for r in rows}

    overall_rows, _ = cube_mod.aggregate(cube_df, None, None, ["recall"], 1)
    overall_recall = overall_rows[0].get("recall") if overall_rows else None
    if target_recall is None:
        target_recall = overall_recall if overall_recall is not None else 0.9

    marg = _marginals(dim_counts, cube_df, dims)

    # Index synthetic supplements by cell id (only those matching these dims).
    supp_by_cell: Dict[str, Dict] = {}
    for s in supplements or []:
        c = s.get("cell") or {}
        if set(c.keys()) == set(dims):
            cid = cell_id(c, dims)
            agg = supp_by_cell.setdefault(cid, {"n_added": 0, "tp_added": 0, "fn_added": 0,
                                                "datasets": []})
            agg["n_added"] += int(s.get("n_added", 0))
            agg["tp_added"] += int(s.get("tp_added", 0))
            agg["fn_added"] += int(s.get("fn_added", 0))
            if s.get("dataset_id"):
                agg["datasets"].append(s["dataset_id"])

    cells: List[Dict] = []
    for combo in itertools.product(*(DIMENSIONS[d] for d in dims)):
        cell = dict(zip(dims, combo))
        cid = cell_id(cell, dims)
        r = observed.get(combo, {})
        n = int(r.get("n", 0))
        tp = int(r.get("tp", 0))
        fn = int(r.get("fn", 0))

        supp = supp_by_cell.get(cid)
        if supp:
            n += supp["n_added"]
            tp += supp["tp_added"]
            fn += supp["fn_added"]

        gt_n = tp + fn
        recall = (tp / gt_n) if gt_n > 0 else None
        lo, hi = wilson_interval(tp, gt_n)
        ci_width = hi - lo
        share = math.prod(marg[d].get(cell[d], 0.0) for d in dims)

        reasons = []
        if n < min_samples:
            reasons.append("insufficient_samples")
        if ci_width > max_ci_width:
            reasons.append("ci_too_wide")
        adequate = not reasons
        deficit = max(0.0, float(target_recall) - lo)
        risk = share * deficit

        rec = {
            "cell_id": cid,
            "cell": cell,
            "n": n,
            "tp": tp,
            "fn": fn,
            "recall": None if recall is None else round(recall, 4),
            "mean_iou": r.get("mean_iou"),
            "wilson_ci": [round(lo, 4), round(hi, 4)],
            "ci_width": round(ci_width, 4),
            "production_share": round(share, 6),
            "adequate": adequate,
            "is_gap": not adequate,
            "gap_reasons": reasons,
            "performance_deficit": round(deficit, 4),
            "risk": round(risk, 6),
        }
        if supp:
            rec["synthetic_supplement"] = supp
        cells.append(rec)

    populated = [c for c in cells if c["n"] > 0]
    adequate_cells = [c for c in cells if c["adequate"]]
    gaps = sorted((c for c in cells if c["is_gap"]), key=lambda c: -c["risk"])

    total_share = sum(c["production_share"] for c in cells) or 1.0
    weighted_cov = sum(c["production_share"] for c in adequate_cells) / total_share

    fill_requests = [gap_fill_request(g, min_samples) for g in gaps[:max_gaps]]

    return {
        "dims": dims,
        "thresholds": {"min_samples": min_samples, "max_ci_width": max_ci_width,
                       "target_recall": round(float(target_recall), 4)},
        "summary": {
            "total_cells": len(cells),
            "populated_cells": len(populated),
            "empty_cells": len(cells) - len(populated),
            "adequate_cells": len(adequate_cells),
            "gap_cells": len(gaps),
            "coverage_rate": round(len(adequate_cells) / len(cells), 4),
            "production_weighted_coverage": round(weighted_cov, 4),
            "overall_recall": overall_recall,
        },
        "gaps": gaps[:max_gaps],
        "fill_requests": fill_requests,
        "cells": cells,
        "method": "combinatorial cell coverage over the metric cube; recall CI = "
                  "Wilson; production share = product of population marginals "
                  "(independence assumption); risk = share x (target_recall - "
                  "Wilson recall lower bound)",
    }


def gap_fill_request(gap: Dict, min_samples: int) -> Dict:
    """A scenario REQUEST the synthetic generator can fulfil for one gap cell."""
    cell = gap["cell"]
    needed = max(0, min_samples - gap["n"])
    cls = cell.get("class")
    return {
        "cell_id": gap["cell_id"],
        "cell": cell,
        "needed_samples": needed,
        "risk": gap["risk"],
        "generator": "sensorflow.evaluation.synthetic.generate_dataset",
        "request": {
            "weather": cell.get("weather"),
            "time_of_day": ("night" if cell.get("lighting") == "night"
                            else cell.get("lighting", "day")),
            "target_class": _CLASS_TO_GENERATOR.get(cls) if cls else None,
            "target_class_note": ("bus mapped to truck proxy" if cls == "bus" else None),
            "road_type": cell.get("road_type"),
            "road_type_note": "recorded request attribute; the synthetic generator "
                              "does not model road geometry",
            "num_sequences": 2,
            "frames_per_sequence": max(10, min(40, needed)),
        },
    }


# ------------------------------------------------------------------ run plumbing


def _supplements_key(run_id: str) -> Tuple[str, str]:
    return ("odd", f"{run_id}.json")


def load_supplements(run_id: str) -> List[Dict]:
    return read_json(*_supplements_key(run_id)) or []


def coverage_for_run(store, run, dims: Optional[List[str]] = None,
                     min_samples: int = DEFAULT_MIN_SAMPLES,
                     max_ci_width: float = DEFAULT_MAX_CI_WIDTH,
                     target_recall: Optional[float] = None,
                     include_cells: bool = False,
                     max_gaps: int = 50) -> Dict:
    """ODD coverage for a published megaeval run (uses cube + dataset card)."""
    art = store.artifacts(run.run_id)
    meta = pop_mod.load_meta(run.population_id) or {}
    result = compute_coverage(
        art["cube"], dims=dims, dim_counts=meta.get("dim_counts"),
        min_samples=min_samples, max_ci_width=max_ci_width,
        target_recall=target_recall, supplements=load_supplements(run.run_id),
        max_gaps=max_gaps)
    result["run_id"] = run.run_id
    result["population_id"] = run.population_id
    result["model_version"] = run.model_version
    if not include_cells:
        result.pop("cells")
    return result


def fill_gap(store, run, cell: Dict[str, str], num_sequences: int = 2,
             frames_per_sequence: int = 20, seed: Optional[int] = None) -> Dict:
    """Generate targeted synthetic data for a gap cell and re-compute coverage.

    Actually wired: calls the existing synthetic generator + label generator on
    the labeleval store, retargets the generated frames' conditions to the gap
    cell, measures how many ground-truth objects of the target class were added
    and how many the simulated model labeled, persists the supplement, and
    returns before/after coverage for that cell.
    """
    from sensorflow.evaluation import synthetic
    from sensorflow.evaluation.process_units import ProcessMeter
    from sensorflow.evaluation.records import get_store as get_legacy_store

    dims = [d for d in DIM_NAMES if d in cell]
    if not dims:
        raise ValueError("cell must specify at least one known ODD dimension")
    for d in dims:
        if cell[d] not in DIMENSIONS[d]:
            raise ValueError(f"unknown value {cell[d]!r} for dimension {d}")
    cell = {d: cell[d] for d in dims}

    before = coverage_for_run(store, run, dims=dims, include_cells=True)
    cid = cell_id(cell, dims)
    before_cell = next(c for c in before["cells"] if c["cell_id"] == cid)

    legacy = get_legacy_store()
    seed = seed if seed is not None else (run.seed + len(load_supplements(run.run_id))) % (2**31)
    dataset = synthetic.generate_dataset(
        legacy, name=f"odd-gap-{cid}", num_sequences=num_sequences,
        frames_per_sequence=frames_per_sequence, seed=seed,
        version="gap-fill", generated_from_model=run.model_version)

    # Retarget conditions to the requested cell (deterministic post-adjustment;
    # the generator itself does not take weather/lighting parameters).
    weather = cell.get("weather")
    lighting = cell.get("lighting")
    for frame in legacy.where("frames", dataset_id=dataset.dataset_id):
        if weather:
            frame.weather = weather
            if weather != "clear" and "adverse_weather" not in frame.scenario_tags:
                frame.scenario_tags.append("adverse_weather")
        if lighting:
            frame.time_of_day = "night" if lighting == "night" else lighting
            if lighting == "night" and "night_glare" not in frame.scenario_tags:
                frame.scenario_tags.append("night_glare")
        if "odd_gap_fill" not in frame.scenario_tags:
            frame.scenario_tags.append("odd_gap_fill")
        legacy.put("frames", frame)

    annotations = synthetic.generate_labels(legacy, dataset,
                                            model_version=run.model_version)
    labeled_gt = {a.matched_gt_id for a in annotations if a.matched_gt_id}

    target_class = _CLASS_TO_GENERATOR.get(cell.get("class", ""), None)
    n_added = tp_added = 0
    for frame in legacy.where("frames", dataset_id=dataset.dataset_id):
        for gt in frame.gt_boxes:
            if target_class is not None and gt.class_name != target_class:
                continue
            n_added += 1
            if gt.gt_id in labeled_gt:
                tp_added += 1
    fn_added = n_added - tp_added

    supplement = {
        "cell_id": cid,
        "cell": cell,
        "dataset_id": dataset.dataset_id,
        "n_added": n_added,
        "tp_added": tp_added,
        "fn_added": fn_added,
        "seed": seed,
        "source": "synthetic",
        "created_at": _now(),
        "note": "synthetic supplement generated by sensorflow.evaluation.synthetic; "
                "counted separately from the immutable evaluation population",
    }
    supplements = load_supplements(run.run_id)
    supplements.append(supplement)
    write_json(supplements, *_supplements_key(run.run_id))

    ProcessMeter(legacy, run_id=run.run_id).record("odd_gap_fill", n_added, factor=1.2)
    legacy.audit("odd_gap_filled", "Dataset", dataset.dataset_id,
                 f"cell={cid} n_added={n_added}")
    legacy.save()

    # Register the generated scenario in the scenario database.
    try:
        from sensorflow.safety import scenario_db
        scenario_db.get_db().add_gap_fill(run.run_id, cell, dataset.dataset_id, n_added)
    except Exception:
        pass

    after = coverage_for_run(store, run, dims=dims, include_cells=True)
    after_cell = next(c for c in after["cells"] if c["cell_id"] == cid)
    after.pop("cells")

    return {
        "run_id": run.run_id,
        "cell_id": cid,
        "cell": cell,
        "generated_dataset_id": dataset.dataset_id,
        "objects_added": n_added,
        "labeled_added": tp_added,
        "supplement": supplement,
        "cell_before": before_cell,
        "cell_after": after_cell,
        "coverage_after": after,
    }
