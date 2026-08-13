"""Distribution + concentration analysis (deterministic).

Determines whether a failure pattern is uniform or concentrated across the
instrumented dimensions (construction zone, lighting, geography), computing
per-stratum failure rate vs the population baseline with relative risk, odds
ratio, Wilson CI, absolute risk difference and small-sample instability flags.

All statistics are exact deterministic computations; the Wilson interval is
reused from megaeval (single source of truth for that formula).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from sensorflow.agentic import data as data_mod
from sensorflow.agentic.models import ConcentrationAnalysis, StratumRisk
from sensorflow.megaeval.sampling import wilson_interval

MIN_EVENTS_FOR_STABLE_ESTIMATE = 5
RELATIVE_RISK_CONCENTRATION_THRESHOLD = 3.0


def _stratum_row(dim: str, name: str, mask: np.ndarray, flips: np.ndarray,
                 overall_rate: float) -> StratumRisk:
    exposure = int(mask.sum())
    events = int(flips[mask].sum())
    rate = events / exposure if exposure else 0.0
    lo, hi = wilson_interval(events, exposure)

    out_events = int(flips[~mask].sum())
    out_exposure = int((~mask).sum())
    out_rate = out_events / out_exposure if out_exposure else 0.0

    rr = (rate / out_rate) if out_rate > 0 else None
    # odds ratio with Haldane-Anscombe 0.5 correction when any cell is zero
    a, b = events, exposure - events
    c, d = out_events, out_exposure - out_events
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds = (a / b) / (c / d) if b > 0 and d > 0 and c > 0 else None

    return StratumRisk(
        dimension=dim, stratum=name,
        exposure=exposure,
        exposure_share=round(exposure / flips.size, 4),
        events=events,
        stratum_rate=round(rate, 8),
        baseline_rate=round(overall_rate, 8),
        relative_risk=None if rr is None else round(rr, 3),
        odds_ratio=None if odds is None else round(odds, 3),
        risk_difference=round(rate - out_rate, 8),
        rate_wilson_ci=[round(lo, 8), round(hi, 8)],
        small_sample_flag=events < MIN_EVENTS_FOR_STABLE_ESTIMATE,
    )


def analyze_concentration(failure_id: str, seed: int = data_mod.DEFAULT_SEED
                          ) -> ConcentrationAnalysis:
    arrays = data_mod.get_rate_arrays(seed)
    flips = arrays.candidate_flip
    overall_rate = float(flips.mean())

    strata: List[StratumRisk] = []
    dims: Dict[str, List] = {
        "construction_zone": [("construction", arrays.construction),
                              ("non_construction", ~arrays.construction)],
        "lighting": [("night", arrays.night), ("day", ~arrays.night)],
        "geo_bucket": [(data_mod.GEO_BUCKETS[i], arrays.geo == i)
                       for i in range(len(data_mod.GEO_BUCKETS))],
    }
    for dim, entries in dims.items():
        for name, mask in entries:
            strata.append(_stratum_row(dim, name, mask, flips, overall_rate))

    total_events = int(flips.sum())
    if total_events < MIN_EVENTS_FOR_STABLE_ESTIMATE:
        determination = "insufficient_data"
        concentrated = []
    else:
        concentrated = sorted({
            s.dimension for s in strata
            if s.relative_risk is not None
            and s.relative_risk >= RELATIVE_RISK_CONCENTRATION_THRESHOLD
            and not s.small_sample_flag})
        determination = "concentrated" if concentrated else "uniform"

    return ConcentrationAnalysis(
        failure_id=failure_id,
        determination=determination,
        concentrated_dimensions=concentrated,
        strata=strata,
        method=(
            "per-stratum rate vs complement with relative risk, "
            "Haldane-Anscombe-corrected odds ratio, Wilson CI and absolute "
            f"risk difference; a dimension is 'concentrated' when RR >= "
            f"{RELATIVE_RISK_CONCENTRATION_THRESHOLD} on a stratum with >= "
            f"{MIN_EVENTS_FOR_STABLE_ESTIMATE} events"),
    )
