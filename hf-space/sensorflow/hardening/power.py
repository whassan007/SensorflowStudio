"""Sample-size derivation for evaluation sizing (audit F-010 support).

The platform's evaluation tiers previously used fixed counts with no power
justification. This module derives the required n from the actual decision
parameters, so tier sizes follow from (baseline rate, MDE, alpha, power,
prevalence, pairing correlation) instead of tradition.

Sequential/anytime aspects are OUT of scope here by design: for continuous
monitoring, delegate to sensorflow.seqeval (e-processes remain valid under
optional stopping; the fixed-n formulas below do not).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

# Inverse standard normal CDF (Acklam's rational approximation, |err| < 1.2e-8;
# avoids a scipy dependency for a handful of quantiles).
def _norm_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def required_n_two_proportions(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> int:
    """Required n PER ARM to detect an absolute change `mde` in a proportion.

    Standard two-sample proportion formula:

        n = (z_{1-alpha[/2]} * sqrt(2 * p_bar * (1 - p_bar))
             + z_{power} * sqrt(p1*(1-p1) + p2*(1-p2)))^2 / mde^2

    where p1 = baseline_rate, p2 = p1 +/- mde (worse direction),
    p_bar = (p1 + p2) / 2. Larger MDE -> smaller n; smaller alpha or higher
    power -> larger n.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1)")
    if mde <= 0:
        raise ValueError("mde must be positive")
    p1 = baseline_rate
    p2 = min(max(p1 + mde, 1e-9), 1 - 1e-9)
    p_bar = (p1 + p2) / 2
    z_a = _norm_ppf(1 - alpha / (2 if two_sided else 1))
    z_b = _norm_ppf(power)
    num = (z_a * math.sqrt(2 * p_bar * (1 - p_bar))
           + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / mde ** 2)


def required_n_paired(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    pairing_correlation: float = 0.0,
) -> int:
    """Required PAIRS when both models are evaluated on the SAME items.

    Pairing reduces the variance of the difference by (1 - rho):

        var(p2_hat - p1_hat) ~ [p1(1-p1) + p2(1-p2) - 2*rho*sd1*sd2] / n

    so n_paired ~ n_unpaired * (1 - rho) for sd1 ~ sd2. Positive correlation
    between paired outcomes (the common case: both models fail on the same
    hard frames) strictly reduces the required n; rho=0 recovers the
    independent formula.
    """
    if not -1 < pairing_correlation < 1:
        raise ValueError("pairing_correlation must be in (-1, 1)")
    n_indep = required_n_two_proportions(baseline_rate, mde, alpha, power)
    return max(1, math.ceil(n_indep * (1 - pairing_correlation)))


def required_events_rare(
    prevalence: float,
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> Dict[str, int]:
    """Sizing when the metric only exists on a rare subpopulation.

    If the failure mode occurs on a slice with `prevalence` in the fleet
    stream, the n from the proportion formula applies to SLICE items, so the
    raw stream requirement inflates by 1/prevalence:

        n_slice  = required_n_two_proportions(...)
        n_stream = ceil(n_slice / prevalence)

    This is exactly why stratified oversampling (sampling.py) exists: mine
    slice items directly instead of paying the 1/prevalence stream tax.
    """
    if not 0 < prevalence <= 1:
        raise ValueError("prevalence must be in (0, 1]")
    n_slice = required_n_two_proportions(baseline_rate, mde, alpha, power)
    return {"n_slice": n_slice, "n_stream": math.ceil(n_slice / prevalence)}


def cluster_design_effect(mean_cluster_size: float, icc: float) -> float:
    """Kish design effect for cluster-correlated data:

        DEFF = 1 + (m_bar - 1) * ICC

    Frames within a driving sequence are correlated (same scene, weather,
    model state); effective n = n / DEFF. Use rca.stats for ICC estimation.
    """
    if mean_cluster_size < 1:
        raise ValueError("mean_cluster_size must be >= 1")
    if not 0 <= icc <= 1:
        raise ValueError("icc must be in [0, 1]")
    return 1.0 + (mean_cluster_size - 1.0) * icc


@dataclass
class TierSpec:
    name: str
    purpose: str
    baseline_rate: float
    mde: float
    alpha: float
    power: float
    prevalence: float = 1.0
    pairing_correlation: float = 0.0
    mean_cluster_size: float = 1.0
    icc: float = 0.0


def tier_sizing(spec: TierSpec) -> Dict:
    """Derived (never fixed) size for one evaluation tier.

    Pipeline: paired proportion n -> inflate by cluster DEFF -> inflate by
    1/prevalence for the raw stream. All inputs and intermediate numbers are
    returned so the sizing is auditable.
    """
    n_pairs = required_n_paired(spec.baseline_rate, spec.mde, spec.alpha,
                                spec.power, spec.pairing_correlation)
    deff = cluster_design_effect(spec.mean_cluster_size, spec.icc)
    n_effective = math.ceil(n_pairs * deff)
    n_stream = math.ceil(n_effective / spec.prevalence)
    return {
        "tier": spec.name,
        "purpose": spec.purpose,
        "inputs": {
            "baseline_rate": spec.baseline_rate, "mde": spec.mde,
            "alpha": spec.alpha, "power": spec.power,
            "prevalence": spec.prevalence,
            "pairing_correlation": spec.pairing_correlation,
            "mean_cluster_size": spec.mean_cluster_size, "icc": spec.icc,
        },
        "n_pairs": n_pairs,
        "design_effect": round(deff, 4),
        "n_slice_items": n_effective,
        "n_stream_items": n_stream,
        "sequential_note": "For continuous monitoring at this tier, use "
                           "sensorflow.seqeval; these are fixed-n sizes.",
    }


def default_tiers() -> Dict[str, TierSpec]:
    """Tier 0-3 sizing driven by decision parameters, not fixed counts.

    The parameter choices are explicit policy: smoke tests accept coarse
    MDEs; launch gates demand small MDEs on rare safety slices.
    """
    return {
        "tier0": TierSpec("tier0", "smoke: catch gross breakage fast",
                          baseline_rate=0.90, mde=0.10, alpha=0.05, power=0.80,
                          pairing_correlation=0.5),
        "tier1": TierSpec("tier1", "PR gate: moderate regressions on common slices",
                          baseline_rate=0.90, mde=0.03, alpha=0.05, power=0.80,
                          pairing_correlation=0.5, mean_cluster_size=20, icc=0.1),
        "tier2": TierSpec("tier2", "release candidate: small regressions incl. rare slices",
                          baseline_rate=0.95, mde=0.01, alpha=0.01, power=0.90,
                          prevalence=0.05, pairing_correlation=0.5,
                          mean_cluster_size=20, icc=0.1),
        "tier3": TierSpec("tier3", "launch gate: safety-critical slices",
                          baseline_rate=0.99, mde=0.005, alpha=0.01, power=0.95,
                          prevalence=0.01, pairing_correlation=0.5,
                          mean_cluster_size=20, icc=0.1),
    }
