"""Anytime-valid sequential inference for paired mean differences.

Why not repeated fixed-sample tests: a CLT/Wilson interval is valid at ONE
pre-committed n. Recomputing it after every batch and stopping on the first
significant look inflates type-I error without bound (the law of the iterated
logarithm guarantees an eventual false crossing). Everything here is therefore
ANYTIME-VALID: the guarantee holds simultaneously over all sample sizes and
data-dependent stopping rules.

Machinery (Waudby-Smith & Ramdas, "Estimating means of bounded random variables
by betting", JRSS-B 2023):

  * `EmpiricalBernsteinCS` — predictable plug-in empirical-Bernstein confidence
    sequence for a mean of [0,1]-bounded observations. Variance-adaptive: for
    paired differences (variance << 1/4) it is far tighter than Hoeffding.
  * `OneSidedEProcess` — a betting supermartingale (e-process) against a
    one-sided null "mean >= m" or "mean <= m". Ville's inequality gives
    P(e-value ever >= 1/alpha) <= alpha under the null, at any stopping time.
  * `PairedSequentialTest` — combines both on paired differences d in [-1,1]
    (mapped to x=(d+1)/2), producing the mandatory THREE-OUTCOME decision
    against a practical-significance margin delta:
        REGRESSION            e-process against  H0: Delta >= -delta  crossed 1/alpha_reg
        PASS                  e-process against  H0: Delta <= -delta  crossed 1/alpha_pass
                              (an equivalence-style claim: the drop, if any, is
                               confidently smaller than the margin)
        INSUFFICIENT_EVIDENCE neither — never interpreted as proven equivalence.
  * A Beta-Binomial posterior P(Delta < -delta | data) is computed alongside as
    an operational, human-readable complement (it is NOT the gate criterion).

Observations fed in are CLUSTER means (see units.py), so temporal/spatial
correlation is handled by construction rather than by post-hoc correction.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from scipy import stats as _sstats

DECISION_REGRESSION = "REGRESSION"
DECISION_PASS = "PASS"
DECISION_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


class EmpiricalBernsteinCS:
    """Predictable plug-in empirical-Bernstein confidence sequence on [0,1].

    Running-intersection bounds are kept (intersecting a CS over time preserves
    coverage), so the interval never widens as data accrue.
    """

    def __init__(self, alpha: float = 0.05, truncation: float = 0.5):
        self.alpha = alpha
        self.truncation = truncation
        self.t = 0
        self._sum_x = 0.0
        self._mu_hat = 0.5           # mu_hat_0 = 1/2 (prior-regularized running mean)
        self._var_acc = 0.25         # 1/4 + sum_i (x_i - mu_hat_i)^2
        self._sigma2_hat = 0.25
        self._sum_lam = 0.0
        self._sum_lam_x = 0.0
        self._sum_v_psi = 0.0
        self.lower = 0.0
        self.upper = 1.0

    def update(self, x: float) -> None:
        x = min(1.0, max(0.0, float(x)))
        t = self.t + 1
        log2a = math.log(2.0 / self.alpha)
        lam = math.sqrt(2.0 * log2a / (max(self._sigma2_hat, 1e-12) * t * math.log1p(t)))
        lam = min(lam, self.truncation)
        v = 4.0 * (x - self._mu_hat) ** 2
        psi = (-math.log1p(-lam) - lam) / 4.0
        self._sum_lam += lam
        self._sum_lam_x += lam * x
        self._sum_v_psi += v * psi
        # advance predictable estimates AFTER betting with the old ones
        self.t = t
        self._sum_x += x
        self._mu_hat = (0.5 + self._sum_x) / (t + 1)
        self._var_acc += (x - self._mu_hat) ** 2
        self._sigma2_hat = self._var_acc / (t + 1)
        if self._sum_lam > 0:
            center = self._sum_lam_x / self._sum_lam
            radius = (log2a + self._sum_v_psi) / self._sum_lam
            self.lower = max(self.lower, center - radius)
            self.upper = min(self.upper, center + radius)

    def interval(self) -> tuple:
        return (max(0.0, self.lower), min(1.0, self.upper))


class OneSidedEProcess:
    """Betting e-process for a one-sided mean null on [0,1] observations.

    side="below": H0: mu >= m, wealth grows when observations fall below m.
    side="above": H0: mu <= m, wealth grows when observations exceed m.

    Bet sizing: max of the empirical-Bernstein recipe (drives early growth
    while the sample variance estimate is still coarse) and an aGRAPA /
    approximate-Kelly bet lambda ~ drift / (sigma^2 + drift^2) (near-optimal
    asymptotic growth rate ~ KL divergence per observation once a real drift
    emerges). Both are predictable and clipped to the admissible range, so the
    wealth process remains a nonnegative supermartingale under the null — the
    sizing choice affects power only, never validity.
    """

    def __init__(self, m: float, side: str, alpha_ref: float = 0.05,
                 truncation: float = 0.75):
        assert side in ("below", "above")
        self.m = float(m)
        self.side = side
        self.alpha_ref = alpha_ref
        self.truncation = truncation
        self.t = 0
        self.log_e = 0.0
        self._sum_x = 0.0
        self._mu_hat = 0.5
        self._var_acc = 0.25
        self._sigma2_hat = 0.25

    def update(self, x: float) -> None:
        x = min(1.0, max(0.0, float(x)))
        t = self.t + 1
        loga = math.log(1.0 / self.alpha_ref)
        sigma2 = max(self._sigma2_hat, 1e-12)
        lam_eb = math.sqrt(2.0 * loga / (sigma2 * t * math.log1p(t)))
        drift = (self.m - self._mu_hat) if self.side == "below" else (self._mu_hat - self.m)
        lam_kelly = max(drift, 0.0) / (sigma2 + drift * drift)
        lam = max(lam_eb, lam_kelly)
        if self.side == "below":
            lam = min(lam, self.truncation / max(1.0 - self.m, 1e-6))
            factor = 1.0 - lam * (x - self.m)
        else:
            lam = min(lam, self.truncation / max(self.m, 1e-6))
            factor = 1.0 + lam * (x - self.m)
        self.log_e += math.log(max(factor, 1e-12))
        self.t = t
        self._sum_x += x
        self._mu_hat = (0.5 + self._sum_x) / (t + 1)
        self._var_acc += (x - self._mu_hat) ** 2
        self._sigma2_hat = self._var_acc / (t + 1)

    @property
    def e_value(self) -> float:
        return math.exp(min(self.log_e, 700.0))

    def p_analogue(self) -> float:
        """Anytime-valid p-value analogue: min(1, 1/e)."""
        return min(1.0, 1.0 / max(self.e_value, 1e-300))


class PairedSequentialTest:
    """Three-outcome anytime-valid test on paired differences d in [-1, 1].

    Delta = E[d] (candidate minus baseline). Regression margin `delta` in
    absolute metric points: the H0 boundary sits at Delta = -delta.
    """

    def __init__(self, delta: float = 0.005, alpha: float = 0.05,
                 alpha_pass: Optional[float] = None):
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.alpha_pass = float(alpha_pass if alpha_pass is not None else alpha)
        m = (1.0 - self.delta) / 2.0  # Delta = -delta  <=>  mu = (1-delta)/2
        self.cs = EmpiricalBernsteinCS(alpha=self.alpha)
        self.e_reg = OneSidedEProcess(m, side="below", alpha_ref=self.alpha)
        self.e_pass = OneSidedEProcess(m, side="above", alpha_ref=self.alpha_pass)
        # object-level accounting (for reporting + Bayesian complement)
        self.n_objects = 0
        self.n_clusters = 0
        self.sum_baseline = 0.0
        self.sum_candidate = 0.0
        self.n01 = 0  # baseline correct, candidate wrong  (regression pairs)
        self.n10 = 0  # candidate correct, baseline wrong  (improvement pairs)
        self.decision = DECISION_INSUFFICIENT
        self.decided_at_n: Optional[int] = None
        self.trajectory: List[Dict] = []

    # ---- updates -------------------------------------------------------

    def update_clusters(self, cluster_means) -> None:
        """Feed per-cluster mean paired differences (each in [-1, 1])."""
        for d in cluster_means:
            x = (float(d) + 1.0) / 2.0
            self.cs.update(x)
            self.e_reg.update(x)
            self.e_pass.update(x)
            self.n_clusters += 1

    def record_objects(self, b, c) -> None:
        """Track object-level paired outcomes for counts and the posterior."""
        import numpy as np
        b = np.asarray(b, dtype=bool)
        c = np.asarray(c, dtype=bool)
        self.n_objects += int(b.size)
        self.sum_baseline += float(b.sum())
        self.sum_candidate += float(c.sum())
        self.n01 += int(np.sum(b & ~c))
        self.n10 += int(np.sum(~b & c))

    # ---- state ---------------------------------------------------------

    def delta_interval(self) -> tuple:
        lo, hi = self.cs.interval()
        return (2.0 * lo - 1.0, 2.0 * hi - 1.0)

    def delta_estimate(self) -> Optional[float]:
        if self.n_objects == 0:
            return None
        return (self.sum_candidate - self.sum_baseline) / self.n_objects

    def bayes_p_regression(self) -> Optional[float]:
        """Beta-Binomial posterior P(Delta < -delta | data).

        McNemar decomposition: among D = n01 + n10 discordant pairs let
        theta = P(pair is a 01 regression pair). With plug-in discordance rate
        nu = D/n, Delta = nu * (1 - 2*theta), so
            Delta < -delta  <=>  theta > (1 + delta*n/D) / 2.
        Prior theta ~ Beta(1,1).
        """
        d_total = self.n01 + self.n10
        if self.n_objects == 0 or d_total == 0:
            return None
        threshold = (1.0 + self.delta * self.n_objects / d_total) / 2.0
        if threshold >= 1.0:
            return 0.0
        return float(_sstats.beta.sf(threshold, 1 + self.n01, 1 + self.n10))

    def evaluate(self, alpha_reg: Optional[float] = None,
                 alpha_pass: Optional[float] = None) -> str:
        """Update the sticky three-outcome decision at the given levels."""
        if self.decision != DECISION_INSUFFICIENT:
            return self.decision
        a_r = alpha_reg if alpha_reg is not None else self.alpha
        a_p = alpha_pass if alpha_pass is not None else self.alpha_pass
        if self.e_reg.e_value >= 1.0 / a_r:
            self.decision = DECISION_REGRESSION
            self.decided_at_n = self.n_objects
        elif self.e_pass.e_value >= 1.0 / a_p:
            self.decision = DECISION_PASS
            self.decided_at_n = self.n_objects
        return self.decision

    def snapshot(self) -> Dict:
        lo, hi = self.delta_interval()
        est = self.delta_estimate()
        return {
            "n": self.n_objects,
            "n_clusters": self.n_clusters,
            "delta_estimate": None if est is None else round(est, 5),
            "delta_ci": [round(lo, 5), round(hi, 5)],
            "e_regression": round(self.e_reg.e_value, 4),
            "e_pass": round(self.e_pass.e_value, 4),
            "p_analogue": round(self.e_reg.p_analogue(), 6),
            "bayes_p_regression": (None if self.bayes_p_regression() is None
                                   else round(self.bayes_p_regression(), 5)),
            "decision": self.decision,
        }

    def record_trajectory_point(self) -> None:
        """Append an evidence-vs-samples point (for the dashboard chart)."""
        self.trajectory.append({
            "n": self.n_objects,
            "n_clusters": self.n_clusters,
            "delta_estimate": self.delta_estimate(),
            "delta_lower": round(self.delta_interval()[0], 5),
            "delta_upper": round(self.delta_interval()[1], 5),
            "log_e_regression": round(self.e_reg.log_e, 4),
            "log_e_pass": round(self.e_pass.log_e, 4),
            "decision": self.decision,
        })


def approx_mde(var_d: float, n_eff: float, delta: float, alpha: float,
               power: float = 0.9, anytime_inflation: float = 1.4) -> Optional[float]:
    """Approximate minimum detectable absolute effect at the current n_eff.

    Fixed-n normal approximation (z_{1-a} + z_{power}) * sqrt(var/n_eff),
    inflated for the anytime-valid overhead, plus the practical margin delta
    (effects must exceed the margin to be declarable)."""
    if n_eff <= 1 or var_d <= 0:
        return None
    z = _sstats.norm.ppf(1 - alpha) + _sstats.norm.ppf(power)
    return float(delta + anytime_inflation * z * math.sqrt(var_d / n_eff))
