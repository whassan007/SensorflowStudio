"""StatisticalRegressionAgent — mostly deterministic.

Establishes, before anyone treats "~0.01%" as real: the denominator, the
sample size, confidence intervals, statistical significance, power/MDE and
small-sample instability.

This agent is a THIN ORCHESTRATOR over sensorflow.seqeval — the anytime-valid
sequential machinery (PairedSequentialTest: empirical-Bernstein confidence
sequence + betting e-processes) and the MDE approximation are imported from
seqeval, not reimplemented. Wilson intervals come from megaeval.sampling
(single source of truth).

Rare-event handling: when event counts are small, the practical-margin
sequential test is honest but low-powered, so significance for the rate
DELTA is additionally established with the exact conditional binomial
(McNemar-exact) test over discordant pairs; both results are reported with
their methods, and the seqeval three-outcome decision is never coerced.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy import stats as sstats

from sensorflow.agentic import data as data_mod
from sensorflow.agentic.agents.base import BaseAgent, compact, no_escalation
from sensorflow.agentic.models import (AgentEscalation, RateEstimate,
                                       StatisticalAssessment)
from sensorflow.megaeval.sampling import wilson_interval
from sensorflow.seqeval.sequential import PairedSequentialTest, approx_mde

ALPHA = 0.05
SEQ_DELTA_MARGIN = 5.0e-5      # practical-significance margin for the flip rate
SMALL_SAMPLE_EVENT_FLOOR = 50


class StatisticalRegressionAgent(BaseAgent):
    name = "statistical_regression"
    version = "1.0.0"
    epistemic_status = "OBSERVED"   # this agent reports measurements

    def assess(self, seed: int = data_mod.DEFAULT_SEED) -> StatisticalAssessment:
        arrays = data_mod.get_rate_arrays(seed)
        n = arrays.n
        b_events = int(arrays.baseline_flip.sum())
        c_events = int(arrays.candidate_flip.sum())
        b_rate, c_rate = b_events / n, c_events / n
        b_lo, b_hi = wilson_interval(b_events, n)
        c_lo, c_hi = wilson_interval(c_events, n)

        # ---- exact rare-event significance (McNemar-exact / conditional
        # binomial over discordant pairs) --------------------------------
        n01 = int((arrays.baseline_flip & ~arrays.candidate_flip).sum())
        n10 = int((~arrays.baseline_flip & arrays.candidate_flip).sum())
        if n01 + n10 > 0:
            exact_p = float(sstats.binomtest(n10, n01 + n10, 0.5,
                                             alternative="greater").pvalue)
        else:
            exact_p = None
        significant = exact_p is not None and exact_p < ALPHA

        # ---- seqeval delegation: anytime-valid paired sequential test ----
        # success = "no flip"; d = candidate - baseline (negative = worse)
        b_success = (~arrays.baseline_flip).astype(np.float64)
        c_success = (~arrays.candidate_flip).astype(np.float64)
        d = c_success - b_success
        cluster_sums = np.bincount(arrays.cluster_id, weights=d)
        cluster_counts = np.bincount(arrays.cluster_id)
        cluster_means = cluster_sums / np.maximum(cluster_counts, 1)

        seq_test = PairedSequentialTest(delta=SEQ_DELTA_MARGIN, alpha=ALPHA)
        seq_test.update_clusters(cluster_means.tolist())
        seq_test.record_objects(b_success.astype(bool), c_success.astype(bool))
        seq_decision = seq_test.evaluate()
        seq_snapshot = seq_test.snapshot()

        # ---- power / MDE at the achieved sample size ---------------------
        nu = (seq_test.n01 + seq_test.n10) / max(seq_test.n_objects, 1)
        est = seq_test.delta_estimate() or 0.0
        var_d = max(nu - est ** 2, 1e-12)
        mde = approx_mde(var_d, n_eff=float(len(cluster_means)),
                         delta=SEQ_DELTA_MARGIN, alpha=ALPHA)
        observed_abs_delta = abs(c_rate - b_rate)
        power_note = (
            "sequential-margin test is UNDERPOWERED at this n for the "
            "observed delta" if (mde is not None and mde > observed_abs_delta)
            else "sequential-margin test adequately powered")

        # ---- small-sample instability flags -------------------------------
        flags = []
        if c_events < SMALL_SAMPLE_EVENT_FLOOR:
            flags.append(f"candidate event count {c_events} < "
                         f"{SMALL_SAMPLE_EVENT_FLOOR}: rate estimate unstable")
        if b_events < 10:
            flags.append(f"baseline event count {b_events} < 10: relative "
                         "risk denominator unstable")
        if (c_hi - c_lo) > c_rate:
            flags.append("candidate CI width exceeds the point estimate")

        return StatisticalAssessment(
            baseline=RateEstimate(events=b_events, denominator=n,
                                  rate=round(b_rate, 8),
                                  wilson_ci=[round(b_lo, 8), round(b_hi, 8)]),
            candidate=RateEstimate(events=c_events, denominator=n,
                                   rate=round(c_rate, 8),
                                   wilson_ci=[round(c_lo, 8), round(c_hi, 8)]),
            absolute_delta=round(c_rate - b_rate, 8),
            relative_delta=(round(c_rate / b_rate, 3) if b_rate > 0 else None),
            significant=significant,
            significance_method=(
                "exact conditional binomial (McNemar-exact) over discordant "
                f"pairs: n10={n10} candidate-only vs n01={n01} baseline-only, "
                f"H0 p=0.5, one-sided"),
            exact_binomial_p=exact_p,
            seqeval={
                "delegated_to": "sensorflow.seqeval.sequential.PairedSequentialTest",
                "test_method": ("paired cluster-mean empirical-Bernstein "
                                "confidence sequence + one-sided betting "
                                "e-process (anytime-valid)"),
                "delta_margin": SEQ_DELTA_MARGIN,
                "alpha": ALPHA,
                "decision": seq_decision,
                "snapshot": seq_snapshot,
                "clusters_fed": int(len(cluster_means)),
                "note": ("three-outcome decision reported as-is; "
                         "INSUFFICIENT_EVIDENCE is never coerced to PASS"),
            },
            power_mde={
                "mde_abs": mde,
                "observed_abs_delta": round(observed_abs_delta, 8),
                "assessment": power_note,
                "method": "sensorflow.seqeval.sequential.approx_mde",
            },
            small_sample_flags=flags,
            rare_event_handling=(
                "event counts are small, so exact binomial inference is used "
                "for the significance claim; the seqeval sequential test is "
                "reported alongside with its own (unforced) decision"),
        )

    def _analyze(self, failure_id: str, seed: int = data_mod.DEFAULT_SEED,
                 **inputs) -> Tuple[Dict[str, Any], float, str, AgentEscalation]:
        assessment = self.assess(seed)
        conf = 0.9 if assessment.significant and not assessment.small_sample_flags \
            else (0.7 if assessment.significant else 0.4)
        return (assessment.model_dump(), conf,
                "deterministic statistics; confidence reflects significance "
                "and sample stability",
                no_escalation())

    def _llm_prompt(self, failure_id, output, **inputs):
        return ("Explain these regression statistics to a launch-review "
                "audience without changing any number: " + compact(output))
