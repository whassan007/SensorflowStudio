"""Hierarchical gatekeeping multiple-testing controller.

Four levels of hypotheses, each with its own alpha budget (budgets sum to the
family alpha, so the union bound gives family-wise control across levels):

    L1  overall population delta                (single test)
    L2  per-class deltas                        (e-BH within level)
    L3  per class x condition strata            (e-BH within level)
    L4  difficulty bands (easy/medium/hard)     (e-BH within level)

Within a level, discoveries are controlled with e-BH (Benjamini-Hochberg on
e-values, Wang & Ramdas 2022): sort e-values descending and reject the k
hypotheses with e_(k) >= M / (k * alpha_level). e-BH is valid under ARBITRARY
dependence — essential here because the levels aggregate the same units and
the strata share scenes — and composes cleanly with optional stopping because
an e-process value at any stopping time is a valid e-value.

Safety-critical strata are PRE-REGISTERED PRIMARIES: each gets its own reserved
alpha and is tested individually, outside e-BH. They can therefore never be
masked by an improving overall metric or by many well-behaved sibling strata
(this is what makes the "overall improved, pedestrian-night regressed" case
detectable by construction).

Deltas are candidate-minus-baseline, so all rejection decisions are REGRESSION
claims; PASS is the separate equivalence-style claim per node (e_pass at the
node's pass alpha). Decisions are sticky: an anytime-valid rejection stays
rejected as more data arrive.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from sensorflow.seqeval import units as units_mod
from sensorflow.seqeval.sequential import (DECISION_INSUFFICIENT, DECISION_PASS,
                                           DECISION_REGRESSION, PairedSequentialTest)

LEVELS = {1: "overall", 2: "class", 3: "stratum", 4: "difficulty"}

# Default alpha budget shares (of the family alpha).
DEFAULT_ALPHA_SHARES = {
    "overall": 0.30,
    "safety": 0.30,     # split equally among pre-registered safety primaries
    "class": 0.15,
    "stratum": 0.15,
    "difficulty": 0.10,
}

MULTIPLE_TESTING_METHOD = ("hierarchical alpha budgeting (L1 overall / L2 class / "
                           "L3 stratum / L4 difficulty) with e-BH within levels; "
                           "safety primaries pre-registered at reserved alpha")


class Node:
    def __init__(self, key: str, level: int, delta: float, alpha_family: float,
                 cls_code: Optional[int] = None, cond_code: Optional[int] = None,
                 band: Optional[int] = None, weighted: bool = False,
                 safety_primary: bool = False):
        self.key = key
        self.level = level
        self.cls_code = cls_code
        self.cond_code = cond_code
        self.band = band
        self.weighted = weighted
        self.safety_primary = safety_primary
        self.test = PairedSequentialTest(delta=delta, alpha=alpha_family)
        self.suspect = False
        # clustering diagnostics (accumulated per batch)
        self._d_values: List[np.ndarray] = []
        self._d_clusters: List[np.ndarray] = []

    def select(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        mask = np.ones(batch["d"].size, dtype=bool)
        if self.cls_code is not None:
            mask &= batch["class"] == self.cls_code
        if self.cond_code is not None:
            mask &= batch["cond"] == self.cond_code
        if self.band is not None:
            mask &= batch["band"] == self.band
        return mask

    def update(self, batch: Dict[str, np.ndarray]) -> None:
        mask = self.select(batch)
        if not mask.any():
            return
        d = batch["d"][mask]
        containers = batch["container_id"][mask]
        w = batch["weight"][mask] if self.weighted else None
        means, _ = units_mod.cluster_units(d, containers, weights=w)
        self.test.update_clusters(means)
        self.test.record_objects(batch["b"][mask], batch["c"][mask])
        self._d_values.append(d)
        self._d_clusters.append(containers)

    def clustering(self) -> Dict:
        if not self._d_values:
            return {"n": 0, "n_clusters": 0, "icc": 0.0, "design_effect": 1.0,
                    "n_effective": 0.0, "mean_cluster_size": 0.0}
        return units_mod.cluster_summary(np.concatenate(self._d_values),
                                         np.concatenate(self._d_clusters))


def e_bh_reject(e_values: Dict[str, float], alpha: float) -> List[str]:
    """e-BH: reject the k hypotheses with largest e-values where
    e_(k) >= M / (k * alpha). Valid under arbitrary dependence."""
    if not e_values or alpha <= 0:
        return []
    items = sorted(e_values.items(), key=lambda kv: -kv[1])
    m = len(items)
    k_star = 0
    for k, (_, e) in enumerate(items, start=1):
        if e >= m / (k * alpha):
            k_star = k
    return [key for key, _ in items[:k_star]]


class HierarchyController:
    def __init__(self, strata: Dict[str, Dict], n_cond: int, delta: float,
                 alpha: float, alpha_pass: float,
                 alpha_shares: Optional[Dict[str, float]] = None,
                 band_names: Optional[List[str]] = None):
        self.alpha = float(alpha)
        self.alpha_pass = float(alpha_pass)
        self.shares = {**DEFAULT_ALPHA_SHARES, **(alpha_shares or {})}
        self.nodes: Dict[str, Node] = {}
        band_names = band_names or ["easy", "medium", "hard"]

        self.nodes["overall"] = Node("overall", 1, delta, alpha, weighted=True)
        classes = sorted({(int(s) // n_cond, v["class"]) for s, v in strata.items()})
        for code, name in classes:
            self.nodes[f"class:{name}"] = Node(f"class:{name}", 2, delta, alpha,
                                               cls_code=code, weighted=True)
        self.safety_keys: List[str] = []
        for s, v in strata.items():
            s = int(s)
            key = f"stratum:{v['key']}"
            node = Node(key, 3, delta, alpha, cls_code=s // n_cond,
                        cond_code=s % n_cond, safety_primary=v.get("safety_primary", False))
            self.nodes[key] = node
            if node.safety_primary:
                self.safety_keys.append(key)
        for b, name in enumerate(band_names):
            self.nodes[f"difficulty:{name}"] = Node(f"difficulty:{name}", 4, delta,
                                                    alpha, band=b, weighted=True)

    # ---- alpha allocation ------------------------------------------------

    def alpha_for(self, node: Node) -> float:
        a = self.alpha
        if node.safety_primary:
            return a * self.shares["safety"] / max(len(self.safety_keys), 1)
        if node.level == 1:
            return a * self.shares["overall"]
        if node.level == 2:
            return a * self.shares["class"]      # level budget, shared via e-BH
        if node.level == 3:
            return a * self.shares["stratum"]
        return a * self.shares["difficulty"]

    # ---- updates -----------------------------------------------------------

    def update_with_batch(self, batch: Dict[str, np.ndarray]) -> None:
        for node in self.nodes.values():
            node.update(batch)

    def evaluate(self, screening_e_threshold: float = 1.5) -> Dict[str, str]:
        """Apply per-level testing; return {node_key: decision}. Sticky."""
        # pre-registered primaries + single-hypothesis levels: individual tests
        for node in self.nodes.values():
            if node.safety_primary or node.level == 1:
                node.test.evaluate(alpha_reg=self.alpha_for(node),
                                   alpha_pass=self.alpha_pass)
        # e-BH within L2 / L3(non-safety) / L4
        for level, share_key in ((2, "class"), (3, "stratum"), (4, "difficulty")):
            group = {k: n for k, n in self.nodes.items()
                     if n.level == level and not n.safety_primary}
            undecided = {k: n.test.e_reg.e_value for k, n in group.items()
                         if n.test.decision == DECISION_INSUFFICIENT}
            already = {k: n.test.e_reg.e_value for k, n in group.items()
                       if n.test.decision == DECISION_REGRESSION}
            # include already-rejected e-values so e-BH stays monotone/sticky
            rejected = e_bh_reject({**undecided, **already},
                                   self.alpha * self.shares[share_key])
            for k in rejected:
                node = group[k]
                if node.test.decision == DECISION_INSUFFICIENT:
                    node.test.decision = DECISION_REGRESSION
                    node.test.decided_at_n = node.test.n_objects
            # PASS claims are per-node equivalence statements
            for k, n in group.items():
                if n.test.decision == DECISION_INSUFFICIENT \
                        and n.test.e_pass.e_value >= 1.0 / self.alpha_pass:
                    n.test.decision = DECISION_PASS
                    n.test.decided_at_n = n.test.n_objects
        # screening flags (soft evidence, drives escalation priority)
        for node in self.nodes.values():
            if node.test.e_reg.e_value >= screening_e_threshold:
                node.suspect = True
        return {k: n.test.decision for k, n in self.nodes.items()}

    def record_trajectories(self) -> None:
        for node in self.nodes.values():
            if node.test.n_objects > 0:
                node.test.record_trajectory_point()

    # ---- summaries -----------------------------------------------------

    def decisions(self) -> Dict[str, str]:
        return {k: n.test.decision for k, n in self.nodes.items()}

    def any_regression(self) -> bool:
        return any(n.test.decision == DECISION_REGRESSION for n in self.nodes.values())

    def required_pass(self) -> bool:
        """Overall + every safety primary must individually reach PASS."""
        required = [self.nodes["overall"]] + [self.nodes[k] for k in self.safety_keys]
        return all(n.test.decision == DECISION_PASS for n in required)

    def undecided_keys(self) -> List[str]:
        return [k for k, n in self.nodes.items()
                if n.test.n_objects > 0 and n.test.decision == DECISION_INSUFFICIENT]
