"""seqeval — budgeted, anytime-valid sequential regression detection.

Detects small (~2pp) per-stratum model regressions without evaluating the full
corpus, with validity under sequential looks, clustering, stratification and
multiple comparisons. See DESIGN.md in this package for the full statistical
design and its justifications.

Module map:
    units.py       statistical units: clusters, ICC, design effect
    planner.py     frozen stratified sampling plans (Neyman + risk floors)
    paired.py      paired harness + fingerprint-keyed prediction cache
    sequential.py  empirical-Bernstein confidence sequences, e-processes,
                   three-outcome decisions, Beta-Binomial complement
    hierarchy.py   hierarchical gatekeeping with e-BH within levels
    attribution.py the regression map
    ledger.py      machine-readable evidence records + lineage
    controller.py  sanity -> screening -> sequential -> escalation state machine
    api.py         REST surface under /api/seqeval

Entry point for the safety Regression Gate:
    from sensorflow.seqeval import evaluate_regression
"""

from sensorflow.seqeval.controller import evaluate_regression  # noqa: F401
