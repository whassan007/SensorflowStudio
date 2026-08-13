"""ROTR — Right-of-the-Road violation detection, triage, evaluation & training.

Causal-layer separation is the load-bearing constraint of this package: a
ROTR violation is NEVER auto-attributed to perception. Detection (rules.py),
attribution (attribution.py), consequence (consequence.py), statistics
(metrics.py, delegating to seqeval), mining (taxonomy.py) and governance
(flywheel.py, stopship.py) are separate modules with typed contracts
(models.py). Architecture: docs/architecture/rotr-architecture.md.
"""

SOFTWARE_VERSION = "rotr-0.1.0"
