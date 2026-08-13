"""Regression Root Cause Analysis (RCA) investigation workbench.

A skeptical, staged methodology for answering "offline says +5%, shadow says
-2% -- which is real, and why?" without assuming either measurement is valid.

Modules:
    models      investigation entity, stage state machine, evidence records
    scenario    synthetic offline+shadow dataset generator with planted causes
    stats       Wilson CIs, PSI / JS / KS, cluster-aware effective sample size
    diagnostics per-stage diagnostic computations (pure functions on data)
    scoring     root-cause scoring board, decision tree, experiment ranking
    report      final markdown + JSON report assembly
    store       persistence under runs/rca/
    api         FastAPI router mounted at /api/rca
"""
