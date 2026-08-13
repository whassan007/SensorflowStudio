"""Safety & compliance layer for Sensorflow Studio.

Closes competitive feature gaps identified against AV perception evaluation
platforms (Applied Intuition, Deepen AI, Scale Nucleus, Voxel51):

- odd.py          ODD taxonomy + combinatorial coverage (ISO 34503 / ASAM OpenODD-inspired)
- ssam_ext.py     Extended SSAM surrogate safety measures (FHWA SSAM: DRAC, DeltaS,
                  MaxS, TTC/PET, Conflict Severity Index)
- calibration.py  Multi-sensor extrinsic calibration validation (Deepen-style gate)
- gates.py        Layered release gating + Safety Evidence Package (ISO 26262 /
                  ISO 21448 SOTIF / UL 4600 mappings — the package SUPPORTS a
                  safety case, it does not certify compliance)
- discrepancy.py  Online-vs-offline auto-label discrepancy mining
- scenario_db.py  Curated scenario database (Safety Pool-inspired, local)
- semantic.py     Neuro-symbolic two-stage semantic scenario mining
- api.py          FastAPI router exposing everything under /api/safety/*

Foundation-model-style components are lightweight deterministic simulations
behind clean interfaces, clearly marked as such — consistent with the rest of
the platform.
"""
