"""studio2 — the Studio 2.0 integration layer: unified control plane +
deterministic release governance.

This package deliberately builds NO new evaluation math. It composes the
engines that already exist:

    registry.py      versioned control-plane entities (models, datasets with
                     roles + contamination guard, scenarios, policies,
                     experiments, evaluation runs with reproducibility tuples,
                     safety cases, release decisions) + retroactive ingest of
                     the existing runs/ stores
    release_gate.py  ReleaseGate.evaluate(...) -> GO/REVIEW/NO-GO composed
                     from safety gates + seqeval verdicts + megaeval shift
                     (+ agentic policy / nextgen closed loop when importable);
                     GO never authorizes deployment — human approval is a
                     separate recorded action
    hardware.py      hardware/domain strata as first-class gate dimensions;
                     per-combination gate matrix with insufficient-evidence
                     reporting
    funnel.py        unified observability funnel aggregated best-effort from
                     the real stores (absent sources are UNAVAILABLE, never
                     fabricated)
    demo.py          one seeded closed-loop run stitching the real engines
                     end-to-end
    api.py           REST surface under /api/studio2

Landed dependencies are imported directly; in-flight packages (agentic,
nextgen, hardening, retro) are imported in try/except and their absence
degrades the release decision to REVIEW with the gap named.
"""

from sensorflow.studio2.registry import Registry, get_registry  # noqa: F401
from sensorflow.studio2.release_gate import ReleaseGate  # noqa: F401
