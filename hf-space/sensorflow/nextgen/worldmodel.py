"""World-model integration boundary for counterfactual scene generation.

The platform's counterfactual engine (counterfactual.py) needs one thing from
a "world model": given a source scene and a transformation recipe, produce a
physically coherent transformed scene with full ground truth. This module
defines that boundary as an explicit interface with exactly one live
implementation:

* :class:`DeterministicSceneTransformer` — the internal, deterministic,
  physics-rule-based transformer. It operates on reconstructed world-frame
  actor kinematics from :mod:`sensorflow.bevfusion.scenes` sequences and is
  fully seeded/reproducible. This is what runs today.

* :class:`ExternalWorldModelAdapter` — a stub adapter documenting where a
  learned generative world model (e.g. a proprietary video/world model such
  as an internal WWM) would plug in. It is intentionally NOT implemented: we
  assume no proprietary APIs. The contract it would have to satisfy is
  documented on the class, and everything downstream (validity gate,
  closed-loop evaluation, provenance labels) is already generator-agnostic —
  a generated scenario from ANY implementation goes through the same
  validity gate before it can enter an evaluation suite.

The generative-architecture trade study (diffusion vs transformer world model
vs hybrid) lives in docs/architecture/nextgen-worldmodel-generative-comparison.md
and is served by GET /api/nextgen/architecture/docs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from sensorflow.bevfusion.scenes import SceneSequence
from sensorflow.nextgen.models import TransformationStep


@dataclass
class ActorTrack:
    """World-frame kinematics of one actor across the scenario horizon.

    states[i] corresponds to frame i: dict with x, y, vx, vy, yaw, occluded.
    """

    instance_id: str
    class_name: str
    dims: List[float]  # l, w, h
    states: List[Dict] = field(default_factory=list)


@dataclass
class TransformedScene:
    """Output contract of a scene transformer: a renderable sequence plus the
    world-frame tracks it was built from (needed by validity + closed loop)."""

    sequence: SceneSequence
    actors: List[ActorTrack]
    environment: Dict[str, str]      # weather / time_of_day / extended tags
    applied: List[TransformationStep]
    notes: List[str] = field(default_factory=list)


class SceneTransformer(ABC):
    """The world-model boundary. Implementations must be pure functions of
    (source, recipe, seed): same inputs -> identical output."""

    name: str = "abstract"
    version: str = "0"

    @abstractmethod
    def transform(self, source: SceneSequence, recipe: List[TransformationStep],
                  seed: int) -> TransformedScene:
        ...


class ExternalWorldModelAdapter(SceneTransformer):
    """Stub for a learned generative world model.

    Contract an external implementation must satisfy:

    1. Input: a source scene (sensor-agnostic ground-truth representation, as
       :class:`SceneSequence`) + a transformation recipe expressed as
       :class:`TransformationStep` s + a seed.
    2. Output: a :class:`TransformedScene` with FULL ground truth (world-frame
       actor kinematics per frame) — not just rendered pixels. Without GT the
       result cannot be evaluated, only admired.
    3. Determinism: same (source, recipe, seed) must reproduce the same
       output, or the run's lineage record is unreproducible and the run is
       INVALID for launch purposes (lineage.py).
    4. Provenance: outputs are labeled GENERATED (vs COUNTERFACTUAL for
       rule-based transforms of a real/synthetic source) so reports can
       distinguish generator classes.
    5. Every output still passes through the validity gate (validity.py);
       being expensive to generate earns no exemption from gating.
    """

    name = "external-worldmodel-stub"
    version = "0 (not implemented)"

    def transform(self, source: SceneSequence, recipe: List[TransformationStep],
                  seed: int) -> TransformedScene:
        raise NotImplementedError(
            "No external world model is wired in. This adapter documents the "
            "integration contract; see class docstring and "
            "docs/architecture/nextgen-worldmodel-generative-comparison.md")
