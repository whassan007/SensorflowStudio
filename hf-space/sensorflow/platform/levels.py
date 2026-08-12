"""Evaluation abstraction across aggregate levels.

Unifies LabelEval (frame/annotation) and MegaEval (container/cohort/population)
into one ladder without duplicating storage.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AggregateLevel(str, Enum):
    FRAME = "frame"
    CLIP = "clip"  # sequence / short clip
    SCENE = "scene"
    DRIVE = "drive"
    CONTAINER = "container"
    DATASET = "dataset"
    COHORT = "cohort"
    POPULATION = "population"

    @classmethod
    def ladder(cls) -> List["AggregateLevel"]:
        return [
            cls.FRAME,
            cls.CLIP,
            cls.SCENE,
            cls.DRIVE,
            cls.CONTAINER,
            cls.DATASET,
            cls.COHORT,
            cls.POPULATION,
        ]

    def parent(self) -> Optional["AggregateLevel"]:
        ladder = self.ladder()
        i = ladder.index(self)
        return ladder[i + 1] if i + 1 < len(ladder) else None

    def child(self) -> Optional["AggregateLevel"]:
        ladder = self.ladder()
        i = ladder.index(self)
        return ladder[i - 1] if i > 0 else None


# Which backend owns primary aggregates at each level
LEVEL_BACKEND: Dict[AggregateLevel, str] = {
    AggregateLevel.FRAME: "labeleval",
    AggregateLevel.CLIP: "labeleval",
    AggregateLevel.SCENE: "labeleval",
    AggregateLevel.DRIVE: "platform",  # stub until Phase 5/6
    AggregateLevel.CONTAINER: "megaeval",
    AggregateLevel.DATASET: "labeleval",
    AggregateLevel.COHORT: "megaeval",
    AggregateLevel.POPULATION: "megaeval",
}


class EvaluationScope(BaseModel):
    """Points at one aggregate cell without inventing a second DB."""

    level: AggregateLevel
    evaluation_id: Optional[str] = None
    dataset_id: Optional[str] = None
    population_id: Optional[str] = None
    run_id: Optional[str] = None
    container_id: Optional[str] = None
    mega_container_id: Optional[int] = None
    drive_id: Optional[str] = None
    scene_id: Optional[str] = None
    sequence_id: Optional[str] = None
    frame_id: Optional[str] = None
    cohort_filters: Dict[str, List[str]] = Field(default_factory=dict)
    model_version: Optional[str] = None
    label_version: Optional[str] = None

    def backend(self) -> str:
        return LEVEL_BACKEND[self.level]

    def drill_down_hint(self) -> AggregateLevel:
        child = self.level.child()
        return child or self.level


def summarize_scope(scope: EvaluationScope) -> Dict[str, Any]:
    return {
        "level": scope.level.value,
        "backend": scope.backend(),
        "parent_level": scope.level.parent().value if scope.level.parent() else None,
        "child_level": scope.level.child().value if scope.level.child() else None,
        "refs": {
            k: v
            for k, v in {
                "evaluation_id": scope.evaluation_id,
                "dataset_id": scope.dataset_id,
                "population_id": scope.population_id,
                "run_id": scope.run_id,
                "container_id": scope.container_id,
                "mega_container_id": scope.mega_container_id,
                "drive_id": scope.drive_id,
                "scene_id": scope.scene_id,
                "sequence_id": scope.sequence_id,
                "frame_id": scope.frame_id,
                "model_version": scope.model_version,
            }.items()
            if v is not None
        },
        "cohort_filters": scope.cohort_filters,
    }
