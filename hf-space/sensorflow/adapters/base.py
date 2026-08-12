"""Adapter protocol for vendor-specific data normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from sensorflow.schemas.unified_frame import UnifiedSequence


class VendorAdapter(ABC):
    @abstractmethod
    def load(self, source: Dict[str, Any], sequence_id: str) -> UnifiedSequence:
        """Convert vendor-specific data into a UnifiedSequence."""
