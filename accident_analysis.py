"""
Accident Analysis Engine Layer
Performs temporal pattern detection, geospatial hotspot clustering, demographic reviews, and caches results.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from crash_analyzer import CrashDataAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataAnalyzer:
    """Performs schema, time, and severity breakdown analysis on accident records."""
    
    def __init__(self, analyzer: CrashDataAnalyzer):
        self.analyzer = analyzer

    def analyze(self) -> Dict[str, Any]:
        """Runs temporal and severity breakdown metrics."""
        temporal = self.analyzer.analyze_temporal_patterns()
        severity = self.analyzer.analyze_severity()
        collision = self.analyzer.analyze_collision_types()
        
        return {
            "temporal_patterns": temporal,
            "severity_distribution": severity,
            "collision_types": collision
        }

class GeospatialAnalyzer:
    """Performs geographic analysis and hotspot identification."""
    
    def __init__(self, analyzer: CrashDataAnalyzer):
        self.analyzer = analyzer

    def analyze(self, grid_size: float = 0.01) -> Dict[str, Any]:
        """Clusters coordinates to locate hotspot sectors."""
        summary = self.analyzer.get_geographic_summary()
        hotspots = self.analyzer.identify_hotspots(grid_size=grid_size)
        return {
            "geographic_summary": summary,
            "hotspots": hotspots
        }

class DemographicsAnalyzer:
    """Performs party, vehicle type, and traveler demographic analysis."""
    
    def __init__(self, analyzer: CrashDataAnalyzer):
        self.analyzer = analyzer

    def analyze(self) -> Dict[str, Any]:
        """Parses driver profiles and vehicle risk attributes."""
        vehicles = self.analyzer.analyze_vehicle_types()
        demographics = self.analyzer.analyze_demographics()
        return {
            "vehicle_risk": vehicles,
            "demographics": demographics
        }

class AccidentAnalysisEngine:
    """Orchestrates temporal, geospatial, and demographic analysis while caching results."""

    def __init__(self, accidents_df: pd.DataFrame, vehicles_df: Optional[pd.DataFrame] = None,
                 parties_df: Optional[pd.DataFrame] = None):
        self.analyzer = CrashDataAnalyzer(accidents_df, vehicles_df, parties_df)
        self.metadata_analyzer = MetadataAnalyzer(self.analyzer)
        self.geospatial_analyzer = GeospatialAnalyzer(self.analyzer)
        self.demographics_analyzer = DemographicsAnalyzer(self.analyzer)
        self._cache: Dict[str, Any] = {}

    def run_full_pipeline(self, bypass_cache: bool = False) -> Dict[str, Any]:
        """
        Executes all sub-analyzers and returns cached or fresh analysis results.

        Args:
            bypass_cache: Force fresh computation.

        Returns:
            Dict containing full analysis indicators.
        """
        if "full_metrics" in self._cache and not bypass_cache:
            logger.info("Returning cached analysis results.")
            return self._cache["full_metrics"]

        logger.info("Computing fresh analysis metrics...")
        meta = self.metadata_analyzer.analyze()
        geo = self.geospatial_analyzer.analyze()
        demo = self.demographics_analyzer.analyze()

        results = {
            "metadata": meta,
            "geospatial": geo,
            "demographics": demo
        }

        self._cache["full_metrics"] = results
        return results
