"""
Accident Insights Layer
Deduce high-level risks, deadliest hours, risk factors, and safety insights.
"""

from typing import Dict, Any, List

class AccidentInsights:
    """Extracts safety-critical insights and risk scores from processed analysis metrics."""

    def __init__(self, metrics: Dict[str, Any]):
        self.metrics = metrics

    def get_risk_profile(self) -> Dict[str, Any]:
        """
        Deduces the high-risk profiles from demographic and temporal stats.

        Returns:
            Dict containing warning metrics and risk profiles.
        """
        meta = self.metrics.get("metadata", {})
        temporal = meta.get("temporal_patterns", {})
        severity = meta.get("severity_distribution", {})
        
        # Deadliest hour
        deadliest_hours = list(temporal.get("deadliest_hours", {}).keys())
        primary_time_risk = f"Hour {deadliest_hours[0]}" if deadliest_hours else "Unknown"

        # High-risk collision types
        collision = meta.get("collision_types", {})
        top_collisions = list(collision.get("top_collision_types", {}).keys())
        primary_collision_risk = top_collisions[0] if top_collisions else "Unknown"

        # Hotspots
        geo = self.metrics.get("geospatial", {})
        hotspots = geo.get("hotspots", [])
        primary_hotspot = f"Lat: {hotspots[0]['center_lat']:.4f}, Lon: {hotspots[0]['center_lon']:.4f}" if hotspots else "None"

        # Risk score (mock safety scale based on fatal crash ratios)
        total = severity.get("total_crashes", 1)
        fatal = severity.get("total_fatalities", 0)
        risk_index = round((fatal / total) * 10, 2) if total > 0 else 0.0

        return {
            "risk_index": risk_index,
            "deadliest_hour": primary_time_risk,
            "primary_collision_threat": primary_collision_risk,
            "primary_geospatial_hotspot": primary_hotspot,
            "safety_recommendations": [
                "Deploy speed enforcement cameras around the primary geospatial hotspot.",
                f"Increase local police patrols during peak danger windows ({primary_time_risk}).",
                "Evaluate street infrastructure modifications to reduce primary collision threat vectors."
            ]
        }
