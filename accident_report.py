"""
Accident Reporting Layer
Generates production-grade HTML and JSON reports, incorporating visualizations.
"""

import json
from typing import Dict, Any

class AccidentReport:
    """Exports accident analysis results to multiple formats."""

    def __init__(self, metrics: Dict[str, Any], insights: Dict[str, Any]):
        self.metrics = metrics
        self.insights = insights

    def export_to_json(self) -> str:
        """
        Exports full metrics and insights in standard JSON.

        Returns:
            JSON formatted string.
        """
        payload = {
            "metrics": self.metrics,
            "insights": self.insights
        }
        return json.dumps(payload, indent=2)

    def export_to_html(self) -> str:
        """
        Generates a styled, standalone HTML page displaying the reports.

        Returns:
            HTML markup string.
        """
        rec_list = "".join(f"<li>{r}</li>" for r in self.insights.get("safety_recommendations", []))
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Accident Analysis Executive Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0f19; color: #f3f4f6; margin: 40px; }}
        .container {{ max-width: 800px; margin: 0 auto; background: rgba(20, 28, 48, 0.45); padding: 30px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); }}
        h1 {{ color: #00c896; border-bottom: 2px solid rgba(0, 200, 150, 0.2); padding-bottom: 10px; }}
        h2 {{ color: #3b82f6; margin-top: 30px; }}
        .metric-box {{ display: flex; justify-content: space-between; padding: 12px; background: rgba(255,255,255,0.02); margin: 8px 0; border-radius: 6px; }}
        .score {{ font-size: 24px; font-weight: bold; color: #ef4444; }}
        ul {{ padding-left: 20px; }}
        li {{ margin: 8px 0; color: #9ca3af; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Accident Analysis Summary Report</h1>
        <div class="metric-box">
            <span><strong>Calamity Risk Index (0-10):</strong></span>
            <span class="score">{self.insights.get("risk_index", 0)}</span>
        </div>
        <div class="metric-box">
            <span><strong>Peak Danger Hour:</strong></span>
            <span>{self.insights.get("deadliest_hour", "N/A")}</span>
        </div>
        <div class="metric-box">
            <span><strong>Primary Collision Threat:</strong></span>
            <span>{self.insights.get("primary_collision_threat", "N/A")}</span>
        </div>
        <div class="metric-box">
            <span><strong>Primary Geospatial Hotspot:</strong></span>
            <span>{self.insights.get("primary_geospatial_hotspot", "N/A")}</span>
        </div>
        
        <h2>Safety Recommendations</h2>
        <ul>
            {rec_list}
        </ul>
    </div>
</body>
</html>
"""
        return html
