# taxonomy_mapping.py
"""
Mapping from YOLO detection class names to taxonomy categories defined in taxonomy.md.
The mapping is intentionally simple, covering the major categories used in the platform.
Add or adjust entries as needed to reflect the full taxonomy.
"""

TAXONOMY_MAP = {
    # Road Users
    "car": "Passenger Vehicles",
    "person": "Vulnerable Road Users",
    "pedestrian": "Vulnerable Road Users",
    "bicycle": "Micro-Mobility",
    "motorcycle": "Micro-Mobility",
    "truck": "Heavy & Commercial Vehicles",
    "bus": "Heavy & Commercial Vehicles",
    "emergency_vehicle": "Emergency Vehicles",
    "animal": "Animals",
    # Static Road Objects
    "traffic_light": "Traffic Control Elements",
    "stop_sign": "Traffic Control Elements",
    "road": "Drivable Surfaces",
    "lane": "Drivable Surfaces",
    "crosswalk": "Drivable Surfaces",
    "guardrail": "Physical Constraints & Barriers",
    "curb": "Physical Constraints & Barriers",
    "sidewalk": "Physical Constraints & Barriers",
    # Dynamic / Transitory
    "construction": "Temporary Infrastructure",
    "debris": "Uncontrolled Hazards"
}
