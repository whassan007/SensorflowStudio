# Crash Data Analysis — Quick Reference

## One-Line Loader
```python
from crash_analyzer import load_switrs_data
analyzer = load_switrs_data('accidents.csv', 'vehicles.csv', 'parties.csv')
```

## Core Methods

### Metadata Analysis
```python
analyzer.analyze_temporal_patterns()  # Crashes by hour/day/month/year
analyzer.analyze_severity()            # Fatal/Injury/Property Damage counts
analyzer.analyze_collision_types()     # Most common & deadliest collisions
```

### Geospatial Analysis
```python
analyzer.get_geographic_summary()      # Crashes by county/district
analyzer.identify_hotspots()           # Geographic clusters of crashes
# → Returns list of dicts with center_lat, center_lon, crash_count, fatal_count
```

### Vehicle/Traveler Analysis
```python
analyzer.analyze_vehicle_types()       # Risk by vehicle type
analyzer.analyze_demographics()        # Age, gender, sobriety patterns
analyzer.analyze_injury_patterns()     # Injuries by demographics
```

### Full Report
```python
analyzer.generate_summary_report()     # All analyses combined
```

---

## Common One-Liners

**Deadliest hour of day**
```python
analyzer.analyze_temporal_patterns()['deadliest_hours']
```

**Most dangerous intersections (top 3)**
```python
hotspots = analyzer.identify_hotspots()
sorted(hotspots, key=lambda x: x['crash_count'], reverse=True)[:3]
```

**Fatality rate by vehicle type**
```python
analyzer.analyze_vehicle_types()['vehicle_risk_analysis']
```

**Age group with highest injuries**
```python
demographics = analyzer.analyze_demographics()
demographics['crashes_by_age_group']  # Find highest count
```

---

## Data Schema Cheat Sheet

| Table | Key Columns |
|-------|------------|
| **Accidents** | `case_id`, `accident_date`, `accident_time`, `latitude`, `longitude`, `severity`, `county`, `collision_type`, `fatality_count` |
| **Vehicles** | `case_id`, `vehicle_type`, `vehicle_year`, `damage_extent` |
| **Parties** | `case_id`, `age`, `gender`, `sobriety`, `injury_level`, `party_type` |

**Severity Codes**: `F` = Fatal, `I` = Injury, `P` = Property Damage Only  
**Sobriety Codes**: `1` = Sober, `2` = Drinking, `3` = Intoxicated, `4` = Unknown

---

## Quick Visualizations

**Crashes by hour** (matplotlib)
```python
analyzer.accidents.groupby('hour').size().plot(kind='bar')
```

**Crashes on map** (folium)
```python
import folium
m = folium.Map(location=[37, -120], zoom_start=6)
for _, row in analyzer.accidents.iterrows():
    folium.CircleMarker([row['latitude'], row['longitude']], radius=2).add_to(m)
m.save('crash_map.html')
```

**Vehicle risk comparison** (plotly)
```python
import plotly.express as px
risk = analyzer.analyze_vehicle_types()['vehicle_risk_analysis']
px.bar(x=list(risk.keys()), y=[v['fatality_rate'] for v in risk.values()]).show()
```

---

## Interpretation Guide

| Finding | What It Means | Action |
|---------|--------------|--------|
| High crashes at 14:30 | Rush hour peak | Increase enforcement then |
| Hotspot at (37.77, -122.41) | Dangerous intersection | Consider signal upgrade |
| Motorcycle fatality rate 10% | High risk vehicle | Motorcycle safety education |
| Age 65+ injuries highest | Elderly vulnerability | Vision/mobility programs |
| 2am crashes high | Impaired driving | DUI checkpoints then |

---

## File Locations

```
Skill:           .claude/skills/crash-data-analyzer.md
Module:          crash_analyzer.py
Example:         example_crash_analysis.py
Full Guide:      CRASH_DATA_GUIDE.md
README:          CRASH_ANALYSIS_README.md
```

---

## Data Sources

- **TIMS**: https://tims.berkeley.edu/
- **SWITRS Help**: https://tims.berkeley.edu/help/crashdata/switrs-geocoding.php
- **CHP**: https://www.chp.ca.gov/

---

## Typical Analysis Workflow

```
1. Load data
   analyzer = load_switrs_data('accidents.csv', 'vehicles.csv', 'parties.csv')

2. Get overview
   report = analyzer.generate_summary_report()

3. Dig deeper
   temporal = analyzer.analyze_temporal_patterns()
   hotspots = analyzer.identify_hotspots()
   demographics = analyzer.analyze_demographics()

4. Visualize
   # Create charts for reporting

5. Report findings
   # Document deadliest times, places, and groups
```

---

**Print this page for quick reference during analysis.**
