# Crash Data Analysis Toolkit

Comprehensive framework for analyzing traffic crash data across three dimensions: **metadata** (temporal/severity patterns), **geospatial** (location-based hotspots), and **vehicle/traveler** (demographics & safety).

## 📦 What's Included

### Core Components

1. **`crash_analyzer.py`** — Python module with `CrashDataAnalyzer` class
   - Load SWITRS data from CSV
   - Analyze temporal patterns (hour, day, month, year)
   - Identify geographic hotspots via grid clustering
   - Analyze vehicle types and traveler demographics
   - Generate comprehensive reports

2. **`.claude/skills/crash-data-analyzer.md`** — Claude skill file
   - Analysis workflows and templates
   - Data structure reference (SWITRS format)
   - SQL/Python code snippets
   - Visualization recommendations

3. **`example_crash_analysis.py`** — Runnable example
   - Demonstrates all three analysis dimensions
   - Creates sample data for testing
   - Shows how to interpret results

4. **`CRASH_DATA_GUIDE.md`** — Complete reference documentation
   - SWITRS data schema (accidents, vehicles, parties tables)
   - Common analyses and use cases
   - Visualization examples
   - Data preparation steps

## 🚀 Quick Start

### 1. Basic Analysis with Sample Data

```bash
python example_crash_analysis.py
```

Output includes:
- Temporal patterns (deadliest hours, crashes by day)
- Severity distribution (fatal, injury, property damage)
- Geographic hotspots with coordinates
- Vehicle risk analysis (by vehicle type)
- Demographic insights (age, gender, impairment)

### 2. Load Your Own SWITRS Data

```python
from crash_analyzer import load_switrs_data

# Expects: accidents.csv, vehicles.csv, parties.csv
analyzer = load_switrs_data('accidents.csv', 'vehicles.csv', 'parties.csv')

# Get complete report
report = analyzer.generate_summary_report()
```

### 3. Target Specific Analyses

```python
from crash_analyzer import CrashDataAnalyzer

# Find dangerous times
temporal = analyzer.analyze_temporal_patterns()
print(f"Deadliest hours: {temporal['deadliest_hours']}")

# Find dangerous places
hotspots = analyzer.identify_hotspots(grid_size=0.01)  # ~1km grid cells
for hotspot in hotspots[:5]:
    print(f"Hotspot at ({hotspot['center_lat']}, {hotspot['center_lon']}): "
          f"{hotspot['crash_count']} crashes")

# Find dangerous demographics
demographics = analyzer.analyze_demographics()
print(f"Average age in crashes: {demographics['age_statistics']['mean']}")
```

## 📊 Three Analysis Dimensions

### 1. METADATA — Temporal & Severity Patterns
```
analyze_temporal_patterns()     → Crashes by hour/day/month/year + deadliest times
analyze_severity()               → Fatal/Injury/Property Damage distribution
analyze_collision_types()        → Most common & deadliest collision types
```

**Use For**:
- Identifying peak crash hours (traffic management)
- Spotting seasonal trends (winter weather impact)
- Understanding severity distribution
- Targeting enforcement activities

### 2. GEOSPATIAL — Location-Based Analysis
```
get_geographic_summary()         → Crashes by county/district
identify_hotspots()              → Grid-based clustering of high-crash areas
```

**Use For**:
- Prioritizing infrastructure improvements
- Planning emergency response deployment
- Identifying dangerous intersections
- Geographic risk assessment

**Example Output**:
```
Hotspot at (37.7749, -122.4194): 45 crashes, 3 fatal
  → Focus enforcement here
  → Consider signal upgrade at this intersection
```

### 3. VEHICLE/TRAVELER — Demographics & Safety
```
analyze_vehicle_types()          → Crash count & fatality rate per vehicle type
analyze_demographics()           → Age, gender, sobriety, injury patterns
analyze_injury_patterns()        → Injury severity by demographics
```

**Use For**:
- Identifying high-risk groups (age, vehicle type)
- Understanding impairment impact
- Evaluating safety equipment effectiveness
- Targeting education programs

## 📋 SWITRS Data Format

Three main tables (typical SWITRS structure):

### Accidents
- `case_id`, `accident_date`, `accident_time`
- `latitude`, `longitude`, `county`, `district`
- `severity` (F/I/P), `collision_type`
- `weather_1`, `surface_condition`, `traffic_control_device`
- `fatality_count`, `injury_count`

### Vehicles
- `case_id`, `vehicle_type`, `vehicle_year`
- `vehicle_make`, `vehicle_model`, `vehicle_color`
- `damage_extent`, `direction_of_travel`

### Parties
- `case_id`, `party_type` (Driver/Passenger/Pedestrian)
- `age`, `gender`, `sobriety`, `injury_level`
- `safety_equipment_used`, `ejection`
- `primary_collision_factor`

See `CRASH_DATA_GUIDE.md` for complete schema reference.

## 🛠️ Usage Examples

### Find Most Dangerous Intersections
```python
hotspots = analyzer.identify_hotspots()
top_5 = sorted(hotspots, key=lambda x: x['fatal_count'], reverse=True)[:5]
for spot in top_5:
    print(f"Priority: {spot['fatal_count']} fatal crashes at "
          f"({spot['center_lat']}, {spot['center_lon']})")
```

### Identify High-Risk Demographics
```python
demographics = analyzer.analyze_demographics()
age_injury = demographics['injury_by_age_group']
# Ages with highest injury rates get targeted education
```

### Compare Vehicle Safety
```python
vehicle_risk = analyzer.analyze_vehicle_types()['vehicle_risk_analysis']
motorcycles = vehicle_risk['Motorcycle']
cars = vehicle_risk['Car']
print(f"Motorcycle fatality rate: {motorcycles['fatality_rate']:.1%}")
print(f"Car fatality rate: {cars['fatality_rate']:.1%}")
```

### Track Trends Over Time
```python
temporal = analyzer.analyze_temporal_patterns()
yearly = temporal['crashes_by_year']
# Analyze if crashes are increasing or decreasing
```

## 📈 Visualization Suggestions

| Analysis | Chart Type | Library |
|----------|-----------|---------|
| Crashes by hour | Line chart | matplotlib, plotly |
| Severity breakdown | Stacked bar | plotly, seaborn |
| Geographic hotspots | Heatmap or cluster map | folium, kepler.gl |
| Vehicle risk | Grouped bar | plotly |
| Age demographics | Histogram | matplotlib, plotly |
| County comparison | Choropleth map | folium |

**Quick Plotly Example**:
```python
import plotly.express as px

# Crashes by hour
hourly = analyzer.accidents.groupby('hour').size().reset_index(name='count')
px.bar(hourly, x='hour', y='count', title='Crashes by Hour of Day').show()

# Hotspots on map
hotspots_df = pd.DataFrame(analyzer.identify_hotspots())
px.scatter_mapbox(
    hotspots_df, 
    lat='center_lat', lon='center_lon',
    size='crash_count',
    hover_name='crash_count',
    title='Crash Hotspots',
    zoom=6
).show()
```

## 📖 Using with Claude Code

Reference the skill in Claude Code:

```
/crash-data-analyzer
```

When you invoke this skill, Claude will have access to:
- Complete analysis workflows
- SWITRS data schema reference
- Code templates for all three dimensions
- Visualization recommendations

**Example Prompt**:
> "I have SWITRS crash data for California. Analyze it for metadata (temporal patterns and severity), geospatial hotspots, and vehicle/traveler demographics."

## 🔗 Data Sources

### Where to Get SWITRS Data
- **TIMS (Transportation Injury Mapping System)**: https://tims.berkeley.edu/
- **California Highway Patrol**: https://www.chp.ca.gov/
- **Local Law Enforcement Agencies**: County/city specific data
- **Transportation Injury Mapping System Help**: https://tims.berkeley.edu/help/crashdata/switrs-geocoding.php

### Required Data
For full analysis, you need:
1. **accidents.csv** — Core crash records
2. **vehicles.csv** — Vehicle details per crash
3. **parties.csv** — Occupant/pedestrian information

(Some analyses work with just accidents data)

## ⚙️ Installation

### Dependencies
```bash
pip install pandas numpy
```

### Optional (for enhanced features)
```bash
pip install geopandas folium plotly scikit-learn
```

### No installation needed for skill
The `.claude/skills/crash-data-analyzer.md` skill is automatically available in Claude Code.

## 📝 Key Metrics

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **Fatality Rate** | Deaths / Total Crashes | Higher = more severe |
| **Injury Rate** | Injury Crashes / Total Crashes | Prevention target |
| **Peak Hour** | Hour with most crashes | Traffic management |
| **Hotspot** | Grid cell with 10+ crashes | Infrastructure priority |
| **Risk by Group** | Deaths per demographic group | Education target |

## ⚠️ Data Quality Notes

Before analysis, verify:
- ✅ No duplicate `case_id` values
- ✅ Valid latitude/longitude coordinates
- ✅ Dates parse correctly
- ✅ Severity values are consistent (F/I/P or 1/2/3)
- ✅ Age values are reasonable (0-120)
- ✅ Merge keys (`case_id`) match across tables

## 🐛 Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'pandas'"
```bash
pip install pandas numpy
```

**Issue**: Hotspots not returning results
```python
# Check if you have valid geographic data
valid_crashes = analyzer.accidents.dropna(subset=['latitude', 'longitude'])
print(f"Valid geo records: {len(valid_crashes)}")

# Try different grid size
hotspots = analyzer.identify_hotspots(grid_size=0.02)  # Larger cells
```

**Issue**: Age data producing warnings
```python
# Ensure age is numeric and in reasonable range
analyzer.parties['age'] = pd.to_numeric(analyzer.parties['age'], errors='coerce')
analyzer.parties = analyzer.parties[(analyzer.parties['age'] > 0) & (analyzer.parties['age'] < 120)]
```

## 📚 Further Reading

- **Berkeley TIMS Documentation**: https://tims.berkeley.edu/help/
- **SWITRS Geocoding Guide**: https://tims.berkeley.edu/help/crashdata/switrs-geocoding.php
- **California Traffic Safety Program**: https://www.chp.ca.gov/
- **Safe Transportation Research & Education Center**: https://safetrec.berkeley.edu

## 📄 File Structure

```
DrivingRepo/
├── crash_analyzer.py              # Core Python module
├── example_crash_analysis.py       # Runnable example
├── CRASH_ANALYSIS_README.md        # This file
├── CRASH_DATA_GUIDE.md             # Complete reference
└── .claude/
    └── skills/
        └── crash-data-analyzer.md  # Claude skill file
```

## 🎯 Next Steps

1. **Gather Data** — Obtain SWITRS CSV files
2. **Load & Explore** — Run `example_crash_analysis.py`
3. **Analyze Specific Dimension** — Use individual analyzer methods
4. **Visualize** — Create charts for reporting
5. **Report Findings** — Use skill as guide for structure

---

**Created**: 2026-07-26  
**For**: SWITRS crash data analysis  
**Dimensions**: Metadata, Geospatial, Vehicle/Traveler
