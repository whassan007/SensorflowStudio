# Crash Data Analysis Guide

Complete guide to analyzing crash data with metadata, geospatial, and vehicle/traveler dimensions.

## Quick Start

```python
from crash_analyzer import load_switrs_data

# Load SWITRS CSV data
analyzer = load_switrs_data('accidents.csv', 'vehicles.csv', 'parties.csv')

# Generate full report
report = analyzer.generate_summary_report()
```

## Data Source: SWITRS

**Reference**: https://tims.berkeley.edu/help/crashdata/switrs-geocoding.php

SWITRS (Statewide Integrated Traffic Records System) provides comprehensive California highway crash data with three main record types:

### 1. Accidents Table (Core Records)
Primary crash incident information.

| Field | Type | Description | Example |
|---|---|---|---|
| `case_id` | int | Unique crash identifier | 123456 |
| `accident_date` | date | Date of crash | 2024-01-15 |
| `accident_time` | time | Time of crash (24-hr) | 14:30 |
| `latitude` | float | Geocoded latitude | 37.7749 |
| `longitude` | float | Geocoded longitude | -122.4194 |
| `county` | string | County name | "San Francisco" |
| `district` | string | CHP district | "SF" |
| `severity` | string | Crash severity | F/I/P (Fatal/Injury/PropertyDamage) |
| `collision_type` | string | Type of collision | "Rear End", "Broadside", etc. |
| `weather_1` | string | Primary weather | "Clear", "Rain", "Fog" |
| `weather_2` | string | Secondary weather | Optional |
| `surface_condition` | string | Road surface | "Dry", "Wet", "Snow" |
| `traffic_control_device` | string | Traffic control status | "Functioning", "Not Functioning" |
| `fatality_count` | int | Number of fatalities | 0-5 |
| `injury_count` | int | Number of injuries | 0-10 |

### 2. Vehicles Table (Vehicle Details)
Vehicle information for each crash.

| Field | Type | Description |
|---|---|---|
| `case_id` | int | Link to accidents table |
| `vehicle_id` | int | Unique vehicle in crash |
| `vehicle_type` | string | Car, Truck, Motorcycle, SUV, Van, etc. |
| `vehicle_year` | int | Vehicle model year |
| `vehicle_make` | string | Manufacturer (Toyota, Ford, etc.) |
| `vehicle_model` | string | Model name |
| `vehicle_color` | string | Color |
| `damage_extent` | string | Severe, Moderate, Minor |
| `direction_of_travel` | string | N, S, E, W, etc. |

### 3. Parties Table (Occupants & Pedestrians)
Information about crash participants (drivers, passengers, pedestrians).

| Field | Type | Description |
|---|---|---|
| `case_id` | int | Link to accidents table |
| `party_number` | int | Sequence in crash |
| `party_type` | string | Driver, Passenger, Pedestrian, Cyclist |
| `age` | int | Age in years | 
| `gender` | string | M/F |
| `sobriety` | int | 1=Sober, 2=Drinking, 3=Intoxicated, 4=Unknown |
| `injury_level` | int | 0=None, 1=Possible, 2=Probable, 3=Severe, 4=Fatal |
| `safety_equipment_used` | string | Seatbelt, Helmet, etc. |
| `ejection` | string | Yes/No/Unknown |
| `primary_collision_factor` | string | Cause (Speeding, DUI, etc.) |

---

## Analysis Dimensions

### 1. METADATA ANALYSIS

**Time-Based Patterns**
```python
# Identify dangerous times
temporal = analyzer.analyze_temporal_patterns()
# Returns: crashes by hour, day, month, year + deadliest times
```

**What to Look For**:
- Rush hour peaks (7-9am, 4-6pm)
- Weekend spikes
- Holiday effect (New Year, July 4th)
- Seasonal trends (winter weather, spring break)

**Severity Analysis**
```python
severity = analyzer.analyze_severity()
# Returns: distribution of Fatal/Injury/Property Damage
```

**What to Report**:
- Fatality rate (total deaths / total crashes)
- Injury-to-crash ratio
- Crash trend over time (increasing/decreasing)

**Collision Types**
```python
collisions = analyzer.analyze_collision_types()
# Returns: most common collision types and deadliest types
```

**Key Metrics**:
- Most common collision type
- Deadliest collision type (by fatality rate)
- Preventable vs. unpreventable patterns

---

### 2. GEOSPATIAL ANALYSIS

**Geographic Summaries**
```python
geo = analyzer.get_geographic_summary()
# Returns: crashes by county/district with severity breakdown
```

**Geographic Hotspot Detection**
```python
hotspots = analyzer.identify_hotspots()
# Returns: grid-based clustering of high-crash areas
```

**Interpretation**:
- Cluster size = grid cell crashes
- Coordinates = center of high-risk area
- Fatal count = number of fatal crashes in area
- Focus enforcement/infrastructure improvements on hotspots

**Use Cases**:
- Identify intersections needing traffic signal upgrades
- Plan emergency response deployments
- Target enforcement activities
- Prioritize road maintenance

**Visualization Recommendations**:
- Heatmap of crash density (Folium, Kepler.gl)
- Cluster markers on interactive map
- Severity overlay (color-coded pins)
- Temporal animation (crashes by month)

---

### 3. VEHICLE/TRAVELER ANALYSIS

**Vehicle Risk Analysis**
```python
vehicles_analysis = analyzer.analyze_vehicle_types()
# Returns: crashes per vehicle type + fatality rates
```

**Insights**:
- Which vehicle types are overrepresented in crashes?
- Which have highest fatality rates?
- Older vs. newer vehicles
- Motorcycle vs. car comparison

**Traveler Demographics**
```python
demographics = analyzer.analyze_demographics()
# Returns: age distribution, gender, sobriety, injury levels
```

**Key Demographics**:
- Age group risk (teenagers, elderly highest risk)
- Gender patterns
- Impairment prevalence
- Safety equipment usage (seatbelt, helmet)

**Injury Patterns**
```python
injuries = analyzer.analyze_injury_patterns()
# Returns: injury severity by age group, party type
```

**Focus Areas**:
- Vulnerable road users (pedestrians, cyclists)
- Age-specific injury severity
- Occupant vs. pedestrian injury rates
- Effectiveness of safety equipment

---

## Data Preparation

### Step 1: Obtain SWITRS Data
Options:
- **TIMS Website**: https://tims.berkeley.edu (public access, some limitations)
- **California Highway Patrol**: Direct request
- **Local Law Enforcement**: May have county-specific data

### Step 2: Data Format
- Expected format: **CSV files** (one file per table type)
- Delimiter: Comma (`,`)
- Character encoding: UTF-8
- Missing values: Blank or `NULL`

### Step 3: Clean Data
```python
import pandas as pd

accidents = pd.read_csv('accidents.csv')

# Remove duplicates
accidents = accidents.drop_duplicates(subset=['case_id'])

# Handle missing geolocation
accidents = accidents.dropna(subset=['latitude', 'longitude'])

# Validate date format
accidents['accident_date'] = pd.to_datetime(accidents['accident_date'], errors='coerce')
```

### Step 4: Load into Analyzer
```python
from crash_analyzer import CrashDataAnalyzer

analyzer = CrashDataAnalyzer(accidents, vehicles, parties)
```

---

## Common Analyses

### Find Most Dangerous Intersections
```python
hotspots = analyzer.identify_hotspots(grid_size=0.005)  # ~500m cells
# Focus on grid cells with >10 crashes
```

### Compare Age Groups
```python
demographics = analyzer.analyze_demographics()
# Look for age groups with highest injury rates
```

### Identify Impairment Patterns
```python
# Filter parties where sobriety == 3 (intoxicated)
impaired_crashes = analyzer.parties[analyzer.parties['sobriety'] == 3]
# Analyze time of day, day of week, location
```

### Seasonal Trends
```python
temporal = analyzer.analyze_temporal_patterns()
# Compare summer vs. winter crash counts
```

### Vehicle Type Safety
```python
vehicles = analyzer.analyze_vehicle_types()
# Motorcycle fatality rate vs. car fatality rate
```

---

## Visualization Examples

### 1. Time Series: Crashes by Hour
```python
import matplotlib.pyplot as plt

hourly = analyzer.accidents.groupby('hour').size()
plt.bar(hourly.index, hourly.values)
plt.xlabel('Hour of Day')
plt.ylabel('Number of Crashes')
plt.title('Crashes by Hour of Day')
```

### 2. Geographic Hotmap (Folium)
```python
import folium

m = folium.Map(location=[37.0, -120.0], zoom_start=6)

for hotspot in analyzer.identify_hotspots():
    folium.CircleMarker(
        location=[hotspot['center_lat'], hotspot['center_lon']],
        radius=5 + hotspot['crash_count'] / 10,
        color='red' if hotspot['fatal_count'] > 0 else 'orange',
        popup=f"{hotspot['crash_count']} crashes"
    ).add_to(m)

m.save('hotspots_map.html')
```

### 3. Vehicle Risk Comparison (Plotly)
```python
import plotly.express as px

vehicle_risk = analyzer.analyze_vehicle_types()['vehicle_risk_analysis']
df = pd.DataFrame(vehicle_risk).T

px.bar(df, y='fatality_rate', title='Fatality Rate by Vehicle Type').show()
```

### 4. Age Demographics (Seaborn)
```python
import seaborn as sns

sns.histplot(analyzer.parties, x='age', hue='injury_level', kde=True)
plt.title('Injury Severity by Age')
```

---

## Key Metrics to Track

| Metric | Formula | Interpretation |
|---|---|---|
| Fatality Rate | Fatal Crashes / Total Crashes | Higher = more dangerous |
| Injury Rate | Injury Crashes / Total Crashes | Focus on prevention |
| Fatality per Crash | Total Fatalities / Total Crashes | Severity measure |
| Peak Hour | Hour with most crashes | Traffic management |
| Hotspot Density | Crashes per sq km | Infrastructure need |
| Demographics Risk | Deaths / Population Group | Target safety programs |

---

## API Reference

### CrashDataAnalyzer Methods

```python
class CrashDataAnalyzer:
    # Metadata
    analyze_temporal_patterns() -> Dict
    analyze_severity() -> Dict
    analyze_collision_types() -> Dict
    
    # Geospatial
    get_geographic_summary() -> Dict
    identify_hotspots(lat_col, lon_col, grid_size) -> List[Dict]
    
    # Vehicle/Traveler
    analyze_vehicle_types() -> Dict
    analyze_demographics() -> Dict
    analyze_injury_patterns() -> Dict
    
    # Combined
    generate_summary_report() -> Dict
```

---

## Tips for Analysis

✅ **Do**:
- Start with summary statistics before deep dives
- Validate data quality before conclusions
- Cross-reference multiple data sources
- Account for population denominators (risk per capita)
- Visualize before presenting findings

❌ **Don't**:
- Confuse correlation with causation
- Ignore data quality issues
- Over-interpret small sample sizes
- Make policy recommendations without context
- Ignore survivor bias in injury data

---

## Resources

- **TIMS**: https://tims.berkeley.edu/
- **SWITRS Documentation**: https://tims.berkeley.edu/help/crashdata/switrs-geocoding.php
- **Berkeley Safe Transportation Research & Education Center**: https://safetrec.berkeley.edu
- **CHP Statewide Integrated Traffic Records System**: https://www.chp.ca.gov/
