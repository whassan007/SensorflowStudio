---
name: crash-data-analyzer
description: Analyze crash data with metadata, geospatial, and vehicle/traveler insights
---

# Crash Data Analysis Skill

This skill helps you analyze crash data from SWITRS (Statewide Integrated Traffic Records System) and similar sources, with structured workflows for three key dimensions.

## Data Source Reference

**SWITRS Documentation**: https://tims.berkeley.edu/help/crashdata/switrs-geocoding.php

### SWITRS Data Structure

SWITRS provides California highway crash records with:
- **Accident data**: date, time, severity, location (county/district)
- **Vehicle data**: vehicle type, year, damage, direction
- **Party data**: age, gender, sobriety, injury severity, safety equipment
- **Victim data**: type (driver/passenger/pedestrian), injury level

---

## Analysis Workflows

### 1. METADATA ANALYSIS
Temporal patterns and incident classification

```python
# Time-based patterns
- Crashes by hour of day (peak times)
- Crashes by day of week
- Seasonal trends (monthly/quarterly)
- Trend over years

# Severity classification
- Count by severity level (fatal, injury, property damage)
- Injuries per crash by type
- Fatality rates by demographics

# Incident classification
- Primary cause (collision type)
- Weather conditions
- Road surface conditions
- Traffic control device status
```

**Key Queries**:
- Deadliest hours/days
- High-risk demographics
- Most common collision types
- Severity distribution

---

### 2. GEOSPATIAL ANALYSIS
Location-based patterns and clustering

```python
# Geographic aggregation
- Crashes by county/district
- Hotspot identification (lat/lon clustering)
- Corridor analysis (highways, intersections)
- Urban vs. rural comparison

# Mapping layers
- Crash density heatmaps
- High-severity zones
- Repeat location analysis (cluster detection)
- School/hospital/major intersection proximity

# Route analysis
- Crash frequency by direction
- Intersection danger ratings
- On-ramp/off-ramp incident patterns
```

**Recommended Tools**:
- Folium (interactive maps)
- GeoPandas (spatial analysis)
- Plotly (3D/clustered visualizations)
- DBSCAN/K-means (hotspot detection)

---

### 3. TRAVELER/VEHICLE DATA ANALYSIS
Demographics and vehicle characteristics

```python
# Vehicle factors
- Vehicle type distribution (car/truck/motorcycle)
- Age of vehicle in crashes
- Common manufacturers/models in incidents
- Damage severity by vehicle type

# Traveler demographics
- Age distribution of drivers/passengers
- Gender breakdown
- Sobriety/impairment patterns
- Seatbelt/helmet usage rates

# Injury patterns
- Injury type by age group
- Fatality rates by demographic
- Vulnerable road user analysis (pedestrians/cyclists)
- Occupant injury severity
```

**Key Metrics**:
- Fatality rate by vehicle type
- Risk by age group
- Impairment prevalence
- Safety equipment effectiveness

---

## Data Loading Template

```python
import pandas as pd
import geopandas as gpd

# Load SWITRS data (typically CSV or database)
accidents = pd.read_csv('accidents.csv')
vehicles = pd.read_csv('vehicles.csv')
parties = pd.read_csv('parties.csv')

# Common field mappings
# Accidents: case_id, accident_date, accident_time, latitude, longitude, 
#            severity, county, collision_type, weather_1, surface_condition
# Vehicles: case_id, vehicle_type, vehicle_year, damage_extent
# Parties: case_id, party_type, age, gender, sobriety, injury_level

# Merge for integrated analysis
crash_data = accidents.merge(
    vehicles, on='case_id', how='left'
).merge(
    parties, on='case_id', how='left'
)
```

---

## Analysis Templates

### Metadata Example
```python
# Deadliest times
crash_data['hour'] = pd.to_datetime(crash_data['accident_time']).dt.hour
top_hours = crash_data.groupby('hour')['fatality_count'].sum().nlargest(5)

# Severity by day of week
dow_severity = crash_data.groupby('day_of_week')['severity'].value_counts()
```

### Geospatial Example
```python
# Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    crash_data,
    geometry=gpd.points_from_xy(crash_data['longitude'], crash_data['latitude']),
    crs='EPSG:4326'
)

# Hotspot clustering
from sklearn.cluster import DBSCAN
coords = gdf[['latitude', 'longitude']].values
hotspots = DBSCAN(eps=0.01, min_samples=10).fit_predict(coords)

# Heatmap
import folium
m = folium.Map(location=[California_center], zoom_start=7)
for idx, row in crash_data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color='red' if row['severity'] == 'fatal' else 'orange'
    ).add_to(m)
```

### Vehicle/Traveler Example
```python
# Risk by vehicle type
risk_by_vehicle = crash_data.groupby('vehicle_type').agg({
    'fatality_count': 'sum',
    'injury_count': 'sum',
    'case_id': 'count'
})
risk_by_vehicle['fatality_rate'] = (
    risk_by_vehicle['fatality_count'] / risk_by_vehicle['case_id']
)

# Age-based injury patterns
age_groups = pd.cut(crash_data['age'], bins=[0, 16, 25, 35, 50, 65, 100])
age_injury = crash_data.groupby(age_groups)['injury_level'].value_counts()
```

---

## Visualization Recommendations

| Analysis Type | Chart | Library |
|---|---|---|
| Time patterns | Line chart (hourly/daily trends) | Matplotlib, Plotly |
| Severity distribution | Stacked bar chart | Plotly, Seaborn |
| Geographic hotspots | Heatmap or cluster map | Folium, Kepler.gl |
| Vehicle type risk | Grouped bar chart | Plotly |
| Age demographics | Histogram or violin plot | Matplotlib, Plotly |
| Injury by location | Choropleth map (by county) | Folium, Geopandas |

---

## Common Questions

**Q: How do I identify dangerous intersections?**
- Filter by collision_type containing "intersection"
- Group by location (lat/lon within 50m)
- Rank by fatality_count or crash frequency

**Q: What trends are worth reporting?**
- Increasing fatality rates in specific demographics
- Seasonal spikes (drunk driving holidays)
- Vehicle type overrepresentation in severe crashes
- Geographic clusters of preventable crashes

**Q: How to handle missing/geocoded data?**
- SWITRS has latitude/longitude fields (geocoded)
- Check for null coordinates and use county-level fallback
- Document data quality issues in analysis notes

---

## Next Steps

When ready to analyze:
1. Clarify data source (CSV, database, API)
2. Specify geographic focus (state, county, corridor)
3. Define time range
4. Choose primary analysis dimension (metadata/geo/vehicle-traveler)
5. Request visualizations or dashboards
