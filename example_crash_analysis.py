"""
Example: Crash Data Analysis with SWITRS Data

This script demonstrates how to use CrashDataAnalyzer for:
1. Metadata analysis (temporal patterns, severity)
2. Geospatial analysis (hotspots, county breakdowns)
3. Vehicle/traveler analysis (demographics, injury patterns)
"""

import pandas as pd
import json
from crash_analyzer import CrashDataAnalyzer, load_switrs_data


def example_with_sample_data():
    """Create a sample dataset and run analysis"""

    # Create sample crash data
    accidents = pd.DataFrame({
        'case_id': range(1, 101),
        'accident_date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'accident_time': ['14:30', '08:15', '22:45'] * 33 + ['10:00'],
        'latitude': [37.7749 + i*0.01 for i in range(100)],
        'longitude': [-122.4194 + i*0.01 for i in range(100)],
        'county': ['San Francisco', 'Los Angeles', 'San Diego', 'Sacramento'] * 25,
        'severity': ['I', 'P', 'F', 'I', 'P'] * 20,
        'collision_type': ['Same Direction - Sideswipe', 'Broadside', 'Hit Object',
                          'Rear End', 'Pedestrian'] * 20,
        'fatality_count': [0, 0, 1, 0, 0, 1, 0] * 14 + [0, 0],
        'injury_count': [1, 0, 2, 1, 0] * 20,
    })

    # Create sample vehicle data
    vehicles = pd.DataFrame({
        'case_id': range(1, 101),
        'vehicle_type': ['Car', 'Truck', 'Motorcycle', 'SUV', 'Van'] * 20,
        'vehicle_year': [2015 + i%10 for i in range(100)],
        'damage_extent': ['Severe', 'Moderate', 'Minor'] * 33 + ['Severe'],
    })

    # Create sample parties/travelers data
    parties = pd.DataFrame({
        'case_id': [i for i in range(1, 101) for _ in range(2)],  # 2 parties per crash
        'age': [25 + i%60 for i in range(200)],
        'gender': ['M', 'F'] * 100,
        'sobriety': [1, 2, 3, 1, 4] * 40,
        'injury_level': [0, 1, 2, 3] * 50,
        'party_type': ['Driver', 'Passenger'] * 100,
    })

    # Create analyzer
    analyzer = CrashDataAnalyzer(accidents, vehicles, parties)

    # Print results
    print("=" * 80)
    print("CRASH DATA ANALYSIS REPORT")
    print("=" * 80)

    # 1. METADATA ANALYSIS
    print("\n### METADATA ANALYSIS ###\n")

    temporal = analyzer.analyze_temporal_patterns()
    print("Temporal Patterns:")
    print(f"  Deadliest Hours: {temporal.get('deadliest_hours', {})}")
    print(f"  Crashes by Day of Week: {temporal.get('crashes_by_day', {})}")

    severity = analyzer.analyze_severity()
    print(f"\nSeverity Distribution:")
    for k, v in severity.get('severity_distribution', {}).items():
        print(f"  {k}: {v}")
    print(f"  Total Fatalities: {severity.get('total_fatalities', 0)}")
    print(f"  Total Injuries: {severity.get('total_injuries', 0)}")

    collisions = analyzer.analyze_collision_types()
    print(f"\nTop Collision Types:")
    for collision_type, count in list(collisions.get('top_collision_types', {}).items())[:5]:
        print(f"  {collision_type}: {count}")

    # 2. GEOSPATIAL ANALYSIS
    print("\n\n### GEOSPATIAL ANALYSIS ###\n")

    geo = analyzer.get_geographic_summary()
    print("Crashes by County:")
    for county, count in list(geo.get('crashes_by_county', {}).items())[:5]:
        print(f"  {county}: {count}")

    hotspots = analyzer.identify_hotspots()
    print(f"\nTop Hotspots (grid-based {len(hotspots)} found):")
    for i, hotspot in enumerate(hotspots[:3], 1):
        print(f"  {i}. Lat: {hotspot['center_lat']:.4f}, Lon: {hotspot['center_lon']:.4f} "
              f"- {hotspot['crash_count']} crashes, {hotspot['fatal_count']} fatal")

    # 3. VEHICLE/TRAVELER ANALYSIS
    print("\n\n### VEHICLE/TRAVELER ANALYSIS ###\n")

    vehicles_analysis = analyzer.analyze_vehicle_types()
    print("Vehicle Risk Analysis:")
    for vtype, risk in list(vehicles_analysis.get('vehicle_risk_analysis', {}).items())[:5]:
        data = risk
        print(f"  {vtype}:")
        print(f"    Total Crashes: {data['total_crashes']}")
        print(f"    Fatality Rate: {data['fatality_rate']:.2%}")

    demographics = analyzer.analyze_demographics()
    print(f"\nDemographics:")
    print(f"  Average Age: {demographics.get('age_statistics', {}).get('mean', 'N/A'):.1f}")
    print(f"  Crashes by Age Group: {demographics.get('crashes_by_age_group', {})}")
    print(f"  Crashes by Gender: {demographics.get('crashes_by_gender', {})}")
    print(f"  Crashes by Sobriety: {demographics.get('crashes_by_sobriety', {})}")

    # 4. COMPLETE SUMMARY REPORT
    print("\n\n### COMPLETE SUMMARY ###\n")
    report = analyzer.generate_summary_report()
    print(json.dumps(report['overview'], indent=2))


def example_with_csv_files():
    """Load actual SWITRS CSV files if available"""
    try:
        analyzer = load_switrs_data(
            'accidents.csv',
            'vehicles.csv',
            'parties.csv'
        )
        report = analyzer.generate_summary_report()
        print(json.dumps(report, indent=2, default=str))
    except FileNotFoundError as e:
        print(f"CSV files not found: {e}")
        print("Please provide SWITRS CSV files in the current directory:")
        print("  - accidents.csv")
        print("  - vehicles.csv")
        print("  - parties.csv")


if __name__ == '__main__':
    print("Running crash data analysis examples...\n")
    example_with_sample_data()

    # To use with actual SWITRS data, uncomment:
    # example_with_csv_files()
