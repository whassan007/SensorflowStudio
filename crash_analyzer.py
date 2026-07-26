"""
SWITRS Crash Data Analyzer
Analyze crash data with metadata, geospatial, and vehicle/traveler dimensions
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime


class CrashDataAnalyzer:
    """Core crash data analysis engine"""

    def __init__(self, accidents_df: pd.DataFrame, vehicles_df: Optional[pd.DataFrame] = None,
                 parties_df: Optional[pd.DataFrame] = None):
        """
        Initialize analyzer with crash data

        Args:
            accidents_df: DataFrame with accident records (case_id, date, time, lat, lon, severity, etc.)
            vehicles_df: DataFrame with vehicle records (case_id, vehicle_type, year, damage, etc.)
            parties_df: DataFrame with party/occupant records (case_id, age, gender, injury, etc.)
        """
        self.accidents = accidents_df.copy()
        self.vehicles = vehicles_df.copy() if vehicles_df is not None else None
        self.parties = parties_df.copy() if parties_df is not None else None
        self.merged_data = self._merge_datasets()

    def _merge_datasets(self) -> pd.DataFrame:
        """Merge accidents with vehicles and parties on case_id"""
        df = self.accidents.copy()
        if self.vehicles is not None:
            df = df.merge(self.vehicles, on='case_id', how='left')
        if self.parties is not None:
            df = df.merge(self.parties, on='case_id', how='left')
        return df

    # ========== METADATA ANALYSIS ==========

    def analyze_temporal_patterns(self) -> Dict:
        """Analyze crashes by time (hour, day, month, year)"""
        results = {}

        if 'accident_time' in self.accidents.columns:
            # Parse time to hour
            times = pd.to_datetime(self.accidents['accident_time'], errors='coerce')
            self.accidents['hour'] = times.dt.hour
            results['crashes_by_hour'] = self.accidents['hour'].value_counts().sort_index().to_dict()
            results['deadliest_hours'] = self.accidents.groupby('hour')['fatality_count'].sum().nlargest(3).to_dict()

        if 'accident_date' in self.accidents.columns:
            dates = pd.to_datetime(self.accidents['accident_date'], errors='coerce')
            self.accidents['day_of_week'] = dates.dt.day_name()
            self.accidents['month'] = dates.dt.month
            self.accidents['year'] = dates.dt.year

            results['crashes_by_day'] = self.accidents['day_of_week'].value_counts().to_dict()
            results['crashes_by_month'] = self.accidents['month'].value_counts().sort_index().to_dict()
            results['crashes_by_year'] = self.accidents['year'].value_counts().sort_index().to_dict()

        return results

    def analyze_severity(self) -> Dict:
        """Analyze crash severity distribution"""
        severity_map = {
            'F': 'Fatal',
            'I': 'Injury',
            'P': 'Property Damage Only',
            1: 'Fatal',
            2: 'Injury',
            3: 'Property Damage'
        }

        results = {}
        if 'severity' in self.accidents.columns:
            severity_counts = self.accidents['severity'].value_counts()
            results['severity_distribution'] = {
                severity_map.get(k, str(k)): int(v)
                for k, v in severity_counts.items()
            }
            results['total_crashes'] = int(self.accidents.shape[0])

            if 'fatality_count' in self.accidents.columns:
                results['total_fatalities'] = int(self.accidents['fatality_count'].sum())
            if 'injury_count' in self.accidents.columns:
                results['total_injuries'] = int(self.accidents['injury_count'].sum())

        return results

    def analyze_collision_types(self) -> Dict:
        """Analyze primary collision types"""
        if 'collision_type' not in self.accidents.columns:
            return {}

        collision_counts = self.accidents['collision_type'].value_counts()
        fatal_collisions = self.accidents[self.accidents['severity'].isin(['F', 1])]['collision_type'].value_counts()

        return {
            'top_collision_types': collision_counts.head(10).to_dict(),
            'deadliest_collision_types': fatal_collisions.head(5).to_dict()
        }

    # ========== GEOSPATIAL ANALYSIS ==========

    def get_geographic_summary(self) -> Dict:
        """Geographic breakdown by county/district"""
        results = {}

        if 'county' in self.accidents.columns:
            county_crashes = self.accidents['county'].value_counts()
            results['crashes_by_county'] = county_crashes.head(10).to_dict()

            # Severity by county
            county_severity = self.accidents.groupby('county')['severity'].value_counts().unstack(fill_value=0)
            results['severity_by_county'] = county_severity.to_dict()

        if 'district' in self.accidents.columns:
            district_crashes = self.accidents['district'].value_counts()
            results['crashes_by_district'] = district_crashes.head(10).to_dict()

        return results

    def identify_hotspots(self, lat_col: str = 'latitude', lon_col: str = 'longitude',
                         grid_size: float = 0.01) -> List[Dict]:
        """
        Identify geographic hotspots using grid clustering

        Args:
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            grid_size: Size of grid cells (in degrees, ~1km = 0.009 degrees)

        Returns:
            List of hotspots with location and crash count
        """
        try:
            valid_geo = self.accidents[
                (self.accidents[lat_col].notna()) &
                (self.accidents[lon_col].notna())
            ].copy()

            if valid_geo.empty:
                return []

            # Create grid cells
            valid_geo['lat_grid'] = (valid_geo[lat_col] / grid_size).astype(int) * grid_size
            valid_geo['lon_grid'] = (valid_geo[lon_col] / grid_size).astype(int) * grid_size

            # Count crashes per grid cell
            hotspots = valid_geo.groupby(['lat_grid', 'lon_grid']).agg({
                'case_id': 'count',
                'severity': lambda x: (x.isin(['F', 1])).sum(),  # Fatal count
                lat_col: 'mean',
                lon_col: 'mean'
            }).reset_index(drop=True)

            hotspots.columns = ['crash_count', 'fatal_count', 'center_lat', 'center_lon']
            hotspots = hotspots.sort_values('crash_count', ascending=False)

            return hotspots.head(20).to_dict('records')

        except Exception as e:
            return {'error': str(e)}

    # ========== VEHICLE/TRAVELER ANALYSIS ==========

    def analyze_vehicle_types(self) -> Dict:
        """Analyze crashes by vehicle type"""
        if self.vehicles is None or 'vehicle_type' not in self.vehicles.columns:
            return {}

        vehicle_data = self.merged_data.dropna(subset=['vehicle_type'])
        vehicle_counts = vehicle_data['vehicle_type'].value_counts()

        # Calculate risk metrics
        risk_analysis = {}
        for vtype in vehicle_counts.index[:10]:
            vtype_crashes = vehicle_data[vehicle_data['vehicle_type'] == vtype]
            total = len(vtype_crashes)
            fatal = len(vtype_crashes[vtype_crashes['severity'].isin(['F', 1])])

            risk_analysis[str(vtype)] = {
                'total_crashes': int(total),
                'fatal_crashes': int(fatal),
                'fatality_rate': round(fatal / total, 4) if total > 0 else 0
            }

        return {
            'vehicle_crash_distribution': vehicle_counts.head(10).to_dict(),
            'vehicle_risk_analysis': risk_analysis
        }

    def analyze_demographics(self) -> Dict:
        """Analyze traveler demographics (age, gender, etc.)"""
        if self.parties is None:
            return {}

        results = {}

        if 'age' in self.parties.columns:
            age_stats = self.parties['age'].describe().to_dict()
            results['age_statistics'] = {k: float(v) if not np.isnan(v) else None
                                        for k, v in age_stats.items()}

            # Age groups
            age_groups = pd.cut(self.parties['age'],
                               bins=[0, 16, 25, 35, 50, 65, 100],
                               labels=['<16', '16-24', '25-34', '35-49', '50-64', '65+'])
            results['crashes_by_age_group'] = age_groups.value_counts().sort_index().to_dict()

        if 'gender' in self.parties.columns:
            results['crashes_by_gender'] = self.parties['gender'].value_counts().to_dict()

        if 'sobriety' in self.parties.columns:
            sobriety_mapping = {1: 'Sober', 2: 'Drinking', 3: 'Intoxicated', 4: 'Unknown'}
            sobriety_counts = self.parties['sobriety'].value_counts()
            results['crashes_by_sobriety'] = {
                sobriety_mapping.get(k, str(k)): int(v)
                for k, v in sobriety_counts.items()
            }

        return results

    def analyze_injury_patterns(self) -> Dict:
        """Analyze injury patterns by demographics"""
        if self.parties is None or 'injury_level' not in self.parties.columns:
            return {}

        results = {}

        # Overall injury distribution
        injury_counts = self.parties['injury_level'].value_counts()
        results['injury_distribution'] = injury_counts.to_dict()

        # Injuries by age group (if available)
        if 'age' in self.parties.columns:
            age_groups = pd.cut(self.parties['age'],
                               bins=[0, 16, 25, 35, 50, 65, 100],
                               labels=['<16', '16-24', '25-34', '35-49', '50-64', '65+'])
            injury_by_age = self.parties.groupby(age_groups)['injury_level'].value_counts()
            results['injury_by_age_group'] = injury_by_age.to_dict()

        return results

    # ========== COMBINED ANALYSIS ==========

    def generate_summary_report(self) -> Dict:
        """Generate comprehensive analysis summary"""
        return {
            'overview': {
                'total_crashes': int(self.accidents.shape[0]),
                'analysis_date': datetime.now().isoformat()
            },
            'metadata': {
                'temporal': self.analyze_temporal_patterns(),
                'severity': self.analyze_severity(),
                'collision_types': self.analyze_collision_types()
            },
            'geospatial': self.get_geographic_summary(),
            'hotspots': self.identify_hotspots(),
            'vehicles': self.analyze_vehicle_types(),
            'demographics': self.analyze_demographics(),
            'injuries': self.analyze_injury_patterns()
        }


# ========== UTILITY FUNCTIONS ==========

def load_switrs_data(accidents_path: str, vehicles_path: Optional[str] = None,
                     parties_path: Optional[str] = None) -> CrashDataAnalyzer:
    """
    Load SWITRS CSV files and create analyzer

    Args:
        accidents_path: Path to accidents CSV
        vehicles_path: Path to vehicles CSV (optional)
        parties_path: Path to parties CSV (optional)

    Returns:
        CrashDataAnalyzer instance
    """
    accidents = pd.read_csv(accidents_path)
    vehicles = pd.read_csv(vehicles_path) if vehicles_path else None
    parties = pd.read_csv(parties_path) if parties_path else None

    return CrashDataAnalyzer(accidents, vehicles, parties)


if __name__ == '__main__':
    # Example usage
    print("Crash Data Analyzer loaded. Use CrashDataAnalyzer class to analyze crash data.")
    print("Example: analyzer = CrashDataAnalyzer(accidents_df)")
    print("         report = analyzer.generate_summary_report()")
