"""
Accident Data Validation Layer
Validates required columns, data types, coordinates, date ranges, and calculates a data quality score.
"""

import logging
from typing import Dict, Any, List
import pandas as pd
from data_extractor import DataValidator, ValidationReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccidentDataValidator:
    """Validates accident dataframes for schema compliance and geographic/temporal correctness."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.validator = DataValidator(self.df)

    def validate_schema(self) -> ValidationReport:
        """
        Runs standard validation checks on columns, types, coordinates, and nulls.

        Returns:
            ValidationReport outlining validity state, errors, and warnings.
        """
        logger.info("Starting accident schema validation...")
        
        # 1. Required Columns
        required = ["case_id", "accident_date", "severity", "latitude", "longitude"]
        self.validator.validate_required_columns(required)
        
        # 2. Check no null values in primary keys
        self.validator.validate_no_nulls(["case_id"])
        
        # 3. Data Types Check
        type_map = {
            "case_id": "object",
            "latitude": "float64",
            "longitude": "float64"
        }
        self.validator.validate_data_types(type_map)
        
        # 4. Check Coordinate Ranges
        if "latitude" in self.df.columns and "longitude" in self.df.columns:
            # Valid US lat/lon boundaries approx
            lat_invalid = self.df[(self.df["latitude"] < 24.0) | (self.df["latitude"] > 49.0)]
            lon_invalid = self.df[(self.df["longitude"] < -125.0) | (self.df["longitude"] > -66.0)]
            
            if not lat_invalid.empty:
                self.validator.errors.append(f"Latitude out of valid US range (24-49): {len(lat_invalid)} violations")
            if not lon_invalid.empty:
                self.validator.errors.append(f"Longitude out of valid US range (-125 to -66): {len(lon_invalid)} violations")

        # 5. Check Date Ranges
        if "accident_date" in self.df.columns:
            try:
                parsed_dates = pd.to_datetime(self.df["accident_date"], errors="coerce")
                null_dates = parsed_dates.isna().sum()
                if null_dates > 0:
                    self.validator.errors.append(f"accident_date: {null_dates} rows contain unparseable dates")
                
                # Check for future dates
                future_dates = (parsed_dates > pd.Timestamp.now()).sum()
                if future_dates > 0:
                    self.validator.errors.append(f"accident_date: {future_dates} rows contain future dates")
            except Exception as e:
                self.validator.errors.append(f"Failed to validate date column: {str(e)}")

        # 6. Calculate Quality Score
        report = self.validator.get_report()
        
        # Compute dynamic quality score (percentage of successful validation elements)
        total_checks = 6
        failed_checks = len(report.errors)
        quality_score = max(0.0, min(1.0, (total_checks - failed_checks) / total_checks))
        
        # Inject quality score attribute
        report.quality_score = quality_score
        
        logger.info(f"Validation complete. Valid: {report.valid}, Quality Score: {quality_score * 100:.1f}%")
        return report
