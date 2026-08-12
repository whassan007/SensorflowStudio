"""
Accident Data Cleaning & Normalization Layer
Cleans raw dataframes, normalizes severity codes, strips spaces, and standardizes schemas.
"""

import logging
import pandas as pd
from typing import Optional, List
from data_extractor import DataCleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccidentDataCleaner:
    """Provides routines to clean and standardize accident datasets."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def clean_dataset(self) -> pd.DataFrame:
        """
        Executes standard pipeline cleanups: column standardization, whitespace trim,
        duplicate removal, and severity normalization.

        Returns:
            pd.DataFrame of cleaned data.
        """
        logger.info(f"Cleaning dataset of size {self.df.shape}...")
        
        # 1. Standardize Column Names
        self.df = DataCleaner.standardize_column_names(self.df)
        
        # 2. Trim Whitespace
        self.df = DataCleaner.trim_whitespace(self.df)
        
        # 3. Remove duplicates
        if "case_id" in self.df.columns:
            self.df = DataCleaner.remove_duplicates(self.df, subset=["case_id"])
            
        # 4. Fill missing severities with 'Property Damage' (P) as fallback
        if "severity" in self.df.columns:
            self.df["severity"] = self.df["severity"].fillna("P")
            # Map standard values
            severity_map = {
                "fatal": "F", "injury": "I", "property damage": "P",
                "f": "F", "i": "I", "p": "P",
                "1": "F", "2": "I", "3": "P",
                1: "F", 2: "I", 3: "P"
            }
            self.df["severity"] = self.df["severity"].map(lambda x: severity_map.get(str(x).lower().strip(), "P"))

        # 5. Coordinate cleanups
        if "latitude" in self.df.columns:
            self.df["latitude"] = pd.to_numeric(self.df["latitude"], errors="coerce")
        if "longitude" in self.df.columns:
            self.df["longitude"] = pd.to_numeric(self.df["longitude"], errors="coerce")

        # Drop rows where coordinates are NaN to keep analysis accurate
        self.df = self.df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
        
        logger.info(f"Cleaning complete. Returning clean dataframe of size {self.df.shape}")
        return self.df
