"""
Accident Data Ingestion Layer
Loads SWITRS accident datasets from multiple sources including CSV, JSON, Parquet, SQL databases, and paginated APIs.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
from data_extractor import DataExtractionPipeline, FileExtractor, APIExtractor, DatabaseExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccidentDataImporter:
    """Ingests accident datasets from various formats and database/API endpoints."""

    def __init__(self, pipeline_name: str = "AccidentIngestionPipeline"):
        self.pipeline = DataExtractionPipeline(pipeline_name)

    def import_from_file(self, filepath: str, format_type: Optional[str] = None) -> pd.DataFrame:
        """
        Loads dataset from a local file, auto-detecting the extension if format_type is not provided.

        Args:
            filepath: Path to the dataset file.
            format_type: Optional format override (e.g. 'csv', 'json', 'parquet').

        Returns:
            pd.DataFrame of loaded data.
        """
        logger.info(f"Importing accident data from file: {filepath}")
        if not format_type:
            _, ext = os.path.splitext(filepath)
            format_type = ext.lower().replace(".", "")
            if format_type == "jsonl":
                format_type = "jsonl"
            elif format_type not in ["csv", "json", "parquet", "xlsx", "xls"]:
                raise ValueError(f"Could not auto-detect format for: {filepath}")

        try:
            if format_type == "csv":
                self.pipeline.data = FileExtractor.extract_csv(filepath)
            elif format_type == "json":
                self.pipeline.data = FileExtractor.extract_json(filepath)
            elif format_type == "jsonl":
                self.pipeline.data = FileExtractor.extract_jsonl(filepath)
            elif format_type == "parquet":
                self.pipeline.data = FileExtractor.extract_parquet(filepath)
            elif format_type in ["xlsx", "xls"]:
                self.pipeline.data = FileExtractor.extract_excel(filepath)
            else:
                raise ValueError(f"Unsupported file format: {format_type}")

            logger.info(f"Successfully imported {len(self.pipeline.data)} records from {filepath}")
            return self.pipeline.data
        except Exception as e:
            logger.error(f"Failed to import data from file {filepath}: {str(e)}")
            raise

    def import_from_api(self, base_url: str, endpoint: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Fetches accident data from a paginated REST API endpoint.

        Args:
            base_url: The base URL of the API.
            endpoint: Paginated endpoint path.
            params: Optional query parameters.

        Returns:
            pd.DataFrame of fetched data.
        """
        logger.info(f"Importing accident data from API: {base_url}{endpoint}")
        try:
            extractor = APIExtractor(base_url)
            self.pipeline.data = extractor.fetch_paginated(endpoint, params=params)
            logger.info(f"Successfully fetched {len(self.pipeline.data)} records from API")
            return self.pipeline.data
        except Exception as e:
            logger.error(f"Failed to fetch data from API: {str(e)}")
            raise

    def import_from_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """
        Queries accident records from a SQL database.

        Args:
            connection_string: SQL connection URI.
            query: SQL query statement.

        Returns:
            pd.DataFrame of queried records.
        """
        logger.info(f"Importing accident data from database query: {query}")
        try:
            extractor = DatabaseExtractor(connection_string)
            self.pipeline.data = extractor.query_to_dataframe(query)
            logger.info(f"Successfully queried {len(self.pipeline.data)} records from database")
            return self.pipeline.data
        except Exception as e:
            logger.error(f"Failed to query database: {str(e)}")
            raise
