"""
Universal Data Extraction Toolkit
Extract, validate, transform, and export data from multiple sources
"""

import requests
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
import time
from dataclasses import dataclass


# ========== API EXTRACTION ==========

class APIExtractor:
    """Extract data from REST APIs with pagination and retry support"""

    def __init__(self, base_url: str, auth: Optional[tuple] = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        if auth:
            self.session.auth = auth
        self.session.headers.update({
            'User-Agent': 'DataExtractor/1.0'
        })

    def fetch_paginated(self, endpoint: str, params: Optional[Dict] = None,
                       page_key: str = 'page', per_page_key: str = 'per_page') -> pd.DataFrame:
        """Fetch all pages from paginated API endpoint"""
        results = []
        page = 1
        params = params or {}

        while True:
            try:
                params[page_key] = page
                response = self.session.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()

                # Handle different response structures
                if isinstance(data, list):
                    if not data:
                        break
                    results.extend(data)
                elif isinstance(data, dict):
                    if 'data' in data:
                        if not data['data']:
                            break
                        results.extend(data['data'] if isinstance(data['data'], list) else [data['data']])
                    else:
                        results.append(data)
                else:
                    results.append(data)

                page += 1

                # Check if pagination exhausted
                if isinstance(data, dict) and 'total_pages' in data:
                    if page > data['total_pages']:
                        break

            except requests.exceptions.RequestException as e:
                print(f"Error fetching page {page}: {e}")
                break

        return pd.DataFrame(results) if results else pd.DataFrame()

    def fetch_with_retry(self, endpoint: str, max_retries: int = 3,
                        backoff_factor: float = 2, **kwargs) -> Dict:
        """Fetch with exponential backoff retry"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"{self.base_url}{endpoint}",
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = backoff_factor ** attempt
                print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                time.sleep(wait_time)


# ========== DATABASE EXTRACTION ==========

class DatabaseExtractor:
    """Extract data from SQL databases"""

    def __init__(self, connection_string: str):
        try:
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError("sqlalchemy required: pip install sqlalchemy")

        self.engine = create_engine(connection_string)

    def query_to_dataframe(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        return pd.read_sql(sql_query, self.engine)

    def chunked_extract(self, sql_query: str, chunksize: int = 10000) -> pd.DataFrame:
        """Extract large datasets in chunks"""
        chunks = []
        for chunk in pd.read_sql(sql_query, self.engine, chunksize=chunksize):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def table_to_dataframe(self, table_name: str) -> pd.DataFrame:
        """Load entire table"""
        return pd.read_sql_table(table_name, self.engine)


# ========== FILE EXTRACTION ==========

class FileExtractor:
    """Extract data from various file formats"""

    @staticmethod
    def extract_csv(filepath: str, encoding: str = 'utf-8', **kwargs) -> pd.DataFrame:
        """Extract CSV file"""
        return pd.read_csv(filepath, encoding=encoding, **kwargs)

    @staticmethod
    def extract_json(filepath: str) -> pd.DataFrame:
        """Extract JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame([data])

    @staticmethod
    def extract_jsonl(filepath: str) -> pd.DataFrame:
        """Extract JSON Lines file (one object per line)"""
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    @staticmethod
    def extract_excel(filepath: str, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """Extract Excel file"""
        return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)

    @staticmethod
    def extract_parquet(filepath: str) -> pd.DataFrame:
        """Extract Parquet file"""
        return pd.read_parquet(filepath)

    @staticmethod
    def batch_extract(pattern: str, filetype: str = 'csv') -> pd.DataFrame:
        """Extract multiple files matching pattern"""
        paths = list(Path('.').glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No files matching pattern: {pattern}")

        dfs = []
        for filepath in paths:
            try:
                if filetype == 'csv':
                    df = pd.read_csv(filepath)
                elif filetype == 'json':
                    df = pd.read_json(filepath)
                elif filetype == 'parquet':
                    df = pd.read_parquet(filepath)
                else:
                    raise ValueError(f"Unsupported filetype: {filetype}")

                df['source_file'] = str(filepath)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ========== WEB SCRAPING ==========

class WebExtractor:
    """Extract data from web pages"""

    def __init__(self, user_agent: Optional[str] = None):
        self.headers = {
            'User-Agent': user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def scrape_table(self, url: str, table_index: int = 0) -> pd.DataFrame:
        """Extract HTML table from webpage"""
        try:
            tables = pd.read_html(url)
            return tables[table_index] if table_index < len(tables) else pd.DataFrame()
        except Exception as e:
            print(f"Error scraping table: {e}")
            return pd.DataFrame()

    def scrape_all_tables(self, url: str) -> List[pd.DataFrame]:
        """Extract all tables from page"""
        try:
            return pd.read_html(url)
        except Exception as e:
            print(f"Error scraping tables: {e}")
            return []


# ========== DATA VALIDATION ==========

@dataclass
class ValidationReport:
    """Validation result"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    row_count: int
    column_count: int


class DataValidator:
    """Validate extracted data"""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.errors = []
        self.warnings = []

    def validate_required_columns(self, required_cols: List[str]) -> bool:
        """Check all required columns present"""
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            self.errors.append(f"Missing columns: {missing}")
        return len(missing) == 0

    def validate_no_nulls(self, columns: List[str]) -> bool:
        """Check no null values in columns"""
        has_errors = False
        for col in columns:
            if col not in self.df.columns:
                continue
            null_count = self.df[col].isna().sum()
            if null_count > 0:
                self.errors.append(f"{col}: {null_count} null values")
                has_errors = True
        return not has_errors

    def validate_data_types(self, type_map: Dict[str, str]) -> bool:
        """Validate column data types"""
        has_errors = False
        for col, expected_type in type_map.items():
            if col not in self.df.columns:
                continue

            try:
                self.df[col].astype(expected_type)
            except (ValueError, TypeError):
                self.errors.append(f"{col}: cannot convert to {expected_type}")
                has_errors = True
        return not has_errors

    def validate_unique(self, columns: List[str]) -> bool:
        """Check uniqueness constraints"""
        duplicates = self.df.duplicated(subset=columns).sum()
        if duplicates > 0:
            self.warnings.append(f"Duplicates in {columns}: {duplicates}")
        return duplicates == 0

    def validate_value_range(self, col: str, min_val: Optional[float] = None,
                            max_val: Optional[float] = None) -> bool:
        """Check values within range"""
        if col not in self.df.columns:
            return True

        has_errors = False
        if min_val is not None:
            below = (self.df[col] < min_val).sum()
            if below > 0:
                self.errors.append(f"{col}: {below} values below {min_val}")
                has_errors = True

        if max_val is not None:
            above = (self.df[col] > max_val).sum()
            if above > 0:
                self.errors.append(f"{col}: {above} values above {max_val}")
                has_errors = True

        return not has_errors

    def get_report(self) -> ValidationReport:
        """Get validation report"""
        return ValidationReport(
            valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
            row_count=len(self.df),
            column_count=len(self.df.columns)
        )


# ========== DATA TRANSFORMATION ==========

class DataCleaner:
    """Clean and normalize data"""

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None,
                         keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate rows"""
        return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop',
                             fill_value: Any = None) -> pd.DataFrame:
        """Handle missing data"""
        if strategy == 'drop':
            return df.dropna()
        elif strategy == 'fill':
            return df.fillna(fill_value)
        elif strategy == 'forward':
            return df.fillna(method='ffill')
        elif strategy == 'backward':
            return df.fillna(method='bfill')
        return df

    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Convert to snake_case, lowercase"""
        df.columns = (df.columns
                     .str.lower()
                     .str.replace(' ', '_')
                     .str.replace('-', '_')
                     .str.replace(r'[^\w_]', '', regex=True))
        return df

    @staticmethod
    def trim_whitespace(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove leading/trailing whitespace"""
        cols = columns or df.select_dtypes(include=['object']).columns
        for col in cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        return df

    @staticmethod
    def convert_dtypes(df: pd.DataFrame, type_map: Dict[str, str]) -> pd.DataFrame:
        """Convert multiple columns to specified types"""
        for col, dtype in type_map.items():
            if col in df.columns:
                try:
                    if dtype == 'date':
                        df[col] = pd.to_datetime(df[col])
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert {col} to {dtype}: {e}")
        return df


# ========== DATA EXPORT ==========

class DataExporter:
    """Export data in multiple formats"""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def to_csv(self, filepath: str, **kwargs) -> str:
        """Export to CSV"""
        self.df.to_csv(filepath, index=False, **kwargs)
        return filepath

    def to_json(self, filepath: str, orient: str = 'records', **kwargs) -> str:
        """Export to JSON"""
        self.df.to_json(filepath, orient=orient, **kwargs)
        return filepath

    def to_parquet(self, filepath: str, **kwargs) -> str:
        """Export to Parquet"""
        self.df.to_parquet(filepath, index=False, **kwargs)
        return filepath

    def to_excel(self, filepath: str, sheet_name: str = 'Sheet1', **kwargs) -> str:
        """Export to Excel"""
        self.df.to_excel(filepath, sheet_name=sheet_name, index=False, **kwargs)
        return filepath

    def to_dict_list(self) -> List[Dict]:
        """Convert to list of dictionaries"""
        return self.df.to_dict('records')


# ========== COMPLETE PIPELINE ==========

class DataExtractionPipeline:
    """Complete extraction, validation, transformation, and export pipeline"""

    def __init__(self, name: str):
        self.name = name
        self.data: Optional[pd.DataFrame] = None
        self.validation_report: Optional[ValidationReport] = None
        self.transformation_log: List[str] = []

    def extract(self, source: str, source_type: str = 'csv', **kwargs) -> 'DataExtractionPipeline':
        """Step 1: Extract data from source"""
        print(f"\n[{self.name}] Extracting from {source_type}...")

        try:
            if source_type == 'api':
                extractor = APIExtractor(source, **kwargs.get('api_params', {}))
                self.data = extractor.fetch_paginated(
                    kwargs.get('endpoint', ''),
                    params=kwargs.get('params')
                )
            elif source_type == 'database':
                extractor = DatabaseExtractor(source)
                self.data = extractor.query_to_dataframe(kwargs.get('sql_query', 'SELECT * FROM table'))
            elif source_type == 'csv':
                self.data = FileExtractor.extract_csv(source, **kwargs)
            elif source_type == 'json':
                self.data = FileExtractor.extract_json(source)
            elif source_type == 'jsonl':
                self.data = FileExtractor.extract_jsonl(source)
            elif source_type == 'excel':
                self.data = FileExtractor.extract_excel(source, **kwargs)
            elif source_type == 'parquet':
                self.data = FileExtractor.extract_parquet(source)
            elif source_type == 'web':
                extractor = WebExtractor()
                self.data = extractor.scrape_table(source, **kwargs)
            elif source_type == 'batch':
                self.data = FileExtractor.batch_extract(source, kwargs.get('filetype', 'csv'))
            else:
                raise ValueError(f"Unknown source type: {source_type}")

            print(f"  ✓ Extracted {len(self.data)} rows × {len(self.data.columns)} columns")

        except Exception as e:
            print(f"  ✗ Extraction failed: {e}")
            self.data = pd.DataFrame()

        return self

    def validate(self, validations: Dict[str, Any]) -> 'DataExtractionPipeline':
        """Step 2: Validate extracted data"""
        if self.data is None or self.data.empty:
            print(f"[{self.name}] No data to validate")
            return self

        print(f"[{self.name}] Validating...")

        validator = DataValidator(self.data)

        for validation_type, params in validations.items():
            if validation_type == 'required_columns':
                validator.validate_required_columns(params)
            elif validation_type == 'no_nulls':
                validator.validate_no_nulls(params)
            elif validation_type == 'data_types':
                validator.validate_data_types(params)
            elif validation_type == 'unique':
                validator.validate_unique(params)
            elif validation_type == 'value_range':
                for col, (min_v, max_v) in params.items():
                    validator.validate_value_range(col, min_v, max_v)

        self.validation_report = validator.get_report()

        if self.validation_report.valid:
            print(f"  ✓ Validation passed")
        else:
            print(f"  ⚠ Validation found {len(self.validation_report.errors)} errors:")
            for error in self.validation_report.errors[:5]:
                print(f"    - {error}")

        return self

    def transform(self, transformations: Dict[str, Any]) -> 'DataExtractionPipeline':
        """Step 3: Transform data"""
        if self.data is None or self.data.empty:
            return self

        print(f"[{self.name}] Transforming...")

        cleaner = DataCleaner()

        for transform_type, params in transformations.items():
            try:
                if transform_type == 'remove_duplicates':
                    self.data = cleaner.remove_duplicates(self.data, **params)
                    self.transformation_log.append("Removed duplicates")

                elif transform_type == 'standardize_names':
                    self.data = cleaner.standardize_column_names(self.data)
                    self.transformation_log.append("Standardized column names")

                elif transform_type == 'trim_whitespace':
                    self.data = cleaner.trim_whitespace(self.data, **params)
                    self.transformation_log.append("Trimmed whitespace")

                elif transform_type == 'handle_missing':
                    self.data = cleaner.handle_missing_values(self.data, **params)
                    self.transformation_log.append(f"Handled missing values ({params.get('strategy')})")

                elif transform_type == 'convert_types':
                    self.data = cleaner.convert_dtypes(self.data, params)
                    self.transformation_log.append("Converted data types")

            except Exception as e:
                print(f"  ⚠ Transform {transform_type} failed: {e}")

        print(f"  ✓ Applied {len(self.transformation_log)} transformations")
        return self

    def export(self, filepath: str, format: str = 'csv') -> str:
        """Step 4: Export data"""
        if self.data is None or self.data.empty:
            print(f"[{self.name}] No data to export")
            return ""

        print(f"[{self.name}] Exporting to {format}...")

        try:
            exporter = DataExporter(self.data)

            if format == 'csv':
                result = exporter.to_csv(filepath)
            elif format == 'json':
                result = exporter.to_json(filepath)
            elif format == 'parquet':
                result = exporter.to_parquet(filepath)
            elif format == 'excel':
                result = exporter.to_excel(filepath)
            else:
                raise ValueError(f"Unknown export format: {format}")

            print(f"  ✓ Exported to {result}")
            return result

        except Exception as e:
            print(f"  ✗ Export failed: {e}")
            return ""

    def get_summary(self) -> Dict:
        """Get pipeline summary"""
        return {
            'pipeline': self.name,
            'rows': len(self.data) if self.data is not None else 0,
            'columns': len(self.data.columns) if self.data is not None else 0,
            'validation': self.validation_report.__dict__ if self.validation_report else None,
            'transformations': self.transformation_log
        }


if __name__ == '__main__':
    print("Data Extraction Toolkit loaded.")
    print("Use: from data_extractor import DataExtractionPipeline")
