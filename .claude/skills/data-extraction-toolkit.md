---
name: data-extraction-toolkit
description: Extract, validate, and structure data from multiple sources using proven patterns
---

# Data Extraction Toolkit Skill

Universal framework for extracting data from APIs, databases, files, and web sources with validation, transformation, and quality assurance.

## Core Extraction Methods

### 1. REST API Extraction
```python
import requests
import pandas as pd

class APIExtractor:
    def __init__(self, base_url, auth=None):
        self.base_url = base_url
        self.auth = auth
        self.session = requests.Session()
        if auth:
            self.session.auth = auth

    def fetch_paginated(self, endpoint, params=None, page_key='page'):
        """Fetch all pages from paginated API"""
        results = []
        page = 1
        
        while True:
            params = params or {}
            params[page_key] = page
            
            response = self.session.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response structures
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict) and 'data' in data:
                results.extend(data['data'])
            else:
                results.append(data)
            
            # Check if more pages
            if len(data) == 0 or (isinstance(data, dict) and 'page' not in data):
                break
            
            page += 1
        
        return pd.DataFrame(results)

    def fetch_with_retry(self, endpoint, max_retries=3, backoff=2):
        """Fetch with exponential backoff retry"""
        import time
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = backoff ** attempt
                print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                time.sleep(wait_time)
```

### 2. Database Extraction
```python
import pandas as pd
from sqlalchemy import create_engine

class DatabaseExtractor:
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)

    def query_to_dataframe(self, sql_query):
        """Execute SQL and return DataFrame"""
        return pd.read_sql(sql_query, self.engine)

    def table_to_dataframe(self, table_name):
        """Load entire table"""
        return pd.read_table(table_name, self.engine)

    def chunked_extract(self, sql_query, chunksize=10000):
        """Extract large datasets in chunks"""
        chunks = []
        for chunk in pd.read_sql(sql_query, self.engine, chunksize=chunksize):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    def execute_procedure(self, procedure_name, params=None):
        """Execute stored procedure"""
        with self.engine.connect() as conn:
            if params:
                result = conn.execute(f"CALL {procedure_name}({params})")
            else:
                result = conn.execute(f"CALL {procedure_name}")
            return result.fetchall()

# Connection examples
# PostgreSQL: postgresql://user:password@localhost:5432/dbname
# MySQL: mysql+pymysql://user:password@localhost:3306/dbname
# SQLite: sqlite:///path/to/database.db
```

### 3. File-Based Extraction
```python
import pandas as pd
import json
import csv
from pathlib import Path

class FileExtractor:
    @staticmethod
    def extract_csv(filepath, encoding='utf-8', **kwargs):
        """Extract CSV with flexible parsing"""
        return pd.read_csv(filepath, encoding=encoding, **kwargs)

    @staticmethod
    def extract_json(filepath):
        """Extract JSON (file or list of objects)"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame([data])

    @staticmethod
    def extract_excel(filepath, sheet_name=0, **kwargs):
        """Extract Excel sheet"""
        return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)

    @staticmethod
    def extract_jsonl(filepath):
        """Extract JSON Lines (one object per line)"""
        records = []
        with open(filepath, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        return pd.DataFrame(records)

    @staticmethod
    def extract_parquet(filepath):
        """Extract Parquet (columnar format)"""
        return pd.read_parquet(filepath)

    @staticmethod
    def batch_extract(pattern, filetype='csv'):
        """Extract multiple files matching pattern"""
        paths = Path('.').glob(pattern)
        dfs = []
        
        for filepath in paths:
            if filetype == 'csv':
                df = pd.read_csv(filepath)
            elif filetype == 'json':
                df = pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported filetype: {filetype}")
            
            df['source_file'] = str(filepath)
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
```

### 4. Web Scraping Extraction
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

class WebExtractor:
    def __init__(self, user_agent=None):
        self.headers = {
            'User-Agent': user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def scrape_table(self, url, table_index=0):
        """Extract HTML table from webpage"""
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        tables = pd.read_html(response.text)
        return tables[table_index]

    def scrape_with_beautifulsoup(self, url, selector):
        """Scrape with CSS selector"""
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        elements = soup.select(selector)
        data = [elem.get_text(strip=True) for elem in elements]
        
        return pd.DataFrame({'content': data})

    def scrape_all_tables(self, url):
        """Extract all tables from page"""
        response = requests.get(url, headers=self.headers)
        tables = pd.read_html(response.text)
        return tables  # List of DataFrames

# Usage:
# web = WebExtractor()
# df = web.scrape_table('https://example.com/data')
```

### 5. Structured Data Extraction (from Text/HTML)
```python
import re
from typing import List, Dict

class StructuredExtractor:
    @staticmethod
    def extract_with_regex(text, pattern, group_names=None):
        """Extract data using regex"""
        matches = re.findall(pattern, text)
        
        if group_names and matches:
            return [dict(zip(group_names, match)) for match in matches]
        return matches

    @staticmethod
    def extract_key_value_pairs(text, delimiter=':', line_sep='\n'):
        """Extract key-value pairs from text"""
        data = {}
        for line in text.split(line_sep):
            if delimiter in line:
                key, value = line.split(delimiter, 1)
                data[key.strip()] = value.strip()
        return data

    @staticmethod
    def extract_columns(text, column_positions):
        """Extract fixed-width columns"""
        lines = text.strip().split('\n')
        data = []
        
        for line in lines:
            row = {}
            for col_name, (start, end) in column_positions.items():
                row[col_name] = line[start:end].strip()
            data.append(row)
        
        return pd.DataFrame(data)

# Regex Example:
# emails = StructuredExtractor.extract_with_regex(
#     text, 
#     r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
#     ['email']
# )
```

---

## Data Validation & Quality

### Validation Framework
```python
from typing import Any, Callable, Dict, List
import pandas as pd

class DataValidator:
    def __init__(self, dataframe):
        self.df = dataframe
        self.errors = []

    def validate_required_columns(self, required_cols):
        """Check all required columns present"""
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            self.errors.append(f"Missing columns: {missing}")
        return len(missing) == 0

    def validate_no_nulls(self, columns):
        """Check no null values in columns"""
        for col in columns:
            null_count = self.df[col].isna().sum()
            if null_count > 0:
                self.errors.append(f"{col}: {null_count} nulls")
        return len([e for e in self.errors if 'nulls' in e]) == 0

    def validate_data_types(self, type_map):
        """Validate column data types"""
        for col, expected_type in type_map.items():
            if col not in self.df.columns:
                continue
            
            try:
                self.df[col].astype(expected_type)
            except (ValueError, TypeError):
                self.errors.append(f"{col}: cannot convert to {expected_type}")

    def validate_value_range(self, col, min_val=None, max_val=None):
        """Check values within range"""
        if min_val is not None:
            below = (self.df[col] < min_val).sum()
            if below > 0:
                self.errors.append(f"{col}: {below} values below {min_val}")
        
        if max_val is not None:
            above = (self.df[col] > max_val).sum()
            if above > 0:
                self.errors.append(f"{col}: {above} values above {max_val}")

    def validate_unique(self, columns):
        """Check uniqueness constraints"""
        duplicates = self.df.duplicated(subset=columns).sum()
        if duplicates > 0:
            self.errors.append(f"Duplicates in {columns}: {duplicates}")

    def validate_custom(self, col, validator_func):
        """Apply custom validation function"""
        invalid = ~self.df[col].apply(validator_func)
        if invalid.sum() > 0:
            self.errors.append(f"{col}: {invalid.sum()} invalid values")

    def get_report(self):
        """Get validation report"""
        return {
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'row_count': len(self.df),
            'column_count': len(self.df.columns)
        }

# Usage:
# validator = DataValidator(df)
# validator.validate_required_columns(['id', 'name', 'date'])
# validator.validate_no_nulls(['id'])
# validator.validate_data_types({'age': 'int', 'email': 'string'})
# print(validator.get_report())
```

---

## Data Transformation

### Cleaning Pipeline
```python
import pandas as pd
import numpy as np

class DataCleaner:
    @staticmethod
    def remove_duplicates(df, subset=None, keep='first'):
        """Remove duplicate rows"""
        return df.drop_duplicates(subset=subset, keep=keep)

    @staticmethod
    def handle_missing_values(df, strategy='drop', fill_value=None):
        """Handle missing data"""
        if strategy == 'drop':
            return df.dropna()
        elif strategy == 'fill':
            return df.fillna(fill_value)
        elif strategy == 'forward':
            return df.fillna(method='ffill')
        elif strategy == 'backward':
            return df.fillna(method='bfill')

    @staticmethod
    def standardize_column_names(df):
        """Convert to snake_case, lowercase"""
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        return df

    @staticmethod
    def trim_whitespace(df, columns=None):
        """Remove leading/trailing whitespace"""
        cols = columns or df.select_dtypes(include=['object']).columns
        for col in cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        return df

    @staticmethod
    def normalize_text(df, columns, lowercase=True, remove_special=False):
        """Normalize text columns"""
        for col in columns:
            if lowercase:
                df[col] = df[col].str.lower()
            if remove_special:
                df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
        return df

    @staticmethod
    def convert_dtypes(df, type_map):
        """Convert multiple columns to specified types"""
        for col, dtype in type_map.items():
            if col in df.columns:
                if dtype == 'date':
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(dtype)
        return df

    @staticmethod
    def parse_datetime(df, col, format=None):
        """Parse datetime column"""
        df[col] = pd.to_datetime(df[col], format=format, errors='coerce')
        return df
```

### Data Enrichment
```python
class DataEnricher:
    @staticmethod
    def add_derived_columns(df, derivations):
        """Add calculated columns"""
        for col_name, func in derivations.items():
            df[col_name] = df.apply(func, axis=1)
        return df

    @staticmethod
    def categorize(df, col, bins, labels):
        """Categorize continuous values"""
        df[f'{col}_category'] = pd.cut(df[col], bins=bins, labels=labels)
        return df

    @staticmethod
    def join_datasets(df1, df2, on, how='inner'):
        """Merge two datasets"""
        return df1.merge(df2, on=on, how=how)

    @staticmethod
    def denormalize(df, columns_to_aggregate):
        """Denormalize data by aggregating"""
        group_by = [c for c in df.columns if c not in columns_to_aggregate]
        return df.groupby(group_by, as_index=False).agg(columns_to_aggregate)
```

---

## Export Formats

### Universal Export
```python
class DataExporter:
    def __init__(self, dataframe):
        self.df = dataframe

    def to_csv(self, filepath, **kwargs):
        """Export to CSV"""
        self.df.to_csv(filepath, index=False, **kwargs)
        return filepath

    def to_json(self, filepath, orient='records', **kwargs):
        """Export to JSON"""
        self.df.to_json(filepath, orient=orient, **kwargs)
        return filepath

    def to_parquet(self, filepath, **kwargs):
        """Export to Parquet (efficient columnar format)"""
        self.df.to_parquet(filepath, index=False, **kwargs)
        return filepath

    def to_excel(self, filepath, sheet_name='Sheet1', **kwargs):
        """Export to Excel"""
        self.df.to_excel(filepath, sheet_name=sheet_name, index=False, **kwargs)
        return filepath

    def to_sql(self, table_name, connection, if_exists='replace'):
        """Export to database"""
        self.df.to_sql(table_name, connection, if_exists=if_exists, index=False)
        return table_name

    def to_dict_list(self):
        """Convert to list of dictionaries"""
        return self.df.to_dict('records')

    def to_markdown_table(self):
        """Export as Markdown table"""
        return self.df.to_markdown(index=False)
```

---

## Complete Extraction Pipeline

### Template
```python
class DataExtractionPipeline:
    def __init__(self, name):
        self.name = name
        self.data = None
        self.validation_report = None
        self.transformation_log = []

    def extract(self, source, source_type='api', **kwargs):
        """Step 1: Extract data"""
        print(f"[{self.name}] Extracting from {source_type}...")
        
        if source_type == 'api':
            extractor = APIExtractor(source, **kwargs)
            self.data = extractor.fetch_paginated(**kwargs)
        elif source_type == 'database':
            extractor = DatabaseExtractor(source)
            self.data = extractor.query_to_dataframe(**kwargs)
        elif source_type == 'csv':
            self.data = FileExtractor.extract_csv(source, **kwargs)
        elif source_type == 'web':
            extractor = WebExtractor()
            self.data = extractor.scrape_table(source, **kwargs)
        
        print(f"  ✓ Extracted {len(self.data)} rows × {len(self.data.columns)} columns")
        return self

    def validate(self, validations):
        """Step 2: Validate extracted data"""
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
        
        self.validation_report = validator.get_report()
        
        if self.validation_report['valid']:
            print(f"  ✓ Validation passed")
        else:
            print(f"  ⚠ Validation errors:")
            for error in self.validation_report['errors']:
                print(f"    - {error}")
        
        return self

    def transform(self, transformations):
        """Step 3: Transform data"""
        print(f"[{self.name}] Transforming...")
        
        cleaner = DataCleaner()
        
        for transform_type, params in transformations.items():
            if transform_type == 'remove_duplicates':
                self.data = cleaner.remove_duplicates(self.data, **params)
                self.transformation_log.append(f"Removed duplicates")
            elif transform_type == 'standardize_names':
                self.data = cleaner.standardize_column_names(self.data)
                self.transformation_log.append(f"Standardized column names")
            elif transform_type == 'trim_whitespace':
                self.data = cleaner.trim_whitespace(self.data, **params)
                self.transformation_log.append(f"Trimmed whitespace")
            elif transform_type == 'convert_types':
                self.data = cleaner.convert_dtypes(self.data, params)
                self.transformation_log.append(f"Converted data types")
        
        print(f"  ✓ {len(self.transformation_log)} transformations applied")
        return self

    def export(self, filepath, format='csv'):
        """Step 4: Export data"""
        print(f"[{self.name}] Exporting to {format}...")
        
        exporter = DataExporter(self.data)
        
        if format == 'csv':
            result = exporter.to_csv(filepath)
        elif format == 'json':
            result = exporter.to_json(filepath)
        elif format == 'parquet':
            result = exporter.to_parquet(filepath)
        elif format == 'excel':
            result = exporter.to_excel(filepath)
        
        print(f"  ✓ Exported to {result}")
        return filepath

    def get_summary(self):
        """Get pipeline summary"""
        return {
            'pipeline': self.name,
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'validation': self.validation_report,
            'transformations': self.transformation_log
        }

# Usage:
# pipeline = DataExtractionPipeline('MyData')
# pipeline.extract('https://api.example.com/data', source_type='api')
#         .validate({
#             'required_columns': ['id', 'name', 'date'],
#             'no_nulls': ['id']
#         })
#         .transform({
#             'remove_duplicates': {'subset': ['id']},
#             'standardize_names': {},
#             'trim_whitespace': {}
#         })
#         .export('output.csv', format='csv')
# 
# print(pipeline.get_summary())
```

---

## Common Extraction Scenarios

### Scenario 1: Extract from Public API with Pagination
```python
pipeline = DataExtractionPipeline('PublicAPI')
pipeline.extract(
    'https://api.example.com',
    source_type='api',
    endpoint='/data',
    params={'limit': 100},
    page_key='page'
).validate({
    'required_columns': ['id', 'name']
}).transform({
    'standardize_names': {}
}).export('data.csv', format='csv')
```

### Scenario 2: Extract from SQL Database
```python
pipeline = DataExtractionPipeline('DatabaseExtract')
pipeline.extract(
    'postgresql://user:pass@localhost/mydb',
    source_type='database',
    sql_query='SELECT * FROM table WHERE date > now() - interval 7 day'
).validate({
    'no_nulls': ['user_id'],
    'data_types': {'created_at': 'datetime64'}
}).transform({
    'remove_duplicates': {'subset': ['user_id']},
    'convert_types': {'amount': 'float', 'created_at': 'date'}
}).export('weekly_data.parquet', format='parquet')
```

### Scenario 3: Extract Multiple CSVs
```python
pipeline = DataExtractionPipeline('BatchCSV')
all_data = FileExtractor.batch_extract('data/*.csv', filetype='csv')
pipeline.data = all_data

pipeline.validate({
    'required_columns': ['id', 'value']
}).transform({
    'remove_duplicates': {},
    'standardize_names': {}
}).export('combined.csv', format='csv')
```

### Scenario 4: Web Table Scraping
```python
pipeline = DataExtractionPipeline('WebScrape')
pipeline.extract(
    'https://example.com/table',
    source_type='web',
    table_index=0
).validate({
    'required_columns': ['col1', 'col2']
}).transform({
    'trim_whitespace': {}
}).export('scraped.csv', format='csv')
```

---

## Best Practices

✅ **Do**:
- Validate immediately after extraction
- Log all transformations applied
- Handle errors gracefully with retries
- Use appropriate data types (don't store dates as strings)
- Export in efficient formats (Parquet for large data)
- Document data sources and extraction date
- Implement pagination for large API endpoints
- Test pipelines with small datasets first

❌ **Don't**:
- Extract without validating
- Ignore missing values silently
- Store sensitive data in plain text
- Skip error handling
- Use inefficient formats for large datasets
- Mix multiple data types in one column
- Hardcode credentials in scripts
- Extract everything when you only need part of it

---

## Error Handling

```python
class ExtractionError(Exception):
    """Base extraction error"""
    pass

class ValidationError(ExtractionError):
    """Data validation failed"""
    pass

class TransformationError(ExtractionError):
    """Data transformation failed"""
    pass

# In pipeline:
try:
    pipeline.extract(source)
except ConnectionError as e:
    print(f"Failed to connect to source: {e}")
    # Retry or use backup source
except ExtractionError as e:
    print(f"Extraction failed: {e}")
    # Log and alert
```

---

## Performance Tips

| Scenario | Approach |
|----------|----------|
| Large API dataset | Use pagination + chunked processing |
| Large CSV file | Use `chunksize` parameter or Parquet |
| Multiple sources | Use threading/async for parallel extraction |
| Frequent updates | Implement delta/incremental extraction |
| Complex validation | Cache validation results |
| Real-time data | Use streaming/event-driven extraction |

---

**Use with**: Python scripts, notebooks, data pipelines, ETL workflows, analytics tools
