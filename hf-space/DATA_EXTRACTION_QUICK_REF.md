# Data Extraction Toolkit — Quick Reference

## One-Liner Pipeline
```python
from data_extractor import DataExtractionPipeline
DataExtractionPipeline('Data').extract('input.csv').validate({'required_columns': ['id']}).transform({'standardize_names': {}}).export('output.csv')
```

---

## Extract from Source

### CSV
```python
from data_extractor import FileExtractor
df = FileExtractor.extract_csv('file.csv')
```

### JSON
```python
df = FileExtractor.extract_json('file.json')
df = FileExtractor.extract_jsonl('file.jsonl')
```

### Excel
```python
df = FileExtractor.extract_excel('file.xlsx', sheet_name=0)
```

### Parquet
```python
df = FileExtractor.extract_parquet('file.parquet')
```

### API
```python
from data_extractor import APIExtractor
api = APIExtractor('https://api.example.com')
df = api.fetch_paginated('/data', params={'limit': 100})
```

### Database
```python
from data_extractor import DatabaseExtractor
db = DatabaseExtractor('postgresql://user:pass@localhost/db')
df = db.query_to_dataframe('SELECT * FROM table')
```

### Batch (Multiple Files)
```python
df = FileExtractor.batch_extract('data/*.csv', filetype='csv')
```

---

## Validate Data

```python
from data_extractor import DataValidator
validator = DataValidator(df)

# Required columns
validator.validate_required_columns(['id', 'name'])

# No nulls
validator.validate_no_nulls(['id'])

# Data types
validator.validate_data_types({'age': 'int', 'date': 'datetime64'})

# Unique values
validator.validate_unique(['email'])

# Value range
validator.validate_value_range('age', min_val=0, max_val=150)

# Get report
report = validator.get_report()
print(f"Valid: {report.valid}, Errors: {report.errors}")
```

---

## Transform Data

```python
from data_extractor import DataCleaner
cleaner = DataCleaner()

# Remove duplicates
df = cleaner.remove_duplicates(df, subset=['id'])

# Standardize column names
df = cleaner.standardize_column_names(df)

# Trim whitespace
df = cleaner.trim_whitespace(df)

# Handle missing values
df = cleaner.handle_missing_values(df, strategy='drop')

# Convert types
df = cleaner.convert_dtypes(df, {'date': 'date', 'amount': 'float'})
```

---

## Export Data

```python
from data_extractor import DataExporter
exporter = DataExporter(df)

exporter.to_csv('output.csv')
exporter.to_json('output.json', orient='records')
exporter.to_parquet('output.parquet')  # Smallest file size
exporter.to_excel('output.xlsx', sheet_name='Data')
exporter.to_dict_list()  # List of dictionaries
```

---

## Complete Pipeline

```python
from data_extractor import DataExtractionPipeline

pipeline = DataExtractionPipeline('MyProject')

# Step 1: Extract
pipeline.extract('input.csv', source_type='csv')

# Step 2: Validate
pipeline.validate({
    'required_columns': ['id', 'name'],
    'no_nulls': ['id'],
    'data_types': {'age': 'int'}
})

# Step 3: Transform
pipeline.transform({
    'remove_duplicates': {},
    'standardize_names': {},
    'trim_whitespace': {}
})

# Step 4: Export
pipeline.export('output.csv', format='csv')

# View summary
print(pipeline.get_summary())
```

---

## Chained Pipeline

```python
DataExtractionPipeline('MyData') \
    .extract('input.csv') \
    .validate({'required_columns': ['id']}) \
    .transform({'standardize_names': {}}) \
    .export('output.csv')
```

---

## Source Types

| Type | Use | Example |
|------|-----|---------|
| `csv` | CSV files | `data.csv` |
| `json` | JSON | `data.json` |
| `jsonl` | JSON Lines | `data.jsonl` |
| `excel` | Excel | `data.xlsx` |
| `parquet` | Parquet | `data.parquet` |
| `database` | SQL DB | `postgresql://...` |
| `api` | REST API | `https://api.example.com` |
| `web` | HTML table | `https://example.com` |
| `batch` | Multiple files | `data/*.csv` |

---

## Export Formats

| Format | Best For | Speed | Size |
|--------|----------|-------|------|
| CSV | Excel, sharing | Fast | Large |
| JSON | APIs, web | Medium | Large |
| Parquet | Big data | Very fast | Small |
| Excel | Business | Slow | Medium |

---

## Validation Rules

```python
# Check columns exist
'required_columns': ['id', 'name']

# Check no nulls
'no_nulls': ['id', 'email']

# Check types
'data_types': {'age': 'int', 'salary': 'float'}

# Check unique
'unique': ['email', 'username']

# Check range
'value_range': {'age': (0, 150)}
```

---

## Transform Operations

```python
# Remove duplicate rows
'remove_duplicates': {}
'remove_duplicates': {'subset': ['id']}

# Standardize column names (snake_case)
'standardize_names': {}

# Trim whitespace
'trim_whitespace': {}
'trim_whitespace': {'columns': ['name']}

# Handle missing values
'handle_missing': {'strategy': 'drop'}  # or 'fill', 'forward', 'backward'

# Convert data types
'convert_types': {'date': 'date', 'amount': 'float'}
```

---

## Common Patterns

### Pattern 1: Clean CSV
```python
DataExtractionPipeline('Clean') \
    .extract('raw.csv') \
    .validate({'required_columns': ['id']}) \
    .transform({'remove_duplicates': {}, 'standardize_names': {}}) \
    .export('clean.csv')
```

### Pattern 2: API to CSV
```python
DataExtractionPipeline('API') \
    .extract('https://api.example.com', source_type='api', 
             endpoint='/data', params={'limit': 100}) \
    .validate({'no_nulls': ['id']}) \
    .export('api_data.csv')
```

### Pattern 3: Combine Files
```python
DataExtractionPipeline('Combine') \
    .extract('data/*.csv', source_type='batch', filetype='csv') \
    .validate({'unique': ['id']}) \
    .export('combined.csv')
```

### Pattern 4: Database to Parquet
```python
DataExtractionPipeline('DB') \
    .extract('postgresql://user:pass@host/db', source_type='database',
             sql_query='SELECT * FROM table') \
    .validate({'no_nulls': ['id']}) \
    .transform({'convert_types': {'amount': 'float'}}) \
    .export('data.parquet', format='parquet')
```

---

## Error Handling

```python
try:
    pipeline.extract('file.csv') \
            .validate({'required_columns': ['id']}) \
            .export('output.csv')
except Exception as e:
    print(f"Error: {e}")
    print(f"Data: {pipeline.data.head()}")
```

---

## Debugging

```python
# Check columns
print(df.columns)

# Check data types
print(df.dtypes)

# Check first rows
print(df.head())

# Check nulls
print(df.isnull().sum())

# Check shape
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
```

---

## Performance

| Scenario | Action |
|----------|--------|
| Large CSV | Use Parquet format export |
| Many files | Use batch extraction |
| Slow API | Use `fetch_with_retry()` |
| Memory issues | Use `chunked_extract()` |

---

## Dependencies

```bash
pip install pandas requests
pip install openpyxl          # Excel
pip install pyarrow            # Parquet
pip install sqlalchemy         # Database
```

---

## Files

```
Skill:      .claude/skills/data-extraction-toolkit.md
Module:     data_extractor.py
Examples:   example_data_extraction.py
Guide:      DATA_EXTRACTION_GUIDE.md
README:     DATA_EXTRACTION_TOOLKIT_README.md
```

---

**Print this page for quick reference!**
