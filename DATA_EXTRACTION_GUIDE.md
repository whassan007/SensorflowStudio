# Data Extraction Toolkit - Complete Guide

Universal framework for extracting data from multiple sources (APIs, databases, files, web) with built-in validation, transformation, and export.

## 🚀 Quick Start

### One-Minute Setup

```python
from data_extractor import DataExtractionPipeline

pipeline = DataExtractionPipeline('MyData')
pipeline.extract('data.csv', source_type='csv') \
        .validate({'required_columns': ['id', 'name']}) \
        .transform({'standardize_names': {}, 'remove_duplicates': {}}) \
        .export('output.csv', format='csv')

print(pipeline.get_summary())
```

### Supported Sources

| Source | Type | Example |
|--------|------|---------|
| CSV files | `csv` | `data.csv` |
| JSON files | `json` | `data.json` |
| JSON Lines | `jsonl` | `data.jsonl` |
| Excel | `excel` | `data.xlsx` |
| Parquet | `parquet` | `data.parquet` |
| SQL Database | `database` | `postgresql://user:pass@host/db` |
| REST API | `api` | `https://api.example.com` |
| Web tables | `web` | `https://example.com/table` |
| Multiple files | `batch` | `data/*.csv` |

---

## 📚 Core Components

### 1. Extractors

#### CSV Extraction
```python
from data_extractor import FileExtractor

df = FileExtractor.extract_csv('data.csv', encoding='utf-8')
df = FileExtractor.extract_csv('data.csv', sep=';', decimal=',')  # Delimiter variations
```

#### JSON Extraction
```python
df = FileExtractor.extract_json('data.json')  # Array or single object
df = FileExtractor.extract_jsonl('data.jsonl')  # One object per line
```

#### API Extraction
```python
from data_extractor import APIExtractor

api = APIExtractor('https://api.example.com', auth=('user', 'pass'))
df = api.fetch_paginated('/data', params={'limit': 100})
data = api.fetch_with_retry('/data', max_retries=3)
```

#### Database Extraction
```python
from data_extractor import DatabaseExtractor

db = DatabaseExtractor('postgresql://user:pass@localhost/mydb')
df = db.query_to_dataframe('SELECT * FROM users WHERE status = "active"')
df = db.chunked_extract('SELECT * FROM huge_table', chunksize=50000)
```

#### Batch Extraction
```python
# Combine multiple CSV files
df = FileExtractor.batch_extract('data/*.csv', filetype='csv')
```

### 2. Validation

```python
from data_extractor import DataValidator

validator = DataValidator(df)
validator.validate_required_columns(['id', 'name', 'email'])
validator.validate_no_nulls(['id', 'email'])
validator.validate_data_types({'age': 'int', 'created_at': 'datetime64'})
validator.validate_unique(['email'])
validator.validate_value_range('age', min_val=0, max_val=150)

report = validator.get_report()
print(f"Valid: {report.valid}")
print(f"Errors: {report.errors}")
```

### 3. Transformation

```python
from data_extractor import DataCleaner

cleaner = DataCleaner()
df = cleaner.remove_duplicates(df, subset=['id'])
df = cleaner.standardize_column_names(df)  # snake_case, lowercase
df = cleaner.trim_whitespace(df)
df = cleaner.handle_missing_values(df, strategy='drop')
df = cleaner.convert_dtypes(df, {'date_col': 'date', 'amount': 'float'})
```

### 4. Export

```python
from data_extractor import DataExporter

exporter = DataExporter(df)
exporter.to_csv('output.csv')
exporter.to_json('output.json', orient='records')
exporter.to_parquet('output.parquet')  # Efficient for large data
exporter.to_excel('output.xlsx', sheet_name='Data')
```

---

## 🔄 Complete Pipeline

### The 4-Step Pattern

```python
pipeline = DataExtractionPipeline('MyProject')

# Step 1: Extract
pipeline.extract('source.csv', source_type='csv')

# Step 2: Validate
pipeline.validate({
    'required_columns': ['id', 'name'],
    'no_nulls': ['id'],
    'data_types': {'amount': 'float'}
})

# Step 3: Transform
pipeline.transform({
    'remove_duplicates': {},
    'standardize_names': {},
    'trim_whitespace': {},
    'convert_types': {'date': 'date'}
})

# Step 4: Export
pipeline.export('output.csv', format='csv')

# View results
print(pipeline.get_summary())
```

### Inline Chaining

```python
pipeline = DataExtractionPipeline('Data') \
    .extract('input.csv') \
    .validate({'required_columns': ['id']}) \
    .transform({'standardize_names': {}}) \
    .export('output.csv')
```

---

## 📋 Validation Rules

### Required Columns
```python
validator.validate_required_columns(['id', 'name', 'email'])
# ✓ All columns must exist
```

### No Nulls
```python
validator.validate_no_nulls(['id', 'email'])
# ✓ No NULL/NaN values in these columns
```

### Data Types
```python
validator.validate_data_types({
    'age': 'int',
    'salary': 'float',
    'hired_date': 'datetime64',
    'name': 'object'
})
```

### Unique Values
```python
validator.validate_unique(['email', 'username'])
# ✓ Combination of these columns must be unique
```

### Value Range
```python
validator.validate_value_range('age', min_val=0, max_val=150)
validator.validate_value_range('percentage', min_val=0, max_val=100)
```

---

## 🛠️ Transformation Operations

| Operation | Purpose | Code |
|-----------|---------|------|
| **Remove Duplicates** | Eliminate duplicate rows | `cleaner.remove_duplicates(df, subset=['id'])` |
| **Standardize Names** | Convert to snake_case | `cleaner.standardize_column_names(df)` |
| **Trim Whitespace** | Remove leading/trailing spaces | `cleaner.trim_whitespace(df)` |
| **Handle Missing** | Drop or fill null values | `cleaner.handle_missing_values(df, strategy='drop')` |
| **Convert Types** | Cast columns to types | `cleaner.convert_dtypes(df, {'age': 'int'})` |

---

## 💾 Export Formats

### CSV
```python
exporter.to_csv('file.csv')
exporter.to_csv('file.csv', sep=';', encoding='utf-8')
```
**Best for**: Spreadsheets, Excel import, human-readable
**Size**: Large

### JSON
```python
exporter.to_json('file.json', orient='records')  # [{...}, {...}]
exporter.to_json('file.json', orient='table')    # {schema: ..., data: [...]}
```
**Best for**: Web APIs, JavaScript
**Size**: Large

### Parquet
```python
exporter.to_parquet('file.parquet')
```
**Best for**: Big data, data science (pandas, Spark)
**Size**: Small (compressed)

### Excel
```python
exporter.to_excel('file.xlsx', sheet_name='Data')
```
**Best for**: Business reports, presentations
**Size**: Medium

---

## 📊 Real-World Examples

### Example 1: Clean Customer Data
```python
pipeline = DataExtractionPipeline('CustomerCleanup')
pipeline.extract('raw_customers.csv') \
        .validate({
            'required_columns': ['customer_id', 'email', 'phone'],
            'no_nulls': ['customer_id', 'email'],
            'unique': ['customer_id', 'email']
        }) \
        .transform({
            'remove_duplicates': {'subset': ['email']},
            'standardize_names': {},
            'trim_whitespace': {'columns': ['name', 'address']},
            'convert_types': {'signup_date': 'date', 'lifetime_value': 'float'}
        }) \
        .export('clean_customers.csv', format='csv')
```

### Example 2: Fetch API Data
```python
pipeline = DataExtractionPipeline('APIFetch')
pipeline.extract(
    'https://api.github.com',
    source_type='api',
    endpoint='/users',
    params={'per_page': 100}
) \
.validate({
    'required_columns': ['login', 'id'],
    'no_nulls': ['id']
}) \
.transform({
    'standardize_names': {}
}) \
.export('github_users.json', format='json')
```

### Example 3: Combine Multiple Sources
```python
from data_extractor import FileExtractor
import pandas as pd

# Extract from multiple CSVs
df_sales = FileExtractor.extract_csv('sales_2024.csv')
df_returns = FileExtractor.extract_csv('returns_2024.csv')

# Combine
combined = pd.concat([df_sales, df_returns], ignore_index=True)

# Pipeline for cleaning
pipeline = DataExtractionPipeline('CombinedData')
pipeline.data = combined
pipeline.validate({
    'required_columns': ['transaction_id'],
    'no_nulls': ['transaction_id']
}) \
.transform({
    'remove_duplicates': {'subset': ['transaction_id']},
    'convert_types': {'amount': 'float', 'date': 'date'}
}) \
.export('combined_transactions.csv', format='csv')
```

### Example 4: Database Sync
```python
pipeline = DataExtractionPipeline('DBSync')
pipeline.extract(
    'mysql://user:pass@localhost/production',
    source_type='database',
    sql_query='SELECT * FROM orders WHERE created_at > NOW() - INTERVAL 1 DAY'
) \
.validate({
    'no_nulls': ['order_id', 'customer_id'],
    'value_range': {'amount': (0, 1000000)}
}) \
.transform({
    'convert_types': {'created_at': 'date', 'amount': 'float'}
}) \
.export('daily_orders.parquet', format='parquet')
```

---

## 🔧 Advanced Patterns

### Custom Validation
```python
validator = DataValidator(df)

# Add custom validators
def is_valid_email(email):
    return '@' in email

validator.validate_custom('email', is_valid_email)
```

### Conditional Transformations
```python
pipeline.data['age_group'] = pd.cut(
    pipeline.data['age'],
    bins=[0, 18, 35, 50, 100],
    labels=['<18', '18-34', '35-49', '50+']
)
```

### Error Handling
```python
try:
    pipeline.extract('data.csv') \
            .validate({'required_columns': ['id']}) \
            .export('output.csv')
except Exception as e:
    print(f"Pipeline failed: {e}")
    print(pipeline.data.head())  # Debug data
```

---

## 📈 Performance Tips

| Scenario | Recommendation |
|----------|-----------------|
| Large CSV (1GB+) | Use `chunksize` parameter or convert to Parquet |
| Many small files | Use batch extraction (`batch_extract`) |
| Slow API | Implement retry logic with `fetch_with_retry()` |
| Frequent exports | Use Parquet format (80% smaller than CSV) |
| Complex joins | Use database extraction with SQL joins |

---

## ❌ Common Issues & Solutions

### Issue: "Missing columns" error
```python
# Solution: Check column names
print(df.columns)
print(df.columns.str.lower())  # Check if case-sensitive
```

### Issue: "Cannot convert to int" error
```python
# Solution: Clean nulls first, then convert
df = df.dropna(subset=['numeric_col'])
df['numeric_col'] = df['numeric_col'].astype('int')
```

### Issue: Encoding errors
```python
# Solution: Try different encoding
df = FileExtractor.extract_csv('file.csv', encoding='latin-1')
```

### Issue: Duplicate rows after extract
```python
# Solution: Remove in transform step
pipeline.transform({'remove_duplicates': {'subset': ['id']}})
```

---

## 📚 API Reference

### DataExtractionPipeline
```python
pipeline = DataExtractionPipeline(name)
pipeline.extract(source, source_type, **kwargs)  # Load data
pipeline.validate(validations)                   # Check quality
pipeline.transform(transformations)              # Clean/enrich
pipeline.export(filepath, format)                # Save output
pipeline.get_summary()                           # Get stats
```

### FileExtractor (Static Methods)
```python
FileExtractor.extract_csv(filepath, **kwargs)
FileExtractor.extract_json(filepath)
FileExtractor.extract_jsonl(filepath)
FileExtractor.extract_excel(filepath, **kwargs)
FileExtractor.extract_parquet(filepath)
FileExtractor.batch_extract(pattern, filetype)
```

### APIExtractor
```python
api = APIExtractor(base_url, auth, timeout)
api.fetch_paginated(endpoint, params, page_key)
api.fetch_with_retry(endpoint, max_retries, backoff_factor)
```

### DatabaseExtractor
```python
db = DatabaseExtractor(connection_string)
db.query_to_dataframe(sql_query)
db.chunked_extract(sql_query, chunksize)
db.table_to_dataframe(table_name)
```

### DataValidator
```python
validator = DataValidator(df)
validator.validate_required_columns(cols)
validator.validate_no_nulls(cols)
validator.validate_data_types(type_map)
validator.validate_unique(cols)
validator.validate_value_range(col, min, max)
validator.get_report()  # ValidationReport
```

### DataCleaner (Static Methods)
```python
DataCleaner.remove_duplicates(df, subset, keep)
DataCleaner.standardize_column_names(df)
DataCleaner.trim_whitespace(df, columns)
DataCleaner.handle_missing_values(df, strategy, fill_value)
DataCleaner.convert_dtypes(df, type_map)
```

### DataExporter
```python
exporter = DataExporter(df)
exporter.to_csv(filepath, **kwargs)
exporter.to_json(filepath, orient, **kwargs)
exporter.to_parquet(filepath, **kwargs)
exporter.to_excel(filepath, sheet_name, **kwargs)
exporter.to_dict_list()  # List of dicts
```

---

## 🎯 When to Use What

**Use CSV when**: Need Excel compatibility, human-readable, sharing with non-technical users

**Use JSON when**: Working with APIs, JavaScript/web, hierarchical/nested data

**Use Parquet when**: Big data (>1GB), data science workflows, storage efficiency important

**Use Database**: Already using database, need to sync data, complex queries

**Use API**: Real-time data, third-party services, cloud data sources

---

## 📝 Checklists

### Pre-Extraction Checklist
- [ ] Data source is accessible
- [ ] Have proper credentials (if needed)
- [ ] Know expected schema/structure
- [ ] Understand data size (affects memory)
- [ ] Know update frequency (if using API)

### Validation Checklist
- [ ] Required columns present
- [ ] No unexpected nulls
- [ ] Data types correct
- [ ] Value ranges reasonable
- [ ] No duplicates (if unique required)

### Transformation Checklist
- [ ] Column names standardized
- [ ] Whitespace trimmed
- [ ] Missing values handled
- [ ] Data types converted
- [ ] Duplicates removed

### Export Checklist
- [ ] Output location writable
- [ ] File format appropriate
- [ ] Column order correct
- [ ] No sensitive data leaking
- [ ] File size reasonable

---

## 🔗 Integration Examples

### With Pandas
```python
import pandas as pd
from data_extractor import DataExtractionPipeline

pipeline = DataExtractionPipeline('Data')
pipeline.extract('data.csv')
df = pipeline.data
# Use pandas operations
df['new_col'] = df['col1'] + df['col2']
pipeline.export('result.csv')
```

### With SQLAlchemy
```python
from sqlalchemy import create_engine
from data_extractor import DataExtractionPipeline

engine = create_engine('sqlite:///database.db')
pipeline = DataExtractionPipeline('Data')
pipeline.extract('data.csv') \
        .transform({'standardize_names': {}}) \
        .data.to_sql('my_table', engine, if_exists='replace')
```

### With Jupyter Notebooks
```python
from data_extractor import DataExtractionPipeline

pipeline = DataExtractionPipeline('Analysis')
pipeline.extract('data.csv') \
        .validate({'required_columns': ['value']}) \
        .transform({'convert_types': {'value': 'float'}})

# Explore
pipeline.data['value'].describe()
pipeline.data.groupby('category')['value'].sum()
```

---

**Version**: 1.0  
**Last Updated**: 2026-07-26  
**Status**: Production Ready
