# Data Extraction Toolkit — Complete Package

Universal data extraction framework for pulling data from **APIs**, **databases**, **files**, and **web sources** with built-in validation, transformation, and export.

## 📦 What's Included

### 1. **Skill File** (`.claude/skills/data-extraction-toolkit.md`)
Claude skill with:
- Complete extraction patterns for all source types
- Validation templates and best practices
- Transformation recipes
- Export format guide
- Real-world scenarios

### 2. **Python Module** (`data_extractor.py`)
Production-ready classes:
- `APIExtractor` — REST API data with pagination & retry
- `DatabaseExtractor` — SQL database queries
- `FileExtractor` — CSV, JSON, Excel, Parquet
- `WebExtractor` — HTML table scraping
- `DataValidator` — Quality checks
- `DataCleaner` — Normalization & transformation
- `DataExporter` — Multi-format export
- `DataExtractionPipeline` — Complete 4-step workflow

### 3. **Examples** (`example_data_extraction.py`)
7 runnable examples:
1. CSV extraction with pipeline
2. API data fetching
3. Batch file combination
4. JSON processing
5. Data cleaning operations
6. Validation framework
7. Multi-format export

### 4. **Complete Guide** (`DATA_EXTRACTION_GUIDE.md`)
- Quick start (1 minute)
- Source types reference
- Component documentation
- Real-world examples
- Performance tips
- API reference

---

## 🚀 Get Started (60 seconds)

### Step 1: Import
```python
from data_extractor import DataExtractionPipeline
```

### Step 2: Create Pipeline
```python
pipeline = DataExtractionPipeline('MyProject')
```

### Step 3: Extract → Validate → Transform → Export
```python
pipeline.extract('data.csv') \
        .validate({'required_columns': ['id', 'name']}) \
        .transform({'standardize_names': {}, 'remove_duplicates': {}}) \
        .export('output.csv', format='csv')
```

### Step 4: Check Results
```python
print(pipeline.get_summary())
```

---

## 📊 Supported Data Sources

### Files
- CSV (with flexible parsing)
- JSON (array or objects)
- JSON Lines (one per line)
- Excel (XLSX/XLS)
- Parquet (columnar)
- **Batch**: Multiple files with glob pattern

### Databases
- PostgreSQL
- MySQL
- SQLite
- Any SQLAlchemy-compatible DB

### APIs
- REST endpoints with pagination
- Automatic retry with backoff
- Custom headers & auth

### Web
- HTML table scraping
- Multiple tables per page

---

## 🔄 The 4-Step Pipeline

### Step 1: Extract
```python
pipeline.extract(source, source_type='csv')
```
- Supported types: `csv`, `json`, `excel`, `database`, `api`, `web`, `batch`
- Returns data in pandas DataFrame

### Step 2: Validate
```python
pipeline.validate({
    'required_columns': ['id', 'name'],
    'no_nulls': ['id'],
    'data_types': {'age': 'int'},
    'unique': ['email']
})
```
- Checks data quality
- Reports errors and warnings
- Doesn't block (data kept for inspection)

### Step 3: Transform
```python
pipeline.transform({
    'remove_duplicates': {},
    'standardize_names': {},
    'trim_whitespace': {},
    'handle_missing': {'strategy': 'drop'},
    'convert_types': {'date': 'date'}
})
```
- Clean and normalize
- Handle missing values
- Standardize formats

### Step 4: Export
```python
pipeline.export('output.csv', format='csv')
# Formats: csv, json, parquet, excel
```
- Save to file
- Multiple format support
- Efficient compression (Parquet)

---

## 💡 Common Patterns

### Pattern 1: Clean CSV Data
```python
DataExtractionPipeline('Customers') \
    .extract('raw_customers.csv') \
    .validate({'required_columns': ['id', 'email']}) \
    .transform({'remove_duplicates': {}, 'standardize_names': {}}) \
    .export('clean_customers.csv')
```

### Pattern 2: Fetch & Process API Data
```python
DataExtractionPipeline('API') \
    .extract('https://api.example.com', source_type='api', 
             endpoint='/data', params={'limit': 100}) \
    .validate({'no_nulls': ['id']}) \
    .export('api_data.json', format='json')
```

### Pattern 3: Combine Multiple Files
```python
DataExtractionPipeline('Batch') \
    .extract('data/*.csv', source_type='batch', filetype='csv') \
    .validate({'unique': ['id']}) \
    .transform({'remove_duplicates': {'subset': ['id']}}) \
    .export('combined.csv')
```

### Pattern 4: Database Sync
```python
DataExtractionPipeline('DBSync') \
    .extract('postgresql://user:pass@localhost/db', 
             source_type='database',
             sql_query='SELECT * FROM orders WHERE created_at > NOW() - INTERVAL 1 DAY') \
    .validate({'no_nulls': ['order_id']}) \
    .transform({'convert_types': {'amount': 'float'}}) \
    .export('daily_orders.parquet', format='parquet')
```

---

## 🛠️ Extractor Classes

### FileExtractor (Static Methods)
```python
from data_extractor import FileExtractor

df = FileExtractor.extract_csv('file.csv', encoding='utf-8')
df = FileExtractor.extract_json('file.json')
df = FileExtractor.extract_jsonl('file.jsonl')
df = FileExtractor.extract_excel('file.xlsx', sheet_name=0)
df = FileExtractor.extract_parquet('file.parquet')
df = FileExtractor.batch_extract('data/*.csv', filetype='csv')
```

### APIExtractor
```python
from data_extractor import APIExtractor

api = APIExtractor('https://api.example.com', auth=('user', 'pass'))
df = api.fetch_paginated('/data', params={'limit': 100}, page_key='page')
data = api.fetch_with_retry('/data', max_retries=3, backoff_factor=2)
```

### DatabaseExtractor
```python
from data_extractor import DatabaseExtractor

db = DatabaseExtractor('postgresql://user:pass@localhost:5432/mydb')
df = db.query_to_dataframe('SELECT * FROM users')
df = db.chunked_extract('SELECT * FROM huge_table', chunksize=50000)
```

### WebExtractor
```python
from data_extractor import WebExtractor

web = WebExtractor()
df = web.scrape_table('https://example.com/table', table_index=0)
tables = web.scrape_all_tables('https://example.com')
```

---

## ✅ Validation

```python
validator = DataValidator(df)

# Check columns exist
validator.validate_required_columns(['id', 'name'])

# Check no nulls
validator.validate_no_nulls(['id', 'email'])

# Check data types
validator.validate_data_types({
    'age': 'int',
    'salary': 'float',
    'hired': 'datetime64'
})

# Check uniqueness
validator.validate_unique(['email'])

# Check value ranges
validator.validate_value_range('age', min_val=0, max_val=150)

# Get results
report = validator.get_report()
print(f"Valid: {report.valid}")
print(f"Errors: {report.errors}")
```

---

## 🧹 Transformation

```python
cleaner = DataCleaner()

# Remove duplicate rows
df = cleaner.remove_duplicates(df, subset=['id'])

# Standardize column names (snake_case)
df = cleaner.standardize_column_names(df)

# Trim whitespace
df = cleaner.trim_whitespace(df)

# Handle missing values
df = cleaner.handle_missing_values(df, strategy='drop')  # or 'fill', 'forward', 'backward'

# Convert data types
df = cleaner.convert_dtypes(df, {
    'date': 'date',
    'amount': 'float',
    'id': 'int'
})
```

---

## 💾 Export Formats

| Format | Use Case | Size | Performance |
|--------|----------|------|-------------|
| **CSV** | Excel, spreadsheets | Large | Fast read |
| **JSON** | APIs, web, JavaScript | Large | Standard |
| **Parquet** | Big data, Spark | Small (compressed) | Fastest |
| **Excel** | Business reports | Medium | Slower |

```python
exporter = DataExporter(df)

exporter.to_csv('data.csv')
exporter.to_json('data.json', orient='records')
exporter.to_parquet('data.parquet')  # 80% smaller than CSV
exporter.to_excel('data.xlsx', sheet_name='Data')
exporter.to_dict_list()  # List of dictionaries
```

---

## 🔗 Use with Claude Code

### With Skill
```
/data-extraction-toolkit
```

Prompt Claude with:
> "I need to extract customer data from an API, validate it, and export to CSV. Help me build this."

Claude will reference the skill and provide complete code.

### Standalone Module
Use directly in your Python scripts:
```python
from data_extractor import DataExtractionPipeline
```

---

## 📚 Dependencies

### Required
```
pandas
requests
```

### Optional (for full functionality)
```
openpyxl          # Excel support
pyarrow            # Parquet support
sqlalchemy         # Database support
beautifulsoup4     # Web scraping
```

Install all:
```bash
pip install pandas requests openpyxl pyarrow sqlalchemy beautifulsoup4
```

---

## 📋 Real-World Scenarios

### E-Commerce: Customer Sync
```python
DataExtractionPipeline('CustomerSync') \
    .extract('postgresql://db/customers', source_type='database',
             sql_query='SELECT * FROM customers WHERE updated_at > NOW() - INTERVAL 1 DAY') \
    .validate({
        'required_columns': ['customer_id', 'email'],
        'unique': ['email']
    }) \
    .transform({
        'remove_duplicates': {'subset': ['email']},
        'convert_types': {'created_at': 'date'}
    }) \
    .export('daily_customers.parquet', format='parquet')
```

### Analytics: Data Lake Ingestion
```python
DataExtractionPipeline('DataLake') \
    .extract('data/raw/*.csv', source_type='batch', filetype='csv') \
    .validate({'no_nulls': ['event_id', 'timestamp']}) \
    .transform({
        'standardize_names': {},
        'convert_types': {'timestamp': 'date', 'value': 'float'}
    }) \
    .export('lake/processed/events.parquet', format='parquet')
```

### API Integration: Third-Party Data
```python
DataExtractionPipeline('ThirdParty') \
    .extract('https://api.partners.com', source_type='api',
             endpoint='/v1/transactions', params={'limit': 1000}) \
    .validate({'no_nulls': ['transaction_id']}) \
    .transform({'standardize_names': {}}) \
    .export('partner_data.json', format='json')
```

### Data Migration: Legacy System
```python
DataExtractionPipeline('Migration') \
    .extract('mysql://legacy_db/old_users', source_type='database',
             sql_query='SELECT * FROM users') \
    .validate({'required_columns': ['id', 'username']}) \
    .transform({
        'remove_duplicates': {'subset': ['email']},
        'convert_types': {'created_at': 'date'}
    }) \
    .export('new_system_import.csv', format='csv')
```

---

## ⚡ Performance Tips

| Task | Recommendation |
|------|-----------------|
| **Large CSV** (1GB+) | Use Parquet export; use `chunksize` for extraction |
| **Many files** | Batch extraction with glob patterns |
| **Slow API** | Implement pagination; use retry with backoff |
| **Frequent exports** | Parquet format (80% smaller) |
| **Complex joins** | Extract via SQL, not Python |
| **Real-time data** | Set up incremental extraction |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "ModuleNotFoundError" | `pip install pandas requests` |
| Encoding errors | `extract_csv(file, encoding='latin-1')` |
| Missing columns | Check with `print(df.columns)` |
| Type conversion fails | Clean nulls first: `df.dropna(subset=[col])` |
| API timeout | Use `fetch_with_retry()` with backoff |
| Memory issues | Use `chunked_extract()` or Parquet |

---

## 📖 File Structure

```
DrivingRepo/
├── .claude/skills/
│   └── data-extraction-toolkit.md          # Skill for Claude
├── data_extractor.py                       # Main module
├── example_data_extraction.py               # 7 runnable examples
├── DATA_EXTRACTION_GUIDE.md                 # Complete reference
└── DATA_EXTRACTION_TOOLKIT_README.md        # This file
```

---

## 🎯 Quick Reference

### Extract CSV
```python
from data_extractor import FileExtractor
df = FileExtractor.extract_csv('data.csv')
```

### Extract from API
```python
from data_extractor import APIExtractor
api = APIExtractor('https://api.example.com')
df = api.fetch_paginated('/data')
```

### Validate Data
```python
from data_extractor import DataValidator
validator = DataValidator(df)
validator.validate_required_columns(['id'])
report = validator.get_report()
```

### Clean Data
```python
from data_extractor import DataCleaner
cleaner = DataCleaner()
df = cleaner.remove_duplicates(df)
df = cleaner.standardize_column_names(df)
```

### Export
```python
from data_extractor import DataExporter
exporter = DataExporter(df)
exporter.to_csv('output.csv')
exporter.to_parquet('output.parquet')
```

### Complete Pipeline
```python
from data_extractor import DataExtractionPipeline
pipeline = DataExtractionPipeline('MyData') \
    .extract('input.csv') \
    .validate({'required_columns': ['id']}) \
    .transform({'standardize_names': {}}) \
    .export('output.csv')
```

---

## ✨ Next Steps

1. **Try Examples**: Run `python example_data_extraction.py`
2. **Read Guide**: See `DATA_EXTRACTION_GUIDE.md` for details
3. **Use Skill**: Type `/data-extraction-toolkit` in Claude Code
4. **Customize**: Modify classes for your specific needs
5. **Deploy**: Use in production pipelines

---

**Version**: 1.0  
**Status**: Production Ready  
**Last Updated**: 2026-07-26
