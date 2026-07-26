# Accident Analysis Feature Implementation Prompt for Ollama

**Implementation Model**: `qwen3-coder:30b` (ollama-spark-coding)

**MCP Host**: `http://dgx-spark.tail16d8d9.ts.net:11434`

---

## Quick Summary

Implement a production-grade accident/crash analysis feature using two available toolkits:
1. **CrashDataAnalyzer** (`crash_analyzer.py`) — SWITRS data analysis
2. **DataExtractionPipeline** (`data_extractor.py`) — ETL for any source

Build 6 layers: Ingestion → Validation → Cleaning → Analysis → Insights → API

---

## Direct Prompt for qwen3-coder:30b

Copy and submit to `qwen3-coder:30b` via Ollama MCP:

```
=== ACCIDENT ANALYSIS FEATURE - IMPLEMENTATION ===

Build a production accident/crash analysis platform in Python 3.9+.

AVAILABLE TOOLKITS (in repository):
- crash_analyzer.py: CrashDataAnalyzer for SWITRS data analysis
- data_extractor.py: DataExtractionPipeline, DataValidator, DataCleaner
- Examples: example_crash_analysis.py, example_data_extraction.py

BUILD 6 LAYERS (in order):

LAYER 1: DATA INGESTION (accident_importer.py)
- Class: AccidentDataImporter
- Load from: CSV, APIs (paginated), SQL databases, JSON/Parquet
- Auto-detect format, handle errors, standardize schema
- Base: DataExtractionPipeline

LAYER 2: DATA VALIDATION (accident_validator.py)
- Class: AccidentDataValidator
- Validate: required fields, types, coordinate ranges, date ranges
- Detect: outliers, missing values, anomalies
- Return: quality report with score
- Base: DataValidator

LAYER 3: DATA CLEANING (accident_cleaner.py)
- Class: AccidentDataCleaner
- Standardize: column names (snake_case), severity, vehicle types
- Handle: missing values, duplicates, datetime, coordinates
- Base: DataCleaner

LAYER 4: ANALYSIS ENGINE (accident_analysis.py)
- Classes: MetadataAnalyzer, GeospatialAnalyzer, DemographicsAnalyzer
- Orchestrator: AccidentAnalysisEngine
- Analyze: temporal patterns, hotspots, demographics
- Cache results in memory
- Base: CrashDataAnalyzer

LAYER 5: INSIGHTS & REPORTING (accident_insights.py, accident_report.py)
- AccidentInsights: deadliest times/places/groups, risk scores
- AccidentReport: HTML/PDF/JSON export, visualizations

LAYER 6: API & INTEGRATION (main.py, routes/, models/)
- FastAPI endpoints:
  POST /api/analyze (upload CSV)
  GET /api/hotspots (geographic)
  GET /api/insights (findings)
  GET /api/report (download)
  WebSocket /ws/stream (real-time)

PRODUCTION REQUIREMENTS:
✓ Complete working code (no TODOs)
✓ Type hints on all functions
✓ Google-style docstrings
✓ Error handling (try/except with logging)
✓ 80%+ test coverage (pytest)
✓ Performance: <5s for 10k records
✓ Security: input validation, SQL injection protection
✓ Logging: DEBUG/INFO/ERROR/CRITICAL levels
✓ Config: environment variables

OUTPUT EACH LAYER:
1. Complete working Python code
2. Unit tests (pytest)
3. Integration tests
4. Full docstrings & type hints
5. Error handling with examples
6. Usage examples

Start with LAYER 1. Use existing toolkits. Follow PEP 8.
```

---

## MCP Server Configuration

```json
{
  "mcpServers": {
    "ollama-spark-coding": {
      "command": "npx",
      "args": ["-y", "@iflow-mcp/ollama-mcp"],
      "env": {
        "OLLAMA_HOST": "http://dgx-spark.tail16d8d9.ts.net:11434",
        "OLLAMA_MODEL": "qwen3-coder:30b"
      }
    }
  }
}
```

---

## Model Selection

**Primary**: `qwen3-coder:30b` (code generation specialist)
- Best for Python implementation across 6 layers
- Understands complex architecture and APIs
- Large context window for full specs

**Optional**: `reasoning-core` (planning/validation)
- Good for architecture review
- Validates implementation decisions

**Optional**: `gemma4:26b` (documentation)
- Generate docs, examples, deployment guides

---

## 6-Layer Architecture

```
Data Source (CSV/API/DB)
        ↓
LAYER 1: Data Ingestion
        ↓
LAYER 2: Data Validation (Quality Check)
        ↓
LAYER 3: Data Cleaning (Normalization)
        ↓
LAYER 4: Analysis Engine (Metadata/Geo/Demographics)
        ↓
LAYER 5: Insights & Reports (HTML/PDF/JSON)
        ↓
LAYER 6: REST API (FastAPI endpoints)
```

---

## Key Requirements

- **Language**: Python 3.9+
- **Framework**: FastAPI (API layer)
- **Testing**: pytest (80%+ coverage)
- **Data**: SWITRS format (accidents.csv, vehicles.csv, parties.csv)
- **Performance**: <5 seconds for 10k records
- **Code Quality**: Type hints, docstrings, error handling

---

## Timeline

- Layer 1 (Ingestion): 1-2 days
- Layer 2 (Validation): 1 day
- Layer 3 (Cleaning): 1 day
- Layer 4 (Analysis): 2-3 days
- Layer 5 (Insights): 1-2 days
- Layer 6 (API): 2 days
- Tests & Deployment: 2 days

**Total**: 11-17 days

---

## Success Criteria

✅ Extract from 3+ sources (CSV, API, database)
✅ Validate with >95% accuracy
✅ Analyze temporal, geospatial, demographic patterns
✅ Identify hotspots and high-risk groups
✅ Generate HTML/PDF/JSON reports
✅ Provide REST API
✅ Support real-time analysis
✅ Handle 1GB+ datasets
✅ 80%+ test coverage
✅ Production-ready code

---

**Ready for Ollama Implementation**: qwen3-coder:30b
