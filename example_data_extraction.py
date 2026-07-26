"""
Data Extraction Toolkit Examples
Demonstrates extraction, validation, transformation, and export patterns
"""

import pandas as pd
from data_extractor import (
    DataExtractionPipeline,
    FileExtractor,
    DataCleaner,
    DataValidator,
    DataExporter
)


def example_1_csv_extraction():
    """Example 1: Extract CSV with validation and transformation"""
    print("\n" + "="*80)
    print("EXAMPLE 1: CSV Extraction Pipeline")
    print("="*80)

    # Create sample data
    sample_data = {
        'ID': [1, 2, 3, 4, 5, None],
        'Name': ['  John  ', '  Jane  ', '  Bob  ', '  Alice  ', '  Charlie  ', '  David  '],
        'Email': ['john@example.com', 'jane@example.com', 'bob@example.com',
                 'alice@example.com', 'charlie@example.com', 'david@example.com'],
        'Age': [25, 30, 35, 28, 32, 29],
        'Join Date': ['2024-01-15', '2024-02-20', '2024-01-10', '2024-03-05', '2024-02-28', '2024-01-20']
    }
    df = pd.DataFrame(sample_data)
    df.to_csv('/tmp/sample_data.csv', index=False)

    # Run pipeline
    pipeline = DataExtractionPipeline('CSV Import')
    pipeline.extract('/tmp/sample_data.csv', source_type='csv') \
            .validate({
                'required_columns': ['ID', 'Name', 'Email'],
                'no_nulls': ['Email'],
                'data_types': {'Age': 'int'}
            }) \
            .transform({
                'standardize_names': {},
                'trim_whitespace': {'columns': ['Name']},
                'handle_missing': {'strategy': 'drop'},
                'convert_types': {'join_date': 'date'}
            }) \
            .export('/tmp/cleaned_data.csv', format='csv')

    print("\nPipeline Summary:")
    import json
    summary = pipeline.get_summary()
    print(json.dumps({k: v for k, v in summary.items() if k != 'validation'}, indent=2))


def example_2_api_extraction():
    """Example 2: API extraction with pagination"""
    print("\n" + "="*80)
    print("EXAMPLE 2: API Extraction (Simulated)")
    print("="*80)

    # Create simulated API response
    print("Pipeline setup for API extraction:")
    print("""
    pipeline = DataExtractionPipeline('JSONPlaceholder API')
    pipeline.extract(
        'https://jsonplaceholder.typicode.com',
        source_type='api',
        endpoint='/posts',
        params={'_limit': 10}
    ).validate({
        'required_columns': ['userId', 'title', 'body']
    }).export('api_data.json', format='json')
    """)


def example_3_batch_extraction():
    """Example 3: Extract and combine multiple files"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch File Extraction")
    print("="*80)

    # Create sample CSV files
    for i in range(3):
        data = pd.DataFrame({
            'id': range(i*10, (i+1)*10),
            'value': range(100 + i*100, 100 + (i+1)*100),
            'category': ['A', 'B'] * 5
        })
        data.to_csv(f'/tmp/batch_data_{i}.csv', index=False)

    # Run batch extraction
    pipeline = DataExtractionPipeline('Batch Import')
    pipeline.extract(
        '/tmp/batch_data_*.csv',
        source_type='batch',
        filetype='csv'
    ).validate({
        'required_columns': ['id', 'value'],
        'unique': ['id']
    }).transform({
        'standardize_names': {},
        'remove_duplicates': {'subset': ['id']}
    }).export('/tmp/combined_data.csv', format='csv')

    print("\nCombined data info:")
    print(f"  Total rows: {len(pipeline.data)}")
    print(f"  Columns: {list(pipeline.data.columns)}")


def example_4_json_extraction():
    """Example 4: Extract and transform JSON data"""
    print("\n" + "="*80)
    print("EXAMPLE 4: JSON Extraction")
    print("="*80)

    # Create sample JSON file
    sample_json = [
        {'user_id': 1, 'user_name': 'Alice', 'signup_date': '2024-01-15', 'account_status': 'active'},
        {'user_id': 2, 'user_name': 'Bob', 'signup_date': '2024-02-20', 'account_status': 'active'},
        {'user_id': 3, 'user_name': 'Charlie', 'signup_date': '2024-01-10', 'account_status': 'inactive'},
    ]
    with open('/tmp/users.json', 'w') as f:
        import json
        json.dump(sample_json, f)

    # Run pipeline
    pipeline = DataExtractionPipeline('JSON Users')
    pipeline.extract('/tmp/users.json', source_type='json') \
            .validate({
                'required_columns': ['user_id', 'user_name'],
                'no_nulls': ['user_id']
            }) \
            .transform({
                'standardize_names': {},
                'convert_types': {'signup_date': 'date'}
            }) \
            .export('/tmp/users_processed.json', format='json')

    print("\nProcessed JSON data:")
    print(pipeline.data.to_string())


def example_5_data_cleaning():
    """Example 5: Comprehensive data cleaning"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Data Cleaning Operations")
    print("="*80)

    # Create messy data
    messy_data = pd.DataFrame({
        'User ID': [1, 1, 2, 3, 3, 4, None],  # Duplicates and null
        'User Name': ['  Alice  ', 'Alice', '  Bob  ', 'Charlie', '  Charlie  ', 'David', 'Eve'],
        'Email Address': ['alice@example.com', 'alice@example.com', 'bob@example.com',
                         'charlie@example.com', 'charlie@example.com', 'david@example.com', None],
        'Sign-Up Date': ['2024-01-15', '2024-01-15', '2024-02-20', '2024-01-10',
                        '2024-01-10', '2024-03-05', '2024-01-20'],
        'Account-Status': ['Active', 'Active', 'Inactive', 'Active', 'Active', 'Pending', 'Active']
    })

    print("Original data:")
    print(messy_data.to_string())

    # Clean data
    cleaner = DataCleaner()
    cleaned = cleaner.remove_duplicates(messy_data, subset=['User ID', 'Email Address'])
    cleaned = cleaner.standardize_column_names(cleaned)
    cleaned = cleaner.trim_whitespace(cleaned)
    cleaned = cleaner.handle_missing_values(cleaned, strategy='drop')
    cleaned = cleaner.convert_dtypes(cleaned, {'sign_up_date': 'date'})

    print("\n\nCleaned data:")
    print(cleaned.to_string())

    print(f"\nCleaning summary:")
    print(f"  Original rows: {len(messy_data)}")
    print(f"  Cleaned rows: {len(cleaned)}")
    print(f"  Rows removed: {len(messy_data) - len(cleaned)}")


def example_6_data_validation():
    """Example 6: Comprehensive data validation"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Data Validation")
    print("="*80)

    # Create sample data with issues
    data = pd.DataFrame({
        'product_id': [1, 2, 3, None, 5, 6],
        'product_name': ['Widget A', 'Widget B', 'Widget C', 'Widget D', 'Widget E', 'Widget F'],
        'price': [10.50, 20.00, 15.75, -5.00, 30.00, 100.00],  # Negative price is invalid
        'stock': [100, 50, 75, 200, None, 150],  # Missing stock
        'category': ['Electronics', 'Electronics', 'Home', 'Home', 'Electronics', 'Electronics']
    })

    # Validate
    validator = DataValidator(data)
    validator.validate_required_columns(['product_id', 'product_name', 'price'])
    validator.validate_no_nulls(['product_id', 'product_name'])
    validator.validate_data_types({'price': 'float', 'stock': 'float'})
    validator.validate_value_range('price', min_val=0, max_val=1000)
    validator.validate_unique(['product_id'])

    report = validator.get_report()

    print("Validation Results:")
    print(f"  Valid: {report.valid}")
    print(f"  Rows: {report.row_count}")
    print(f"  Columns: {report.column_count}")
    print(f"\nErrors ({len(report.errors)}):")
    for error in report.errors:
        print(f"    - {error}")
    print(f"\nWarnings ({len(report.warnings)}):")
    for warning in report.warnings:
        print(f"    - {warning}")


def example_7_export_formats():
    """Example 7: Export to multiple formats"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Export to Multiple Formats")
    print("="*80)

    # Create sample data
    data = pd.DataFrame({
        'id': range(1, 6),
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'score': [85, 92, 78, 88, 95],
        'date': pd.date_range('2024-01-01', periods=5)
    })

    exporter = DataExporter(data)

    # Export to different formats
    csv_path = exporter.to_csv('/tmp/export_data.csv')
    json_path = exporter.to_json('/tmp/export_data.json')

    print("Exported to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")

    # Try Parquet if available
    try:
        parquet_path = exporter.to_parquet('/tmp/export_data.parquet')
        print(f"  Parquet: {parquet_path}")
    except Exception as e:
        print(f"  Parquet: Not available ({e})")

    # Convert to dictionary list
    dict_list = exporter.to_dict_list()
    print(f"\nAs dictionary list (first 2 records):")
    for record in dict_list[:2]:
        print(f"  {record}")


def main():
    """Run all examples"""
    examples = [
        ('1', 'CSV Extraction Pipeline', example_1_csv_extraction),
        ('2', 'API Extraction', example_2_api_extraction),
        ('3', 'Batch File Extraction', example_3_batch_extraction),
        ('4', 'JSON Extraction', example_4_json_extraction),
        ('5', 'Data Cleaning', example_5_data_cleaning),
        ('6', 'Data Validation', example_6_data_validation),
        ('7', 'Export Formats', example_7_export_formats),
    ]

    print("\n" + "="*80)
    print("DATA EXTRACTION TOOLKIT - EXAMPLES")
    print("="*80)
    print("\nAvailable examples:")
    for num, title, _ in examples:
        print(f"  {num}. {title}")

    print("\nRunning all examples...\n")

    for num, title, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n✗ Example {num} failed: {e}")

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)


if __name__ == '__main__':
    main()
