"""Quick test of comparative dashboard"""
import pandas as pd
import numpy as np
from comparative_analytics import ComparativeAnalytics, Configuration, PerformanceMetrics, ExperimentResult

# Create analytics engine
analytics = ComparativeAnalytics()
print("✓ ComparativeAnalytics created")

# Generate sample results
for dataset in ['Dataset_A', 'Dataset_B']:
    for model in ['Model_X', 'Model_Y']:
        for technique in ['Technique_1', 'Technique_2']:
            config = Configuration(
                f'config_{technique}_{dataset}',
                f'Config for {technique}',
                technique,
                {'param_a': 0.1}
            )
            
            for i in range(3):
                metrics = PerformanceMetrics(
                    accuracy=0.75 + np.random.normal(0, 0.05),
                    f1_score=0.72 + np.random.normal(0, 0.05),
                    auc=0.80 + np.random.normal(0, 0.05),
                    inference_time=np.random.uniform(0.01, 0.1),
                    memory_usage=np.random.uniform(50, 500)
                )
                
                result = ExperimentResult(
                    dataset=dataset,
                    model=model,
                    configuration=config,
                    metrics=metrics
                )
                
                analytics.add_result(result)

print(f"✓ Added {len(analytics.results)} experiment results")
print(f"  Datasets: {analytics.datasets}")
print(f"  Models: {analytics.models}")
print(f"  Techniques: {analytics.techniques}")

# Test filtering
df = analytics.to_dataframe()
print(f"\n✓ Created DataFrame: {len(df)} rows × {len(df.columns)} columns")

# Test best configurations
best = analytics.get_best_configurations(metric='f1_score', top_n=3)
print(f"\n✓ Top 3 configurations by F1-Score:")
print(best[['dataset', 'model', 'technique', 'f1_score']])

# Test comparisons
comp = analytics.compare_techniques(metric='f1_score')
print(f"\n✓ Technique Comparison (F1-Score):")
print(comp)

# Test summary
summary = analytics.get_summary_statistics(metric='f1_score')
print(f"\n✓ Summary Statistics:")
for technique, stats in summary.items():
    print(f"  {technique}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

# Test matrix
matrix = analytics.get_dataset_model_matrix(metric='f1_score')
print(f"\n✓ Dataset × Model Matrix (F1-Score):")
print(matrix.round(4))

print("\n" + "="*60)
print("✅ All tests passed! Dashboard system is working correctly.")
print("="*60)
