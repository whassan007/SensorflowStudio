"""Comparative Analytics Dashboard - Core Engine"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from datetime import datetime


class MetricType(Enum):
    """Metric categories"""
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    AUC = "auc"
    INFERENCE_TIME = "inference_time"
    MEMORY_USAGE = "memory_usage"


@dataclass
class Configuration:
    """Configuration parameters"""
    config_id: str
    name: str
    technique: str
    parameters: Dict[str, Any]
    def __hash__(self):
        return hash(self.config_id)


@dataclass
class PerformanceMetrics:
    """Evaluation metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc: Optional[float] = None
    inference_time: Optional[float] = None
    memory_usage: Optional[float] = None
    training_time: Optional[float] = None
    throughput: Optional[float] = None
    latency: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def get_metric(self, metric_name: str) -> Optional[float]:
        return getattr(self, metric_name, None)


@dataclass
class ExperimentResult:
    """Single experiment result"""
    dataset: str
    model: str
    configuration: Configuration
    metrics: PerformanceMetrics
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    def __hash__(self):
        return hash((self.dataset, self.model, self.configuration.config_id))


class ComparativeAnalytics:
    """Analytics engine for comparing results"""

    def __init__(self):
        self.results: List[ExperimentResult] = []
        self.datasets: set = set()
        self.models: set = set()
        self.techniques: set = set()
        self.configurations: Dict[str, Configuration] = {}

    def add_result(self, result: ExperimentResult) -> None:
        self.results.append(result)
        self.datasets.add(result.dataset)
        self.models.add(result.model)
        self.techniques.add(result.configuration.technique)
        self.configurations[result.configuration.config_id] = result.configuration

    def add_results_batch(self, results: List[ExperimentResult]) -> None:
        for result in results:
            self.add_result(result)

    def filter_results(self, datasets: Optional[List[str]] = None, models: Optional[List[str]] = None,
                      techniques: Optional[List[str]] = None, configurations: Optional[List[str]] = None) -> List[ExperimentResult]:
        filtered = self.results
        if datasets:
            filtered = [r for r in filtered if r.dataset in datasets]
        if models:
            filtered = [r for r in filtered if r.model in models]
        if techniques:
            filtered = [r for r in filtered if r.configuration.technique in techniques]
        if configurations:
            filtered = [r for r in filtered if r.configuration.config_id in configurations]
        return filtered

    def to_dataframe(self, datasets: Optional[List[str]] = None, models: Optional[List[str]] = None,
                    techniques: Optional[List[str]] = None, configurations: Optional[List[str]] = None) -> pd.DataFrame:
        filtered = self.filter_results(datasets, models, techniques, configurations)
        rows = []
        for result in filtered:
            row = {
                'dataset': result.dataset,
                'model': result.model,
                'technique': result.configuration.technique,
                'configuration': result.configuration.name,
                'config_id': result.configuration.config_id,
                'timestamp': result.timestamp,
            }
            row.update(result.metrics.to_dict())
            rows.append(row)
        return pd.DataFrame(rows)

    def get_best_configurations(self, metric: str = 'f1_score', dataset: Optional[str] = None,
                               model: Optional[str] = None, top_n: int = 5) -> pd.DataFrame:
        filters = {'datasets': [dataset] if dataset else None, 'models': [model] if model else None}
        df = self.to_dataframe(**filters)
        if df.empty or metric not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=[metric])
        df = df.sort_values(metric, ascending=False).head(top_n)
        return df

    def compare_techniques(self, metric: str = 'f1_score', dataset: Optional[str] = None,
                          model: Optional[str] = None) -> pd.DataFrame:
        filters = {'datasets': [dataset] if dataset else None, 'models': [model] if model else None}
        df = self.to_dataframe(**filters)
        if df.empty or metric not in df.columns:
            return pd.DataFrame()
        comparison = df.groupby('technique')[metric].agg([('mean', 'mean'), ('std', 'std'),
                                                          ('min', 'min'), ('max', 'max'), ('count', 'count')]).round(4)
        return comparison.sort_values('mean', ascending=False)

    def get_summary_statistics(self, metric: str = 'f1_score', groupby: str = 'technique') -> Dict:
        df = self.to_dataframe()
        if df.empty or metric not in df.columns:
            return {}
        stats = {}
        for group_name, group_df in df.groupby(groupby):
            group_data = group_df[metric].dropna()
            stats[str(group_name)] = {
                'mean': float(group_data.mean()),
                'std': float(group_data.std()),
                'min': float(group_data.min()),
                'max': float(group_data.max()),
                'median': float(group_data.median()),
                'count': int(len(group_data))
            }
        return stats

    def get_dataset_model_matrix(self, metric: str = 'f1_score') -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty or metric not in df.columns:
            return pd.DataFrame()
        matrix = df.groupby(['dataset', 'model'])[metric].mean().unstack(fill_value=None)
        return matrix


class DashboardBuilder:
    """Build dashboard components"""
    def __init__(self, analytics: ComparativeAnalytics):
        self.analytics = analytics

    def get_summary_report(self) -> Dict[str, Any]:
        df = self.analytics.to_dataframe()
        if df.empty:
            return {}
        return {
            'overview': {
                'total_experiments': len(self.analytics.results),
                'num_datasets': len(self.analytics.datasets),
                'num_models': len(self.analytics.models),
                'num_techniques': len(self.analytics.techniques),
            },
            'datasets': sorted(list(self.analytics.datasets)),
            'models': sorted(list(self.analytics.models)),
            'techniques': sorted(list(self.analytics.techniques)),
        }
