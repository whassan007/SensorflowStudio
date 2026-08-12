"""
Dashboard Visualizations - Complete Implementation
Bar charts, heatmaps, scatter plots, and Pareto frontier
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from comparative_analytics import ComparativeAnalytics

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Install with: pip install plotly")


class DashboardVisualizations:
    """Generate interactive visualizations for comparative analysis"""

    def __init__(self, analytics: ComparativeAnalytics):
        self.analytics = analytics
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations. Install with: pip install plotly")

    def bar_chart_by_technique(self, metric: str = 'f1_score', dataset: Optional[str] = None,
                              model: Optional[str] = None) -> go.Figure:
        """Bar chart comparing metric values by technique"""
        df = self.analytics.to_dataframe(
            datasets=[dataset] if dataset else None,
            models=[model] if model else None
        )

        if df.empty or metric not in df.columns:
            return go.Figure().add_annotation(text="No data available")

        # Group by technique and calculate statistics
        tech_stats = df.groupby('technique')[metric].agg(['mean', 'std']).reset_index()
        tech_stats = tech_stats.sort_values('mean', ascending=False)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=tech_stats['technique'],
            y=tech_stats['mean'],
            error_y=dict(type='data', array=tech_stats['std'], visible=True),
            name='Mean ± Std',
            marker=dict(
                color=tech_stats['mean'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=metric)
            ),
            text=tech_stats['mean'].round(4),
            textposition='auto',
        ))

        title = f"{metric.replace('_', ' ').title()} by Technique"
        if dataset:
            title += f" ({dataset})"
        if model:
            title += f" ({model})"

        fig.update_layout(
            title=title,
            xaxis_title='Technique',
            yaxis_title=metric.replace('_', ' ').title(),
            hovermode='x unified',
            height=500,
            showlegend=True
        )

        return fig

    def bar_chart_by_configuration(self, metric: str = 'f1_score', technique: Optional[str] = None,
                                  top_n: int = 10) -> go.Figure:
        """Bar chart showing top N configurations"""
        df = self.analytics.to_dataframe(
            techniques=[technique] if technique else None
        )

        if df.empty or metric not in df.columns:
            return go.Figure().add_annotation(text="No data available")

        # Get top configurations
        top = df.nlargest(top_n, metric)[['configuration', 'technique', metric]]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=top['configuration'],
            x=top[metric],
            orientation='h',
            marker=dict(
                color=top[metric],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=metric)
            ),
            text=top[metric].round(4),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>' + f'{metric}: %{{x:.4f}}<extra></extra>'
        ))

        title = f"Top {top_n} Configurations by {metric.replace('_', ' ').title()}"
        if technique:
            title += f" ({technique})"

        fig.update_layout(
            title=title,
            xaxis_title=metric.replace('_', ' ').title(),
            yaxis_title='Configuration',
            height=400 + top_n * 20,
            showlegend=False
        )

        return fig

    def heatmap_dataset_technique(self, metric: str = 'f1_score') -> go.Figure:
        """Heatmap: Dataset × Technique"""
        df = self.analytics.to_dataframe()

        if df.empty or metric not in df.columns:
            return go.Figure().add_annotation(text="No data available")

        # Create pivot table
        pivot = df.pivot_table(
            values=metric,
            index='dataset',
            columns='technique',
            aggfunc='mean'
        )

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            text=np.round(pivot.values, 4),
            texttemplate='%{text:.4f}',
            textfont={"size": 10},
            hovertemplate='Dataset: %{y}<br>Technique: %{x}<br>' + f'{metric}: %{{z:.4f}}<extra></extra>',
            colorbar=dict(title=metric)
        ))

        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} Heatmap: Dataset × Technique",
            xaxis_title='Technique',
            yaxis_title='Dataset',
            height=400,
            width=700
        )

        return fig

    def heatmap_dataset_model(self, metric: str = 'f1_score') -> go.Figure:
        """Heatmap: Dataset × Model"""
        df = self.analytics.to_dataframe()

        if df.empty or metric not in df.columns:
            return go.Figure().add_annotation(text="No data available")

        # Create pivot table
        pivot = df.pivot_table(
            values=metric,
            index='dataset',
            columns='model',
            aggfunc='mean'
        )

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            text=np.round(pivot.values, 4),
            texttemplate='%{text:.4f}',
            textfont={"size": 10},
            hovertemplate='Dataset: %{y}<br>Model: %{x}<br>' + f'{metric}: %{{z:.4f}}<extra></extra>',
            colorbar=dict(title=metric)
        ))

        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} Heatmap: Dataset × Model",
            xaxis_title='Model',
            yaxis_title='Dataset',
            height=400,
            width=700
        )

        return fig

    def scatter_plot_trade_off(self, metric_x: str = 'f1_score', metric_y: str = 'inference_time',
                              color_by: str = 'technique') -> go.Figure:
        """Scatter plot for two-metric trade-off"""
        df = self.analytics.to_dataframe()

        if df.empty or metric_x not in df.columns or metric_y not in df.columns:
            return go.Figure().add_annotation(text="No data available")

        df = df.dropna(subset=[metric_x, metric_y])

        fig = px.scatter(
            df,
            x=metric_x,
            y=metric_y,
            color=color_by,
            hover_data=['dataset', 'model', 'configuration'],
            title=f"Trade-off: {metric_x.replace('_', ' ').title()} vs {metric_y.replace('_', ' ').title()}",
            labels={
                metric_x: metric_x.replace('_', ' ').title(),
                metric_y: metric_y.replace('_', ' ').title()
            }
        )

        fig.update_layout(
            height=500,
            hovermode='closest'
        )

        return fig

    def line_chart_trends(self, metric: str = 'f1_score', technique: Optional[str] = None,
                         group_by: str = 'configuration') -> go.Figure:
        """Line chart showing performance trends"""
        df = self.analytics.to_dataframe(
            techniques=[technique] if technique else None
        )

        if df.empty or metric not in df.columns:
            return go.Figure().add_annotation(text="No data available")

        df = df.sort_values('timestamp')

        fig = go.Figure()

        # Line for each configuration
        for config in df[group_by].unique():
            config_df = df[df[group_by] == config]
            fig.add_trace(go.Scatter(
                x=config_df.index,
                y=config_df[metric],
                mode='lines+markers',
                name=config,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              f'{metric}: %{{y:.4f}}<br>' +
                              'Index: %{x}<extra></extra>'
            ))

        title = f"{metric.replace('_', ' ').title()} Trends"
        if technique:
            title += f" ({technique})"

        fig.update_layout(
            title=title,
            xaxis_title='Experiment Index',
            yaxis_title=metric.replace('_', ' ').title(),
            hovermode='x unified',
            height=500
        )

        return fig

    def pareto_frontier_plot(self, metric1: str = 'f1_score', metric2: str = 'inference_time') -> go.Figure:
        """Plot showing Pareto frontier (optimal trade-offs)"""
        df = self.analytics.to_dataframe()

        if df.empty or metric1 not in df.columns or metric2 not in df.columns:
            return go.Figure().add_annotation(text="No data available")

        df = df.dropna(subset=[metric1, metric2])

        # Calculate Pareto frontier
        pareto_mask = np.ones(len(df), dtype=bool)
        for i in range(len(df)):
            for j in range(len(df)):
                if i != j:
                    # Check if point j dominates point i
                    if df.iloc[j][metric1] >= df.iloc[i][metric1] and \
                       df.iloc[j][metric2] <= df.iloc[i][metric2] and \
                       (df.iloc[j][metric1] > df.iloc[i][metric1] or df.iloc[j][metric2] < df.iloc[i][metric2]):
                        pareto_mask[i] = False
                        break

        pareto_df = df[pareto_mask]

        fig = go.Figure()

        # All points
        fig.add_trace(go.Scatter(
            x=df[metric2],
            y=df[metric1],
            mode='markers',
            name='All Results',
            marker=dict(size=8, color='lightblue', opacity=0.6),
            hovertemplate=f'{metric1}: %{{y:.4f}}<br>{metric2}: %{{x:.4f}}<extra></extra>'
        ))

        # Pareto frontier
        if not pareto_df.empty:
            pareto_sorted = pareto_df.sort_values(metric2)
            fig.add_trace(go.Scatter(
                x=pareto_sorted[metric2],
                y=pareto_sorted[metric1],
                mode='lines+markers',
                name='Pareto Frontier',
                marker=dict(size=12, color='red'),
                line=dict(color='red', width=2),
                hovertemplate=f'{metric1}: %{{y:.4f}}<br>{metric2}: %{{x:.4f}}<extra></extra>'
            ))

        fig.update_layout(
            title=f"Pareto Frontier: {metric1.replace('_', ' ').title()} vs {metric2.replace('_', ' ').title()}",
            xaxis_title=metric2.replace('_', ' ').title(),
            yaxis_title=metric1.replace('_', ' ').title(),
            hovermode='closest',
            height=500
        )

        return fig

    def multi_metric_subplots(self, metrics: List[str] = ['f1_score', 'accuracy', 'inference_time']) -> go.Figure:
        """Subplots comparing techniques across multiple metrics"""
        df = self.analytics.to_dataframe()

        if df.empty:
            return go.Figure().add_annotation(text="No data available")

        available_metrics = [m for m in metrics if m in df.columns]

        if not available_metrics:
            return go.Figure().add_annotation(text="No metrics available")

        fig = make_subplots(
            rows=1,
            cols=len(available_metrics),
            subplot_titles=[m.replace('_', ' ').title() for m in available_metrics],
            specs=[[{"type": "bar"}] * len(available_metrics)]
        )

        colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']

        for col_idx, metric in enumerate(available_metrics, 1):
            df_clean = df.dropna(subset=[metric])
            stats = df_clean.groupby('technique')[metric].mean().sort_values(ascending=False)

            fig.add_trace(
                go.Bar(
                    x=stats.index,
                    y=stats.values,
                    name=metric,
                    marker=dict(color=colors[col_idx % len(colors)]),
                    text=stats.round(4),
                    textposition='auto',
                ),
                row=1,
                col=col_idx
            )

            fig.update_xaxes(title_text='Technique', row=1, col=col_idx)
            fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=1, col=col_idx)

        fig.update_layout(
            title='Technique Comparison Across Multiple Metrics',
            height=500,
            showlegend=False
        )

        return fig

    def export_all_charts_html(self, filepath: str, metrics: List[str] = ['f1_score', 'accuracy']) -> str:
        """Export all visualizations to a single HTML file"""
        html_parts = [
            """
            <html>
            <head>
                <title>Comparative Analytics Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                    .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
                    h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
                    .section { margin: 40px 0; }
                    h2 { color: #666; margin-top: 30px; }
                    .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                    .chart-item { border: 1px solid #ddd; padding: 10px; border-radius: 4px; }
                    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #4CAF50; color: white; }
                    tr:nth-child(even) { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📊 Comparative Analytics Dashboard</h1>
            """
        ]

        # Overview
        summary = self.analytics.to_dataframe()
        if not summary.empty:
            html_parts.append(f"""
            <div class="section">
                <h2>Overview</h2>
                <p><strong>Total Experiments:</strong> {len(self.analytics.results)}</p>
                <p><strong>Datasets:</strong> {', '.join(sorted(self.analytics.datasets))}</p>
                <p><strong>Models:</strong> {', '.join(sorted(self.analytics.models))}</p>
                <p><strong>Techniques:</strong> {', '.join(sorted(self.analytics.techniques))}</p>
            </div>
            """)

        # Visualizations
        html_parts.append('<div class="section"><h2>Visualizations</h2></div>')

        # Bar charts by technique
        for metric in metrics:
            fig = self.bar_chart_by_technique(metric=metric)
            html_parts.append(f'<div class="section"><h3>{metric.replace("_", " ").title()} by Technique</h3>')
            html_parts.append(fig.to_html(include_plotlyjs=False, div_id=f"chart_{metric}_tech"))
            html_parts.append('</div>')

        # Heatmaps
        fig_hm_dt = self.heatmap_dataset_technique()
        html_parts.append('<div class="section"><h3>Dataset × Technique Heatmap (F1-Score)</h3>')
        html_parts.append(fig_hm_dt.to_html(include_plotlyjs=False, div_id="heatmap_dt"))
        html_parts.append('</div>')

        fig_hm_dm = self.heatmap_dataset_model()
        html_parts.append('<div class="section"><h3>Dataset × Model Heatmap (F1-Score)</h3>')
        html_parts.append(fig_hm_dm.to_html(include_plotlyjs=False, div_id="heatmap_dm"))
        html_parts.append('</div>')

        # Scatter plot
        fig_scatter = self.scatter_plot_trade_off()
        html_parts.append('<div class="section"><h3>Trade-off Analysis (F1-Score vs Inference Time)</h3>')
        html_parts.append(fig_scatter.to_html(include_plotlyjs=False, div_id="scatter"))
        html_parts.append('</div>')

        # Pareto frontier
        fig_pareto = self.pareto_frontier_plot()
        html_parts.append('<div class="section"><h3>Pareto Frontier</h3>')
        html_parts.append(fig_pareto.to_html(include_plotlyjs=False, div_id="pareto"))
        html_parts.append('</div>')

        # Multi-metric subplots
        fig_multi = self.multi_metric_subplots(metrics=metrics + ['inference_time'])
        html_parts.append('<div class="section"><h2>Multi-Metric Comparison</h2>')
        html_parts.append(fig_multi.to_html(include_plotlyjs=False, div_id="subplots"))
        html_parts.append('</div>')

        # Summary statistics table
        stats = self.analytics.compare_techniques()
        if not stats.empty:
            html_parts.append('<div class="section"><h2>Summary Statistics</h2>')
            html_parts.append(stats.to_html())
            html_parts.append('</div>')

        html_parts.append('</div></body></html>')

        html_content = '\n'.join(html_parts)

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath
