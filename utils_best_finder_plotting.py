"""
Plotting utilities for Best Performance Finder
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from best_performance_finder import BestPerformanceFinder

# Model color mapping for consistency across plots
MODEL_COLORS = {
    'nemotron-nano:8b': 'skyblue',
    'deepseek-r1:8b': 'blue',
    'gemma3:12b': 'green',
    'gpt-oss:20b': 'orange',
    'llama3.1:8b': 'purple',
    'mistral-small3.2:24b': 'brown',
    'phi4:14b': 'pink',
    'qwen3:14b': 'crimson'
}

SUBPLOT_CONFIG = {
    'figure_size': (20, 24),  # Width, height for 6x3 layout
    'dot_size': 30,           # Scatter point size
    'alpha': 0.6,             # Point transparency
    'grid_alpha': 0.2,        # Grid transparency
    'title_fontsize': 10,     # Subplot title font size
    'axis_fontsize': 8,       # Axis label font size
    'legend_fontsize': 9,     # Legend font size
    'subplot_spacing': {
        'hspace': 0.25,        # Height spacing between subplots
        'wspace': 0.2         # Width spacing between subplots
    }
}

BENCHMARK_DISPLAY_NAMES = {
    # CrossNER datasets
    'crossner_conll2003': 'CoNLL2003',
    'crossner_music': 'CrossNER - Music',
    'crossner_politics': 'CrossNER - Politics',
    'crossner_science': 'CrossNER - Science',
    'crossner_literature': 'CrossNER - Literature',
    'crossner_ai': 'CrossNER - AI',
    
    # MIT datasets
    'mitner_movie': 'MIT - Movie',
    'mitner_restaurant': 'MIT - Restaurant',
    
    # Biomedical datasets
    'anatem': 'AnatEM',
    'bc2gm': 'BC2GM',
    'bc4chemd': 'BC4CHEMD',
    'bc5cdr': 'BC5CDR',
    'GENIA': 'GENIA',
    
    # Other datasets
    'ACE05': 'ACE 2005',
    'Broad Twitter': 'Broad Twitter',
    'FabNER': 'FabNER',
    'FindVehicle': 'FindVehicle',
    'HarveyNER': 'HarveyNER',
    'MultiNERD': 'MultiNERD',
    'OntoNotes': 'OntoNotes 5.0'
}

def _create_experiment_level_plot(fig, results, row=1, col=1):
    """Create experiment level scatter plot"""
    strict_f1 = [r.strict_span_f1_avg for r in results]
    best_f1 = [r.best_model_f1 for r in results]
    mv_f1 = [r.mv_f1 for r in results]
    
    # Create hover text for best model points
    hover_text_best = []
    hover_text_mv = []
    
    for r in results:
        base_text = (
            f"Iteration: {r.iteration}<br>"
            f"Models: {r.config.models}<br>"
            f"Supervisor: {r.config.supervisor_model_name}<br>"
            f"Gold Standard: {'Yes' if r.config.supervised_by_gold_standard else 'No'}<br>"
            f"llm_family_config: {r.config.llm_family_config}<br>"
            f"skip_final_goal_update: {r.config.skip_final_goal_update}<br>"
            f"Config: {r.config.get_config_dict_str()}<br>"
            f"Strict F1: {r.strict_span_f1_avg:.3f}"
        )
        hover_text_best.append(f"{base_text}<br>Best Model F1: {r.best_model_f1:.3f}")
        hover_text_mv.append(f"{base_text}<br>MV F1: {r.mv_f1:.3f}")
    
    # Add best model F1 scatter plot
    fig.add_trace(go.Scatter(
        x=strict_f1, y=best_f1, mode='markers', name='Best Model F1',
        text=hover_text_best, hovertemplate='%{text}<extra></extra>',
        marker=dict(size=8, color='blue', opacity=0.7)
    ), row=row, col=col)

def _create_individual_model_plot(fig, results, row=1, col=2):
    """Create individual model scatter plot"""
    added_models = set()
    
    for result in results:
        if not (result.individual_model_results and result.individual_model_strict_f1):
            continue
            
        for model_name, model_f1 in result.individual_model_results.items():
            model_strict_f1 = result.individual_model_strict_f1.get(model_name, 0.0)
            color = MODEL_COLORS.get(model_name, 'gray')
            
            hover_text = (
                f"Model: {model_name}<br>"
                f"Iteration: {result.iteration}<br>"
                f"Models: {result.config.models}<br>"
                f"Supervisor: {result.config.supervisor_model_name}<br>"
                f"Gold Standard: {'Yes' if result.config.supervised_by_gold_standard else 'No'}<br>"
                f"llm_family_config: {result.config.llm_family_config}<br>"
                f"skip_final_goal_update: {result.config.skip_final_goal_update}<br>"
                f"Config: {result.config.get_config_dict_str()}<br>"
                f"Model Strict F1: {model_strict_f1:.3f}<br>"
                f"Model Gold F1: {model_f1:.3f}"
            )
            
            fig.add_trace(go.Scatter(
                x=[model_strict_f1], y=[model_f1], mode='markers',
                name=model_name, text=[hover_text], hovertemplate='%{text}<extra></extra>',
                marker=dict(size=8, color=color, opacity=0.7),
                showlegend=model_name not in added_models
            ), row=row, col=col)
            added_models.add(model_name)

def plot_interactive_performance_by_benchmark(finder: 'BestPerformanceFinder'):
    """Create interactive performance plots for each benchmark"""
    if not finder.results:
        print("No results found. Run search_configurations first.")
        return
    
    benchmark_results = defaultdict(list)
    for result in finder.results:
        benchmark_results[result.config.benchmark].append(result)
    
    for benchmark in sorted(benchmark_results.keys()):
        results = benchmark_results[benchmark]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=[f'{benchmark} - Experiment Level', f'{benchmark} - Individual Models'],
            horizontal_spacing=0.15
        )
        
        # Add experiment level plot
        _create_experiment_level_plot(fig, results, row=1, col=1)
        
        # Add individual model plot
        _create_individual_model_plot(fig, results, row=1, col=2)
        
        # Update axes labels
        fig.update_xaxes(title_text="Average Strict span F1-Score", row=1, col=1)
        fig.update_yaxes(title_text="F1 Score", row=1, col=1)
        fig.update_xaxes(title_text="Models's Avg Pairwise Strict Span F1", row=1, col=2)
        fig.update_yaxes(title_text="Gold Standard F1", row=1, col=2)
        
        # Update layout
        fig.update_layout(
            height=500, width=1400, 
            title_text=f"Performance Analysis - {benchmark}"
        )
        
        fig.show()

def plot_performance_comparison(finder: 'BestPerformanceFinder', metric: str = 'best_model_f1'):
    """Create a comparison plot across all benchmarks"""
    if not finder.results:
        print("No results found. Run search_configurations first.")
        return
    
    benchmark_best = {}
    for result in finder.results:
        benchmark = result.config.benchmark
        if benchmark not in benchmark_best or result.get_score(metric) > benchmark_best[benchmark].get_score(metric):
            benchmark_best[benchmark] = result
    
    benchmarks = sorted(benchmark_best.keys())
    scores = [benchmark_best[b].get_score(metric) for b in benchmarks]
    
    fig = go.Figure(data=go.Bar(x=benchmarks, y=scores))
    fig.update_layout(
        title=f"Best {metric.replace('_', ' ').title()} by Benchmark",
        xaxis_title="Benchmark",
        yaxis_title=f"{metric.replace('_', ' ').title()}",
        xaxis_tickangle=-45
    )
    
    fig.show()

def plot_model_performance_distribution(finder: 'BestPerformanceFinder'):
    """Plot distribution of individual model performances"""
    if not finder.results:
        print("No results found. Run search_configurations first.")
        return
    
    model_performances = defaultdict(list)
    
    for result in finder.results:
        if result.individual_model_results:
            for model_name, f1_score in result.individual_model_results.items():
                model_performances[model_name].append(f1_score)
    
    if not model_performances:
        print("No individual model results found.")
        return
    
    fig = go.Figure()
    
    for model_name in sorted(model_performances.keys()):
        scores = model_performances[model_name]
        color = MODEL_COLORS.get(model_name, 'gray')
        
        fig.add_trace(go.Box(
            y=scores, name=model_name,
            marker_color=color,
            boxmean=True
        ))
    
    fig.update_layout(
        title="Individual Model Performance Distribution",
        xaxis_title="Model",
        yaxis_title="F1 Score",
        showlegend=False
    )
    
    fig.show()

def plot_configuration_impact(finder: 'BestPerformanceFinder', config_param: str, metric: str = 'best_model_f1'):
    """Plot impact of specific configuration parameter on performance"""
    if not finder.results:
        print("No results found. Run search_configurations first.")
        return
    
    config_performances = defaultdict(list)
    
    for result in finder.results:
        if hasattr(result.config, config_param):
            config_value = getattr(result.config, config_param)
            config_performances[str(config_value)].append(result.get_score(metric))
    
    if not config_performances:
        print(f"No results found for configuration parameter: {config_param}")
        return
    
    config_values = sorted(config_performances.keys())
    mean_scores = [sum(config_performances[cv]) / len(config_performances[cv]) for cv in config_values]
    
    fig = go.Figure(data=go.Bar(x=config_values, y=mean_scores))
    fig.update_layout(
        title=f"Impact of {config_param} on {metric.replace('_', ' ').title()}",
        xaxis_title=config_param,
        yaxis_title=f"Mean {metric.replace('_', ' ').title()}"
    )
    
    fig.show()

# Additional utility functions for advanced plotting
def create_performance_heatmap(finder: 'BestPerformanceFinder', metric: str = 'best_model_f1'):
    """Create heatmap of performance across benchmarks and configurations"""
    if not finder.results:
        print("No results found. Run search_configurations first.")
        return
    
    # Group results by benchmark and supervisor
    heatmap_data = defaultdict(lambda: defaultdict(list))
    
    for result in finder.results:
        benchmark = result.config.benchmark
        supervisor = result.config.supervisor_model_name.split('-')[0]  # Simplified supervisor name
        heatmap_data[benchmark][supervisor].append(result.get_score(metric))
    
    # Calculate means
    benchmarks = sorted(heatmap_data.keys())
    supervisors = sorted(set(sup for bench_data in heatmap_data.values() for sup in bench_data.keys()))
    
    z_data = []
    for benchmark in benchmarks:
        row = []
        for supervisor in supervisors:
            if supervisor in heatmap_data[benchmark] and heatmap_data[benchmark][supervisor]:
                mean_score = sum(heatmap_data[benchmark][supervisor]) / len(heatmap_data[benchmark][supervisor])
                row.append(mean_score)
            else:
                row.append(None)
        z_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=supervisors,
        y=benchmarks,
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title=f"Performance Heatmap - {metric.replace('_', ' ').title()}",
        xaxis_title="Supervisor Model",
        yaxis_title="Benchmark"
    )
    
    fig.show()

# Example usage functions that can be called independently
def quick_benchmark_plots(finder: 'BestPerformanceFinder'):
    """Generate a standard set of plots for analysis"""
    plot_interactive_performance_by_benchmark(finder)
    plot_performance_comparison(finder)
    plot_model_performance_distribution(finder)
    create_performance_heatmap(finder)

def save_individual_model_scatter_plots(finder: 'BestPerformanceFinder', save_dir: str = 'plots'):
    """Save individual model scatter plots as TIFF files with DPI 600"""
    import os
    from pathlib import Path
    
    if not finder.results:
        print("No results found. Run search_configurations first.")
        return
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    benchmark_results = defaultdict(list)
    for result in finder.results:
        benchmark_results[result.config.benchmark].append(result)
    
    for benchmark in sorted(benchmark_results.keys()):
        results = benchmark_results[benchmark]
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        added_models = set()
        for result in results:
            if not (result.individual_model_results and result.individual_model_strict_f1):
                continue
                
            for model_name, model_f1 in result.individual_model_results.items():
                model_strict_f1 = result.individual_model_strict_f1.get(model_name, 0.0)
                color = MODEL_COLORS.get(model_name, 'gray')
                
                # Add scatter point
                if model_name not in added_models:
                    ax.scatter(model_strict_f1, model_f1, color=color, 
                              label=model_name, alpha=0.7, s=50)
                    added_models.add(model_name)
                else:
                    ax.scatter(model_strict_f1, model_f1, color=color, alpha=0.7, s=50)
        
        ax.set_xlabel('Average Pairwise Strict Span F1')
        ax.set_ylabel('Gold Standard F1')
        ax.set_title(f'{BENCHMARK_DISPLAY_NAMES[benchmark]}', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Save as TIFF with DPI 600
        plt.tight_layout()
        tiff_path = save_path / f'{benchmark}_individual_models_scatter.tiff'
        plt.savefig(tiff_path, format='tiff', dpi=600, bbox_inches='tight')
        print(f"Saved: {tiff_path}")
        
        # Also save as PDF with text recognition
        pdf_path = save_path / f'{benchmark}_individual_models_scatter.pdf'
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Saved: {pdf_path}")
        plt.show()
        
        plt.close()

def analyze_correlation_by_benchmark(finder: 'BestPerformanceFinder'):
    """Analyze correlation between strict span F1 and gold standard F1 by benchmark"""
    if not finder.results:
        print("No results found. Run search_configurations first.")
        return
    
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS BY BENCHMARK")
    print(f"{'='*80}")
    
    benchmark_results = defaultdict(list)
    for result in finder.results:
        benchmark_results[result.config.benchmark].append(result)
    
    overall_strict_f1 = []
    overall_gold_f1 = []
    
    for benchmark in sorted(benchmark_results.keys()):
        results = benchmark_results[benchmark]
        
        strict_f1_values = []
        gold_f1_values = []
        
        # Collect all individual model data points within this benchmark
        for result in results:
            if result.individual_model_results and result.individual_model_strict_f1:
                for model_name, gold_f1 in result.individual_model_results.items():
                    strict_f1 = result.individual_model_strict_f1.get(model_name, 0.0)
                    if strict_f1 > 0:  # Only include valid data points
                        strict_f1_values.append(strict_f1)
                        gold_f1_values.append(gold_f1)
                        overall_strict_f1.append(strict_f1)
                        overall_gold_f1.append(gold_f1)
        
        if len(strict_f1_values) >= 3:  # Need at least 3 points for meaningful correlation
            # Calculate multiple correlation measures
            pearson_r, pearson_p = pearsonr(strict_f1_values, gold_f1_values)
            spearman_r, spearman_p = spearmanr(strict_f1_values, gold_f1_values)
            kendall_tau, kendall_p = kendalltau(strict_f1_values, gold_f1_values)
            
            print(f"\n{benchmark} (n={len(strict_f1_values)}):")
            print(f"  Pearson correlation:  r={pearson_r:.4f}, p={pearson_p:.4f}")
            print(f"  Spearman correlation: ρ={spearman_r:.4f}, p={spearman_p:.4f}")
            print(f"  Kendall's tau:        τ={kendall_tau:.4f}, p={kendall_p:.4f}")
        else:
            print(f"\n{benchmark}: Insufficient data points (n={len(strict_f1_values)})")
    
    # Overall correlation across all benchmarks
    if len(overall_strict_f1) >= 3:
        overall_pearson_r, overall_pearson_p = pearsonr(overall_strict_f1, overall_gold_f1)
        overall_spearman_r, overall_spearman_p = spearmanr(overall_strict_f1, overall_gold_f1)
        overall_kendall_tau, overall_kendall_p = kendalltau(overall_strict_f1, overall_gold_f1)
        
        print(f"\nOVERALL CORRELATION (n={len(overall_strict_f1)}):")
        print(f"  Pearson correlation:  r={overall_pearson_r:.4f}, p={overall_pearson_p:.4f}")
        print(f"  Spearman correlation: ρ={overall_spearman_r:.4f}, p={overall_spearman_p:.4f}")
        print(f"  Kendall's tau:        τ={overall_kendall_tau:.4f}, p={overall_kendall_p:.4f}")

def analyze_model_level_correlation(finder: 'BestPerformanceFinder'):
    """Analyze correlation at model level within each benchmark, then macro average"""
    if not finder.results:
        print("No results found. Run search_configurations first.")
        return None
    
    print(f"\n{'='*80}")
    print("MODEL-LEVEL CORRELATION ANALYSIS BY BENCHMARK")
    print(f"{'='*80}")
    
    # Group results by benchmark first
    benchmark_results = defaultdict(list)
    for result in finder.results:
        benchmark_results[result.config.benchmark].append(result)
    
    # Store correlations for macro averaging
    all_model_correlations = defaultdict(list)
    benchmark_model_correlations = {}
    
    # Dictionary to store benchmark-level correlations (all models pooled)
    benchmark_pooled_correlations = {}
    
    # Analyze each benchmark separately
    for benchmark in sorted(benchmark_results.keys()):
        print(f"\n{'-'*60}")
        print(f"BENCHMARK: {benchmark}")
        print(f"{'-'*60}")
        
        results = benchmark_results[benchmark]
        
        # Collect data by model within this benchmark
        model_data = defaultdict(lambda: {'strict_f1': [], 'gold_f1': []})
        
        # Also collect ALL data points for pooled correlation
        pooled_strict_f1 = []
        pooled_gold_f1 = []
        
        for result in results:
            if result.individual_model_results and result.individual_model_strict_f1:
                for model_name, gold_f1 in result.individual_model_results.items():
                    strict_f1 = result.individual_model_strict_f1.get(model_name, 0.0)
                    if strict_f1 > 0:
                        model_data[model_name]['strict_f1'].append(strict_f1)
                        model_data[model_name]['gold_f1'].append(gold_f1)
                        pooled_strict_f1.append(strict_f1)
                        pooled_gold_f1.append(gold_f1)
        
        # Calculate pooled correlation for this benchmark (all models together)
        if len(pooled_strict_f1) >= 3:
            pooled_pearson_r, pooled_pearson_p = pearsonr(pooled_strict_f1, pooled_gold_f1)
            pooled_spearman_r, pooled_spearman_p = spearmanr(pooled_strict_f1, pooled_gold_f1)
            pooled_kendall_tau, pooled_kendall_p = kendalltau(pooled_strict_f1, pooled_gold_f1)
            
            benchmark_pooled_correlations[benchmark] = pooled_pearson_r
            
            print(f"\nPooled correlation (all models, n={len(pooled_strict_f1)}):")
            print(f"  Pearson:  r={pooled_pearson_r:.4f}, p={pooled_pearson_p:.4f}")
            print(f"  Spearman: ρ={pooled_spearman_r:.4f}, p={pooled_spearman_p:.4f}")
            print(f"  Kendall:  τ={pooled_kendall_tau:.4f}, p={pooled_kendall_p:.4f}")
        
        benchmark_model_correlations[benchmark] = {}
        
        # Calculate correlation for each model within this benchmark
        for model_name in sorted(model_data.keys()):
            data = model_data[model_name]
            if len(data['strict_f1']) >= 3:
                pearson_r, pearson_p = pearsonr(data['strict_f1'], data['gold_f1'])
                spearman_r, spearman_p = spearmanr(data['strict_f1'], data['gold_f1'])
                kendall_tau, kendall_p = kendalltau(data['strict_f1'], data['gold_f1'])
                
                correlations = {
                    'pearson': {'r': pearson_r, 'p': pearson_p},
                    'spearman': {'r': spearman_r, 'p': spearman_p},
                    'kendall': {'tau': kendall_tau, 'p': kendall_p},
                    'n': len(data['strict_f1'])
                }
                
                benchmark_model_correlations[benchmark][model_name] = correlations
                all_model_correlations[model_name].append(correlations)
                
                print(f"  {model_name} (n={len(data['strict_f1'])}):")
                print(f"    Pearson:  r={pearson_r:.4f}, p={pearson_p:.4f}")
                print(f"    Spearman: ρ={spearman_r:.4f}, p={spearman_p:.4f}")
                print(f"    Kendall:  τ={kendall_tau:.4f}, p={kendall_p:.4f}")
            else:
                print(f"  {model_name}: Insufficient data (n={len(data['strict_f1'])})")
    
    # Overall macro averaging across all models and benchmarks
    print(f"\n{'='*80}")
    print("OVERALL MODEL-LEVEL MACRO AVERAGE CORRELATION")
    print(f"{'='*80}")
    
    # Method 1: Macro average within each benchmark, then average across benchmarks
    benchmark_macro_correlations = {'pearson': [], 'spearman': [], 'kendall': []}
    
    for benchmark, model_corrs in benchmark_model_correlations.items():
        if model_corrs:
            bench_pearson = np.mean([corr['pearson']['r'] for corr in model_corrs.values()])
            bench_spearman = np.mean([corr['spearman']['r'] for corr in model_corrs.values()])
            bench_kendall = np.mean([corr['kendall']['tau'] for corr in model_corrs.values()])
            
            benchmark_macro_correlations['pearson'].append(bench_pearson)
            benchmark_macro_correlations['spearman'].append(bench_spearman)
            benchmark_macro_correlations['kendall'].append(bench_kendall)
            
            print(f"\n{benchmark} macro average (across {len(model_corrs)} models):")
            print(f"  Pearson:  r={bench_pearson:.4f}")
            print(f"  Spearman: ρ={bench_spearman:.4f}")
            print(f"  Kendall:  τ={bench_kendall:.4f}")
    
    # Final macro average across benchmarks
    if benchmark_macro_correlations['pearson']:
        overall_macro_pearson = np.mean(benchmark_macro_correlations['pearson'])
        overall_macro_spearman = np.mean(benchmark_macro_correlations['spearman'])
        overall_macro_kendall = np.mean(benchmark_macro_correlations['kendall'])
        
        print(f"\nFINAL MACRO AVERAGE (across {len(benchmark_macro_correlations['pearson'])} benchmarks):")
        print(f"  Pearson (macro):  r={overall_macro_pearson:.4f}")
        print(f"  Spearman (macro): ρ={overall_macro_spearman:.4f}")
        print(f"  Kendall (macro):  τ={overall_macro_kendall:.4f}")
        
        # Standard deviation across benchmarks
        std_pearson = np.std(benchmark_macro_correlations['pearson'])
        std_spearman = np.std(benchmark_macro_correlations['spearman'])
        std_kendall = np.std(benchmark_macro_correlations['kendall'])
        
        print(f"\nSTANDARD DEVIATION (across benchmarks):")
        print(f"  Pearson (std):  {std_pearson:.4f}")
        print(f"  Spearman (std): {std_spearman:.4f}")
        print(f"  Kendall (std):  {std_kendall:.4f}")
    
    # Method 2: Individual model macro average
    print(f"\n{'-'*60}")
    print("INDIVIDUAL MODEL MACRO AVERAGES (across benchmarks)")
    print(f"{'-'*60}")
    
    model_macro_correlations = {'pearson': [], 'spearman': [], 'kendall': []}
    
    for model_name in sorted(all_model_correlations.keys()):
        model_correlations_list = all_model_correlations[model_name]
        if len(model_correlations_list) >= 1:
            model_pearson = np.mean([corr['pearson']['r'] for corr in model_correlations_list])
            model_spearman = np.mean([corr['spearman']['r'] for corr in model_correlations_list])
            model_kendall = np.mean([corr['kendall']['tau'] for corr in model_correlations_list])
            
            model_macro_correlations['pearson'].append(model_pearson)
            model_macro_correlations['spearman'].append(model_spearman)
            model_macro_correlations['kendall'].append(model_kendall)
            
            print(f"{model_name} (across {len(model_correlations_list)} benchmarks):")
            print(f"  Pearson:  r={model_pearson:.4f}")
            print(f"  Spearman: ρ={model_spearman:.4f}")
            print(f"  Kendall:  τ={model_kendall:.4f}")
    
    # Final model-wise macro average
    if model_macro_correlations['pearson']:
        final_model_macro_pearson = np.mean(model_macro_correlations['pearson'])
        final_model_macro_spearman = np.mean(model_macro_correlations['spearman'])
        final_model_macro_kendall = np.mean(model_macro_correlations['kendall'])
        
        print(f"\nFINAL MODEL-WISE MACRO AVERAGE (across {len(model_macro_correlations['pearson'])} models):")
        print(f"  Pearson (macro):  r={final_model_macro_pearson:.4f}")
        print(f"  Spearman (macro): ρ={final_model_macro_spearman:.4f}")
        print(f"  Kendall (macro):  τ={final_model_macro_kendall:.4f}")
    
    # Return dictionary mapping benchmark to POOLED Pearson correlation
    return benchmark_pooled_correlations

def save_combined_individual_model_scatter_plots(finder: 'BestPerformanceFinder', 
                                                 save_dir: str = 'plots',
                                                 config: dict = None,
                                                 benchmark_correlations: dict = None,
                                                 print_improvement_stats: bool = False):
    """Save all individual model scatter plots with correlation info and optional improvement stats"""
    from pathlib import Path
    
    if not finder.results:
        print("No results found. Run search_configurations first.")
        return
    
    # Use default config if none provided
    if config is None:
        config = SUBPLOT_CONFIG.copy()
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    benchmark_results = defaultdict(list)
    for result in finder.results:
        benchmark_results[result.config.benchmark].append(result)
    
    # Calculate improvement statistics if requested
    if print_improvement_stats:
        print(f"\n{'='*80}")
        print("PERFORMANCE IMPROVEMENT STATISTICS")
        print(f"{'='*80}")
        
        benchmark_improvements = {}
        all_model_improvements = []
        
        for benchmark in sorted(benchmark_results.keys()):
            results = benchmark_results[benchmark]
            
            # Group results by model
            model_performances = defaultdict(lambda: {'iter0': [], 'max': []})
            
            for result in results:
                if not (result.individual_model_results and result.individual_model_strict_f1):
                    continue
                
                iteration = result.iteration
                
                for model_name, gold_f1 in result.individual_model_results.items():
                    if iteration == 0:
                        model_performances[model_name]['iter0'].append(gold_f1)
                    model_performances[model_name]['max'].append(gold_f1)
            
            # Calculate improvements for each model
            model_improvements = []
            
            print(f"\n{benchmark}:")
            for model_name in sorted(model_performances.keys()):
                perf = model_performances[model_name]
                if perf['iter0'] and perf['max']:
                    iter0_avg = np.mean(perf['iter0'])
                    max_perf = np.max(perf['max'])
                    improvement = max_perf - iter0_avg
                    improvement_pct = (improvement / iter0_avg * 100) if iter0_avg > 0 else 0
                    
                    model_improvements.append(improvement)
                    all_model_improvements.append(improvement)
                    
                    print(f"  {model_name:30s}: Iter0 avg={iter0_avg:.4f}, Max={max_perf:.4f}, "
                          f"Improvement={improvement:+.4f} ({improvement_pct:+.2f}%)")
            
            # Calculate benchmark average improvement
            if model_improvements:
                benchmark_avg_improvement = np.mean(model_improvements)
                benchmark_std_improvement = np.std(model_improvements)
                benchmark_improvements[benchmark] = benchmark_avg_improvement
                
                print(f"  {'BENCHMARK AVERAGE':30s}: {benchmark_avg_improvement:+.4f} "
                      f"(std={benchmark_std_improvement:.4f})")
        
        # Overall statistics across all benchmarks
        if benchmark_improvements:
            print(f"\n{'-'*80}")
            print("OVERALL STATISTICS ACROSS ALL BENCHMARKS:")
            print(f"{'-'*80}")
            
            overall_benchmark_avg = np.mean(list(benchmark_improvements.values()))
            overall_benchmark_std = np.std(list(benchmark_improvements.values()))
            overall_model_avg = np.mean(all_model_improvements)
            overall_model_std = np.std(all_model_improvements)
            
            print(f"Average improvement per benchmark: {overall_benchmark_avg:+.4f} "
                  f"(std={overall_benchmark_std:.4f}, n={len(benchmark_improvements)})")
            print(f"Average improvement per model:     {overall_model_avg:+.4f} "
                  f"(std={overall_model_std:.4f}, n={len(all_model_improvements)})")
            
            # Min/Max benchmarks
            best_benchmark = max(benchmark_improvements.items(), key=lambda x: x[1])
            worst_benchmark = min(benchmark_improvements.items(), key=lambda x: x[1])
            
            print(f"\nBest improving benchmark:  {best_benchmark[0]} ({best_benchmark[1]:+.4f})")
            print(f"Worst improving benchmark: {worst_benchmark[0]} ({worst_benchmark[1]:+.4f})")
        
        print(f"{'='*80}\n")
    
    # Define custom order for benchmarks
    benchmark_order = [
        'crossner_ai', 'crossner_literature', 'crossner_music', 'crossner_politics', 'crossner_science',
        'mitner_movie', 'mitner_restaurant', 'Broad Twitter', 'crossner_conll2003', 'FabNER', 
        'MultiNERD', 'ACE05', 'anatem', 'bc2gm', 'bc4chemd', 'bc5cdr', 'GENIA', 'OntoNotes'
    ]
    
    # Filter benchmarks to only include those found in results
    benchmarks = [bench for bench in benchmark_order if bench in benchmark_results]
    
    if len(benchmarks) > 18:
        print(f"Warning: {len(benchmarks)} benchmarks found. Only first 18 will be plotted.")
        benchmarks = benchmarks[:18]
    
    # Create figure with subplots
    fig, axes = plt.subplots(6, 3, figsize=config['figure_size'])
    axes = axes.flatten()
    
    # Track models for unified legend
    all_models = set()
    
    # Plot each benchmark
    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx]
        results = benchmark_results[benchmark]
        
        # Track models added to this subplot
        subplot_models = set()
        
        for result in results:
            if not (result.individual_model_results and result.individual_model_strict_f1):
                continue
                
            for model_name, model_f1 in result.individual_model_results.items():
                model_strict_f1 = result.individual_model_strict_f1.get(model_name, 0.0)
                color = MODEL_COLORS.get(model_name, 'gray')
                
                # Add scatter point
                ax.scatter(model_strict_f1, model_f1, 
                          color=color, 
                          alpha=config['alpha'], 
                          s=config['dot_size'])
                
                subplot_models.add(model_name)
                all_models.add(model_name)
        
        # Configure subplot
        ax.set_xlabel('Avg Pairwise Strict Span F1', fontsize=config['axis_fontsize'])
        ax.set_ylabel('Gold Standard F1', fontsize=config['axis_fontsize'])
        ax.set_title(BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark), 
                    fontsize=config['title_fontsize'], fontweight='bold')
        ax.grid(True, alpha=config['grid_alpha'])
        ax.tick_params(labelsize=config['axis_fontsize']-1)
        
        # Add Pearson correlation in top-left corner if available
        if benchmark_correlations and benchmark in benchmark_correlations:
            corr_value = benchmark_correlations[benchmark]
            ax.text(0.03, 0.95, f'ρ = {corr_value:.3f}', 
                   transform=ax.transAxes,
                   fontsize=config['axis_fontsize']*1.5,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Hide unused subplots
    for idx in range(len(benchmarks), 18):
        axes[idx].set_visible(False)
    
    # Create unified legend
    legend_elements = []
    for model_name in sorted(all_models):
        color = MODEL_COLORS.get(model_name, 'gray')
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=8, 
                                        label=model_name, alpha=config['alpha']))
    
    # Place legend
    fig.legend(handles=legend_elements, 
              loc='center right', 
              bbox_to_anchor=(0.925, 0.15),
              fontsize=config['legend_fontsize'])
    
    # Adjust layout
    plt.subplots_adjust(
        hspace=config['subplot_spacing']['hspace'],
        wspace=config['subplot_spacing']['wspace'],
        right=0.83
    )
    
    # Save files
    tiff_path = save_path / 'combined_individual_models_scatter.tiff'
    plt.savefig(tiff_path, format='tiff', dpi=600, bbox_inches='tight')
    print(f"Saved: {tiff_path}")
    
    pdf_path = save_path / 'combined_individual_models_scatter.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {pdf_path}")
    
    plt.show()
    plt.close()