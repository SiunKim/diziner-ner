"""
Refactored visualization module for iterative NER experiment results
Provides comprehensive analysis of performance evolution across iterations
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import seaborn as sns
from collections import defaultdict

# Import for experiment suffix generation
from supervisor_implementation import generate_experiment_suffix

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class ExperimentConfig:
    """Configuration for iterative experiment"""
    benchmark: str
    dataset_path: str
    num_groups: int
    group_size: int
    models: int
    starting_group_index: int = 0
    max_iterations: int = 5
    supervisor_model_name: str = "gpt-5-2025-08-07"
    supervisor_params: Dict = field(default_factory=dict)


@dataclass
class PlotConfig:
    """Configuration for plotting"""
    figsize: Tuple[int, int] = (10, 8)
    dpi: int = 600
    colors: List[str] = field(default_factory=lambda: [
        'skyblue', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'crimson',
        'olive', 'cyan', 'magenta', 'yellow', 'navy', 'lime'
    ])
    markers: List[str] = field(default_factory=lambda: [
        'o', 's', '^', 'D', 'p', '*', 'h', 'H', 'v', '<', '>', 'd', 'P', 'X'
    ])
    linestyles: List[str] = field(default_factory=lambda: ['-', '--', '-.', ':'])


class ExperimentDataLoader:
    """Handles loading and basic processing of experiment data"""
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def get_iteration_paths(self, iteration_number: int, group_index: int) -> Dict[str, Path]:
        """Get paths for specific iteration with supervisor suffix"""
        enhanced_suffix = generate_experiment_suffix(**self.config.supervisor_params)
        supervisor_safe_name = self.config.supervisor_model_name.replace(':', '_').replace('/', '_').replace(' ', '_')
        
        experiment_name = f"g{self.config.num_groups}_s{self.config.group_size}_grp{group_index}_iter{iteration_number}{enhanced_suffix}"
        base_dir = Path(f'experiment_results/{self.config.benchmark}/models{self.config.models}') / supervisor_safe_name / experiment_name
        
        return {
            'combined_results_file': base_dir / 'combined_results.json',
            'analysis_summary_file': base_dir / 'agreement_analysis' / 'main_results_analysis' / 'analysis_summary.json',
            'pairwise_agreements_file': base_dir / 'agreement_analysis' / 'main_results_analysis' / 'pairwise_agreements.json'
        }
    
    def load_all_iteration_data(self) -> Dict[int, Dict[str, Any]]:
        """Load data from all available iterations including Elite MV"""
        iteration_data = {}
        
        print("Loading iteration data...")
        for iteration in range(self.config.max_iterations):
            group_index = self.config.starting_group_index + iteration
            paths = self.get_iteration_paths(iteration, group_index)
            
            combined_file = paths['combined_results_file']
            if combined_file.exists():
                try:
                    with open(combined_file, 'r', encoding='utf-8') as f:
                        combined_results = json.load(f)
                    
                    agreement_results = None
                    if paths['analysis_summary_file'].exists():
                        with open(paths['analysis_summary_file'], 'r', encoding='utf-8') as f:
                            agreement_results = json.load(f)
                    
                    pairwise_agreements = None
                    if paths['pairwise_agreements_file'].exists():
                        try:
                            with open(paths['pairwise_agreements_file'], 'r', encoding='utf-8') as f:
                                pairwise_agreements = json.load(f)
                        except Exception as e:
                            print(f"  Warning: Failed to load pairwise agreements for iteration {iteration}: {e}")

                    iteration_data[iteration] = {
                        'combined_results': combined_results,
                        'agreement_results': agreement_results,
                        'pairwise_agreements': pairwise_agreements
                    }
                    print(f"  Loaded iteration {iteration} (group {group_index})")
                    
                except Exception as e:
                    print(f"  Warning: Failed to load iteration {iteration}: {e}")
        
        print(f"Successfully loaded {len(iteration_data)} iterations")
        return iteration_data


class MetricsCalculator:
    """Handles calculation of various metrics from raw data"""
    def __init__(self, iteration_data: Dict[int, Dict[str, Any]]):
        self.iteration_data = iteration_data
    
    def extract_performance_data(self) -> Dict[str, Any]:
        """Extract performance metrics from all iterations including Elite MV"""
        performance_data = {
            'iterations': [],
            'models': set(),
            'individual_performance': defaultdict(list),
            'mv_performance': [],
            'elite_mv_performance': [],  # Add Elite MV tracking
            'iteration_summaries': []
        }
        
        for iteration, data in self.iteration_data.items():
            combined_results = data['combined_results']
            agreement_results = data['agreement_results']
            
            performance_data['iterations'].append(iteration)
            results_by_model = combined_results.get('results_by_model', {})
            iteration_models = {}
            
            for model_name, model_result in results_by_model.items():
                if 'error' not in model_result:
                    performance_data['models'].add(model_name)
                    gold_f1 = model_result.get('avg_metrics', {}).get('f1', 0.0)
                    
                    performance_data['individual_performance'][model_name].append({
                        'iteration': iteration,
                        'gold_f1': gold_f1,
                        'precision': model_result.get('avg_metrics', {}).get('precision', 0.0),
                        'recall': model_result.get('avg_metrics', {}).get('recall', 0.0)
                    })
                    
                    iteration_models[model_name] = {'gold_f1': gold_f1}
            
            # Extract MV performance
            mv_gold_f1 = 0.0
            elite_mv_gold_f1 = 0.0
            
            if agreement_results and 'model_average_agreements' in agreement_results:
                model_avg_agreements = agreement_results['model_average_agreements']
                
                # Standard MajorityVote
                if 'MajorityVote' in model_avg_agreements:
                    mv_gold_f1 = model_avg_agreements['MajorityVote'].get('gold_macro_f1', 0.0)
                elif 'DawidSkene' in model_avg_agreements:
                    mv_gold_f1 = model_avg_agreements['DawidSkene'].get('gold_macro_f1', 0.0)
                
                # Elite MajorityVote
                if 'EliteMajorityVote' in model_avg_agreements:
                    elite_mv_gold_f1 = model_avg_agreements['EliteMajorityVote'].get('gold_macro_f1', 0.0)
            
            performance_data['mv_performance'].append({
                'iteration': iteration,
                'gold_f1': mv_gold_f1
            })
            
            performance_data['elite_mv_performance'].append({
                'iteration': iteration,
                'gold_f1': elite_mv_gold_f1
            })
            
            # Create iteration summary
            if iteration_models:
                model_f1s = [m['gold_f1'] for m in iteration_models.values()]
                best_model = max(iteration_models.items(), key=lambda x: x[1]['gold_f1'])
                
                performance_data['iteration_summaries'].append({
                    'iteration': iteration,
                    'best_model': best_model,
                    'avg_performance': np.mean(model_f1s),
                    'mv_performance': mv_gold_f1,
                    'elite_mv_performance': elite_mv_gold_f1,
                    'model_count': len(iteration_models)
                })
        
        performance_data['models'] = sorted(list(performance_data['models']))
        performance_data['iterations'] = sorted(performance_data['iterations'])
        
        return performance_data

    def extract_pairwise_metric_data(self, metric_key: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract per-model average for specified metric from pairwise agreements"""
        metric_data = defaultdict(list)
        
        for iteration, data in self.iteration_data.items():
            pairwise_agreements = data.get('pairwise_agreements')
            if not pairwise_agreements:
                continue

            # Accumulate per-model averages across pairs
            sums = defaultdict(lambda: {'sum': 0.0, 'count': 0})
            for agreement_rec in pairwise_agreements.values():
                m1, m2 = agreement_rec.get('model1'), agreement_rec.get('model2')
                metric_val = agreement_rec.get('avg_agreement', {}).get(metric_key, 0.0)
                if m1 and m2:
                    for m in (m1, m2):
                        sums[m]['sum'] += float(metric_val)
                        sums[m]['count'] += 1

            for model, agg in sums.items():
                if agg['count'] > 0:
                    metric_data[model].append({
                        'iteration': iteration,
                        metric_key: agg['sum'] / agg['count']
                    })

        return dict(metric_data)
    
    def compute_inter_model_variances(self, performance_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Compute inter-model variances per iteration"""
        if not performance_data['iteration_summaries']:
            return {'iterations': [], 'gold_f1_var': [], 'strict_span_f1_var': [], 'cohens_kappa_var': []}

        iterations = [s['iteration'] for s in performance_data['iteration_summaries']]

        # Gold F1 variance across models
        gold_f1_var = []
        for it in iterations:
            model_f1s = []
            for model_name in performance_data['models']:
                rec = next((d for d in performance_data['individual_performance'][model_name]
                            if d['iteration'] == it), None)
                if rec:
                    model_f1s.append(rec['gold_f1'])
            gold_f1_var.append(float(np.var(model_f1s)) if model_f1s else 0.0)

        # Variances from pairwise agreements
        strict_span_f1_var, cohens_kappa_var = [], []
        for it in iterations:
            pdata = self.iteration_data.get(it, {}).get('pairwise_agreements')
            if not pdata:
                strict_span_f1_var.append(0.0)
                cohens_kappa_var.append(0.0)
                continue

            sums = defaultdict(lambda: {'strict_span_f1': [0.0, 0], 'cohens_kappa': [0.0, 0]})
            for rec in pdata.values():
                m1, m2 = rec.get("model1"), rec.get("model2")
                avg = rec.get("avg_agreement", {})
                if not m1 or not m2 or not avg:
                    continue
                strict_v = float(avg.get("strict_span_f1", 0.0))
                kappa_v = float(avg.get("cohens_kappa", 0.0))
                for m in (m1, m2):
                    sums[m]['strict_span_f1'][0] += strict_v
                    sums[m]['strict_span_f1'][1] += 1
                    sums[m]['cohens_kappa'][0] += kappa_v
                    sums[m]['cohens_kappa'][1] += 1

            strict_avgs, kappa_avgs = [], []
            for m, agg in sums.items():
                s_sum, s_cnt = agg['strict_span_f1']
                k_sum, k_cnt = agg['cohens_kappa']
                if s_cnt > 0:
                    strict_avgs.append(s_sum / s_cnt)
                if k_cnt > 0:
                    kappa_avgs.append(k_sum / k_cnt)

            strict_span_f1_var.append(float(np.var(strict_avgs)) if strict_avgs else 0.0)
            cohens_kappa_var.append(float(np.var(kappa_avgs)) if kappa_avgs else 0.0)

        return {
            'iterations': iterations,
            'gold_f1_var': gold_f1_var,
            'strict_span_f1_var': strict_span_f1_var,
            'cohens_kappa_var': cohens_kappa_var,
        }


class TrajectoryCalculator:
    """Handles trajectory calculations for evolution plots"""
    def __init__(self, performance_data: Dict[str, Any]):
        self.performance_data = performance_data
    
    def get_model_trajectory(self, model_name: str, metric_data: Dict, metric_key: str) -> List[Dict]:
        """Get combined metric and gold F1 trajectory for a model"""
        if model_name not in metric_data:
            return []
        
        trajectory = []
        model_metric_records = metric_data[model_name]
        model_gold_records = self.performance_data['individual_performance'][model_name]
        
        for metric_rec in model_metric_records:
            iteration = metric_rec['iteration']
            gold_rec = next((g for g in model_gold_records if g['iteration'] == iteration), None)
            if gold_rec:
                trajectory.append({
                    'iteration': iteration,
                    metric_key: metric_rec[metric_key],
                    'gold_f1': gold_rec['gold_f1']
                })
        
        return sorted(trajectory, key=lambda x: x['iteration'])
    
    def calculate_average_trajectory(self, metric_data: Dict, metric_key: str) -> List[Dict]:
        """Calculate average trajectory across models"""
        metric_by_iter = defaultdict(list)
        gold_by_iter = defaultdict(list)
        
        for model_name in self.performance_data['models']:
            trajectory = self.get_model_trajectory(model_name, metric_data, metric_key)
            for point in trajectory:
                iteration = point['iteration']
                metric_by_iter[iteration].append(point[metric_key])
                gold_by_iter[iteration].append(point['gold_f1'])
        
        return [{
            'iteration': iteration,
            metric_key: float(np.mean(metric_by_iter[iteration])),
            'gold_f1': float(np.mean(gold_by_iter[iteration]))
        } for iteration in sorted(metric_by_iter.keys()) 
        if metric_by_iter[iteration] and gold_by_iter[iteration]]
    
    def calculate_mv_trajectory(self, metric_data: Dict, metric_key: str) -> List[Dict]:
        """Calculate MajorityVote trajectory using average metric values"""
        mv_trajectory = []
        
        for iteration in sorted(self.performance_data['iterations']):
            mv_performance = next((d for d in self.performance_data['mv_performance'] 
                                if d['iteration'] == iteration), None)
            if not mv_performance or mv_performance['gold_f1'] <= 0:
                continue
            
            metric_vals = []
            for model_name in metric_data:
                model_metric_records = metric_data[model_name]
                metric_rec = next((k for k in model_metric_records if k['iteration'] == iteration), None)
                if metric_rec:
                    metric_vals.append(metric_rec[metric_key])
            
            if metric_vals:
                mv_trajectory.append({
                    'iteration': iteration,
                    metric_key: float(np.mean(metric_vals)),
                    'gold_f1': mv_performance['gold_f1']
                })
        
        return mv_trajectory

    def calculate_elite_mv_trajectory(self, metric_data: Dict, metric_key: str) -> List[Dict]:
        """Calculate Elite MajorityVote trajectory using average metric values"""
        elite_mv_trajectory = []
        
        for iteration in sorted(self.performance_data['iterations']):
            elite_mv_performance = next((d for d in self.performance_data['elite_mv_performance'] 
                                        if d['iteration'] == iteration), None)
            if not elite_mv_performance or elite_mv_performance['gold_f1'] <= 0:
                continue
            
            metric_vals = []
            for model_name in metric_data:
                model_metric_records = metric_data[model_name]
                metric_rec = next((k for k in model_metric_records if k['iteration'] == iteration), None)
                if metric_rec:
                    metric_vals.append(metric_rec[metric_key])
            
            if metric_vals:
                elite_mv_trajectory.append({
                    'iteration': iteration,
                    metric_key: float(np.mean(metric_vals)),
                    'gold_f1': elite_mv_performance['gold_f1']
                })
        
        return elite_mv_trajectory


class PlotUtils:
    """Utility functions for plotting"""
    
    @staticmethod
    def plot_trajectory_points(ax, trajectory: List[Dict], metric_key: str, marker: str, 
                             color: str, label: str, iterations: List[int], size: int = 110):
        """Plot trajectory points with intensity based on iteration"""
        if not trajectory:
            return
        
        x_vals = [d[metric_key] for d in trajectory]
        y_vals = [d['gold_f1'] for d in trajectory]
        iter_nums = [d['iteration'] for d in trajectory]
        
        max_iter = max(iterations) if iterations else 1
        min_iter = min(iterations) if iterations else 0
        
        for j, (x, y, iter_num) in enumerate(zip(x_vals, y_vals, iter_nums)):
            intensity = (iter_num - min_iter) / (max_iter - min_iter) if max_iter > min_iter else 0.5
            alpha_val = 0.3 + 0.7 * intensity
            
            ax.scatter(x, y, s=size, marker=marker, color=color, alpha=alpha_val,
                    edgecolors='white', linewidths=0.8, zorder=5,
                    label=label if j == 0 else "")
        
        # Draw evolution arrows
        for j in range(len(x_vals) - 1):
            ax.annotate('', xy=(x_vals[j+1], y_vals[j+1]), xytext=(x_vals[j], y_vals[j]),
                    arrowprops=dict(arrowstyle='->', color='#2a2a2a', alpha=0.6, lw=1))
    
    @staticmethod
    def setup_equal_axes(ax, all_x_vals: List[float], all_y_vals: List[float], 
                        equal_axes: bool, metric_name: str):
        """Setup axes with optional equal scaling and diagonal line"""
        if equal_axes:
            min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
            max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3,
                    label=f'Perfect {metric_name}–Gold Agreement')
        else:
            if all_x_vals and all_y_vals:
                x_min, x_max = min(all_x_vals), max(all_x_vals)
                y_min, y_max = min(all_y_vals), max(all_y_vals)
                x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 0.05
                y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.05
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)
                diag_min = max(x_min - x_pad, y_min - y_pad)
                diag_max = min(x_max + x_pad, y_max + y_pad)
                ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', alpha=0.3,
                        label=f'Perfect {metric_name}–Gold Agreement')


class VisualizationRenderer:
    """Handles all visualization rendering"""
    def __init__(self, config: ExperimentConfig, plot_config: PlotConfig, 
                 performance_data: Dict[str, Any], metrics_calc: MetricsCalculator,
                 trajectory_calc: TrajectoryCalculator, 
                 model_colors: Optional[Dict[str, str]] = None,
                 display_names: Optional[Dict[str, str]] = None):
        self.config = config
        self.plot_config = plot_config
        self.performance_data = performance_data
        self.metrics_calc = metrics_calc
        self.trajectory_calc = trajectory_calc
        self.model_colors = model_colors or {}
        self.display_names = display_names or {}
        
    def plot_performance_over_iterations(self, figsize: Optional[Tuple[int, int]] = None):
        """Plot performance evolution over iterations against Gold Standard with Elite MV"""
        if not self.performance_data['iterations']:
            print("No iteration data available for plotting")
            return
        
        figsize = figsize or self.plot_config.figsize
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        iterations = self.performance_data['iterations']
        
        # Plot individual models
        for i, model_name in enumerate(self.performance_data['models']):
            model_data = self.performance_data['individual_performance'][model_name]
            disp = self.display_names.get(model_name, model_name)
            if model_data:
                model_iterations = [d['iteration'] for d in model_data]
                model_gold_f1s = [d['gold_f1'] for d in model_data]
                
                ax.plot(model_iterations, model_gold_f1s,
                        marker='o', linewidth=2, markersize=6,
                        color=self.model_colors.get(disp, 'black'),
                        label=disp, alpha=0.7) 
        
        # Plot MV performance
        mv_iterations = [d['iteration'] for d in self.performance_data['mv_performance']]
        mv_gold_f1s = [d['gold_f1'] for d in self.performance_data['mv_performance']]
        
        if mv_gold_f1s and any(f1 > 0 for f1 in mv_gold_f1s):
            ax.plot(mv_iterations, mv_gold_f1s, 
                    marker='s', linewidth=2.5, markersize=8, linestyle='--',
                    color='red', label='MajorityVote', alpha=0.9)
        
        # Plot Elite MV performance
        elite_mv_iterations = [d['iteration'] for d in self.performance_data['elite_mv_performance']]
        elite_mv_gold_f1s = [d['gold_f1'] for d in self.performance_data['elite_mv_performance']]
        
        if elite_mv_gold_f1s and any(f1 > 0 for f1 in elite_mv_gold_f1s):
            ax.plot(elite_mv_iterations, elite_mv_gold_f1s, 
                    marker='D', linewidth=2.5, markersize=8, linestyle='-.',
                    color='darkred', label='Elite MajorityVote', alpha=0.9)
        
        # Plot summary performance
        if self.performance_data['iteration_summaries']:
            best_iterations = [s['iteration'] for s in self.performance_data['iteration_summaries']]
            best_f1s = [s['best_model'][1]['gold_f1'] for s in self.performance_data['iteration_summaries']]
            avg_f1s = [s['avg_performance'] for s in self.performance_data['iteration_summaries']]
            
            ax.plot(best_iterations, best_f1s, 
                    marker='^', linewidth=2.5, markersize=8,
                    color='darkgreen', label='Best Model', linestyle='--', alpha=0.9)
            ax.plot(best_iterations, avg_f1s, 
                    marker='d', linewidth=2.5, markersize=6,
                    color='gray', label='Average', linestyle=':', alpha=0.8)
        
        ax.set_xlim(min(iterations) - 0.1, max(iterations) + 0.1)
        ax.set_xticks(iterations)
        ax.set_xlabel('Iteration Number', fontsize=12)
        ax.set_ylabel('Gold Standard F1 Score', fontsize=12)
        ax.set_title(f'Performance Evolution Over Iterations - {self.config.benchmark.upper()}', 
            fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

    def plot_metric_evolution(self, metric_key: str, metric_display_name: str,
                            equal_axes: bool = True, figsize: Optional[Tuple[int, int]] = None):
        """Generic method to plot metric evolution including Elite MV"""
        if not self.performance_data['iterations']:
            print("No iteration data available for plotting")
            return
        
        metric_data = self.metrics_calc.extract_pairwise_metric_data(metric_key)
        if not metric_data:
            print(f"No {metric_display_name} data available for plotting")
            return
        
        figsize = figsize or (11, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        all_x_vals, all_y_vals = [], []
        
        # Plot individual models
        for i, model_name in enumerate(self.performance_data['models']):
            trajectory = self.trajectory_calc.get_model_trajectory(model_name, metric_data, metric_key)
            disp = self.display_names.get(model_name, model_name)
            if len(trajectory) < 2:
                continue
            
            x_vals = [d[metric_key] for d in trajectory]
            y_vals = [d['gold_f1'] for d in trajectory]
            iterations_model = [d['iteration'] for d in trajectory]
            
            all_x_vals.extend(x_vals)
            all_y_vals.extend(y_vals)
            
            marker = self.plot_config.markers[i % len(self.plot_config.markers)]
            base_color = self.model_colors.get(disp, 'black')
            max_iter = max(self.performance_data['iterations'])
            min_iter = min(self.performance_data['iterations'])
            
            # Plot points with varying intensity
            for j, (x, y, iter_num) in enumerate(zip(x_vals, y_vals, iterations_model)):
                intensity = (iter_num - min_iter) / (max_iter - min_iter) if max_iter > min_iter else 0.5
                alpha_val = 0.3 + 0.7 * intensity
                
                ax.scatter(x, y, marker=marker, s=120, c=base_color, alpha=alpha_val,
                        edgecolors='black', linewidth=1, label=disp if j == 0 else "")
            
            # Draw evolution arrows
            for j in range(len(x_vals) - 1):
                ax.annotate('', xy=(x_vals[j+1], y_vals[j+1]), xytext=(x_vals[j], y_vals[j]),
                        arrowprops=dict(arrowstyle='->', color='#2a2a2a', alpha=0.6, lw=1))
        
        # Plot average, MV, and Elite MV trajectories
        avg_trajectory = self.trajectory_calc.calculate_average_trajectory(metric_data, metric_key)
        mv_trajectory = self.trajectory_calc.calculate_mv_trajectory(metric_data, metric_key)
        elite_mv_trajectory = self.trajectory_calc.calculate_elite_mv_trajectory(metric_data, metric_key)
        
        if avg_trajectory:
            all_x_vals.extend([d[metric_key] for d in avg_trajectory])
            all_y_vals.extend([d['gold_f1'] for d in avg_trajectory])
            PlotUtils.plot_trajectory_points(ax, avg_trajectory, metric_key, 'D', 'black', 'Average', 
                                        self.performance_data['iterations'])
        
        if mv_trajectory:
            all_x_vals.extend([d[metric_key] for d in mv_trajectory])
            all_y_vals.extend([d['gold_f1'] for d in mv_trajectory])
            PlotUtils.plot_trajectory_points(ax, mv_trajectory, metric_key, 's', 'red', 'MajorityVote',
                                        self.performance_data['iterations'])
        
        if elite_mv_trajectory:
            all_x_vals.extend([d[metric_key] for d in elite_mv_trajectory])
            all_y_vals.extend([d['gold_f1'] for d in elite_mv_trajectory])
            PlotUtils.plot_trajectory_points(ax, elite_mv_trajectory, metric_key, 'D', 'darkred', 
                                        'Elite MajorityVote', self.performance_data['iterations'], size=130)
        
        # Setup axes
        PlotUtils.setup_equal_axes(ax, all_x_vals, all_y_vals, equal_axes, metric_display_name)
        
        # Formatting
        ax.set_xlabel(metric_display_name, fontsize=12)
        ax.set_ylabel('Gold Standard F1 Score', fontsize=12)
        ax.set_title(f"Model Performance Evolution: {metric_display_name} vs Gold Standard - {self.config.benchmark.upper()}", 
            fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.text(0.02, 0.98, 'Color intensity: Lighter = Earlier iteration, Darker = Later iteration', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

class IterativeExperimentVisualizer:
    """Main coordinator class for iterative NER experiment visualization"""
    
    def __init__(self, config: ExperimentConfig, plot_config: Optional[PlotConfig] = None):
        self.config = config
        self.plot_config = plot_config or PlotConfig()

        self.data_loader = ExperimentDataLoader(config)
        self.iteration_data = self.data_loader.load_all_iteration_data()
        self.metrics_calc = MetricsCalculator(self.iteration_data)
        self.performance_data = self.metrics_calc.extract_performance_data()
        self.trajectory_calc = TrajectoryCalculator(self.performance_data)
        self.display_names = {m: _normalize_model_name(m) for m in self.performance_data['models']}

        palette = self.plot_config.colors
        unique_display = []
        for m in self.performance_data['models']:
            dn = self.display_names[m]
            if dn not in unique_display:
                unique_display.append(dn)
        self.model_colors = {dn: palette[i % len(palette)] for i, dn in enumerate(unique_display)}

        self.renderer = VisualizationRenderer(
            config, self.plot_config, self.performance_data,
            self.metrics_calc, self.trajectory_calc,
            model_colors=self.model_colors,
            display_names=self.display_names
        )

    
    def plot_performance_over_iterations(self, figsize: Optional[Tuple[int, int]] = None):
        """Plot performance evolution over iterations"""
        self.renderer.plot_performance_over_iterations(figsize)
    
    def plot_strict_span_vs_gold_evolution(self, equal_axes: bool = True, 
                                          figsize: Optional[Tuple[int, int]] = None):
        """Plot Strict span F1 vs Gold Standard F1 evolution"""
        self.renderer.plot_metric_evolution('strict_span_f1', 'Strict span F1', equal_axes, figsize)
    
    def plot_kappa_vs_gold_evolution(self, equal_axes: bool = True,
                                   figsize: Optional[Tuple[int, int]] = None):
        """Plot Cohen's Kappa vs Gold Standard F1 evolution"""
        self.renderer.plot_metric_evolution('cohens_kappa', "Cohen's Kappa", equal_axes, figsize)
    
    def plot_pairwise_agreements_over_iterations(self, figsize: Optional[Tuple[int, int]] = None):
        """Plot pairwise agreements over iterations with model weights"""
        if not self.iteration_data:
            print("No iteration data available for plotting")
            return

        metrics = [
            ("strict_span_f1", "Strict span F1"), 
            ("token_macro_f1", "Token macro F1"), 
            ("cohens_kappa", "Cohen's kappa"),
            ("model_weight", "Model Weight")
        ]
        
        figsize = figsize or (8, 12)
        all_models = set()
        per_model_metric_series = {m_key: [] for m_key, _ in metrics}
        avg_metric_series = {m_key: [] for m_key, _ in metrics}
        avg_metric_sds = {m_key: [] for m_key, _ in metrics}

        iterations_sorted = sorted(self.iteration_data.keys())
        for it in iterations_sorted:
            pdata = self.iteration_data[it].get('pairwise_agreements')
            if not pdata:
                continue

            # For agreement metrics (first 3)
            sums = defaultdict(lambda: {m: (0.0, 0) for m, _ in metrics[:-1]})

            for pair_key, rec in pdata.items():
                m1, m2 = rec.get("model1"), rec.get("model2")
                avg = rec.get("avg_agreement", {})

                if not m1 or not m2 or not avg:
                    continue

                all_models.update([m1, m2])

                for m_key, _ in metrics[:-1]:  # Process first 3 metrics
                    val = float(avg.get(m_key, 0.0))
                    for model in [m1, m2]:
                        s, c = sums[model][m_key]
                        sums[model][m_key] = (s + val, c + 1)

            # Calculate model weights based on strict span F1
            model_strict_f1s = {}
            for model in sums.keys():
                s, c = sums[model]["strict_span_f1"]
                model_strict_f1s[model] = s / c if c > 0 else 0.0

            # Calculate weights (normalized so sum = 1)
            total_strict_f1 = sum(model_strict_f1s.values())
            model_weights = {}
            if total_strict_f1 > 0:
                for model, strict_f1 in model_strict_f1s.items():
                    model_weights[model] = strict_f1 / total_strict_f1
            else:
                # Equal weights if all strict_f1 are 0
                num_models = len(model_strict_f1s)
                for model in model_strict_f1s.keys():
                    model_weights[model] = 1.0 / num_models if num_models > 0 else 0.0

            # Process iteration metrics
            iter_metric_vals = {m_key: [] for m_key, _ in metrics}

            # Add agreement metrics
            for model in sums.keys():
                for m_key, _ in metrics[:-1]:  # First 3 metrics
                    s, c = sums[model][m_key]
                    avg_val = s / c if c > 0 else 0.0
                    per_model_metric_series[m_key].append((model, it, avg_val))
                    iter_metric_vals[m_key].append(avg_val)

            # Add model weights
            for model, weight in model_weights.items():
                per_model_metric_series["model_weight"].append((model, it, weight))
                iter_metric_vals["model_weight"].append(weight)

            # Calculate averages and standard deviations for all metrics
            for m_key, _ in metrics:
                if iter_metric_vals[m_key]:
                    avg_val = float(np.mean(iter_metric_vals[m_key]))
                    sd_val = float(np.std(iter_metric_vals[m_key]))
                    avg_metric_series[m_key].append((it, avg_val))
                    avg_metric_sds[m_key].append((it, sd_val))
                else:
                    avg_metric_series[m_key].append((it, np.nan))
                    avg_metric_sds[m_key].append((it, np.nan))

        if not all_models:
            print("No pairwise agreement data found across iterations.")
            return

        # Create 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=False)
        models_sorted = sorted(list(all_models))
        linestyles = ['-', '--', '-.', ':']

        legend_handles = []
        for ax_idx, (m_key, m_label) in enumerate(metrics):
            ax = axes[ax_idx]            
            if ax_idx == 0:
                ax.set_title(f'{m_label} - {self.config.benchmark.upper()}', fontsize=14, fontweight='bold')
            else:
                ax.set_title(m_label, fontsize=14, fontweight='bold')
            
            # Set appropriate y-label based on metric type
            if m_key == "model_weight":
                ax.set_ylabel('Weight', fontsize=12)
            else:
                ax.set_ylabel('Average agreement', fontsize=12)
            
            ax.grid(True, alpha=0.3)

            # Plot per-model lines
            for i, model in enumerate(models_sorted):
                series = [(it, v) for (mdl, it, v) in per_model_metric_series[m_key] if mdl == model]
                disp = self.display_names.get(model, model)
                if not series:
                    continue
                series.sort(key=lambda x: x[0])
                xs, ys = zip(*series)

                line, = ax.plot(xs, ys, marker='o', linewidth=2, markersize=5,
                    color=self.model_colors.get(disp, 'black'),
                    linestyle=linestyles[i % len(linestyles)],
                    label=disp)    
                if ax_idx == 0:
                    legend_handles.append(line)

            # Plot average line with SD-dependent dot sizes (skip for model weight)
            if m_key != "model_weight":
                avg_series_sorted = sorted(avg_metric_series[m_key], key=lambda x: x[0])
                avg_xs = [it for it, _ in avg_series_sorted]
                avg_ys = [val for _, val in avg_series_sorted]

                sd_series_sorted = sorted(avg_metric_sds[m_key], key=lambda x: x[0])
                sd_vals = [val for _, val in sd_series_sorted]

                dot_sizes = [np.clip(2 * 1 / (sd + 1e-3), 10, 300) if not np.isnan(sd) else 50 for sd in sd_vals]

                ax.plot(avg_xs, avg_ys, linewidth=6, alpha=0.12, color='black', zorder=4)
                avg_line = ax.plot(avg_xs, avg_ys, linewidth=3.5, color='black', 
                                linestyle='--', alpha=0.95, zorder=5, label='Average')[0]
                ax.scatter(avg_xs, avg_ys, s=dot_sizes, c='black', marker='D',
                        alpha=0.9, edgecolors='white', linewidths=0.7, zorder=6)

                if ax_idx == 0:
                    legend_handles.append(avg_line)

            # Set x-axis for each subplot
            ax.set_xlim(min(iterations_sorted) - 0.1, max(iterations_sorted) + 0.1)
            ax.set_xticks(iterations_sorted)
            ax.set_xlabel('Iteration', fontsize=12)

        # Add legend at the top
        if legend_handles:
            fig.legend(handles=legend_handles, labels=[h.get_label() for h in legend_handles],
                    loc='upper center', bbox_to_anchor=(0.5, 1.02),
                    ncol=min(4, len(legend_handles)), frameon=True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    def plot_summary_statistics(self, figsize: Optional[Tuple[int, int]] = None):
        """Summary statistics and trends with dual y-axis including Elite MV"""
        if not self.performance_data['iteration_summaries']:
            print("No summary data available for plotting")
            return

        figsize = figsize or (20, 8)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        iterations = [s['iteration'] for s in self.performance_data['iteration_summaries']]
        mv_gold_f1s = [s['mv_performance'] for s in self.performance_data['iteration_summaries']]
        elite_mv_gold_f1s = [s['elite_mv_performance'] for s in self.performance_data['iteration_summaries']]
        best_gold_f1s = [s['best_model'][1]['gold_f1'] for s in self.performance_data['iteration_summaries']]

        # Compute variances
        var_pack = self.metrics_calc.compute_inter_model_variances(self.performance_data)
        strict_variances = var_pack['strict_span_f1_var']
        kappa_variances = var_pack['cohens_kappa_var']

        # Compute average strict span F1 and Cohen's kappa values
        avg_strict_span_f1s = []
        avg_cohens_kappas = []
        
        for iteration in iterations:
            pdata = self.iteration_data.get(iteration, {}).get('pairwise_agreements')
            if not pdata:
                avg_strict_span_f1s.append(0.0)
                avg_cohens_kappas.append(0.0)
                continue

            # Accumulate per-model averages across pairs
            sums = defaultdict(lambda: {'strict_span_f1': [0.0, 0], 'cohens_kappa': [0.0, 0]})
            for rec in pdata.values():
                m1, m2 = rec.get("model1"), rec.get("model2")
                avg = rec.get("avg_agreement", {})
                if not m1 or not m2 or not avg:
                    continue
                strict_v = float(avg.get("strict_span_f1", 0.0))
                kappa_v = float(avg.get("cohens_kappa", 0.0))
                for m in (m1, m2):
                    sums[m]['strict_span_f1'][0] += strict_v
                    sums[m]['strict_span_f1'][1] += 1
                    sums[m]['cohens_kappa'][0] += kappa_v
                    sums[m]['cohens_kappa'][1] += 1

            strict_avgs, kappa_avgs = [], []
            for m, agg in sums.items():
                s_sum, s_cnt = agg['strict_span_f1']
                k_sum, k_cnt = agg['cohens_kappa']
                if s_cnt > 0:
                    strict_avgs.append(s_sum / s_cnt)
                if k_cnt > 0:
                    kappa_avgs.append(k_sum / k_cnt)

            avg_strict_span_f1s.append(float(np.mean(strict_avgs)) if strict_avgs else 0.0)
            avg_cohens_kappas.append(float(np.mean(kappa_avgs)) if kappa_avgs else 0.0)

        # Plot 1: Performance values on left Y-axis
        line_mv = ax1.plot(iterations, mv_gold_f1s, marker='s', linewidth=2.5,
                        color='red', linestyle='-', label='MajorityVote Gold F1')
        line_elite_mv = ax1.plot(iterations, elite_mv_gold_f1s, marker='D', linewidth=2.5,
                                color='darkred', linestyle='-.', label='Elite MajorityVote Gold F1')
        line_best = ax1.plot(iterations, best_gold_f1s, marker='^', linewidth=2.5,
                            color='darkgreen', linestyle='-', label='Best Model Gold F1')
        line_strict = ax1.plot(iterations, avg_strict_span_f1s, marker='o', linewidth=2.5,
                            color='purple', linestyle='-', label='Average Strict span F1')
        line_kappa = ax1.plot(iterations, avg_cohens_kappas, marker='v', linewidth=2.5,
                            color='teal', linestyle='-', label="Average Cohen's kappa")

        ax1.set_title(f'Performance and Variance Analysis - {self.config.benchmark.upper()}', fontweight='bold')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Performance Values', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)

        # Twin axis for variances
        ax1_twin = ax1.twinx()
        line_var_strict = ax1_twin.plot(iterations, strict_variances, marker='P', linewidth=2,
                                        color='purple', linestyle='--', label='Strict span F1 Variance')
        line_var_kappa = ax1_twin.plot(iterations, kappa_variances, marker='X', linewidth=2,
                                    color='teal', linestyle='--', label="Cohen's kappa Variance")
        ax1_twin.set_ylabel('Inter-model Variance', color='black')
        ax1_twin.tick_params(axis='y', labelcolor='black')

        # Unified legend
        lines1 = line_best + line_mv + line_elite_mv + line_strict + line_kappa + line_var_strict + line_var_kappa
        labels1 = [l.get_label() for l in lines1]
        ax1.legend(lines1, labels1, loc='upper left')

    def print_performance_summary(self):
        """Print detailed performance summary including Elite MV"""
        print("\n" + "="*80)
        print("ITERATIVE EXPERIMENT PERFORMANCE SUMMARY")
        print("="*80)
        
        print(f"Dataset: {self.config.dataset_path}")
        print(f"Configuration: {self.config.num_groups} groups, size {self.config.group_size}")
        print(f"Starting group index: {self.config.starting_group_index}")
        print(f"Iterations analyzed: {len(self.performance_data['iterations'])}")
        print(f"Models tracked: {len(self.performance_data['models'])}")
        
        if not self.performance_data['iterations']:
            print("No iteration data available.")
            return
        
        print(f"\nModels: {', '.join(self.performance_data['models'])}")
        
        # Performance evolution summary
        print(f"\n{'PERFORMANCE EVOLUTION (vs Gold Standard)':<50}")
        print("-" * 95)
        print(f"{'Iteration':<10} {'Best Model':<15} {'Best F1':<10} {'Avg F1':<10} {'MV F1':<10} {'Elite MV F1':<12} {'Models':<8}")
        print("-" * 95)
        
        for summary in self.performance_data['iteration_summaries']:
            iteration = summary['iteration']
            best_name = summary['best_model'][0][:14]
            best_f1 = summary['best_model'][1]['gold_f1']
            avg_f1 = summary['avg_performance']
            mv_f1 = summary['mv_performance']
            elite_mv_f1 = summary['elite_mv_performance']
            model_count = summary['model_count']
            
            print(f"{iteration:<10} {best_name:<15} {best_f1:<10.3f} {avg_f1:<10.3f} "
                f"{mv_f1:<10.3f} {elite_mv_f1:<12.3f} {model_count:<8}")
        
        # Consensus comparison
        print(f"\n{'CONSENSUS METHODS COMPARISON':<50}")
        print("-" * 100)
        print(f"{'Iteration':<10} {'Best F1':<10} {'MV F1':<10} {'Elite MV F1':<12} {'MV vs Best':<12} {'Elite vs Best':<15} {'Elite vs MV':<12}")
        print("-" * 100)
        
        for summary in self.performance_data['iteration_summaries']:
            iteration = summary['iteration']
            best_f1 = summary['best_model'][1]['gold_f1']
            mv_f1 = summary['mv_performance']
            elite_mv_f1 = summary['elite_mv_performance']
            
            mv_diff = mv_f1 - best_f1 if mv_f1 > 0 else 0.0
            elite_diff = elite_mv_f1 - best_f1 if elite_mv_f1 > 0 else 0.0
            elite_mv_diff = elite_mv_f1 - mv_f1 if elite_mv_f1 > 0 and mv_f1 > 0 else 0.0
            
            print(f"{iteration:<10} {best_f1:<10.3f} {mv_f1:<10.3f} {elite_mv_f1:<12.3f} "
                f"{mv_diff:+<12.3f} {elite_diff:+<15.3f} {elite_mv_diff:+<12.3f}")
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        self.print_performance_summary()
        self.plot_performance_over_iterations()
        self.plot_strict_span_vs_gold_evolution(equal_axes=False)
        self.plot_kappa_vs_gold_evolution(equal_axes=False)
        self.plot_pairwise_agreements_over_iterations()
        self.plot_summary_statistics()


def _normalize_model_name(name: str) -> str:
    return name.split('/')[-1] if '/' in name else name

def create_visualization(
    benchmark: str,
    dataset_path: str, 
    num_groups: int = 20, 
    group_size: int = None,
    models: int = 8, 
    starting_group_index: int = 0, 
    max_iterations: int = 5,
    supervisor_model_name: str = "gpt-5-2025-08-07",
    max_common_instructions: int = 5,
    max_patterns: int = 10,
    model_specific_for_all: bool = False,
    max_model_specific_instructions: int = 3,
    limit_instruction_changes: bool = False,
    max_change_ratio: float = 0.2,
    drop_worst_annr: bool = False,
    plot_config: Optional[PlotConfig] = None
) -> IterativeExperimentVisualizer:
    """Convenience function to create all visualizations with supervisor parameters"""
    
    config = ExperimentConfig(
        benchmark=benchmark,
        dataset_path=dataset_path,
        num_groups=num_groups,
        group_size=group_size,
        models=models,
        starting_group_index=starting_group_index,
        max_iterations=max_iterations,
        supervisor_model_name=supervisor_model_name,
        supervisor_params={
            'max_common_instructions': max_common_instructions,
            'max_patterns': max_patterns,
            'model_specific_for_all': model_specific_for_all,
            'max_model_specific_instructions': max_model_specific_instructions,
            'limit_instruction_changes': limit_instruction_changes,
            'max_change_ratio': max_change_ratio,
            'drop_worst_annr': drop_worst_annr
        }
    )
    
    visualizer = IterativeExperimentVisualizer(config, plot_config)
    visualizer.generate_all_plots()
    return visualizer


# Example usage
if __name__ == "__main__":
    # Custom plot configuration
    custom_plot_config = PlotConfig(
        figsize=(12, 8),
        dpi=600
    )
    
    for benchmark in [
        # 'crossner_conll2003',
        # 'crossner_literature',
        # 'crossner_music',
        'crossner_politics',
        # 'crossner_science',
    ]:
        visualizer = create_visualization(
            benchmark=benchmark,
            dataset_path=f"datasets/crossner/{benchmark}_ner_dataset.pkl",
            num_groups=20,
            group_size=25,
            models=8,
            starting_group_index=0,
            max_iterations=6,
            max_common_instructions=3,
            max_patterns=5,
            model_specific_for_all=False,
            max_model_specific_instructions=2,
            limit_instruction_changes=True,
            max_change_ratio=0.1,
            drop_worst_annr=False,
            plot_config=custom_plot_config
        )
        
    for benchmark in [
        # 'crossner_conll2003',
        'crossner_literature',
        # 'crossner_music',
        # 'crossner_politics',
        # 'crossner_science',
    ]:
        visualizer = create_visualization(
            benchmark=benchmark,
            dataset_path=f"datasets/crossner/{benchmark}_ner_dataset.pkl",
            num_groups=20,
            group_size=25,
            models=8,
            starting_group_index=0,
            max_iterations=6,
            max_common_instructions=3,
            max_patterns=5,
            model_specific_for_all=False,
            max_model_specific_instructions=2,
            limit_instruction_changes=False,
            max_change_ratio=0.1,
            drop_worst_annr=False,
            plot_config=custom_plot_config
        )