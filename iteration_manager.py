"""
Iteration Manager for Annotation-Supervisor Cycles
전체 반복 프로세스를 관리하는 상위 레벨 모듈
"""
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import from existing modules
from experiments_250818 import (
    run_iterative_annotation_supervisor_cycle,
    load_config,
    get_iteration_paths,
    check_convergence
)

# Import supervisor functionality
try:
    from supervisor_implementation import (
        run_supervisor_analysis,
        estimate_analysis_cost
    )
    SUPERVISOR_AVAILABLE = True
except ImportError:
    print("Warning: Supervisor module not available")
    SUPERVISOR_AVAILABLE = False


class IterationManager:
    """Manages complete annotation-supervisor iteration cycles"""
    
    def __init__(self, 
                 dataset_path: str,
                 models: List[str],
                 ner_scheme: Dict[str, Any],
                 config_path: str = 'experiment_settings/conllpp_default_config.json',
                 supervisor_model: str = "gpt-4o-2024-11-20",
                 base_output_dir: str = "iteration_experiments"):
        """
        Initialize Iteration Manager
        
        Args:
            dataset_path: Path to dataset
            models: List of models to test
            ner_scheme: NER scheme definition
            config_path: Path to experiment configuration
            supervisor_model: Model to use for supervision
            base_output_dir: Base directory for all iteration experiments
        """
        self.dataset_path = dataset_path
        self.models = models
        self.ner_scheme = ner_scheme
        self.config_path = config_path
        self.supervisor_model = supervisor_model
        self.base_output_dir = Path(base_output_dir)
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Create output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Iteration Manager initialized:")
        print(f"  Dataset: {dataset_path}")
        print(f"  Models: {models}")
        print(f"  Supervisor: {supervisor_model}")
        print(f"  Output: {base_output_dir}")
    
    def run_complete_cycle(self,
                          max_iterations: int = 3,
                          convergence_threshold: float = 0.05,
                          group_index: int = 0,
                          estimate_cost_only: bool = False,
                          interactive_confirm: bool = True,
                          **experiment_kwargs) -> Dict[str, Any]:
        """
        Run complete iteration cycle with cost estimation and confirmation
        
        Args:
            max_iterations: Maximum number of iterations
            convergence_threshold: F1 improvement threshold for convergence
            group_index: Group index for lexical diversity
            estimate_cost_only: Only estimate costs without running
            interactive_confirm: Ask for confirmation before running
            **experiment_kwargs: Additional experiment parameters
            
        Returns:
            Complete results dictionary
        """
        experiment_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cycle_output_dir = self.base_output_dir / experiment_id
        
        print(f"\n{'='*100}")
        print("STARTING COMPLETE ANNOTATION-SUPERVISOR CYCLE")
        print(f"{'='*100}")
        print(f"Experiment ID: {experiment_id}")
        print(f"Max iterations: {max_iterations}")
        print(f"Models: {self.models}")
        print(f"Group index: {group_index}")
        
        # Estimate costs
        if SUPERVISOR_AVAILABLE:
            cost_estimates = self._estimate_cycle_costs(
                max_iterations, group_index, **experiment_kwargs
            )
            
            print(f"\nCost Estimates:")
            print(f"  Per iteration (annotation): ~${cost_estimates['annotation_cost_per_iter']:.2f}")
            print(f"  Per iteration (supervisor): ~${cost_estimates['supervisor_cost_per_iter']:.2f}")
            print(f"  Total estimated cost: ~${cost_estimates['total_estimated_cost']:.2f}")
            
            if estimate_cost_only:
                return cost_estimates
        
        # Interactive confirmation
        if interactive_confirm:
            response = input(f"\nProceed with the complete cycle? (y/n): ").lower()
            if response != 'y':
                print("Cycle cancelled by user.")
                return {"cancelled": True}
        
        # Create cycle directory
        cycle_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cycle configuration
        cycle_config = {
            'experiment_id': experiment_id,
            'dataset_path': self.dataset_path,
            'models': self.models,
            'ner_scheme': self.ner_scheme,
            'supervisor_model': self.supervisor_model,
            'parameters': {
                'max_iterations': max_iterations,
                'convergence_threshold': convergence_threshold,
                'group_index': group_index,
                **experiment_kwargs
            },
            'start_timestamp': datetime.now().isoformat()
        }
        
        with open(cycle_output_dir / 'cycle_config.json', 'w', encoding='utf-8') as f:
            json.dump(cycle_config, f, indent=2, ensure_ascii=False)
        
        # Run the iterative cycle
        try:
            iteration_results = run_iterative_annotation_supervisor_cycle(
                dataset_path=self.dataset_path,
                models=self.models,
                ner_scheme=self.ner_scheme,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                supervisor_model_name=self.supervisor_model,
                group_index=group_index,
                **experiment_kwargs
            )
            
            # Analyze results across iterations
            cycle_analysis = self._analyze_cycle_results(iteration_results)
            
            # Create comprehensive results
            complete_results = {
                'cycle_config': cycle_config,
                'iteration_results': iteration_results,
                'cycle_analysis': cycle_analysis,
                'completion_timestamp': datetime.now().isoformat(),
                'output_directory': str(cycle_output_dir)
            }
            
            # Save complete results
            with open(cycle_output_dir / 'complete_cycle_results.json', 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False, default=str)
            
            # Generate summary report
            self._generate_cycle_report(complete_results, cycle_output_dir)
            
            print(f"\n{'='*100}")
            print("COMPLETE CYCLE FINISHED")
            print(f"{'='*100}")
            print(f"Results saved to: {cycle_output_dir}")
            
            return complete_results
            
        except Exception as e:
            error_info = {
                'cycle_config': cycle_config,
                'error': str(e),
                'error_timestamp': datetime.now().isoformat()
            }
            
            with open(cycle_output_dir / 'cycle_error.json', 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
            
            print(f"Cycle failed: {e}")
            return error_info
    
    def _estimate_cycle_costs(self, max_iterations: int, group_index: int, **kwargs) -> Dict[str, float]:
        """Estimate costs for complete cycle"""
        if not SUPERVISOR_AVAILABLE:
            return {
                'annotation_cost_per_iter': 0.0,
                'supervisor_cost_per_iter': 0.0,
                'total_estimated_cost': 0.0,
                'note': 'Supervisor module not available for cost estimation'
            }
        
        try:
            # Get sample paths for cost estimation
            sample_paths = get_iteration_paths(
                self.dataset_path,
                kwargs.get('num_groups', 20),
                kwargs.get('group_size', None),
                0,  # Use iteration 0 for estimation
                group_index
            )
            
            # Estimate supervisor cost (rough estimation)
            # This would need actual disagreement doc and error analysis for precise estimation
            supervisor_cost_estimate = 15.0  # Rough estimate per iteration
            
            # Annotation cost is typically much lower (mainly LLM inference)
            annotation_cost_estimate = 2.0  # Rough estimate per iteration
            
            total_cost = (supervisor_cost_estimate + annotation_cost_estimate) * max_iterations
            
            return {
                'annotation_cost_per_iter': annotation_cost_estimate,
                'supervisor_cost_per_iter': supervisor_cost_estimate,
                'total_estimated_cost': total_cost,
                'max_iterations': max_iterations,
                'estimation_method': 'rough_approximation'
            }
            
        except Exception as e:
            print(f"Cost estimation failed: {e}")
            return {
                'annotation_cost_per_iter': 0.0,
                'supervisor_cost_per_iter': 0.0,
                'total_estimated_cost': 0.0,
                'error': str(e)
            }
    
    def _analyze_cycle_results(self, iteration_results: Dict[int, Dict]) -> Dict[str, Any]:
        """Analyze results across all iterations"""
        analysis = {
            'total_iterations': len(iteration_results),
            'convergence_achieved': False,
            'convergence_iteration': None,
            'performance_progression': {},
            'instruction_evolution': {},
            'summary_statistics': {}
        }
        
        # Track F1 scores across iterations
        f1_progression = {}
        for iteration, results in iteration_results.items():
            annotation_results = results.get('annotation_results')
            if annotation_results and annotation_results.get('experiment_results'):
                model_results = annotation_results['experiment_results'].get('results_by_model', {})
                
                for model_name, model_result in model_results.items():
                    if model_name not in f1_progression:
                        f1_progression[model_name] = []
                    
                    f1_score = model_result.get('avg_metrics', {}).get('f1', 0.0)
                    f1_progression[model_name].append({
                        'iteration': iteration,
                        'f1_score': f1_score
                    })
        
        analysis['performance_progression'] = f1_progression
        
        # Check convergence
        if len(iteration_results) > 1:
            last_iteration = max(iteration_results.keys())
            convergence_info = iteration_results[last_iteration].get('convergence_analysis')
            if convergence_info:
                analysis['convergence_achieved'] = convergence_info.get('converged', False)
                if analysis['convergence_achieved']:
                    analysis['convergence_iteration'] = last_iteration
        
        # Calculate summary statistics
        if f1_progression:
            model_improvements = {}
            for model_name, scores in f1_progression.items():
                if len(scores) >= 2:
                    initial_f1 = scores[0]['f1_score']
                    final_f1 = scores[-1]['f1_score']
                    improvement = final_f1 - initial_f1
                    model_improvements[model_name] = {
                        'initial_f1': initial_f1,
                        'final_f1': final_f1,
                        'total_improvement': improvement,
                        'relative_improvement': improvement / initial_f1 if initial_f1 > 0 else 0.0
                    }
            
            analysis['summary_statistics'] = {
                'model_improvements': model_improvements,
                'avg_improvement': sum(imp['total_improvement'] for imp in model_improvements.values()) / len(model_improvements) if model_improvements else 0.0,
                'best_performing_model': max(model_improvements.keys(), key=lambda k: model_improvements[k]['final_f1']) if model_improvements else None
            }
        
        return analysis
    
    def _generate_cycle_report(self, complete_results: Dict[str, Any], output_dir: Path):
        """Generate human-readable summary report"""
        report_lines = []
        
        # Header
        report_lines.extend([
            "ANNOTATION-SUPERVISOR CYCLE REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Experiment ID: {complete_results['cycle_config']['experiment_id']}",
            ""
        ])
        
        # Configuration
        config = complete_results['cycle_config']
        report_lines.extend([
            "CONFIGURATION:",
            f"  Dataset: {config['dataset_path']}",
            f"  Models: {', '.join(config['models'])}",
            f"  Supervisor: {config['supervisor_model']}",
            f"  Max iterations: {config['parameters']['max_iterations']}",
            f"  Convergence threshold: {config['parameters']['convergence_threshold']}",
            ""
        ])
        
        # Results summary
        analysis = complete_results['cycle_analysis']
        report_lines.extend([
            "RESULTS SUMMARY:",
            f"  Total iterations completed: {analysis['total_iterations']}",
            f"  Convergence achieved: {analysis['convergence_achieved']}",
        ])
        
        if analysis['convergence_achieved']:
            report_lines.append(f"  Convergence at iteration: {analysis['convergence_iteration']}")
        
        report_lines.append("")
        
        # Performance progression
        if analysis['performance_progression']:
            report_lines.extend([
                "PERFORMANCE PROGRESSION (F1 Scores):",
            ])
            
            for model_name, progression in analysis['performance_progression'].items():
                report_lines.append(f"  {model_name}:")
                for point in progression:
                    report_lines.append(f"    Iteration {point['iteration']}: {point['f1_score']:.3f}")
                report_lines.append("")
        
        # Summary statistics
        if analysis.get('summary_statistics', {}).get('model_improvements'):
            report_lines.extend([
                "IMPROVEMENT SUMMARY:",
            ])
            
            improvements = analysis['summary_statistics']['model_improvements']
            for model_name, stats in improvements.items():
                report_lines.extend([
                    f"  {model_name}:",
                    f"    Initial F1: {stats['initial_f1']:.3f}",
                    f"    Final F1: {stats['final_f1']:.3f}",
                    f"    Improvement: {stats['total_improvement']:+.3f} ({stats['relative_improvement']:+.1%})",
                    ""
                ])
            
            avg_improvement = analysis['summary_statistics']['avg_improvement']
            best_model = analysis['summary_statistics']['best_performing_model']
            report_lines.extend([
                f"  Average improvement: {avg_improvement:+.3f}",
                f"  Best performing model: {best_model}",
                ""
            ])
        
        # Save report
        report_text = "\n".join(report_lines)
        with open(output_dir / 'cycle_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\nCycle report saved to: {output_dir / 'cycle_report.txt'}")
        
        # Print summary to console
        print("\nCYCLE SUMMARY:")
        if analysis.get('summary_statistics', {}).get('model_improvements'):
            for model_name, stats in analysis['summary_statistics']['model_improvements'].items():
                print(f"  {model_name}: {stats['initial_f1']:.3f} → {stats['final_f1']:.3f} ({stats['total_improvement']:+.3f})")


def main_iteration_experiment(
    dataset_path: str = 'datasets/conllpp/conllpp_ner_dataset.pkl',
    models: List[str] = None,
    max_iterations: int = 3,
    group_index: int = 0,
    estimate_only: bool = False,
    config_path: str = 'experiment_settings/conllpp_default_config.json',
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for running iteration experiments
    
    Args:
        dataset_path: Path to dataset
        models: List of models (uses config default if None)
        max_iterations: Maximum iterations
        group_index: Group index for diversity
        estimate_only: Only estimate costs
        config_path: Configuration file path
        **kwargs: Additional parameters
        
    Returns:
        Experiment results
    """
    # Load config for defaults
    config = load_config(config_path)
    
    if models is None:
        models = config['models']
    
    # Initialize manager
    manager = IterationManager(
        dataset_path=dataset_path,
        models=models,
        ner_scheme=config['ner_scheme'],
        config_path=config_path
    )
    
    # Run complete cycle
    return manager.run_complete_cycle(
        max_iterations=max_iterations,
        group_index=group_index,
        estimate_cost_only=estimate_only,
        final_task_goal=config['final_task_goal'],
        **{**config['experiment'], **config['analysis'], **kwargs}
    )


if __name__ == "__main__":
    # Example usage
    
    # Simple run with defaults
    results = main_iteration_experiment()
    
    # Custom configuration
    # results = main_iteration_experiment(
    #     models=['llama3.1:8b', 'phi4:14b'],
    #     max_iterations=5,
    #     group_index=0,
    #     estimate_only=False
    # )
    
    # Cost estimation only
    # cost_estimates = main_iteration_experiment(
    #     estimate_only=True,
    #     max_iterations=3
    # )