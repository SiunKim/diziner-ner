"""
Best Performance Finder Module - Main functionality
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

from utils_best_perf_finder import (
    ExperimentConfig, PerformanceResult, TestPerformanceResult,
    BestPerformanceFinderBase, _normalize_model_name
)

class BestPerformanceFinder(BestPerformanceFinderBase):
    def __init__(self, benchmarks: List[str]):
        super().__init__(benchmarks)
        self.test_results = []  # For test performance results
    
    def search_configurations(self, **search_params):
        """Search for configurations matching the specified parameters (validation performance)"""
        available_configs = self.discover_available_configurations()
        
        # Auto-detect available options
        all_model_counts = set()
        all_supervisors = set()
        for benchmark_config in available_configs.values():
            all_model_counts.update(benchmark_config['model_counts'])
            all_supervisors.update(benchmark_config['supervisors'])
        
        defaults = {
            'models': list(all_model_counts) or [8],
            'supervisor_model_name': list(all_supervisors) or ["gpt-5-2025-08-07"]
        }
        
        models_list = search_params.get('models', defaults['models'])
        supervisors_list = search_params.get('supervisor_model_name', defaults['supervisor_model_name'])
        filter_criteria = {k: v for k, v in search_params.items() 
                        if k not in ['models', 'supervisor_model_name'] and v is not None}
        
        print(f"Auto-detected model counts: {sorted(all_model_counts)}")
        print(f"Auto-detected supervisors: {sorted(all_supervisors)}")
        if filter_criteria:
            print(f"Applied filters: {filter_criteria}")
        
        self.results = []
        filtered_out_count = 0
        
        for benchmark in self.benchmarks:
            for model_count in models_list:
                for supervisor in supervisors_list:
                    supervisor_safe_name = supervisor.replace(':', '_').replace('/', '_').replace(' ', '_').replace('.', '_')
                    base_dir = Path(f'experiment_results/{benchmark}/models{model_count}') / supervisor_safe_name
                    
                    if not base_dir.exists():
                        continue
                    
                    processed_count = 0
                    for folder_path in base_dir.iterdir():
                        if not folder_path.is_dir():
                            continue
                        
                        # Create base config
                        base_config = ExperimentConfig(
                            benchmark=benchmark, num_groups=20, group_size=25, models=model_count,
                            starting_group_index=0, max_iterations=6, supervisor_model_name=supervisor,
                            max_common_instructions=5, max_patterns=10, model_specific_for_all=False,
                            max_model_specific_instructions=3, limit_instruction_changes=False,
                            max_change_ratio=0.2, drop_worst_annr=False, supervised_by_gold_standard=False,
                            llm_family_config=None, skip_final_goal_update=False
                        )
                        
                        try:
                            config, iteration, group_index = self._extract_config_from_folder_name(folder_path.name, base_config)
                            
                            if not self._matches_filter_criteria(config, **filter_criteria):
                                filtered_out_count += 1
                                continue
                            
                            performance_data = self._extract_performance_from_folder(folder_path)
                            
                            if performance_data:
                                best_f1, best_name, mv_f1, strict_f1, individual_results, individual_strict = performance_data
                                
                                result = PerformanceResult(
                                    config=config, iteration=iteration, group_index=group_index,
                                    best_model_f1=best_f1, best_model_name=best_name,
                                    mv_f1=mv_f1, strict_span_f1_avg=strict_f1,
                                    individual_model_results=individual_results,
                                    individual_model_strict_f1=individual_strict,
                                    folder_path=str(folder_path)
                                )
                                self.results.append(result)
                                processed_count += 1
                        except Exception:
                            continue
                    
                    if processed_count > 0:
                        print(f"  Processed {processed_count} folders for {benchmark}/models{model_count}/{supervisor}")
        
        print(f"\nFound {len(self.results)} valid results (filtered out {filtered_out_count} results)")

    def search_test_configurations(self, name_of_search_condition: str, 
                             test_experiment_prompts_path: str = 'test_experiment_prompts_and_results',
                             final_test_prompts_path: str = 'final-test-prompts', 
                             **search_params):
        """Search for test configurations by scanning test_experiment_prompts_and_results directly"""
        print(f"\n{'='*80}")
        print("SEARCHING TEST PERFORMANCE RESULTS")
        print(f"{'='*80}")
        print(f"Test experiment prompts path: {test_experiment_prompts_path}")
        print(f"Final test prompts path: {final_test_prompts_path}")
        print(f"Search condition: {name_of_search_condition}")
        
        # Step 1: Scan test_experiment_prompts_and_results folder structure
        if name_of_search_condition == "None":
            # Scan all folders in test_experiment_prompts_path
            test_folders = self._scan_all_test_experiment_folders(test_experiment_prompts_path)
            print("Scanning ALL folders in test_experiment_prompts_path")
        else:
            # Original behavior - scan specific condition folder
            test_folders = self._scan_test_experiment_folders(test_experiment_prompts_path, name_of_search_condition)
        
        # Step 2: Group by clean condition and extract performance
        self.test_results = []
        find_by_gold_results = []
        find_by_agreement_results = []
        
        total_expected = 0
        total_found_json = 0
        total_missing_json = 0
        
        benchmark_stats = defaultdict(lambda: {'expected': 0, 'found_json': 0, 'missing_json': 0})
        
        for benchmark, folder_groups in test_folders.items():
            for clean_condition, folder_info_list in folder_groups.items():
                for folder_info in folder_info_list:
                    total_expected += 1
                    benchmark_stats[benchmark]['expected'] += 1
                    
                    # Map to final-test-prompts path
                    final_prompt_path = self._map_test_folder_to_final_prompts(
                        benchmark, folder_info, final_test_prompts_path
                    )
                    
                    if final_prompt_path and final_prompt_path.exists():
                        # Look for JSON files
                        json_files = list(final_prompt_path.glob('*_inference_results.json'))
                        
                        if json_files:
                            total_found_json += 1
                            benchmark_stats[benchmark]['found_json'] += 1
                            
                            # Extract performance from JSON
                            for json_file in json_files:
                                test_result = self._extract_test_performance_from_json(
                                    json_file, folder_info, benchmark
                                )
                                if test_result:
                                    self.test_results.append(test_result)
                                    
                                    # Group by find method
                                    if folder_info['find_by_suffix'] == '_fdbygst':
                                        find_by_gold_results.append(test_result)
                                    elif folder_info['find_by_suffix'] == '_fdbypma':
                                        find_by_agreement_results.append(test_result)
                                    elif folder_info['find_by_suffix'] == '_fdbygst_fdbypma':
                                        # Include in both categories
                                        find_by_gold_results.append(test_result)
                                        find_by_agreement_results.append(test_result)
                        else:
                            total_missing_json += 1
                            benchmark_stats[benchmark]['missing_json'] += 1
                            print(f"  Missing JSON: {benchmark} - {clean_condition} ({folder_info['find_by_suffix']})")
                    else:
                        total_missing_json += 1
                        benchmark_stats[benchmark]['missing_json'] += 1
                        print(f"  Missing path: {final_prompt_path}")
        
        # Print statistics
        print(f"\nTest Performance Search Summary:")
        print(f"  Total expected conditions: {total_expected}")
        print(f"  Found JSON files: {total_found_json}")
        print(f"  Missing JSON files: {total_missing_json}")
        
        print(f"\nBenchmark Statistics:")
        for benchmark in sorted(benchmark_stats.keys()):
            stats = benchmark_stats[benchmark]
            print(f"  {benchmark}:")
            print(f"    Expected: {stats['expected']}, Found JSON: {stats['found_json']}, Missing: {stats['missing_json']}")
        
        # Print results by find method
        self._print_test_performance_by_find_method(find_by_gold_results, find_by_agreement_results)
        
        return {
            'total_expected': total_expected,
            'total_found_json': total_found_json,
            'total_missing_json': total_missing_json,
            'benchmark_stats': dict(benchmark_stats),
            'find_by_gold_count': len(find_by_gold_results),
            'find_by_agreement_count': len(find_by_agreement_results)
        }


    def _scan_all_test_experiment_folders(self, test_experiment_prompts_path: str) -> Dict[str, Dict[str, List[Dict]]]:
        """Scan ALL folders in test_experiment_prompts_and_results folder structure"""
        base_path = Path(test_experiment_prompts_path)
        
        if not base_path.exists():
            print(f"Warning: Test experiment prompts path not found: {base_path}")
            return {}
        
        test_folders = defaultdict(lambda: defaultdict(list))
        
        # Scan all condition folders under base_path
        for condition_dir in base_path.iterdir():
            if not condition_dir.is_dir():
                continue
                
            print(f"Scanning condition folder: {condition_dir.name}")
            
            # Scan: condition_folder/benchmark/models{N}/supervisor/experiment_folder/prompts/
            for benchmark_dir in condition_dir.iterdir():
                if not benchmark_dir.is_dir():
                    continue
                    
                benchmark = benchmark_dir.name
                
                for model_dir in benchmark_dir.iterdir():
                    if not model_dir.is_dir() or not model_dir.name.startswith('models'):
                        continue
                    
                    model_count = int(model_dir.name.replace('models', ''))
                    
                    for supervisor_dir in model_dir.iterdir():
                        if not supervisor_dir.is_dir():
                            continue
                        
                        supervisor_name = supervisor_dir.name
                        
                        for experiment_dir in supervisor_dir.iterdir():
                            if not experiment_dir.is_dir():
                                continue
                            
                            # Extract clean condition from folder name
                            folder_name = experiment_dir.name
                            clean_condition, find_by_suffix = self._extract_clean_condition_from_folder(folder_name)
                            
                            # Create folder info with condition information
                            folder_info = {
                                'folder_path': experiment_dir,
                                'folder_name': folder_name,
                                'clean_condition': clean_condition,
                                'find_by_suffix': find_by_suffix,
                                'benchmark': benchmark,
                                'model_count': model_count,
                                'supervisor_name': supervisor_name,
                                'condition_folder': condition_dir.name  # Add condition folder info
                            }
                            
                            test_folders[benchmark][clean_condition].append(folder_info)
        
        # Print scan summary
        total_folders = sum(len(folders) for benchmark_folders in test_folders.values() 
                        for folders in benchmark_folders.values())
        print(f"Scanned {len(test_folders)} benchmarks with {total_folders} total experiment folders from ALL conditions")
        
        return test_folders


    def _scan_test_experiment_folders(self, test_experiment_prompts_path: str, name_of_search_condition: str) -> Dict[str, Dict[str, List[Dict]]]:
        """Scan test_experiment_prompts_and_results folder structure"""
        base_path = Path(test_experiment_prompts_path) / name_of_search_condition
        
        if not base_path.exists():
            print(f"Warning: Search condition folder not found: {base_path}")
            return {}
        
        test_folders = defaultdict(lambda: defaultdict(list))
        
        # Scan: benchmark/models{N}/supervisor/experiment_folder/prompts/
        for benchmark_dir in base_path.iterdir():
            if not benchmark_dir.is_dir():
                continue
                
            benchmark = benchmark_dir.name
            
            for model_dir in benchmark_dir.iterdir():
                if not model_dir.is_dir() or not model_dir.name.startswith('models'):
                    continue
                
                model_count = int(model_dir.name.replace('models', ''))
                
                for supervisor_dir in model_dir.iterdir():
                    if not supervisor_dir.is_dir():
                        continue
                    
                    supervisor_name = supervisor_dir.name
                    
                    for experiment_dir in supervisor_dir.iterdir():
                        if not experiment_dir.is_dir():
                            continue
                        
                        # Extract clean condition from folder name
                        folder_name = experiment_dir.name
                        clean_condition, find_by_suffix = self._extract_clean_condition_from_folder(folder_name)
                        
                        # Create folder info
                        folder_info = {
                            'folder_path': experiment_dir,
                            'folder_name': folder_name,
                            'clean_condition': clean_condition,
                            'find_by_suffix': find_by_suffix,
                            'benchmark': benchmark,
                            'model_count': model_count,
                            'supervisor_name': supervisor_name
                        }
                        
                        test_folders[benchmark][clean_condition].append(folder_info)
        
        # Print scan summary
        total_folders = sum(len(folders) for benchmark_folders in test_folders.values() 
                           for folders in benchmark_folders.values())
        print(f"Scanned {len(test_folders)} benchmarks with {total_folders} total experiment folders")
        
        return test_folders

    def _extract_clean_condition_from_folder(self, folder_name: str) -> tuple[str, str]:
        """Extract clean condition and find-by suffix from folder name"""
        # Check for find-by suffixes in order of specificity
        if folder_name.endswith('_fdbygst_fdbypma'):
            clean_condition = folder_name[:-len('_fdbygst_fdbypma')]
            find_by_suffix = '_fdbygst_fdbypma'
        elif folder_name.endswith('_fdbypma_fdbygst'):
            clean_condition = folder_name[:-len('_fdbypma_fdbygst')]
            find_by_suffix = '_fdbygst_fdbypma'  # Normalize to consistent order
        elif folder_name.endswith('_fdbygst'):
            clean_condition = folder_name[:-len('_fdbygst')]
            find_by_suffix = '_fdbygst'
        elif folder_name.endswith('_fdbypma'):
            clean_condition = folder_name[:-len('_fdbypma')]
            find_by_suffix = '_fdbypma'
        else:
            clean_condition = folder_name
            find_by_suffix = ''
        
        return clean_condition, find_by_suffix

    def _map_test_folder_to_final_prompts(self, benchmark: str, folder_info: Dict, final_test_prompts_path: str) -> Optional[Path]:
        """Map test experiment folder to final-test-prompts path structure"""
        clean_condition = folder_info['clean_condition']
        model_count = folder_info['model_count']
        supervisor_name = folder_info['supervisor_name']
        
        base_path = Path(final_test_prompts_path)
        
        # Check if it's iter0 condition
        if "_iter0_" in clean_condition or clean_condition.endswith("_iter0"):
            # For iter0: benchmark/clean_iter0_condition/prompts/
            clean_iter0_condition = self._get_clean_iter0_condition(clean_condition)
            return base_path / benchmark / clean_iter0_condition / "prompts"
        else:
            # For non-iter0: benchmark/model_count/supervisor/clean_condition/prompts/
            return base_path / benchmark / f'models{model_count}' / supervisor_name / clean_condition / "prompts"

    def _get_clean_iter0_condition(self, condition_name: str) -> str:
        """Extract clean iter0 condition (e.g., g20_s25_grp0_iter0)"""
        parts = condition_name.split('_')
        clean_parts = []
        for part in parts:
            clean_parts.append(part)
            if part == 'iter0':
                break
        return '_'.join(clean_parts)

    
    def _extract_iteration_from_condition(self, condition_name: str) -> int:
        """Extract iteration number from condition name"""
        import re
        match = re.search(r'_iter(\d+)(?:_|$)', condition_name)
        if match:
            return int(match.group(1))
        return 0

    def _extract_test_performance_from_json(self, json_file: Path, folder_info: Dict, benchmark: str) -> Optional[TestPerformanceResult]:
        """Extract test performance metrics from JSON file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract performance metrics
            performance_metrics = data.get('performance_metrics', {})
            micro_metrics = performance_metrics.get('micro', {})
            
            f1_score = micro_metrics.get('f1', 0.0)
            precision = micro_metrics.get('precision', 0.0)
            recall = micro_metrics.get('recall', 0.0)
            
            # Extract model information
            model_name = data.get('original_model_name', data.get('model_name', 'unknown'))
            
            # Extract test metadata
            test_samples_count = data.get('test_samples_count', 0)
            successful_samples = data.get('inference_metadata', {}).get('successful_samples', 0)
            
            # Extract iteration from condition_name
            iteration = self._extract_iteration_from_condition(folder_info['clean_condition'])
            
            return TestPerformanceResult(
                benchmark=benchmark,
                model_name=model_name,
                find_by_method=folder_info['find_by_suffix'],
                f1_score=f1_score,
                precision=precision,
                recall=recall,
                test_samples_count=test_samples_count,
                successful_samples=successful_samples,
                json_file_path=str(json_file),
                condition_name=folder_info['clean_condition'],
                model_count=folder_info['model_count'],
                supervisor_name=folder_info['supervisor_name'],
                iteration=iteration
            )
            
        except Exception as e:
            print(f"  Error reading {json_file}: {e}")
            return None

    def _print_test_performance_by_find_method(self, find_by_gold_results: List[TestPerformanceResult], 
                                find_by_agreement_results: List[TestPerformanceResult]):
        """Print test performance results grouped by find method with top 10 limit and deduplication"""
        
        print(f"\n{'='*80}")
        print("TEST PERFORMANCE RESULTS BY FIND METHOD")
        print(f"{'='*80}")
        
        # Find by Gold Standard Results
        print(f"\nFind by Gold Standard Results ({len(find_by_gold_results)} results):")
        print("-" * 60)
        if find_by_gold_results:
            # Group by benchmark
            gold_by_benchmark = defaultdict(list)
            for result in find_by_gold_results:
                # Filter out results with 'sonoma' in model name
                if 'sonoma' not in result.model_name.lower():
                    gold_by_benchmark[result.benchmark].append(result)
            
            for benchmark in sorted(gold_by_benchmark.keys()):
                results = sorted(gold_by_benchmark[benchmark], key=lambda x: x.f1_score, reverse=True)
                
                # Deduplicate results based on key attributes
                unique_results = []
                seen_combinations = set()
                
                for result in results:
                    # Create unique key based on F1 score, model, iteration, and supervisor
                    key = (result.f1_score, result.model_name, result.iteration, result.supervisor_name, result.condition_name)
                    if key not in seen_combinations:
                        seen_combinations.add(key)
                        unique_results.append(result)
                
                # Limit to top 20
                unique_results = unique_results[:20]
                
                if unique_results:
                    best_result = unique_results[0]
                    supervised_flag = " [SUPERVISED]" if 'goldstd' in best_result.condition_name else ""
                    print(f"  {benchmark}: F1={best_result.f1_score*100:.1f} (model: {best_result.model_name}, iter: {best_result.iteration}, supervisor: {best_result.supervisor_name}, samples: {best_result.successful_samples}/{best_result.test_samples_count}){supervised_flag}")
                    
                    if len(unique_results) > 1:
                        for i, result in enumerate(unique_results[1:], 2):
                            supervised_flag = " [SUPERVISED]" if 'goldstd' in result.condition_name else ""
                            print(f"    #{i}: F1={result.f1_score*100:.1f} (model: {result.model_name}, iter: {result.iteration}, supervisor: {result.supervisor_name}){supervised_flag}")
        else:
            print("  No results found")
        
        # Find by Pairwise Agreement Results
        print(f"\nFind by Pairwise Agreement Results ({len(find_by_agreement_results)} results):")
        print("-" * 60)
        if find_by_agreement_results:
            # Group by benchmark
            agreement_by_benchmark = defaultdict(list)
            for result in find_by_agreement_results:
                # Filter out results with 'sonoma' in model name
                if 'sonoma' not in result.model_name.lower():
                    agreement_by_benchmark[result.benchmark].append(result)
            
            for benchmark in sorted(agreement_by_benchmark.keys()):
                results = sorted(agreement_by_benchmark[benchmark], key=lambda x: x.f1_score, reverse=True)
                
                # Deduplicate results based on key attributes
                unique_results = []
                seen_combinations = set()
                
                for result in results:
                    # Create unique key based on F1 score, model, iteration, and supervisor
                    key = (result.f1_score, result.model_name, result.iteration, result.supervisor_name, result.condition_name)
                    if key not in seen_combinations:
                        seen_combinations.add(key)
                        unique_results.append(result)
                
                # Limit to top 20
                unique_results = unique_results[:20]
                
                if unique_results:
                    best_result = unique_results[0]
                    supervised_flag = " [SUPERVISED]" if 'goldstd' in best_result.condition_name else ""
                    print(f"  {benchmark}: F1={best_result.f1_score*100:.1f} (model: {best_result.model_name}, iter: {best_result.iteration}, supervisor: {best_result.supervisor_name}, samples: {best_result.successful_samples}/{best_result.test_samples_count}){supervised_flag}")
                    
                    if len(unique_results) > 1:
                        for i, result in enumerate(unique_results[1:], 2):
                            supervised_flag = " [SUPERVISED]" if 'goldstd' in result.condition_name else ""
                            print(f"    #{i}: F1={result.f1_score*100:.1f} (model: {result.model_name}, iter: {result.iteration}, supervisor: {result.supervisor_name}){supervised_flag}")
        else:
            print("  No results found")

def find_best_performance(benchmarks: List[str], 
                          metric: str = 'best_model_f1',
                          top_k: int = 10,
                          plot_graphs: bool = True,
                          interactive_plots: bool = True,
                          save_best_config_and_prompt: bool = False,
                          find_best_test_performance: bool = False,
                          name_of_search_condition: str = None,
                          test_experiment_prompts_path: str = 'test_experiment_prompts_and_results',
                          final_test_prompts_path: str = 'final-test-prompts',
                          **search_params) -> BestPerformanceFinder:
    """Find best performance with auto-detection and optional test performance analysis"""
    
    # Check if name_of_search_condition is provided when needed
    if (save_best_config_and_prompt or find_best_test_performance) and name_of_search_condition is None:
        raise ValueError("name_of_search_condition must be provided when save_best_config_and_prompt=True or find_best_test_performance=True")
    
    finder = BestPerformanceFinder(benchmarks)
    
    if find_best_test_performance:
        # Test performance mode
        print("=== TEST PERFORMANCE ANALYSIS MODE ===")
        test_stats = finder.search_test_configurations(
            name_of_search_condition, test_experiment_prompts_path, final_test_prompts_path, **search_params
        )
        
        # Print summary
        print(f"\nTest Performance Analysis Summary:")
        print(f"  Search condition: {name_of_search_condition}")
        print(f"  Total conditions expected: {test_stats['total_expected']}")
        print(f"  JSON files found: {test_stats['total_found_json']}")
        print(f"  Results by gold standard: {test_stats['find_by_gold_count']}")
        print(f"  Results by agreement: {test_stats['find_by_agreement_count']}")
        
    else:
        # Validation performance mode (original functionality)
        print("=== VALIDATION PERFORMANCE ANALYSIS MODE ===")
        finder.print_available_configurations()
        if benchmarks:
            finder.debug_folder_structure(benchmarks[0])
        
        finder.search_configurations(**search_params)
        finder.print_benchmark_top_results(metric=metric, top_k=top_k)
        finder.print_benchmark_summary(metric=metric)
        
        # Save best configs and prompts if requested
        if save_best_config_and_prompt:
            # Prepare search parameters for saving
            all_search_params = {
                'benchmarks': benchmarks,
                'metric': metric,
                'top_k': top_k,
                'plot_graphs': plot_graphs,
                'interactive_plots': interactive_plots,
                'save_best_config_and_prompt': save_best_config_and_prompt,
                'find_best_test_performance': find_best_test_performance,
                'name_of_search_condition': name_of_search_condition,
                'test_experiment_prompts_path': test_experiment_prompts_path,
                'final_test_prompts_path': final_test_prompts_path,
                **search_params
            }
            
            # Save search query first
            finder.save_search_query(name_of_search_condition, all_search_params)
            
            # Save best configs and prompts
            finder.save_best_configs_and_prompts(top_k=top_k, name_of_search_condition=name_of_search_condition)
        
        # Import and use plotting functions if requested
        if interactive_plots:
            try:
                from utils_best_finder_plotting import (
                    plot_interactive_performance_by_benchmark,
                    save_individual_model_scatter_plots,
                    save_combined_individual_model_scatter_plots,
                    analyze_correlation_by_benchmark,
                    analyze_model_level_correlation
                )
                # Correlation analysis - get benchmark correlations
                analyze_correlation_by_benchmark(finder)
                benchmark_correlations = analyze_model_level_correlation(finder)
                
                # Save combined plots with correlation info
                save_combined_individual_model_scatter_plots(
                    finder, 
                    save_dir='plots',
                    benchmark_correlations=benchmark_correlations,
                    print_improvement_stats=True)
            except ImportError as e:
                print(f"Warning: utils_best_finder_plotting not available. Error: {e}")
            except Exception as e:
                print(f"ERROR during plotting: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
    
    return finder



if __name__ == "__main__":
    crossner_benchmarks = [
        'crossner_ai',
        'crossner_literature', 
        'crossner_music',
        'crossner_politics',
        'crossner_science', 
    ]
     
    other_benchmarks = [
        'mitner_movie',
        'mitner_restaurant',
        'Broad Twitter', 
        'crossner_conll2003',
        'FabNER',
        'MultiNERD',
        'ACE05', 
        'anatem',
        'bc2gm',
        'bc4chemd',
        'bc5cdr',
        'GENIA',
        'OntoNotes'
    ]
    
    all_benchmarks = crossner_benchmarks + other_benchmarks
    print(f"Numbe of benchmarks: {len(all_benchmarks)}")
    
    print("Final Performance Analysis with Config Dictionary and Filtering")
    finder = find_best_performance(
        benchmarks=all_benchmarks,
        interactive_plots=True,
        metric='best_model_f1',
        top_k=3,

        find_best_test_performance=False,  # Enable test performance mode
        final_test_prompts_path="final-test-prompts_0920",  # Specify custom path
        test_experiment_prompts_path="test_experiment_prompts_and_results_0920",

        save_best_config_and_prompt=False,  # Enable prompt saving

        # name_of_search_condition="None",  # Add search condition name
        
        # name_of_search_condition="main_result_zeroshot",  # Add search condition name
        # group_size=[25], #main setting
        # models=[8], #main setting
        # supervised_by_gold_standard=[False], #main setting
        # supervisor_model_name=['gpt-5-mini-2025-08-07'], #main setting
        # max_common_instructions=list(range(1,21)), #all settings
        # max_patterns=list(range(1,21)), #all settings
        # max_model_specific_instructions=list(range(1,11)), #all settings
        # limit_instruction_changes=[True, False], #all settings
        # max_change_ratio=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], #all settings
        # drop_worst_annr=[True, False], #all settings
        # skip_final_goal_update=[False], #main setting
        # llm_family_config=[None], #main setting
        
        # name_of_search_condition="main_result_supervised",  # Add search condition name
        # group_size=[25], #main setting
        # models=[8], #main setting
        # supervised_by_gold_standard=[True], #<<<<< supervised
        # supervisor_model_name=['gpt-5-mini-2025-08-07'], #main setting
        # max_common_instructions=list(range(1,21)), #all settings
        # max_patterns=list(range(1,21)), #all settings
        # max_model_specific_instructions=list(range(1,11)), #all settings
        # limit_instruction_changes=[True, False], #all settings
        # max_change_ratio=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], #all settings
        # drop_worst_annr=[True, False], #all settings
        # skip_final_goal_update=[False], #main setting
        # llm_family_config=[None], #main setting
        
        name_of_search_condition="main_setting_for_comparison",  # Add search condition name
        group_size=[25], #main setting
        models=[8], #main setting
        supervised_by_gold_standard=[False], #main setting
        supervisor_model_name=['gpt-5-mini-2025-08-07'], #main setting
        max_common_instructions=[3], #main setting
        max_patterns=[5], #main setting
        max_model_specific_instructions=[2], #main setting
        limit_instruction_changes=[True], #main setting
        max_change_ratio=[0.1], #main setting
        drop_worst_annr=[False], #main setting
        skip_final_goal_update=[False], #main setting
        llm_family_config=[None], #main setting
        
        # name_of_search_condition="gpt5_supervisor",  # Add search condition name
        # group_size=[25], #main setting
        # models=[8], #main setting
        # supervised_by_gold_standard=[False], #main setting
        # supervisor_model_name=['gpt-5-2025-08-07'], #<<<<< gpt5
        # max_common_instructions=[3], #main setting
        # max_patterns=[5], #main setting
        # max_model_specific_instructions=[2], #main setting
        # limit_instruction_changes=[True], #main setting
        # max_change_ratio=[0.1], #main setting
        # drop_worst_annr=[False, True], #main setting
        # skip_final_goal_update=[False], #main setting
        # llm_family_config=[None], #main setting

        # name_of_search_condition="worst_model_drop",  # Add search condition name
        # group_size=[25], #main setting
        # models=[8], #main setting
        # supervised_by_gold_standard=[False], #main setting
        # supervisor_model_name=['gpt-5-mini-2025-08-07'], #main setting
        # max_common_instructions=[3], #main setting
        # max_patterns=[5], #main setting
        # max_model_specific_instructions=[2], #main setting
        # limit_instruction_changes=[True], #main setting
        # max_change_ratio=[0.1], #main setting
        # drop_worst_annr=[True], #<<<<<< worst model drop
        # skip_final_goal_update=[False], #main setting
        # llm_family_config=[None], #main setting

        # name_of_search_condition="diff_annr_number_4",  # Add search condition name
        # group_size=[25], #main setting
        # models=[4], #<<< annotator model number 4
        # supervised_by_gold_standard=[False], #main setting
        # supervisor_model_name=['gpt-5-mini-2025-08-07'], #main setting
        # max_common_instructions=[3], #main setting
        # max_patterns=[5], #main setting
        # max_model_specific_instructions=[2], #main setting
        # limit_instruction_changes=[True], #main setting
        # max_change_ratio=[0.1], #main setting
        # drop_worst_annr=[False], #main setting
        # skip_final_goal_update=[False], #main setting
        # llm_family_config=[None], #main settin
        
        # name_of_search_condition="diff_annrnumber_12",  # Add search condition name
        # group_size=[25], #main setting
        # models=[12], #<<< annotator model number 12
        # supervised_by_gold_standard=[False], #main setting
        # supervisor_model_name=['gpt-5-mini-2025-08-07'], #main setting
        # max_common_instructions=[3], #main setting
        # max_patterns=[5], #main setting
        # max_model_specific_instructions=[2], #main setting
        # limit_instruction_changes=[True], #main setting
        # max_change_ratio=[0.1], #main setting
        # drop_worst_annr=[False], #main setting
        # skip_final_goal_update=[False], #main setting
        # llm_family_config=[None], #main settin
        
        # name_of_search_condition="diff_annrnumber_16",  # Add search condition name
        # group_size=[25], #main setting
        # models=[16], #<<< annotator model number 16
        # supervised_by_gold_standard=[False], #main setting
        # supervisor_model_name=['gpt-5-mini-2025-08-07'], #main setting
        # max_common_instructions=[3], #main setting
        # max_patterns=[5], #main setting
        # max_model_specific_instructions=[2], #main setting
        # limit_instruction_changes=[True], #main setting
        # max_change_ratio=[0.1], #main setting
        # drop_worst_annr=[False], #main setting
        # skip_final_goal_update=[False], #main setting
        # llm_family_config=[None], #main settin

        # name_of_search_condition="diff_grouopsize_15",  # Add search condition name
        # group_size=[15], #<<< groupsize 15
        # models=[8], #main setting
        # supervised_by_gold_standard=[False], #main setting
        # supervisor_model_name=['gpt-5-mini-2025-08-07'], #main setting
        # max_common_instructions=[3], #main setting
        # max_patterns=[5], #main setting
        # max_model_specific_instructions=[2], #main setting
        # limit_instruction_changes=[True], #main setting
        # max_change_ratio=[0.1], #main setting
        # drop_worst_annr=[False], #main setting
        # skip_final_goal_update=[False], #main setting
        # llm_family_config=[None], #main setting
        
        # name_of_search_condition="diff_grouopsize_50",  # Add search condition name
        # group_size=[50], #<<< groupsize 50
        # models=[8], #main setting
        # supervised_by_gold_standard=[False], #main setting
        # supervisor_model_name=['gpt-5-mini-2025-08-07'], #main setting
        # max_common_instructions=[3], #main setting
        # max_patterns=[5], #main setting
        # max_model_specific_instructions=[2], #main setting
        # limit_instruction_changes=[True], #main setting
        # max_change_ratio=[0.1], #main setting
        # drop_worst_annr=[False], #main setting
        # skip_final_goal_update=[False], #main setting
        # llm_family_config=[None], #main setting
        
        # name_of_search_condition="diff_grouopsize_100",  # Add search condition name
        # group_size=[100], #<<< groupsize 50
        # models=[8], #main setting
        # supervised_by_gold_standard=[False], #main setting
        # supervisor_model_name=['gpt-5-mini-2025-08-07'], #main setting
        # max_common_instructions=[3], #main setting
        # max_patterns=[5], #main setting
        # max_model_specific_instructions=[2], #main setting
        # limit_instruction_changes=[True], #main setting
        # max_change_ratio=[0.1], #main setting
        # drop_worst_annr=[False], #main setting
        # skip_final_goal_update=[False], #main setting
        # llm_family_config=[None], #main setting

        # name_of_search_condition="llm_family_llama",  # Add search condition name
        # group_size=[25], #main setting
        # models=[8], #main setting
        # supervised_by_gold_standard=[False], #main setting
        # supervisor_model_name=['gpt-5-mini-2025-08-07'], #main setting
        # max_common_instructions=[3], #main setting
        # max_patterns=[5], #main setting
        # max_model_specific_instructions=[2], #main setting
        # limit_instruction_changes=[True], #main setting
        # max_change_ratio=[0.1], #main setting
        # drop_worst_annr=[False], #main setting
        # skip_final_goal_update=[False], #main setting
        # llm_family_config=['llama'], #<<<< llama family
        
        # name_of_search_condition="llm_family_qwen",  # Add search condition name
        # group_size=[25], #main setting
        # models=[8], #main setting
        # supervised_by_gold_standard=[False], #main setting
        # supervisor_model_name=['gpt-5-mini-2025-08-07'], #main setting
        # max_common_instructions=[3], #main setting
        # max_patterns=[5], #main setting
        # max_model_specific_instructions=[2], #main setting
        # limit_instruction_changes=[True], #main setting
        # max_change_ratio=[0.1], #main setting
        # drop_worst_annr=[False], #main setting
        # skip_final_goal_update=[False], #main setting
        # llm_family_config=['qwen'], #<<<< qwen family


        # group_size=[25], #main setting
        # # group_size=[15, 25, 50, 100], #all settings
        
        # models=[8], #main setting
        # # models=[4, 8, 12, 16], #all settings
        
        # supervised_by_gold_standard=[False], #main setting
        # # supervised_by_gold_standard=[True, False], #all settings
        
        # supervisor_model_name=['gpt-5-mini-2025-08-07'], #main setting
        # # supervisor_model_name=['gpt-5-mini-2025-08-07', 'gpt-5-2025-08-07'], #all settings
        
        # max_common_instructions=[3], #main setting
        # # max_common_instructions=list(range(1,21)), #all settings
        
        # max_patterns=[5], #main setting
        # # max_patterns=list(range(1,21)), #all settings
        
        # max_model_specific_instructions=[2], #main setting
        # # max_model_specific_instructions=list(range(1,11)), #all settings
        
        # limit_instruction_changes=[True], #main setting
        # # limit_instruction_changes=[True, False], #all settings
        
        # max_change_ratio=[0.1], #main setting
        # # max_change_ratio=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], #all settings
        
        # drop_worst_annr=[False], #main setting
        # # drop_worst_annr=[True, False], #all settings
        
        # skip_final_goal_update=[False], #main setting
        # # skip_final_goal_update=[True, False], #all settings
        
        # llm_family_config=[None], #main setting
        # # llm_family_config=[None, 'qwen', 'llama'], #all settings
    )
    print("Performance analysis completed.")