"""
Best Performance Finder Utility Module - Base classes and helper functions
"""
import json
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import re
from datetime import datetime

from supervisor_implementation import generate_experiment_suffix

@dataclass
class ExperimentConfig:
    benchmark: str
    num_groups: int
    group_size: int
    models: int
    starting_group_index: int
    max_iterations: int
    supervisor_model_name: str
    max_common_instructions: int
    max_patterns: int
    model_specific_for_all: bool
    max_model_specific_instructions: int
    limit_instruction_changes: bool
    max_change_ratio: float
    drop_worst_annr: bool
    supervised_by_gold_standard: bool = False
    llm_family_config: bool = None
    skip_final_goal_update: bool = False
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() 
                if k not in ['benchmark', 'num_groups', 'group_size', 'models', 
                           'starting_group_index', 'max_iterations', 'supervisor_model_name']}
    
    def get_config_dict_str(self):
        """Return configuration as dictionary string"""
        config = {}
        if self.max_common_instructions != 5:
            config['cin'] = self.max_common_instructions
        if self.max_patterns != 10:
            config['mxpt'] = self.max_patterns
        if self.model_specific_for_all:
            config['msifa'] = True
        if self.max_model_specific_instructions != 3:
            config['mxmsi'] = self.max_model_specific_instructions
        if self.limit_instruction_changes:
            config['lic'] = int(self.max_change_ratio * 100)
        if self.drop_worst_annr:
            config['dwa'] = True
        return str(config) if config else "{}"

@dataclass
class PerformanceResult:
    config: ExperimentConfig
    iteration: int
    group_index: int
    best_model_f1: float
    best_model_name: str
    mv_f1: float
    strict_span_f1_avg: float
    individual_model_results: Dict[str, float] = None
    individual_model_strict_f1: Dict[str, float] = None
    folder_path: str = None
    
    def get_score(self, metric: str) -> float:
        metric_map = {
            'best_model_f1': self.best_model_f1,
            'mv_f1': self.mv_f1,
            'strict_span_f1_avg': self.strict_span_f1_avg
        }
        if metric not in metric_map:
            raise ValueError(f"Unknown metric: {metric}")
        return metric_map[metric]

@dataclass
class TestPerformanceResult:
    """Results from test performance analysis"""
    benchmark: str
    model_name: str
    find_by_method: str  # '_fdbygst', '_fdbypma', or '_fdbygst_fdbypma'
    f1_score: float
    precision: float
    recall: float
    test_samples_count: int
    successful_samples: int
    json_file_path: str
    condition_name: str
    model_count: int
    supervisor_name: str
    iteration: int = 0 # Default to 0 if not applicable

def _normalize_model_name(name: str) -> str:
    """Normalize model name by extracting the part after '/' if present"""
    return name.split('/')[-1] if '/' in name else name

class BestPerformanceFinderBase:
    """Base class with common functionality for validation performance analysis"""
    
    def __init__(self, benchmarks: List[str]):
        self.benchmarks = benchmarks
        self.results = []
    
    def discover_available_configurations(self) -> Dict[str, Dict[str, List[str]]]:
        """Discover available model counts and supervisors for each benchmark"""
        available_configs = {}
        
        for benchmark in self.benchmarks:
            benchmark_path = Path(f'experiment_results/{benchmark}')
            available_configs[benchmark] = {'model_counts': [], 'supervisors': []}
            
            if not benchmark_path.exists():
                continue
            
            for model_dir in benchmark_path.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith('models'):
                    model_count_str = model_dir.name.replace('models', '')
                    if model_count_str.isdigit():
                        model_count = int(model_count_str)
                        if model_count not in available_configs[benchmark]['model_counts']:
                            available_configs[benchmark]['model_counts'].append(model_count)
                        
                        for supervisor_dir in model_dir.iterdir():
                            if supervisor_dir.is_dir():
                                supervisor_name = self._convert_safe_name_to_original(supervisor_dir.name)
                                if supervisor_name not in available_configs[benchmark]['supervisors']:
                                    available_configs[benchmark]['supervisors'].append(supervisor_name)
            
            available_configs[benchmark]['model_counts'].sort()
            available_configs[benchmark]['supervisors'].sort()
        
        return available_configs
    
    def _convert_safe_name_to_original(self, safe_name: str) -> str:
        """Convert filesystem-safe name back to original model name"""
        conversions = {
            'gpt-5-2025-08-07': 'gpt-5-2025-08-07',
            'gpt-5-mini-2025-08-07': 'gpt-5-mini-2025-08-07',
            'gpt-4o-2024-08-06': 'gpt-4o-2024-08-06',
            'claude-3-5-sonnet-20241022': 'claude-3-5-sonnet-20241022',
            'gemini-1_5-pro-002': 'gemini-1.5-pro-002',
            'llama-3_1_8b-instruct': 'llama-3.1:8b-instruct'
        }
        
        if safe_name in conversions:
            return conversions[safe_name]
        
        # Handle pattern like model_version (e.g., qwen3_14b -> qwen3:14b)
        if '_' in safe_name and any(char.isdigit() for char in safe_name):
            parts = safe_name.split('_')
            if len(parts) >= 2 and parts[-1].replace('b', '').isdigit():
                return ':'.join([parts[0], parts[-1]])
        
        # Convert underscores to dots for version numbers
        result = safe_name.replace('_', '.')
        if re.match(r'.*-\d{4}-\d{2}-\d{2}$', result):
            return result
        
        return safe_name
    
    def print_available_configurations(self):
        """Print all available configurations"""
        configs = self.discover_available_configurations()
        print(f"\n{'='*80}")
        print("AVAILABLE CONFIGURATIONS")
        print(f"{'='*80}")
        
        for benchmark in sorted(configs.keys()):
            config = configs[benchmark]
            print(f"\nBenchmark: {benchmark}")
            print(f"  Model counts: {config['model_counts']}")
            print(f"  Supervisors: {config['supervisors']}")
    
    def debug_folder_structure(self, benchmark='crossner_literature'):
        """Debug folder structure for a specific benchmark"""
        base_path = Path(f'experiment_results/{benchmark}')
        if base_path.exists():
            print(f"Found structure in {base_path}:")
            for model_dir in base_path.iterdir():
                if model_dir.is_dir():
                    print(f"  {model_dir.name}/")
                    for supervisor_dir in model_dir.iterdir():
                        if supervisor_dir.is_dir():
                            print(f"    {supervisor_dir.name}/")
                            exp_folders = [f.name for f in supervisor_dir.iterdir() if f.is_dir()][:3]
                            for folder in exp_folders:
                                print(f"      {folder}")
                            if len(exp_folders) > 3:
                                print(f"      ... and {len(list(supervisor_dir.iterdir())) - 3} more")
        else:
            print(f"Base path does not exist: {base_path}")
    
    def _extract_config_from_folder_name(self, folder_name: str, base_config: ExperimentConfig) -> Optional[Tuple[ExperimentConfig, int, int]]:
        """Extract configuration parameters from folder name"""
        parts = folder_name.split('_')
        
        # Start with default values
        config_dict = {
            'max_common_instructions': 5,
            'max_patterns': 10,
            'model_specific_for_all': False,
            'max_model_specific_instructions': 3,
            'limit_instruction_changes': False,
            'max_change_ratio': 0.2,
            'drop_worst_annr': False,
            'supervised_by_gold_standard': False,
            'llm_family_config': None,
            'skip_final_goal_update': False,
        }
        
        # Parse folder name for configuration flags
        if 'goldstd' in folder_name:
            config_dict['supervised_by_gold_standard'] = True
        if 'famllama' in folder_name:
            config_dict['llm_family_config'] = 'llama'
        if 'famqwen' in folder_name:
            config_dict['llm_family_config'] = 'qwen'
        if 'skipgoal' in folder_name:
            config_dict['skip_final_goal_update'] = True
        
        config_dict['model_specific_for_all'] = 'msifa' in folder_name
        config_dict['drop_worst_annr'] = 'dwa' in folder_name
        
        # Extract g20_s25 pattern first
        g_s_pattern = re.search(r'g(\d+)_s(\d+)', folder_name)
        if g_s_pattern:
            num_groups = int(g_s_pattern.group(1))
            group_size = int(g_s_pattern.group(2))
        
        # Parse numeric parameters
        for part in parts:
            if part.startswith('cin'):
                config_dict['max_common_instructions'] = int(part[3:])
            elif part.startswith('mxpt'):
                config_dict['max_patterns'] = int(part[4:])
            elif part.startswith('mxmsi'):
                config_dict['max_model_specific_instructions'] = int(part[5:])
            elif part.startswith('lic'):
                config_dict['limit_instruction_changes'] = True
                config_dict['max_change_ratio'] = int(part[3:]) / 100.0
        
        # Extract iteration and group_index
        iteration = group_index = 0
        for part in parts:
            if part.startswith('iter') and part[4:].isdigit():
                iteration = int(part[4:])
            elif part.startswith('grp') and part[3:].isdigit():
                group_index = int(part[3:])
        
        # Create new config with extracted parameters
        new_config = ExperimentConfig(
            benchmark=base_config.benchmark,
            num_groups=num_groups,
            group_size=group_size,
            models=base_config.models,
            starting_group_index=group_index - iteration,
            max_iterations=base_config.max_iterations,
            supervisor_model_name=base_config.supervisor_model_name,
            **config_dict
        )
        
        return new_config, iteration, group_index
    
    def _extract_performance_from_folder(self, folder_path: Path) -> Optional[Tuple]:
        """Extract performance metrics from result files"""
        try:
            combined_file = folder_path / 'combined_results.json'
            analysis_file = folder_path / 'agreement_analysis' / 'main_results_analysis' / 'analysis_summary.json'
            pairwise_file = folder_path / 'agreement_analysis' / 'main_results_analysis' / 'pairwise_agreements.json'
            
            if not combined_file.exists():
                return None
            
            with open(combined_file, 'r', encoding='utf-8') as f:
                combined_results = json.load(f)
            
            # Load optional analysis files
            agreement_results = pairwise_agreements = None
            if analysis_file.exists():
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    agreement_results = json.load(f)
            if pairwise_file.exists():
                with open(pairwise_file, 'r', encoding='utf-8') as f:
                    pairwise_agreements = json.load(f)
            
            # Extract best model performance
            results_by_model = combined_results.get('results_by_model', {})
            best_model_f1, best_model_name = 0.0, ""
            individual_model_results = {}
            
            for model_name, model_result in results_by_model.items():
                if 'error' not in model_result:
                    gold_f1 = model_result.get('avg_metrics', {}).get('f1', 0.0)
                    normalized_name = _normalize_model_name(model_name)
                    individual_model_results[normalized_name] = gold_f1
                    
                    if gold_f1 > best_model_f1:
                        best_model_f1, best_model_name = gold_f1, normalized_name
            
            # Extract MV performance
            mv_f1 = 0.0
            if agreement_results and 'model_average_agreements' in agreement_results:
                model_avg_agreements = agreement_results['model_average_agreements']
                mv_f1 = model_avg_agreements.get('MajorityVote', model_avg_agreements.get('DawidSkene', {})).get('gold_macro_f1', 0.0)
            
            # Extract strict span F1
            strict_span_f1_avg = 0.0
            individual_model_strict_f1 = {}
            
            if pairwise_agreements:
                model_strict_sums = defaultdict(lambda: {'sum': 0.0, 'count': 0})
                
                for agreement_rec in pairwise_agreements.values():
                    m1, m2 = agreement_rec.get('model1'), agreement_rec.get('model2')
                    strict_val = agreement_rec.get('avg_agreement', {}).get('strict_span_f1', 0.0)
                    
                    if m1 and m2:
                        for m in [_normalize_model_name(m1), _normalize_model_name(m2)]:
                            model_strict_sums[m]['sum'] += float(strict_val)
                            model_strict_sums[m]['count'] += 1
                
                strict_averages = []
                for model, agg in model_strict_sums.items():
                    if agg['count'] > 0:
                        model_avg = agg['sum'] / agg['count']
                        individual_model_strict_f1[model] = model_avg
                        strict_averages.append(model_avg)
                
                if strict_averages:
                    strict_span_f1_avg = float(np.mean(strict_averages))
            
            return best_model_f1, best_model_name, mv_f1, strict_span_f1_avg, individual_model_results, individual_model_strict_f1
            
        except Exception:
            return None

    def _matches_filter_criteria(self, config: ExperimentConfig, **filter_criteria) -> bool:
        """Check if configuration matches filter criteria"""
        for key, expected_value in filter_criteria.items():
            if key in ['models', 'supervisor_model_name']:
                continue  # Handled separately in search_configurations
            
            # Skip filtering if value is None (no restriction)
            if expected_value is None:
                continue
                
            if hasattr(config, key):
                actual_value = getattr(config, key)
                if isinstance(expected_value, list):
                    if actual_value not in expected_value:
                        return False
                else:
                    if actual_value != expected_value:
                        return False
        return True

    def _save_prompts_for_result(self, result: PerformanceResult, suffix: str, name_of_search_condition: str):
        """Save prompts for a specific result with given suffix"""
        try:
            source_folder = Path(result.folder_path)
            prompts_source = source_folder / 'prompts'
            
            if not prompts_source.exists():
                print(f"Warning: No prompts folder found in {source_folder}")
                return
            
            # Create target path with search condition folder
            supervisor_safe_name = result.config.supervisor_model_name.replace(':', '_').replace('/', '_').replace(' ', '_').replace('.', '_')
            target_base = Path('test_experiment_prompts_and_results') / name_of_search_condition / result.config.benchmark / f'models{result.config.models}' / supervisor_safe_name
            
            # Create folder name with suffix
            folder_name_with_suffix = source_folder.name + suffix
            target_folder = target_base / folder_name_with_suffix / 'prompts'
            
            # Create target directory
            target_folder.mkdir(parents=True, exist_ok=True)
            
            # Convert best model name for file matching (replace : with _)
            best_model_name_for_file = result.best_model_name.replace(':', '_').replace('/', '_').replace('\\', '_')
            if 'nemotron' in best_model_name_for_file:
                best_model_name_for_file = 'Randomblock1_' + best_model_name_for_file
            copied_files = 0
            
            for prompt_file in prompts_source.glob('*_prompt_template.txt'):
                # Check if this file belongs to the best model
                if prompt_file.name.startswith(best_model_name_for_file) and '_iter' in prompt_file.name:
                    target_file = target_folder / prompt_file.name
                    shutil.copy2(prompt_file, target_file)
                    copied_files += 1
                    print(f"  Copied prompt: {prompt_file.name}")            
            if copied_files == 0:
                print(f"Warning: No prompt template found for best model '{best_model_name_for_file}' in {prompts_source}")
            else:
                # Print performance scores with the save confirmation
                print(f"Saved {copied_files} best model prompt template file to {target_folder}")
                print(f"  Performance - Best Model F1: {result.best_model_f1:.4f}, Strict F1: {result.strict_span_f1_avg:.4f}")
            
        except Exception as e:
            print(f"Error saving prompts for {result.folder_path}: {e}")

    def save_best_configs_and_prompts(self, top_k: int = 1, name_of_search_condition: str = None):
        """Save prompts for best configurations based on both metrics"""
        if not self.results:
            print("No results found. Run search_configurations first.")
            return
        
        if name_of_search_condition is None:
            raise ValueError("name_of_search_condition is required when save_best_config_and_prompt=True")
        
        # Group results by benchmark
        benchmark_results = defaultdict(list)
        for result in self.results:
            benchmark_results[result.config.benchmark].append(result)
        
        print(f"\n{'='*80}")
        print("SAVING BEST CONFIGS AND PROMPTS")
        print(f"{'='*80}")
        
        for benchmark in sorted(benchmark_results.keys()):
            results = benchmark_results[benchmark]
            
            print(f"\nProcessing benchmark: {benchmark}")
            
            # Get top results by gold standard (best_model_f1)
            top_by_gold = sorted(results, key=lambda x: x.best_model_f1, reverse=True)[:top_k]
            gold_folders = set(r.folder_path for r in top_by_gold)
            
            # Get top results by pairwise agreement (strict_span_f1_avg)
            top_by_strict = sorted(results, key=lambda x: x.strict_span_f1_avg, reverse=True)[:top_k]
            strict_folders = set(r.folder_path for r in top_by_strict)
            
            # Combine all unique folders that need to be processed
            all_folders_to_process = set()
            all_folders_to_process.update(gold_folders)
            all_folders_to_process.update(strict_folders)
            
            # Create a mapping from folder_path to result
            folder_to_result = {r.folder_path: r for r in results}
            
            # Process each unique folder once with appropriate suffix
            for folder_path in all_folders_to_process:
                result = folder_to_result[folder_path]
                
                suffixes = []
                if folder_path in gold_folders:
                    suffixes.append('_fdbygst')
                if folder_path in strict_folders:
                    suffixes.append('_fdbypma')
                
                suffix = ''.join(suffixes)
                if suffix:
                    print(f"  Saving prompts for: {Path(folder_path).name}{suffix}")
                    self._save_prompts_for_result(result, suffix, name_of_search_condition)

    def save_search_query(self, name_of_search_condition: str, search_params: dict):
        """Save search query parameters to JSON file"""
        try:
            search_condition_folder = Path('test_experiment_prompts_and_results') / name_of_search_condition
            search_condition_folder.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            search_query_file = search_condition_folder / f'search_query_{timestamp}.json'
            
            with open(search_query_file, 'w', encoding='utf-8') as f:
                json.dump(search_params, f, indent=2, ensure_ascii=False)
            
            print(f"Saved search query parameters to {search_query_file}")
            
        except Exception as e:
            print(f"Error saving search query: {e}")

    def _print_table(self, title: str, headers: List[str], rows: List[List[str]], width: int = 200):
        """Print formatted table"""
        print(f"\n{'='*width}")
        print(title)
        print(f"{'='*width}")
        
        # Column widths
        col_widths = [4, 20, 4, 6, 8, 8, 8, 8, 8, 8, 8, 15, 6, 25, 0]  # 0 = no limit for last column
        
        # Print headers
        header_line = ""
        for i, (h, w) in enumerate(zip(headers, col_widths)):
            if w == 0:
                header_line += h
            else:
                header_line += f"{h:<{w}}"
        print(header_line)
        print("-" * width)
        
        # Print rows
        for row in rows:
            row_line = ""
            for i, (r, w) in enumerate(zip(row, col_widths)):
                if w == 0:
                    row_line += str(r)
                else:
                    row_line += f"{str(r)[:w-1]:<{w}}"
            print(row_line)

    def print_benchmark_top_results(self, metric: str = 'best_model_f1', top_k: int = 10):
        """Print top results for each benchmark"""
        if not self.results:
            print("No results found. Run search_configurations first.")
            return
        
        benchmark_results = defaultdict(list)
        for result in self.results:
            benchmark_results[result.config.benchmark].append(result)
        
        for benchmark in sorted(benchmark_results.keys()):
            results = sorted(benchmark_results[benchmark], key=lambda x: x.get_score(metric), reverse=True)[:top_k]
            
            headers = ['Rank', 'Benchmark', 'Iter', 'IterNum', 'GoldStd', 'LLMFamily', 'SkipGoal', 'Score', 
                      'Best F1', 'MV F1', 'Strict F1', 'Best Model', 'Models', 'Supervisor', 'Config']
            rows = []
            
            for i, r in enumerate(results, 1):
                rows.append([
                    i, r.config.benchmark[:19], r.iteration, r.iteration,
                    'Yes' if r.config.supervised_by_gold_standard else 'No',
                    r.config.llm_family_config, r.config.skip_final_goal_update,
                    f"{r.get_score(metric):.3f}", f"{r.best_model_f1:.3f}", 
                    f"{r.mv_f1:.3f}", f"{r.strict_span_f1_avg:.3f}",
                    r.best_model_name[:14], r.config.models, r.config.supervisor_model_name[:24],
                    r.config.get_config_dict_str()
                ])
            
            self._print_table(f"TOP {len(results)} RESULTS FOR {benchmark.upper()} - {metric.upper()}", headers, rows)

    def print_benchmark_summary(self, metric: str = 'best_model_f1'):
        """Print summary of best result per benchmark"""
        if not self.results:
            return
        
        benchmark_best = {}
        for result in self.results:
            benchmark = result.config.benchmark
            if benchmark not in benchmark_best or result.get_score(metric) > benchmark_best[benchmark].get_score(metric):
                benchmark_best[benchmark] = result
        
        headers = ['Benchmark', 'Iter', 'IterNum', 'GoldStd', 'Score', 'LLMFamily', 'SkipGoal',
                  'Best F1', 'MV F1', 'Models', 'Supervisor', 'Best Model', 'Config']
        rows = []
        
        for benchmark in sorted(benchmark_best.keys()):
            r = benchmark_best[benchmark]
            rows.append([
                benchmark[:19], r.iteration, r.iteration,
                'Yes' if r.config.supervised_by_gold_standard else 'No',
                r.config.llm_family_config, r.config.skip_final_goal_update,
                f"{r.get_score(metric):.3f}", f"{r.best_model_f1:.3f}", f"{r.mv_f1:.3f}", 
                r.config.models, r.config.supervisor_model_name[:19], r.best_model_name[:14],
                r.config.get_config_dict_str()
            ])
        
        self._print_table(f"BEST RESULT PER BENCHMARK FOR {metric.upper()}", headers, rows, 180)