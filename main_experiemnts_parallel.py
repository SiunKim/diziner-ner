import concurrent.futures
import threading
from datetime import datetime
import traceback
import logging

# Import your main function
from main_experiments import main_iterative_experiment  # Replace with actual module name

# Setup logging for parallel execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.results = {}
        self.lock = threading.Lock()
        
    def run_single_benchmark(self, benchmark, config, experiment_params):
        """Run a single benchmark experiment - FIXED to ensure benchmark isolation"""
        thread_id = threading.get_ident()
        start_time = datetime.now()
        
        try:
            logger.info(f"[Thread {thread_id}] Starting benchmark: {benchmark}")
            logger.info(f"[Thread {thread_id}] Config: {config}")
            # FIXED: Explicitly pass benchmark parameter to ensure proper path generation
            result = main_iterative_experiment(
                benchmark=benchmark,  # FIXED: Ensure benchmark is passed correctly
                **experiment_params,
                **config
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"[Thread {thread_id}] Completed benchmark: {benchmark} in {duration:.1f}s")
            
            return {
                'benchmark': benchmark,
                'config': config,
                'result': result,
                'status': 'success',
                'duration': duration,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Error in benchmark {benchmark}: {str(e)}"
            logger.error(f"[Thread {thread_id}] {error_msg}")
            logger.error(f"[Thread {thread_id}] Traceback: {traceback.format_exc()}")
            
            return {
                'benchmark': benchmark,
                'config': config,
                'result': None,
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'duration': duration,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            }
    
    def run_parallel_experiments(self, benchmarks, experiment_configs, experiment_params):
        """Run experiments in parallel across benchmarks - FIXED for proper benchmark handling"""
        
        # Create all benchmark-config combinations
        experiment_tasks = []
        for benchmark in benchmarks:
            # FIXED: Verify benchmark parameter is properly maintained throughout
            for i, config in enumerate(experiment_configs, 1):
                # Each task maintains its benchmark identity
                experiment_tasks.append((benchmark, config, experiment_params))
        
        total_tasks = len(experiment_tasks)
        logger.info(f"Starting {total_tasks} experiments across {len(benchmarks)} benchmarks")
        logger.info(f"Using max_workers: {self.max_workers}")
        
        # FIXED: Ensure each benchmark gets its own results tracking
        results_by_benchmark = {benchmark: [] for benchmark in benchmarks}
        
        # Run experiments in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks with proper benchmark tracking
            future_to_task = {
                executor.submit(self.run_single_benchmark, benchmark, config, experiment_params): 
                (benchmark, i, config) 
                for benchmark, config, experiment_params in experiment_tasks
                for i, (b, c, p) in enumerate([(benchmark, config, experiment_params)], 1)
            }
            
            # Process completed tasks
            completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                benchmark, config_idx, config = future_to_task[future]
                
                try:
                    result = future.result()
                    # FIXED: Ensure results are stored under correct benchmark
                    results_by_benchmark[benchmark].append(result)
                    
                    completed += 1
                    logger.info(f"Progress: {completed}/{total_tasks} completed")
                    
                    if result['status'] == 'success':
                        logger.info(f"✓ {benchmark} (Config {config_idx}) - {result['duration']:.1f}s")
                    else:
                        logger.error(f"✗ {benchmark} (Config {config_idx}) - FAILED")
                        
                except Exception as e:
                    logger.error(f"Task execution error for {benchmark}: {e}")
                    # FIXED: Store errors under correct benchmark
                    results_by_benchmark[benchmark].append({
                        'benchmark': benchmark,
                        'config': config,
                        'result': None,
                        'status': 'execution_failed',
                        'error': str(e)
                    })
        
        return results_by_benchmark
    
    def print_summary(self, results_by_benchmark):
        """Print experiment summary - Enhanced to show benchmark isolation"""
        print(f"\n{'='*80}")
        print("PARALLEL EXPERIMENTS SUMMARY")
        print(f"{'='*80}")
        
        total_experiments = 0
        successful_experiments = 0
        failed_experiments = 0
        
        for benchmark, results in results_by_benchmark.items():
            print(f"\nBenchmark: {benchmark}")
            print(f"  Total configs: {len(results)}")
            
            success_count = sum(1 for r in results if r['status'] == 'success')
            failed_count = len(results) - success_count
            
            print(f"  Successful: {success_count}")
            print(f"  Failed: {failed_count}")
            
            # FIXED: Show unique result paths to verify benchmark isolation
            if success_count > 0:
                avg_duration = sum(r['duration'] for r in results if r['status'] == 'success') / success_count
                print(f"  Avg duration: {avg_duration:.1f}s")
                
                # Show sample result path to verify proper benchmark separation
                for result in results:
                    if result['status'] == 'success' and result.get('result'):
                        if 'annotation_results' in result['result'].get(0, {}):
                            sample_path = result['result'][0]['annotation_results'].get('experiment_directory', '')
                            if sample_path:
                                print(f"  Sample path: {sample_path}")
                                break
            
            total_experiments += len(results)
            successful_experiments += success_count
            failed_experiments += failed_count
        
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY:")
        print(f"  Total experiments: {total_experiments}")
        print(f"  Successful: {successful_experiments}")
        print(f"  Failed: {failed_experiments}")
        print(f"  Success rate: {successful_experiments/total_experiments*100:.1f}%")
        print(f"  Benchmarks processed: {len(results_by_benchmark)}")
        print(f"{'='*80}")

def run_parallel_benchmark_experiments():
    """Main function to run parallel benchmark experiments"""
    # Hyperparameter combinations list
    experiment_configs = [
        # 1) 안정적 보수 캡
        dict(group_size=25, max_common_instructions=3, max_patterns=5,
             max_model_specific_instructions=2, limit_instruction_changes=True,
             max_change_ratio=0.10, drop_worst_annr=False,
             max_iterations=4,
             supervised_by_gold_standard=False),
        
        # 2) 안정적 보수 캡 - superviesd
        dict(group_size=25, max_common_instructions=3, max_patterns=5,
             max_model_specific_instructions=2, limit_instruction_changes=True,
             max_change_ratio=0.10, drop_worst_annr=False,
             max_iterations=6,
             supervised_by_gold_standard=True),

        # 3) 변경 제한 off + 완화된 캡
        dict(group_size=25, max_common_instructions=5, max_patterns=8,
             max_model_specific_instructions=3, limit_instruction_changes=False,
             max_change_ratio=0.20, drop_worst_annr=False,
             max_iterations=4,
             supervised_by_gold_standard=False),
        
        # 5) 공격적 자유 확장
        dict(group_size=25, 
             max_common_instructions=10,
             max_patterns=20,
             max_model_specific_instructions=10,
             limit_instruction_changes=False,
             max_change_ratio=0.50,            
             drop_worst_annr=False,
             max_iterations=4,                 
             supervised_by_gold_standard=False),
        
        # 4) 안정적 보수 캡 + drop worst annr True
        dict(group_size=25, max_common_instructions=3, max_patterns=5,
             max_model_specific_instructions=2, limit_instruction_changes=True,
             max_change_ratio=0.10, drop_worst_annr=True,
             max_iterations=6,
             supervised_by_gold_standard=False),
    ]
    
    experiment_configs = experiment_configs[:]
    # max_iterations = 4
    num_models = 8
    # Benchmark list
    benchmarks = [
        # 'crossner_conll2003',
        # 'crossner_ai',
        # 'crossner_literature',
        # 'crossner_music',
        # 'crossner_politics',
        # 'crossner_science',
        # 'mitner_movie',
        # 'mitner_restaurant',
        'ACE05',
        # 'anatem',
        # 'bc2gm',
        # 'bc4chemd',
        # 'bc5cdr',
        # 'Broad Twitter',
        # 'FabNER',
        # 'GENIA',
        # 'MultiNERD',
        # 'OntoNotes'
    ]
    
    # Experiment parameters
    experiment_params = {
        'starting_group_index': 0,
        # 'max_iterations': max_iterations,
        'num_models': num_models,
        'convergence_threshold': None,
        'prompts_config_path': "prompts/instruction_supervision_0905.json",
        # 'supervisor_model_name': "gpt-5-2025-08-07",
        'supervisor_model_name': "gpt-5-mini-2025-08-07",
        'llm_infer_by_openrouter': True,
        'prefer_paid_models': True,
        # 'supervised_by_gold_standard': False
    }
    
    print(f"Benchmarks to run: {benchmarks}")
    print(f"Total configs per benchmark: {len(experiment_configs)}")
    print(f"Total experiments: {len(benchmarks) * len(experiment_configs)}")

    max_workers = 8
    
    # Create runner and execute
    runner = BenchmarkRunner(max_workers=max_workers)
    start_time = datetime.now()
    
    try:
        results = runner.run_parallel_experiments(
            benchmarks=benchmarks,
            experiment_configs=experiment_configs,
            experiment_params=experiment_params
        )
        # Print summary with benchmark isolation verification
        runner.print_summary(results)
        
        total_duration = (datetime.now() - start_time).total_seconds()
        print(f"\nTotal execution time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        
        # FIXED: Verify benchmark isolation by checking result paths
        print(f"\nBENCHMARK ISOLATION VERIFICATION:")
        print(f"{'='*50}")
        for benchmark, benchmark_results in results.items():
            if benchmark_results and benchmark_results[0]['status'] == 'success':
                result = benchmark_results[0]['result']
                if result and isinstance(result, dict) and 0 in result:
                    experiment_dir = result[0].get('annotation_results', {}).get('experiment_directory', '')
                    if experiment_dir:
                        print(f"{benchmark}: {experiment_dir}")
                    else:
                        print(f"{benchmark}: No experiment directory found")
                else:
                    print(f"{benchmark}: No valid result structure")
            else:
                print(f"{benchmark}: Failed or no results")
        
        return results
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        return None
    except Exception as e:
        print(f"\nExecution failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_parallel_benchmark_experiments()
