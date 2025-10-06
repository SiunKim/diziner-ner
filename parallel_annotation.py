"""
Parallel annotation module for running multiple NER models concurrently
"""
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from annotation_runner import run_model_annotation
from utils_experiments import (
    get_model_result_path,
    load_existing_model_result
    )

def wait_for_supervisor_file(supervisor_results_path: str, timeout_minutes: int = 30) -> bool:
    """
    Wait for supervisor file to be created and validated
    
    Args:
        supervisor_results_path: Path to supervisor results file
        timeout_minutes: Maximum time to wait in minutes
        
    Returns:
        True if file exists and is valid, False if timeout
    """
    if not supervisor_results_path:
        return True  # No supervisor file needed
    
    timeout_seconds = timeout_minutes * 60
    start_time = time.time()
    check_interval = 10  # Check every 10 seconds
    
    print(f"Waiting for supervisor file: {supervisor_results_path}")
    
    while time.time() - start_time < timeout_seconds:
        if os.path.exists(supervisor_results_path):
            # File exists, now validate structure
            try:
                with open(supervisor_results_path, 'r', encoding='utf-8') as f:
                    supervisor_data = json.load(f)
                
                # Check for enhanced_guidelines structure (4-phase)
                enhanced_guidelines = supervisor_data.get("enhanced_guidelines", {})
                if enhanced_guidelines:
                    has_common = "hierarchical_common_instructions" in enhanced_guidelines
                    has_model_specific = "prioritized_model_instructions" in enhanced_guidelines
                    
                    if has_common or has_model_specific:
                        elapsed = time.time() - start_time
                        print(f"Supervisor file validated after {elapsed:.1f}s")
                        return True
                
                # Check alternative structure
                if "final_guidelines" in supervisor_data:
                    elapsed = time.time() - start_time
                    print(f"Supervisor file validated (alternative structure) after {elapsed:.1f}s")
                    return True
                
                print(f"Supervisor file exists but invalid structure, continuing to wait...")
                
            except Exception as e:
                print(f"Error validating supervisor file: {e}, continuing to wait...")
        
        # Wait before next check
        time.sleep(check_interval)
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed
        print(f"Still waiting for supervisor file... ({remaining/60:.1f} minutes remaining)")
    
    print(f"Timeout waiting for supervisor file after {timeout_minutes} minutes")
    return False


def run_single_model_worker(
    model_name: str,
    test_samples: List[Dict],
    ner_scheme: Dict[str, Any],
    model_source_map: Optional[Dict[str, str]] = None,
    final_task_goal: Optional[str] = None,
    supervisor_results_path: Optional[str] = None,
    iteration_number: int = 0,
    llm_infer_by_openrouter: bool = False,
    skip_final_goal_update: bool = False,
    max_annotation_retries: int = 3,
    experiment_dir: Optional[Path] = None,
    force_rerun: bool = False
) -> Dict[str, Any]:
    """
    Worker function for processing a single model in parallel
    Enhanced to support iteration 0 result reuse
    """
    start_time = time.time()
    
    try:
        # Check for existing results if experiment_dir provided
        if experiment_dir and not force_rerun:
            model_result_path = get_model_result_path(experiment_dir, model_name)
            if model_result_path.exists():
                print(f"[{model_name}] Loading existing result")
                with open(model_result_path, 'r', encoding='utf-8') as f:
                    # MODIFIED: Use load_existing_model_result for proper metadata handling
                    if iteration_number == 0:
                        result = load_existing_model_result(str(model_result_path), model_name)
                    else:
                        result = json.load(f)
                    
                    result['loaded_from_cache'] = True
                    return result
        
        print(f"[{model_name}] Starting annotation...")
        
        # Run model annotation
        result = run_model_annotation(
            model_name=model_name,
            test_samples=test_samples,
            ner_scheme=ner_scheme,
            model_source_map=model_source_map,
            final_task_goal=final_task_goal,
            supervisor_results_path=supervisor_results_path,
            iteration_number=iteration_number,
            llm_infer_by_openrouter=llm_infer_by_openrouter,
            skip_final_goal_update=skip_final_goal_update,
            max_annotation_retries=max_annotation_retries
        )
        
        # Save result if experiment_dir provided
        if experiment_dir:
            model_result_path = get_model_result_path(experiment_dir, model_name)
            model_result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        duration = time.time() - start_time
        avg_f1 = result.get('avg_metrics', {}).get('f1', 0.0)
        print(f"[{model_name}] Completed - F1: {avg_f1:.3f}, Time: {duration:.1f}s")
        
        result['parallel_processing_time'] = duration
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"[{model_name}] Failed after {duration:.1f}s: {str(e)[:100]}...")
        
        # Return error result
        return {
            'model_name': model_name,
            'error': str(e),
            'iteration_number': iteration_number,
            'parallel_processing_time': duration,
            'avg_metrics': {
                'precision': 0.0, 
                'recall': 0.0, 
                'f1': 0.0,
                'successful_samples': 0,
                'total_samples': len(test_samples),
                'token_accuracy': 0.0
            },
            'detailed_results': [],
            'all_confusing_cases': []
        }


def run_models_parallel(
    models: List[str],
    test_samples: List[Dict],
    ner_scheme: Dict[str, Any],
    model_source_map: Optional[Dict[str, str]] = None,
    final_task_goal: Optional[str] = None,
    supervisor_results_path: Optional[str] = None,
    iteration_number: int = 0,
    llm_infer_by_openrouter: bool = False,
    skip_final_goal_update: bool = False,
    max_annotation_retries: int = 3,
    experiment_dir: Optional[Path] = None,
    force_rerun: bool = False,
    max_workers: Optional[int] = None,
    verbose: int = 1,
    supervisor_timeout_minutes: int = 30
) -> Dict[str, Any]:
    """
    Run multiple NER models in parallel
    
    Args:
        models: List of model names to process
        test_samples: Test samples for annotation
        ner_scheme: NER scheme configuration
        model_source_map: Mapping of models to sources
        final_task_goal: Task goal description
        supervisor_results_path: Path to supervisor results
        iteration_number: Current iteration number
        llm_infer_by_openrouter: Whether to use OpenRouter
        max_annotation_retries: Maximum retry attempts
        experiment_dir: Experiment directory for saving results
        force_rerun: Whether to force rerun even if results exist
        max_workers: Maximum number of parallel workers (default: number of models)
        verbose: Verbosity level
        supervisor_timeout_minutes: Maximum time to wait for supervisor file
        
    Returns:
        Dictionary containing results for all models and timing information
    """
    if not llm_infer_by_openrouter:
        raise ValueError("Parallel processing is only supported with llm_infer_by_openrouter=True")
    
    if max_workers is None:
        max_workers = min(len(models), 8)  # Reasonable upper limit
    
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"PARALLEL MODEL PROCESSING - ITERATION {iteration_number}")
        print(f"{'='*60}")
        print(f"Models: {models}")
        print(f"Max workers: {max_workers}")
        print(f"OpenRouter inference: {llm_infer_by_openrouter}")
        if supervisor_results_path:
            print(f"Supervisor guidance: {supervisor_results_path}")
    
    # CRITICAL: Wait for supervisor file if iteration > 0
    if iteration_number > 0 and supervisor_results_path:
        if verbose >= 1:
            print(f"\nWaiting for supervisor file from previous iteration...")
            print(f"Supervisor file path: {supervisor_results_path}")
            print(f"Timeout: {supervisor_timeout_minutes} minutes")
        
        if not wait_for_supervisor_file(supervisor_results_path, supervisor_timeout_minutes):
            error_msg = f"Supervisor file not ready after {supervisor_timeout_minutes} minutes timeout"
            print(f"ERROR: {error_msg}")
            
            # Return error result for all models
            error_results = {}
            for model in models:
                error_results[model] = {
                    'model_name': model,
                    'error': error_msg,
                    'iteration_number': iteration_number,
                    'avg_metrics': {
                        'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                        'successful_samples': 0, 'total_samples': len(test_samples),
                        'token_accuracy': 0.0
                    },
                    'detailed_results': [],
                    'all_confusing_cases': []
                }
            
            return {
                'results_by_model': error_results,
                'parallel_summary': {
                    'total_processing_time': 0.0,
                    'max_workers_used': 0,
                    'models_processed': len(models),
                    'successful_models': 0,
                    'failed_models': len(models),
                    'failed_model_names': models,
                    'iteration_number': iteration_number,
                    'timestamp': datetime.now().isoformat(),
                    'supervisor_timeout': True,
                    'supervisor_timeout_minutes': supervisor_timeout_minutes
                }
            }
        
        if verbose >= 1:
            print(f"Supervisor file validated, proceeding with parallel annotation...")
    
    start_time = time.time()
    results_by_model = {}
    failed_models = []
    
    # Submit all model tasks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(
                run_single_model_worker,
                model_name=model,
                test_samples=test_samples,
                ner_scheme=ner_scheme,
                model_source_map=model_source_map,
                final_task_goal=final_task_goal,
                supervisor_results_path=supervisor_results_path,
                iteration_number=iteration_number,
                llm_infer_by_openrouter=llm_infer_by_openrouter,
                skip_final_goal_update=skip_final_goal_update,
                max_annotation_retries=max_annotation_retries,
                experiment_dir=experiment_dir,
                force_rerun=force_rerun
            ): model for model in models
        }
        
        # Collect results as they complete
        completed_count = 0
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                results_by_model[model_name] = result
                
                if 'error' in result:
                    failed_models.append(model_name)
                
                completed_count += 1
                
                if verbose >= 1:
                    remaining = len(models) - completed_count
                    status = "FAILED" if 'error' in result else "SUCCESS"
                    print(f"[{completed_count}/{len(models)}] {model_name}: {status} ({remaining} remaining)")
                    
            except Exception as e:
                print(f"[ERROR] Unexpected failure for {model_name}: {str(e)[:100]}...")
                failed_models.append(model_name)
                results_by_model[model_name] = {
                    'model_name': model_name,
                    'error': f"Executor exception: {str(e)}",
                    'iteration_number': iteration_number,
                    'avg_metrics': {
                        'precision': 0.0, 
                        'recall': 0.0, 
                        'f1': 0.0,
                        'successful_samples': 0,
                        'total_samples': len(test_samples),
                        'token_accuracy': 0.0
                    },
                    'detailed_results': [],
                    'all_confusing_cases': []
                }
                completed_count += 1
    
    total_time = time.time() - start_time
    
    # Calculate summary statistics
    successful_models = [m for m in models if m not in failed_models]
    
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"PARALLEL PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Successful models: {len(successful_models)}/{len(models)}")
        if failed_models:
            print(f"Failed models: {failed_models}")
        
        # Show F1 scores for successful models
        if successful_models:
            print(f"\nF1 Scores:")
            for model in successful_models:
                f1 = results_by_model[model].get('avg_metrics', {}).get('f1', 0.0)
                cached = " (cached)" if results_by_model[model].get('loaded_from_cache') else ""
                print(f"  {model}: {f1:.3f}{cached}")
    
    # Create summary result
    summary_result = {
        'results_by_model': results_by_model,
        'parallel_summary': {
            'total_processing_time': total_time,
            'max_workers_used': max_workers,
            'models_processed': len(models),
            'successful_models': len(successful_models),
            'failed_models': len(failed_models),
            'failed_model_names': failed_models,
            'iteration_number': iteration_number,
            'timestamp': datetime.now().isoformat(),
            'supervisor_file_waited': iteration_number > 0 and supervisor_results_path is not None
        }
    }
    
    if llm_infer_by_openrouter and verbose >= 1:
        from annotation_runner import collect_openrouter_statistics, print_openrouter_summary
        openrouter_stats = collect_openrouter_statistics(results_by_model)
        print_openrouter_summary(openrouter_stats)
        summary_result['openrouter_statistics'] = openrouter_stats
    
    return summary_result


def get_parallel_processing_stats(results_by_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract parallel processing statistics from model results
    
    Args:
        results_by_model: Dictionary of model results
        
    Returns:
        Dictionary with parallel processing statistics
    """
    processing_times = []
    cached_count = 0
    error_count = 0
    
    for model_name, result in results_by_model.items():
        if result.get('loaded_from_cache'):
            cached_count += 1
        elif 'error' in result:
            error_count += 1
        
        proc_time = result.get('parallel_processing_time', 0.0)
        if proc_time > 0:
            processing_times.append(proc_time)
    
    stats = {
        'total_models': len(results_by_model),
        'models_from_cache': cached_count,
        'models_with_errors': error_count,
        'models_newly_processed': len(processing_times)
    }
    
    if processing_times:
        stats.update({
            'avg_processing_time': sum(processing_times) / len(processing_times),
            'max_processing_time': max(processing_times),
            'min_processing_time': min(processing_times),
            'total_computation_time': sum(processing_times)
        })
    
    return stats
