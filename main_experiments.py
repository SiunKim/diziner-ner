import pickle
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import os

# Import modules
from annotation_runner import run_model_annotation
from agreement_analysis_test import test_analysis
from disagreement_analysis_in_pipeline import analyze_experiment_disagreement
from error_analysis import analyze_ner_errors_from_file
from supervisor_implementation import run_supervisor_analysis
from parallel_annotation import run_models_parallel, get_parallel_processing_stats
import llm_clients
from utils_experiments import (
    load_config,
    get_model_result_path,
    print_experiment_summary,
    save_experiment_results,
    get_final_goal_for_iteration,
    save_iteration_metadata,
    get_iteration_paths,
    create_grouped_samples_by_index,
    get_previous_iteration_supervisor_path,
    create_combined_results_structure,
    print_cost_estimate,
    find_existing_iter0_results,
    copy_iter0_model_results,
    load_existing_model_result
)
from utils_model_dropping import (
    should_drop_worst_model,
    log_model_dropping_decision,
    extract_model_pairwise_strict_f1
)
from utils_annotator import save_prompt_templates_from_experiment

# Environment setup
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
DEBUG = True

def run_analysis_pipeline(
    main_output_file: str,
    experiment_dir: Path,
    models: List[str],
    run_agreement_analysis: bool = True,
    run_disagreement_analysis: bool = True,
    generate_documentation: bool = True,
    run_error_analysis: bool = True,
    hotspot_percentile: float = 80,
    coalition_cutoff: float = 0.5,
    supervised_by_gold_standard: bool = False,
    gold_standard_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Run all analysis steps and return results - SIMPLIFIED unified pipeline"""
    analysis_results = {}
    
    if len(models) < 2:
        print("Skipping all analyses - need at least 2 models for comparison")
        return analysis_results
    
    # Agreement analysis
    if run_agreement_analysis:
        print(f"\n{'='*60}\nRUNNING AGREEMENT ANALYSIS\n{'='*60}")
        try:
            analysis_results['agreement'] = test_analysis(main_output_file,
                                                          verbose=1,
                                                          save_results=True)
            print("Agreement analysis completed!")
        except Exception as e:
            print(f"Agreement analysis failed: {e}")
            analysis_results['agreement'] = None
    
    # UNIFIED DISAGREEMENT ANALYSIS (always runs, with gold standard option)
    if run_disagreement_analysis:
        analysis_type = "UNIFIED ANALYSIS (with Gold Standard)" if supervised_by_gold_standard else "DISAGREEMENT ANALYSIS"
        
        print(f"\n{'='*60}\nRUNNING {analysis_type}\n{'='*60}")
        try:
            disagreement_results = analyze_experiment_disagreement(
                result_file_path=main_output_file,
                base_output_dir="./disagreement_analysis",
                weights=None,
                save_results=True,
                create_visualizations=True,
                use_experiment_structure=True,
                hotspot_percentile=hotspot_percentile,
                coalition_cutoff=coalition_cutoff,
                supervised_by_gold_standard=supervised_by_gold_standard,
                gold_standard_config=gold_standard_config
            )
            analysis_results['disagreement'] = disagreement_results
            print(f"{analysis_type} completed!")
                
        except Exception as e:
            print(f"{analysis_type} failed: {e}")
            analysis_results['disagreement'] = None
    
    # Documentation generation (SIMPLIFIED - already done in disagreement analysis)
    if generate_documentation and analysis_results.get('disagreement'):
        hotspot_doc = analysis_results['disagreement'].get('hotspot_documentation')
        if hotspot_doc:
            analysis_results['documentation'] = hotspot_doc
            doc_type = "Gold Standard Enhanced" if supervised_by_gold_standard else "Standard"
            print(f"Hotspot documentation ({doc_type}) already generated!")
        else:
            print("Warning: Hotspot documentation not found in analysis results")
            analysis_results['documentation'] = None
    
    # Error analysis
    if run_error_analysis:
        analysis_type_name = "ERROR ANALYSIS (Gold Standard Enhanced)" if supervised_by_gold_standard else "ERROR ANALYSIS"
        print(f"\n{'='*60}\nRUNNING {analysis_type_name}\n{'='*60}")
        try:
            error_output_dir = str(experiment_dir / "error_analysis")
            analysis_results['error_analysis'] = analyze_ner_errors_from_file(
                main_output_file, 
                error_output_dir, 
                verbose=1,
                supervised_by_gold_standard=supervised_by_gold_standard
            )
            print(f"{analysis_type_name} completed!")
        except Exception as e:
            print(f"{analysis_type_name} failed: {e}")
            analysis_results['error_analysis'] = None      
    
    return analysis_results

def run_experiment(
    benchmark: str,
    dataset_path: str,
    models: List[str],
    ner_scheme: Dict[str, Any],
    model_source_map: Optional[Dict[str, str]] = None,
    group_index: int = 0,
    final_task_goal: Optional[str] = None,
    iteration_number: int = 0,
    supervisor_results_path: Optional[str] = None,   
    models_count: Optional[int] = None,
    supervisor_model_name: str = "gpt-5-2025-08-07",
    llm_infer_by_openrouter: bool = False, 
    num_groups: int = 20,
    group_size: int = None,
    grouping_model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    device: str = 'auto',
    use_gpu_clustering: bool = True,
    random_seed: int = 42,
    force_regenerate_samples: bool = False,
    run_agreement_analysis: bool = True,
    run_disagreement_analysis: bool = True,
    generate_documentation: bool = True,
    run_error_analysis: bool = True,
    hotspot_percentile: float = 80,
    coalition_cutoff: float = 0.5,
    supervised_by_gold_standard: bool = False,
    gold_standard_config: Dict[str, Any] = None,
    max_common_instructions: int = 5,
    max_patterns: int = 10,
    model_specific_for_all: bool = False,
    max_model_specific_instructions: int = 3,
    limit_instruction_changes: bool = False,
    max_change_ratio: float = 0.2,
    drop_worst_annr: bool = False,
    llm_family_config: Optional[str] = None,
    skip_final_goal_update: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Run comprehensive NER experiment with full analysis pipeline"""
    print(f"\n{'='*80}")
    print(f"RUNNING COMPREHENSIVE NER EXPERIMENT - ITERATION {iteration_number}")
    if llm_infer_by_openrouter:
        print("Inference Method: OpenRouter")
    
    # Add supervised_by_gold_standard info to output
    analysis_mode = "Gold Standard Supervision" if supervised_by_gold_standard else "Disagreement Analysis"
    print(f"Analysis Mode: {analysis_mode}")
    print(f"{'='*80}")
    
    print(f"Models: {models}")
    print(f"Strategy: Lexical Diversity Grouping → Use Group {group_index}")
    
    if iteration_number > 0 and supervisor_results_path:
        print(f"Supervisor guidance: {supervisor_results_path}")
    
    start_time = datetime.now()
    print(f"Number of models: {models_count}")
    

    # Get experiment paths for this iteration
    experiment_paths = get_iteration_paths(
        benchmark=benchmark,
        num_groups=num_groups,
        group_size=group_size,
        iteration_number=iteration_number,
        group_index=group_index,
        models_count=models_count or len(models),
        supervisor_model_name=supervisor_model_name,
        max_common_instructions=max_common_instructions,
        max_patterns=max_patterns,
        model_specific_for_all=model_specific_for_all,
        max_model_specific_instructions=max_model_specific_instructions,
        limit_instruction_changes=limit_instruction_changes,
        max_change_ratio=max_change_ratio,
        drop_worst_annr=drop_worst_annr,
        supervised_by_gold_standard=supervised_by_gold_standard,
        llm_family_config=llm_family_config,
        skip_final_goal_update=skip_final_goal_update,
    )    
    experiment_dir = experiment_paths['experiment_dir']
    
    # Load or create test samples
    test_samples_file = experiment_paths['test_samples_file']
    
    if test_samples_file.exists() and not force_regenerate_samples:
        print("Loading existing test samples...")
        with open(test_samples_file, 'rb') as f:
            test_samples = pickle.load(f)
    else:
        # For iteration > 0, try to load from iteration 0
        if iteration_number > 0:
            iteration_0_paths = get_iteration_paths(
                benchmark=benchmark,
                num_groups=num_groups,
                group_size=group_size,
                iteration_number=0,
                group_index=group_index,
                models_count=models_count or len(models),
                supervisor_model_name=supervisor_model_name,
                max_common_instructions=max_common_instructions,
                max_patterns=max_patterns,
                model_specific_for_all=model_specific_for_all,
                max_model_specific_instructions=max_model_specific_instructions,
                limit_instruction_changes=limit_instruction_changes,
                max_change_ratio=max_change_ratio,
                drop_worst_annr=drop_worst_annr,
                supervised_by_gold_standard=supervised_by_gold_standard,
                skip_final_goal_update=skip_final_goal_update
            )
            iteration_0_samples_file = iteration_0_paths['test_samples_file']
            
            if iteration_0_samples_file.exists():
                print("Loading test samples from iteration 0...")
                with open(iteration_0_samples_file, 'rb') as f:
                    test_samples = pickle.load(f)
                
                # Save to current iteration
                test_samples_file.parent.mkdir(parents=True, exist_ok=True)
                with open(test_samples_file, 'wb') as f:
                    pickle.dump(test_samples, f)
            else:
                test_samples = create_grouped_samples_by_index(
                    dataset_path, group_index, num_groups, group_size,
                    grouping_model_name, batch_size, device, use_gpu_clustering, random_seed
                )
                test_samples_file.parent.mkdir(parents=True, exist_ok=True)
                with open(test_samples_file, 'wb') as f:
                    pickle.dump(test_samples, f)
        else:
            test_samples = create_grouped_samples_by_index(
                dataset_path, group_index, num_groups, group_size,
                grouping_model_name, batch_size, device, use_gpu_clustering, random_seed
            )
            test_samples_file.parent.mkdir(parents=True, exist_ok=True)
            with open(test_samples_file, 'wb') as f:
                pickle.dump(test_samples, f)
    
    # Determine final goal based on iteration and supervisor results
    effective_final_goal = get_final_goal_for_iteration(
        iteration_number, supervisor_results_path, final_task_goal or ""
    ) if iteration_number > 0 and supervisor_results_path else final_task_goal
    
    
    print(f"\n{'='*60}")
    print(f"PROCESSING MODELS - ITERATION {iteration_number}")
    print(f"{'='*60}")
    
    #Check for existing iteration 0 results if this is iteration 0
    models_to_process = models.copy()
    results_by_model = {}
    if iteration_number == 0:
        print(f"Checking for existing iteration 0 results to reuse...")
        existing_results = find_existing_iter0_results(
            benchmark=benchmark,
            models_count=models_count or len(models),
            num_groups=num_groups,
            group_size=group_size,
            target_models=models,
            current_experiment_dir=experiment_dir
        )
        
        if existing_results:
            print(f"Found {len(existing_results)} existing iteration 0 results")
            
            # Copy existing results to current experiment directory
            successfully_copied = copy_iter0_model_results(
                existing_results, 
                experiment_dir / 'model_results'
            )
            
            # Load copied results into results_by_model
            for model_name in successfully_copied:
                model_result_path = get_model_result_path(experiment_dir, model_name)
                if model_result_path.exists():
                    try:
                        result = load_existing_model_result(str(model_result_path), model_name)
                        results_by_model[model_name] = result
                        models_to_process.remove(model_name)
                        print(f"Reused existing result for: {model_name}")
                    except Exception as e:
                        print(f"Failed to load copied result for {model_name}: {e}")
            
            if successfully_copied:
                print(f"Reused {len(successfully_copied)} existing results: {successfully_copied}")
                print(f"Still need to process {len(models_to_process)} models: {models_to_process}")
        else:
            print("No existing iteration 0 results found for reuse")

    
    # Process models with iteration support
    if models_to_process:
        if llm_infer_by_openrouter:
            # Use parallel processing for OpenRouter
            parallel_results = run_models_parallel(
                models=models,
                test_samples=test_samples,
                ner_scheme=ner_scheme,
                model_source_map=model_source_map,
                final_task_goal=effective_final_goal,
                supervisor_results_path=supervisor_results_path,
                iteration_number=iteration_number,
                llm_infer_by_openrouter=llm_infer_by_openrouter,
                skip_final_goal_update=skip_final_goal_update,
                supervisor_timeout_minutes=40,
                experiment_dir=experiment_dir,
                force_rerun=kwargs.get('force_rerun_models', False),
                verbose=1
            )
            results_by_model = parallel_results['results_by_model']
            parallel_stats = get_parallel_processing_stats(results_by_model)
            print_cost_estimate(results_by_model, iteration_number)
        else:
            # Use sequential processing for non-OpenRouter models
            results_by_model = {}
            parallel_stats = None
            for model_name in models:
                print(f"\nProcessing Model: {model_name} (Iteration {iteration_number})")
                
                # Check existing results
                model_result_path = get_model_result_path(experiment_dir, model_name)
                if model_result_path.exists() and not kwargs.get('force_rerun_models', False):
                    with open(model_result_path, 'r', encoding='utf-8') as f:
                        model_result = json.load(f)
                    print(f"Loaded existing result for: {model_name}")
                else:
                    try:
                        model_result = run_model_annotation(
                            model_name=model_name, 
                            test_samples=test_samples, 
                            ner_scheme=ner_scheme, 
                            model_source_map=model_source_map,
                            final_task_goal=effective_final_goal,
                            supervisor_results_path=supervisor_results_path,
                            iteration_number=iteration_number,
                            llm_infer_by_openrouter=llm_infer_by_openrouter,
                            skip_final_goal_update=skip_final_goal_update
                        )
                        # Save result
                        model_result_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(model_result_path, 'w', encoding='utf-8') as f:
                            json.dump(model_result, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        if DEBUG:
                            import traceback
                            traceback.print_exc()
                        model_result = {
                            'model_name': model_name,
                            'error': str(e),
                            'iteration_number': iteration_number,
                            'avg_metrics': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                        }

                results_by_model[model_name] = model_result
    else:
        print("All models have existing results - no new processing needed")
        parallel_stats = None

    # Print summary of what was processed vs reused
    if iteration_number == 0:
        reused_models = [m for m in models if m not in models_to_process]
        if reused_models:
            print(f"\n{'='*60}")
            print(f"ITERATION 0 PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"Reused existing results: {len(reused_models)} models")
            for model in reused_models:
                print(f"  ✓ {model}")
            if models_to_process:
                print(f"Newly processed: {len(models_to_process)} models")
                for model in models_to_process:
                    print(f"  + {model}")
            else:
                print("No new processing required")

    # Convert to list format for compatibility
    all_results = list(results_by_model.values())
    
    # Create experiment info (enhanced with gold standard info)
    experiment_info = {
        'experiment_type': f'lexical_diversity_group_{group_index}_iter_{iteration_number}',
        'dataset': dataset_path,
        'sample_selection_method': f'lexical_diversity_grouping_group_{group_index}',
        'num_groups_created': num_groups,
        'group_size': group_size,
        'selected_group_id': group_index,
        'total_samples': len(test_samples),
        'models_tested': models,
        'ner_scheme': ner_scheme,
        'final_task_goal': effective_final_goal,
        'original_final_task_goal': final_task_goal,
        'final_goal_was_updated': effective_final_goal != final_task_goal if final_task_goal else False,
        'iteration_number': iteration_number,
        'supervisor_results_path': supervisor_results_path,
        'supervised_by_gold_standard': supervised_by_gold_standard,
        'gold_standard_config': gold_standard_config,
        'llm_family_config': llm_family_config,
        'grouping_parameters': {
            'model_name': grouping_model_name,
            'batch_size': batch_size,
            'device': device,
            'use_gpu_clustering': use_gpu_clustering
        },
        'random_seed': random_seed,
        'timestamp': datetime.now().isoformat()
    }

    # Add parallel processing info to experiment_info if used
    if llm_infer_by_openrouter and parallel_stats:
        experiment_info['parallel_processing'] = {
            'enabled': True,
            'stats': parallel_stats
        }
    else:
        experiment_info['parallel_processing'] = {'enabled': False}
    # Create and save combined results
    combined_results = create_combined_results_structure(
        experiment_info, test_samples, results_by_model, all_results
    )
    
    # Save results
    main_output_file = save_experiment_results(combined_results, experiment_paths, group_index)
    try:
        save_prompt_templates_from_experiment(experiment_dir)
    except Exception as e:
        print(f"Warning: Failed to save prompt templates: {e}")

    # Save experiment config (enhanced)
    experiment_config = {
        'dataset_path': dataset_path,
        'models': models,
        'ner_scheme': ner_scheme,
        'final_task_goal': final_task_goal,
        'supervised_by_gold_standard': supervised_by_gold_standard,
        'gold_standard_config': gold_standard_config,
        'iteration_parameters': {
            'iteration_number': iteration_number,
            'supervisor_results_path': supervisor_results_path
        },
        'experiment_parameters': {
            'num_groups': num_groups,
            'group_size': group_size,
            'group_index': group_index,
            'grouping_model_name': grouping_model_name,
            'batch_size': batch_size,
            'device': device,
            'use_gpu_clustering': use_gpu_clustering,
            'random_seed': random_seed
        },
        'analysis_parameters': {
            'run_agreement_analysis': run_agreement_analysis,
            'run_disagreement_analysis': run_disagreement_analysis,
            'generate_documentation': generate_documentation,
            'run_error_analysis': run_error_analysis,
            'hotspot_percentile': hotspot_percentile,
            'coalition_cutoff': coalition_cutoff
        },
        'experiment_timestamp': datetime.now().isoformat()
    }
    
    config_file = experiment_paths['experiment_config_file']
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_config, f, indent=2, ensure_ascii=False)
    
    # Save iteration metadata
    iteration_data = {
        'iteration_number': iteration_number,
        'supervisor_results_path': supervisor_results_path,
        'supervised_by_gold_standard': supervised_by_gold_standard,
        'experiment_config': experiment_config,
        'models': models,
        'performance_summary': {
            model_name: result.get('avg_metrics', {}).get('f1', 0.0) 
            for model_name, result in results_by_model.items()
        }
    }
    save_iteration_metadata(iteration_data, experiment_paths)
    
    print_experiment_summary(all_results)
    print(f"\nResults saved to: {main_output_file}")
    
    # Run analyses (enhanced with gold standard support)
    analysis_results = run_analysis_pipeline(
        main_output_file, experiment_dir, models,
        run_agreement_analysis, run_disagreement_analysis,
        generate_documentation, run_error_analysis,
        hotspot_percentile, coalition_cutoff,
        supervised_by_gold_standard, gold_standard_config
    )
    
    # Final summary
    total_duration = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETED - ITERATION {iteration_number}")
    print(f"{'='*80}")
    print(f"Total duration: {total_duration:.1f} seconds")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Analysis mode: {analysis_mode}")
    
    if iteration_number > 0:
        supervisor_summary = sum(
            result.get('supervisor_instructions_summary', {}).get('total_instructions_applied', 0)
            for result in all_results
        )
        print(f"Total supervisor instructions applied: {supervisor_summary}")
    
    return {
        'experiment_results': combined_results,
        'analysis_results': analysis_results,
        'experiment_output_file': main_output_file,
        'experiment_directory': str(experiment_dir),
        'iteration_number': iteration_number,
        'supervisor_results_path': supervisor_results_path,
        'iteration_metadata': iteration_data,
        'effective_final_goal': effective_final_goal,
        'final_goal_updated': effective_final_goal != final_task_goal if final_task_goal else False,
        'supervised_by_gold_standard': supervised_by_gold_standard
    }

def run_iterative_annotation_supervisor_cycle(
    benchmark: str,
    dataset_path: str,
    models: List[str],
    ner_scheme: Dict[str, Any],
    max_iterations: int = 3,
    convergence_threshold: Optional[float] = 0.05,
    supervisor_model_name: str = "gpt-5-2025-08-07",
    prompts_config_path: str = "prompts/instruction_supervision_0905.json",
    starting_group_index: int = 0,
    models_count: Optional[int] = None,
    llm_infer_by_openrouter: bool = False,
    supervised_by_gold_standard: bool = False,
    gold_standard_config: Dict[str, Any] = None,
    max_common_instructions: int = 5,
    max_patterns: int = 10,
    model_specific_for_all: bool = False,
    max_model_specific_instructions: int = 3,
    limit_instruction_changes: bool = False,
    max_change_ratio: float = 0.2,
    drop_worst_annr: bool = False,
    llm_family_config: Optional[str] = None,
    skip_final_goal_update: bool = False,
    **experiment_kwargs
) -> Dict[int, Dict[str, Any]]:
    """Run complete iterative cycle of annotation and supervision with model dropping support"""
    print(f"\n{'='*100}")
    print("STARTING ITERATIVE ANNOTATION-SUPERVISOR CYCLE")
    if llm_infer_by_openrouter:
        print("Inference Method: OpenRouter")
    
    # Add supervision mode info
    supervision_mode = "Gold Standard Supervision" if supervised_by_gold_standard else "Disagreement-based Supervision"
    print(f"Supervision Mode: {supervision_mode}")
    print(f"{'='*100}")
    
    print(f"Dataset: {dataset_path}")
    print(f"benchmark: {benchmark}")
    print(f"dataset_path: {dataset_path}")
    print(f"Initial models: {models}")
    print(f"Models count: {models_count or len(models)}")
    print(f"Max iterations: {max_iterations}")
    print(f"Convergence threshold: {convergence_threshold if convergence_threshold else 'Disabled'}")
    print(f"Supervisor model: {supervisor_model_name}")
    print(f"Starting group index: {starting_group_index}")
    print(f"Skip final goal update: {skip_final_goal_update}")
    print(f"Drop worst annotator: {drop_worst_annr}")
    experiment_paths = get_iteration_paths(
        benchmark=benchmark,
        num_groups=experiment_kwargs['num_groups'],
        group_size=experiment_kwargs['group_size'],
        iteration_number=0,
        group_index=0,
        models_count=models_count or len(models),
        supervisor_model_name=supervisor_model_name,
        max_common_instructions=max_common_instructions,
        max_patterns=max_patterns,
        model_specific_for_all=model_specific_for_all,
        max_model_specific_instructions=max_model_specific_instructions,
        limit_instruction_changes=limit_instruction_changes,
        max_change_ratio=max_change_ratio,
        drop_worst_annr=drop_worst_annr,
        supervised_by_gold_standard=supervised_by_gold_standard,
        llm_family_config=llm_family_config,
        skip_final_goal_update=skip_final_goal_update
    )
    print(f"experiment_paths: {experiment_paths['experiment_dir']}")
    
    iteration_results = {}
    previous_f1_scores = {}
    
    # Track model dropping
    original_models = models.copy()
    current_models = models.copy()
    dropped_models_history = []
    
    for iteration in range(max_iterations):
        # Model dropping logic (from iteration 2 onwards)
        if drop_worst_annr and iteration >= 2:
            model_to_drop = should_drop_worst_model(
                iteration_results=iteration_results,
                current_models=current_models,
                iteration=iteration,
                min_models=4,
                threshold=0.1
            )
            
            if model_to_drop:
                # Get F1 scores for logging
                previous_iteration = iteration - 1
                if previous_iteration in iteration_results:
                    experiment_dir = iteration_results[previous_iteration]['annotation_results'].get('experiment_directory')
                    if experiment_dir:
                        model_f1_scores = extract_model_pairwise_strict_f1(experiment_dir)
                        log_model_dropping_decision(
                            iteration=iteration,
                            current_models=current_models,
                            model_to_drop=model_to_drop,
                            model_f1_scores=model_f1_scores,
                            threshold=0.05
                        )
                
                # Drop the model
                current_models.remove(model_to_drop)
                dropped_models_history.append({
                    'iteration': iteration,
                    'dropped_model': model_to_drop,
                    'timestamp': datetime.now().isoformat(),
                    'remaining_models': current_models.copy()
                })
            else:
                log_model_dropping_decision(
                    iteration=iteration,
                    current_models=current_models,
                    model_to_drop=None
                )
        
        # Auto-increment group index for each iteration
        current_group_index = starting_group_index + iteration
        
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration} - ANNOTATION PHASE (Group {current_group_index})")
        print(f"{'='*80}")
        print(f"Active models: {current_models}")
        print(f"Supervision mode: {supervision_mode}")
        
        # Get supervisor results path from previous iteration
        supervisor_results_path = None
        if iteration > 0:
            previous_group_index = starting_group_index + (iteration - 1)
            supervisor_results_path = get_previous_iteration_supervisor_path(
                benchmark=benchmark,
                num_groups=experiment_kwargs.get('num_groups', 20),
                group_size=experiment_kwargs.get('group_size', 50),
                current_iteration=iteration,
                previous_group_index=previous_group_index,
                models_count=models_count or len(original_models),
                supervisor_model_name=supervisor_model_name,
                max_common_instructions=max_common_instructions,
                max_patterns=max_patterns,
                model_specific_for_all=model_specific_for_all,
                max_model_specific_instructions=max_model_specific_instructions,
                limit_instruction_changes=limit_instruction_changes,
                max_change_ratio=max_change_ratio,
                drop_worst_annr=drop_worst_annr,
                supervised_by_gold_standard=supervised_by_gold_standard,
                skip_final_goal_update=skip_final_goal_update,
                llm_family_config=llm_family_config
            )
            print(f"supervisor_results_path: {supervisor_results_path}")

            if supervisor_results_path:
                print(f"Using supervisor guidance from: {supervisor_results_path}")
                if not validate_supervisor_file(supervisor_results_path):
                    print("Warning: Supervisor file validation failed, proceeding without guidance")
                    supervisor_results_path = None
            else:
                print("Warning: No supervisor results found from previous iteration")

        # Remove group_index from experiment_kwargs to avoid duplicate
        filtered_kwargs = {k: v for k, v in experiment_kwargs.items() if k != 'group_index'}
        
        # Run annotation experiment for this iteration (enhanced with gold standard support)
        try:
            experiment_result = run_experiment(
                benchmark=benchmark,
                dataset_path=dataset_path,
                models=current_models,
                ner_scheme=ner_scheme,
                group_index=current_group_index,
                iteration_number=iteration,
                supervisor_results_path=supervisor_results_path,
                models_count=models_count or len(original_models),
                supervisor_model_name=supervisor_model_name,
                llm_infer_by_openrouter=llm_infer_by_openrouter,
                supervised_by_gold_standard=supervised_by_gold_standard,
                gold_standard_config=gold_standard_config,
                max_common_instructions=max_common_instructions,
                max_patterns=max_patterns,
                model_specific_for_all=model_specific_for_all,
                max_model_specific_instructions=max_model_specific_instructions,
                limit_instruction_changes=limit_instruction_changes,
                max_change_ratio=max_change_ratio,
                drop_worst_annr=drop_worst_annr,
                llm_family_config=llm_family_config,
                skip_final_goal_update=skip_final_goal_update,
                **filtered_kwargs
            )
            
            iteration_results[iteration] = {
                'annotation_results': experiment_result,
                'supervisor_results': None,
                'convergence_analysis': None,
                'group_index': current_group_index,
                'active_models': current_models.copy(),
                'dropped_models_this_iteration': [h for h in dropped_models_history if h['iteration'] == iteration],
                'supervised_by_gold_standard': supervised_by_gold_standard
            }
            
            # Calculate F1 scores for convergence check
            current_f1_scores = {}
            for model_name, result in experiment_result['experiment_results']['results_by_model'].items():
                current_f1_scores[model_name] = result.get('avg_metrics', {}).get('f1', 0.0)
            
            print(f"\nIteration {iteration} F1 Scores (Group {current_group_index}):")
            for model_name, f1_score in current_f1_scores.items():
                improvement = ""
                if iteration > 0 and model_name in previous_f1_scores:
                    diff = f1_score - previous_f1_scores[model_name]
                    improvement = f" (Δ: {diff:+.3f})"
                print(f"  {model_name}: {f1_score:.3f}{improvement}")
            
            previous_f1_scores = current_f1_scores.copy()
            
        except Exception as e:
            print(f"Annotation experiment failed at iteration {iteration}: {e}")
            iteration_results[iteration] = {
                'annotation_results': None,
                'supervisor_results': None,
                'error': str(e),
                'group_index': current_group_index,
                'active_models': current_models.copy(),
                'supervised_by_gold_standard': supervised_by_gold_standard
            }
            break
        
        # Run supervisor analysis (except for last iteration)
        if iteration < max_iterations - 1:
            print(f"\n{'='*80}")
            supervision_mode = "Gold Standard Supervision" if supervised_by_gold_standard else "Disagreement-based Supervision"
            print(f"ITERATION {iteration} - SUPERVISOR PHASE ({supervision_mode})")
            print(f"{'='*80}")
            
            try:
                experiment_dir = Path(experiment_result['experiment_directory'])
                supervisor_output_dir = str(experiment_dir / 'supervisor_results')
                error_analysis_dir = experiment_dir / 'error_analysis'
                
                # UNIFIED: Always use hotspot disagreement documentation (works for both modes)
                disagreement_doc_path = experiment_dir / 'disagreement_analysis' / 'hotspot_docs' / 'hotspot_disagreement_analysis.md'
                
                if disagreement_doc_path.exists() and error_analysis_dir.exists():
                    # SIMPLIFIED supervisor call - unified path for both modes
                    supervisor_result = run_supervisor_analysis(
                        disagreement_doc_path=str(disagreement_doc_path),  # Same file for both modes
                        error_analysis_dir=str(error_analysis_dir),
                        analysis_type='gold_standard' if supervised_by_gold_standard else 'disagreement',
                        ner_scheme=ner_scheme,
                        final_goal=experiment_kwargs.get('final_task_goal', ''),
                        model_name=supervisor_model_name,
                        num_groups=experiment_kwargs.get('num_groups', 20),
                        group_size=experiment_kwargs.get('group_size', 50),
                        num_models=len(original_models),
                        iteration_number=iteration,
                        group_index=current_group_index,
                        base_output_dir=supervisor_output_dir,
                        prompts_config_path=prompts_config_path,
                        verbose=1,
                        skip_final_goal_update=skip_final_goal_update,
                        use_cache=True,
                        max_common_instructions=max_common_instructions,
                        max_patterns=max_patterns,
                        model_specific_for_all=model_specific_for_all,
                        max_model_specific_instructions=max_model_specific_instructions,
                        limit_instruction_changes=limit_instruction_changes,
                        max_change_ratio=max_change_ratio
                    )
                    
                    iteration_results[iteration]['supervisor_results'] = supervisor_result
                    print(f"{supervision_mode} supervisor analysis completed for iteration {iteration}")
                else:
                    print(f"Warning: Required analysis files not found for supervisor phase")
                    print(f"  Disagreement doc path: {disagreement_doc_path}")
                    print(f"  Error analysis dir exists: {error_analysis_dir.exists() if error_analysis_dir else False}")
                    
            except Exception as e:
                print(f"Error in supervisor phase: {e}")
                iteration_results[iteration]['supervisor_results'] = None
    
    # Add model dropping and supervision mode summary to final results
    for iteration_num, result in iteration_results.items():
        if 'annotation_results' in result and result['annotation_results']:
            result['annotation_results']['model_dropping_summary'] = {
                'original_models': original_models,
                'final_active_models': current_models,
                'dropped_models_history': dropped_models_history,
                'drop_worst_annr_enabled': drop_worst_annr
            }
            result['annotation_results']['supervision_mode'] = supervision_mode
            result['annotation_results']['supervised_by_gold_standard'] = supervised_by_gold_standard
    
    return iteration_results

def validate_supervisor_file(supervisor_path: str) -> bool:
    """Validate that supervisor file exists and has the expected structure"""
    try:
        if not os.path.exists(supervisor_path):
            return False
            
        with open(supervisor_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check for enhanced_guidelines (4-phase structure)
        enhanced_guidelines = data.get("enhanced_guidelines", {})
        if enhanced_guidelines:
            has_hierarchical = "hierarchical_common_instructions" in enhanced_guidelines
            has_prioritized = "prioritized_model_instructions" in enhanced_guidelines
            
            if has_hierarchical or has_prioritized:
                return True
        
        # Check alternative structure
        if "final_guidelines" in data:
            return True
            
        return False
        
    except Exception as e:
        print(f"Supervisor file validation error: {e}")
        return False

def main_iterative_experiment(
    benchmark: str,
    starting_group_index: int = 0,
    max_iterations: int = 3,
    convergence_threshold: Optional[float] = 0.05,
    prompts_config_path: str = "prompts/instruction_supervision_0905.json",
    skip_final_goal_update: bool = False,
    num_models: Optional[int] = None,
    llm_infer_by_openrouter: bool = False,
    supervised_by_gold_standard: bool = False,
    gold_standard_config: Dict[str, Any] = None,
    max_common_instructions: int = 5,
    max_patterns: int = 10,
    model_specific_for_all: bool = False,
    max_model_specific_instructions: int = 3,
    limit_instruction_changes: bool = False,
    max_change_ratio: float = 0.2,
    drop_worst_annr: bool = False,
    prefer_paid_models: bool = True,
    supervisor_model_name: str = "gpt-5-2025-08-07",
    llm_family_config: Optional[str] = None,
    **kwargs
):
    """Main iterative experiment function with gold standard support"""
    if llm_family_config:
        config_path = f'experiment_settings/{benchmark}_default_config_{llm_family_config}_family.json'
    else:
        config_path = f'experiment_settings/{benchmark}_default_config.json'
    print(f"Loading experiment config from: {config_path}")
    config = load_config(config_path)

    # Model selection logic
    if num_models is not None:
        available_models = config.get('models_ollama', config.get('models', []))
        if not available_models:
            raise ValueError("No models found in config under 'models_ollama' or 'models'")
        if num_models <= 0:
            raise ValueError(f"num_models must be positive, got: {num_models}")
        if num_models > len(available_models):
            raise ValueError(
                f"Requested {num_models} models but only {len(available_models)} available in config"
            )
        
        selected_models = available_models[:num_models]
        print(f"Using {num_models} models from config: {selected_models}")
        
        kwargs['models'] = selected_models
        config['models'] = selected_models
        models_count = num_models        
    else:
        models = kwargs.get('models', config['models'])
        models_count = len(models)
        print(f"Using all {models_count} models from config")

    # Setup experiment parameters
    experiment_params = {**config['experiment']}
    print(f"benchmark: {benchmark}")
    print(f"dataset_path: {experiment_params['dataset_path']}")
    for key in ['dataset_path', 'num_groups', 'group_size', 'grouping_model_name',
                'batch_size', 'device', 'use_gpu_clustering', 'random_seed']:
        if key in kwargs:
            experiment_params[key] = kwargs[key]
    
    if 'group_size' not in experiment_params or experiment_params['group_size'] is None:
        experiment_params['group_size'] = config['experiment'].get('group_size', 50)

    # Extract configurations
    model_source_map = config['model_source_map']
    models = kwargs.get('models', config['models'])
    ner_scheme = kwargs.get('ner_scheme', config['ner_scheme'])
    
    # Setup gold standard config if not provided
    if supervised_by_gold_standard and gold_standard_config is None:
        gold_standard_config = {
            'error_weight_strategy': 'uniform',
            'focus_on_recall': True,
            'include_boundary_analysis': True,
            'individual_model_reports': True,
            'context_analysis': True
        }
    
    print("Experiment configuration:")
    print(f"  Skip final goal update: {skip_final_goal_update}")
    print(f"  Models count: {models_count}")
    print(f"  Supervised by gold standard: {supervised_by_gold_standard}")
    if supervised_by_gold_standard:
        print(f"  Gold standard config: {gold_standard_config}")
    
    if llm_infer_by_openrouter:
        llm_clients.PREFER_PAID_MODELS = prefer_paid_models
        print("  OpenRouter inference: ENABLED")
        print(f"  OpenRouter model preference: {'Paid first' if prefer_paid_models else 'Free first'}")
    
    # Merge analysis parameters
    analysis_params = {**config['analysis'], **{k: v for k, v in kwargs.items() 
                                               if k.startswith('run_') or k in ['hotspot_percentile', 'coalition_cutoff']}}
    
    return run_iterative_annotation_supervisor_cycle(
        benchmark=benchmark,
        models=config['models'],
        model_source_map=model_source_map,
        ner_scheme=ner_scheme,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        prompts_config_path=prompts_config_path,
        starting_group_index=starting_group_index,
        final_task_goal=config['final_task_goal'],
        models_count=models_count,
        llm_infer_by_openrouter=llm_infer_by_openrouter,
        supervised_by_gold_standard=supervised_by_gold_standard,
        gold_standard_config=gold_standard_config,
        max_common_instructions=max_common_instructions,
        max_patterns=max_patterns,
        model_specific_for_all=model_specific_for_all,
        max_model_specific_instructions=max_model_specific_instructions,
        limit_instruction_changes=limit_instruction_changes,
        max_change_ratio=max_change_ratio,
        drop_worst_annr=drop_worst_annr,
        supervisor_model_name=supervisor_model_name,
        llm_family_config=llm_family_config,
        skip_final_goal_update=skip_final_goal_update,
        **experiment_params,
        **analysis_params
    )

# main_iterative_experiment(
#     # benchmark='crossner_conll2003',
#     benchmark='FabNER',
#     starting_group_index=0,
#     max_iterations=3,
#     convergence_threshold=0.05,
#     prompts_config_path="prompts/instruction_supervision_0905.json",
#     skip_final_goal_update=False,
#     num_models=None,
#     llm_infer_by_openrouter=True,
#     supervised_by_gold_standard=True,
#     gold_standard_config=None,
#     max_common_instructions=3,
#     max_patterns=5,
#     model_specific_for_all=False,
#     max_model_specific_instructions=2,
#     limit_instruction_changes=True,
#     max_change_ratio=0.1,
#     drop_worst_annr=False,
#     prefer_paid_models=True,
#     group_size=25,
#     )