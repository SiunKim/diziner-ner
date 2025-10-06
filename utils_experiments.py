import pickle
import random
import re
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import shutil
from pathlib import Path
from typing import Dict, List, Optional

# Confusing case analysis setting
ANALYZE_CONFUSING_CASES = False

def print_cost_estimate(results_by_model, iteration_number):
    """print_cost_estimate"""
    total_cost = 0
    total_tokens = 0
    total_calls = 0
    accurate_cost_calls = 0

    for model_name, result in results_by_model.items():
        if 'openrouter_usage' in result:
            usage = result['openrouter_usage']
            total_cost += usage['total_cost_usd']
            total_tokens += usage['total_tokens_used']
            total_calls += usage['api_calls_made']
            accurate_cost_calls += usage['api_cost_calls']

    print(f"\n{'='*60}")
    print(f"OPENROUTER COST SUMMARY - ITERATION {iteration_number}")
    print(f"{'='*60}")

    # Per-model breakdown
    print("Per-Model Costs:")
    for model_name, result in results_by_model.items():
        if 'openrouter_usage' in result:
            usage = result['openrouter_usage']
            cost = usage['total_cost_usd']
            tokens = usage['total_tokens_used']
            calls = usage['api_calls_made']
            accuracy = usage['cost_accuracy_rate']
            print(f"  {model_name:25} | ${cost:8.6f} | {tokens:8,} tokens | {calls:3} calls | {accuracy:5.1%} accurate")
        else:
            print(f"  {model_name:25} | No usage data available")

    print(f"{'-'*60}")
    print(f"Total Cost: ${total_cost:.6f}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Total API Calls: {total_calls}")
    if total_calls > 0:
        accuracy_rate = accurate_cost_calls / total_calls
        print(f"Cost Accuracy: {accuracy_rate:.1%} ({accurate_cost_calls}/{total_calls} accurate)")
        print(f"Average Cost per Call: ${total_cost/total_calls:.6f}")
    print(f"{'='*60}")

def get_previous_iteration_supervisor_path(
    benchmark: str,
    num_groups: int,
    group_size: int,
    current_iteration: int,
    previous_group_index: int = 0,
    models_count: Optional[int] = None,    
    supervisor_model_name: Optional[str] = None,
    max_common_instructions: int = 5,
    max_patterns: int = 10,
    model_specific_for_all: bool = False,
    max_model_specific_instructions: int = 3,
    supervised_by_gold_standard: bool = False,
    limit_instruction_changes: bool = False,
    max_change_ratio: float = 0.2,
    drop_worst_annr: bool = False,
    skip_final_goal_update: bool = False,
    llm_family_config: Optional[str] = None
    ) -> Optional[str]:
    """
    Find supervisor results from previous iteration - FIXED to use benchmark parameter
    """
    if current_iteration <= 0:
        return None

    if group_size is None:
        try:
            with open(f'experiment_settings/{benchmark}_default_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            group_size = config.get('experiment', {}).get('group_size', 50)
            print(f"DEBUG: Loaded group_size from config: {group_size}")
        except:
            group_size = 50
            print(f"DEBUG: Using fallback group_size: {group_size}")

    previous_iteration = current_iteration - 1
    supervisor_model = supervisor_model_name or "gpt-5-2025-08-07"

    # FIXED: Use benchmark directly instead of extracting from dataset_path
    root = Path('experiment_results') / benchmark
    model_roots = [root / f"models{models_count}"] if models_count is not None else list(root.glob("models*"))

    from supervisor_implementation import generate_experiment_suffix 
    enhanced_suffix = generate_experiment_suffix(
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

    for model_root in model_roots:
        exp_dir_name = f"g{num_groups}_s{group_size}_grp{previous_group_index}_iter{previous_iteration}{enhanced_suffix}"
        experiment_dir = model_root / supervisor_model / exp_dir_name
        candidate = experiment_dir / 'supervisor_results' / supervisor_model / 'comprehensive_results.json'
        print(f"DEBUG: Checking path: {candidate}")
        supervisor_file = experiment_dir / 'supervisor_results' / supervisor_model / 'comprehensive_results.json'
        if not supervisor_file.exists():
            return None
        if candidate.exists():
            print(f"Found supervisor results: {candidate}")
            return str(candidate)

    # FIXED: Update debug output to use benchmark
    base_results_dir = Path('experiment_results') / benchmark
    if base_results_dir.exists():
        available_dirs = [str(d.relative_to(base_results_dir)) for d in base_results_dir.rglob('*') if d.is_dir()]
        print("DEBUG: Available subdirs under benchmark root:")
        for d in available_dirs:
            print(f"  - {d}")

    return None

def normalize_hf_model_name(full_model_name: str) -> str:
    """Extract model name from HF path: nvidia/model -> model"""
    if '/' in full_model_name:
        return full_model_name.split('/', 1)[1]
    return full_model_name

def load_models_from_config(config_path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Load models from config and create source mapping
    
    Returns:
        all_models: 정규화된 모델명 리스트
        model_source_map: {정규화된_모델명: "ollama" or HF_전체경로}
    """
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)
    
    ollama_models = config.get("models_ollama", [])
    hf_models_full = config.get("models_huggingface", [])
    
    # Create combined list and source mapping
    all_models = []
    model_source_map = {}
    
    # Add Ollama models
    for model in ollama_models:
        all_models.append(model)
        model_source_map[model] = "ollama"
    
    # Add HF models (normalized names)
    for hf_model in hf_models_full:
        normalized_name = normalize_hf_model_name(hf_model)
        all_models.append(normalized_name)
        model_source_map[normalized_name] = hf_model  # Store full path
    
    return all_models, model_source_map

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)

    all_models, model_source_map = load_models_from_config(config_path)
    
    # Update config with combined models and source mapping
    config['models'] = all_models
    config['model_source_map'] = model_source_map

    return config

def extract_updated_final_goal_from_supervisor(supervisor_results_path: str, 
                                               original_final_goal: str) -> str:
    """Extract updated final goal from supervisor results"""
    if not supervisor_results_path or not os.path.exists(supervisor_results_path):
        return original_final_goal
    
    try:
        with open(supervisor_results_path, 'r', encoding='utf-8') as f:
            supervisor_data = json.load(f)
        
        # Check if goal updates were enabled
        metadata = supervisor_data.get("metadata", {})
        goal_behavior = metadata.get("goal_update_behavior", {})
        
        if goal_behavior.get("skip_final_goal_update", False) or goal_behavior.get("preserve_original_goal", False):
            print("Final goal updates were disabled in supervisor - using original goal")
            return original_final_goal
        
        # Try to extract from Phase 3 results
        phase3_data = supervisor_data.get("phase_results", {}).get("phase3", {})
        if phase3_data and phase3_data.get("success"):
            phase3_result_data = phase3_data.get("result_data", {})
            updated_goal_section = phase3_result_data.get("updated_final_goal", {})
            
            if updated_goal_section and isinstance(updated_goal_section, dict):
                updated_goal_text = updated_goal_section.get("updated_final_goal_text", "")
                
                if updated_goal_text and len(updated_goal_text.strip()) > 0:
                    print(f"Successfully extracted updated final goal from supervisor Phase 3")
                    return updated_goal_text.strip()
        
        return original_final_goal
        
    except Exception as e:
        print(f"Error extracting updated final goal from {supervisor_results_path}: {e}")
        return original_final_goal

def get_final_goal_for_iteration(iteration_number: int, 
                                supervisor_results_path: Optional[str] = None,
                                original_final_goal: str = "") -> str:
    """Get appropriate final goal for current iteration"""
    if iteration_number == 0:
        print("Iteration 0: Using original final goal")
        return original_final_goal
    
    if supervisor_results_path:
        updated_goal = extract_updated_final_goal_from_supervisor(
            supervisor_results_path, original_final_goal
        )
        if updated_goal != original_final_goal:
            print(f"Iteration {iteration_number}: Using updated final goal from supervisor")
            return updated_goal
        else:
            print(f"Iteration {iteration_number}: No goal update available, using original")
            return original_final_goal
    else:
        print(f"Iteration {iteration_number}: No supervisor results available, using original final goal")
        return original_final_goal

def print_experiment_summary(all_results: List[Dict[str, Any]]):
    """Print a comprehensive summary of all model experiments including confusing cases"""
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY - NER MODEL COMPARISON")
    print(f"{'='*80}")
    
    # Create comparison table
    if ANALYZE_CONFUSING_CASES:
        print(f"\n{'Model':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Success Rate':<12} {'Confusing Cases':<15}")
        print("-" * 85)
    else:
        print(f"\n{'Model':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Success Rate':<12}")
        print("-" * 65)
    
    for result in all_results:
        model_name = result['model_name']
        metrics = result['avg_metrics']
        success_rate = metrics['successful_samples'] / metrics['total_samples']
        
        if ANALYZE_CONFUSING_CASES:
            confusing_count = metrics.get('total_confusing_cases', 0)
            avg_confusing = metrics.get('avg_confusing_cases_per_sample', 0)
            print(f"{model_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1']:<10.3f} {success_rate:<12.1%} {confusing_count} ({avg_confusing:.1f}/sample)")
        else:
            print(f"{model_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1']:<10.3f} {success_rate:<12.1%}")
    
    # Find best model
    best_model = max(all_results, key=lambda x: x['avg_metrics']['f1'])
    print(f"\nBest Overall Model: {best_model['model_name']} (F1: {best_model['avg_metrics']['f1']:.3f})")
    
    # Per-type performance summary
    print(f"\n{'='*60}")
    print("PER-TYPE PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    entity_types = ['PER', 'ORG', 'LOC', 'MISC']
    
    for entity_type in entity_types:
        print(f"\n{entity_type} Entities:")
        print(f"{'Model':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 50)
        
        for result in all_results:
            model_name = result['model_name']
            per_type = result['avg_metrics'].get('per_type_avg', {})
            
            if entity_type in per_type:
                type_metrics = per_type[entity_type]
                print(f"{model_name:<15} {type_metrics['precision']:<10.3f} "
                      f"{type_metrics['recall']:<10.3f} {type_metrics['f1']:<10.3f}")
            else:
                print(f"{model_name:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

def get_experiment_paths(dataset_path: str,
                         num_groups: int,
                         group_size: int,
                         group_index: int = 0,
                         iteration_number: int = 0,
                         models_count: Optional[int] = None,
                         supervisor_model_name: str = "gpt-5-2025-08-07",
                         benchmark: Optional[str] = None) -> Dict[str, Path]:
    """
    Get experiment result paths for given configuration - FIXED for benchmark support
    """
    
    # FIXED: Use benchmark if provided, otherwise extract from dataset_path (legacy compatibility)
    if benchmark:
        dataset_name = benchmark
    else:
        dataset_name = Path(dataset_path).stem.replace('_ner_dataset', '')

    if iteration_number > 0:
        exp_name = f"g{num_groups}_s{group_size}_grp{group_index}_iter{iteration_number}"
    else:
        exp_name = f"g{num_groups}_s{group_size}_grp{group_index}"

    if models_count is not None:
        supervisor_safe_name = supervisor_model_name.replace(':', '_').replace('/', '_').replace(' ', '_')
        base_dir = Path('experiment_results') / dataset_name / f"models{models_count}" / supervisor_safe_name / exp_name
    else:
        # legacy
        base_dir = Path('experiment_results') / f"{dataset_name}_{exp_name}"

    paths = {
        'experiment_dir': base_dir,
        'model_results_dir': base_dir / 'model_results',
        'test_samples_file': base_dir / 'test_samples.pkl',
        'experiment_info_file': base_dir / 'experiment_info.json',
        'combined_results_file': base_dir / 'combined_results.json',
        'experiment_config_file': base_dir / 'experiment_config.json',
        'agreement_analysis_dir': base_dir / 'agreement_analysis',
        'disagreement_analysis_dir': base_dir / 'disagreement_analysis',
        'error_analysis_dir': base_dir / 'error_analysis',
        'supervisor_results_dir': base_dir / 'supervisor_results'
    }
    return paths


def get_model_result_path(experiment_dir: Path, model_name: str) -> Path:
    """
    Get model-specific result file path
    
    Args:
        experiment_dir: Experiment directory
        model_name: Name of the model
        
    Returns:
        Path to model result file
    """
    model_safe_name = model_name.replace(':', '_').replace('/', '_').replace(' ', '_')
    return experiment_dir / 'model_results' / f"{model_safe_name}.json"

def save_experiment_results(
    combined_results: Dict[str, Any], 
    experiment_paths: Dict[str, Path],
    group_index: int
) -> str:
    """Save experiment results to multiple locations and return main output file path"""
    
    # Save to experiment directory
    combined_results_file = experiment_paths['combined_results_file']
    with open(combined_results_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    # Save main results to agreement_analysis directory (no timestamp)
    agreement_dir = experiment_paths['agreement_analysis_dir']
    agreement_dir.mkdir(parents=True, exist_ok=True)
    main_output_file = str(agreement_dir / 'main_results.json')
    
    with open(main_output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    return main_output_file

def get_iteration_paths(
    benchmark: str,
    num_groups: int,
    group_size: int,
    iteration_number: int,
    group_index: int = 0,
    models_count: Optional[int] = None,
    supervisor_model_name: str = "gpt-5-2025-08-07",
    max_common_instructions: int = 5,
    drop_worst_annr: bool = False,
    max_patterns: int = 10,
    model_specific_for_all: bool = False,
    max_model_specific_instructions: int = 3,
    limit_instruction_changes: bool = False,
    max_change_ratio: float = 0.2,
    supervised_by_gold_standard: bool = False,
    skip_final_goal_update: bool = False,
    llm_family_config: Optional[str] = None
    ) -> Dict[str, Path]:
    """Get experiment paths for specific iteration - FIXED to use benchmark parameter"""

    if group_size is None:
        try:
            with open(f'experiment_settings/{benchmark}_default_config.json', 'r', encoding='utf-8-sig') as f:
                config = json.load(f)
            group_size = config.get('experiment', {}).get('group_size', 50)
        except:
            group_size = 50

    models_dir = f"models{models_count}" if models_count is not None else "modelsNA"
    
    supervisor_safe_name = supervisor_model_name.replace(':', '_').replace('/', '_').replace(' ', '_')
    
    from supervisor_implementation import generate_experiment_suffix
    enhanced_suffix = generate_experiment_suffix(
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
    experiment_name = f"g{num_groups}_s{group_size}_grp{group_index}_iter{iteration_number}{enhanced_suffix}"
    base_dir = Path('experiment_results') / benchmark / models_dir / supervisor_safe_name / experiment_name

    return {
        'experiment_dir': base_dir,
        'model_results_dir': base_dir / 'model_results',
        'test_samples_file': base_dir / 'test_samples.pkl',
        'experiment_info_file': base_dir / 'experiment_info.json',
        'combined_results_file': base_dir / 'combined_results.json',
        'experiment_config_file': base_dir / 'experiment_config.json',
        'agreement_analysis_dir': base_dir / 'agreement_analysis',
        'disagreement_analysis_dir': base_dir / 'disagreement_analysis',
        'error_analysis_dir': base_dir / 'error_analysis',
        'supervisor_results_dir': base_dir / 'supervisor_results'
    }

def save_iteration_metadata(iteration_data: Dict[str, Any], iteration_paths: Dict[str, Path]):
    """Save metadata about iteration including supervisor results paths"""
    metadata_file = iteration_paths['experiment_dir'] / 'iteration_metadata.json'
    
    metadata = {
        'iteration_number': iteration_data.get('iteration_number', 0),
        'timestamp': datetime.now().isoformat(),
        'supervisor_results_path': iteration_data.get('supervisor_results_path'),
        'experiment_config': iteration_data.get('experiment_config', {}),
        'models_tested': iteration_data.get('models', []),
        'performance_summary': iteration_data.get('performance_summary', {})
    }
    
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def create_grouped_samples_by_index(
    dataset_path: str,
    group_index: int,
    num_groups: int,
    group_size: int = None,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    device: str = 'auto',
    use_gpu_clustering: bool = True,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """Create test samples using lexical diversity grouping for specific group index"""
    print(f"Process: Create diverse groups → Take group {group_index} samples → Use for inference")
    
    if group_index < 0 or group_index >= num_groups:
        raise ValueError(f"group_index {group_index} must be between 0 and {num_groups-1}")
    
    random.seed(random_seed)
    
    # Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    training_data = dataset['train'] + dataset.get('valid', []) + dataset.get('validation', [])
    
    if group_size is None:
        group_size = len(training_data) // num_groups
    
    # Check for existing groups
    dataset_dir = Path(dataset_path).parent
    dataset_name = Path(dataset_path).stem.replace('_ner_dataset', '')
    groups_file = dataset_dir / f"{dataset_name}_groups_{num_groups}_size_{group_size}_groups.pkl"
    
    if groups_file.exists():
        with open(groups_file, 'rb') as f:
            groups = pickle.load(f)['groups']
        print(f"Loaded {len(groups)} existing groups")
    else:
        print("Creating new groups...")
        from lexical_diversity_grouping import create_lexical_diversity_groups
        results = create_lexical_diversity_groups(
            dataset_path=dataset_path,
            num_groups=num_groups,
            group_size=group_size,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            use_gpu_clustering=use_gpu_clustering
        )
        groups = results['groups']
    
    # Extract samples from specified group
    if not groups or len(groups) <= group_index or not groups[group_index]:
        raise ValueError(f"No valid group found at index {group_index}")
    
    selected_indices = groups[group_index]
    selected_samples = [training_data[idx] for idx in selected_indices]
    
    print(f"Using group {group_index}: {len(selected_samples)} samples")
    
    # Add metadata
    for i, sample in enumerate(selected_samples):
        sample['experiment_metadata'] = {
            'selection_method': f'lexical_diversity_grouping_group_{group_index}',
            'group_id': group_index,
            'original_training_index': selected_indices[i],
            'selection_timestamp': datetime.now().isoformat(),
            'grouping_parameters': {
                'num_groups': num_groups,
                'group_size': group_size,
                'model_name': model_name,
                'random_seed': random_seed
            }
        }
    
    return selected_samples

def create_combined_results_structure(
    experiment_info: Dict[str, Any],
    test_samples: List[Dict[str, Any]],
    results_by_model: Dict[str, Any],
    model_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create the standardized combined results dictionary structure"""
    return {
        'experiment_info': experiment_info,
        'test_samples': test_samples,
        'results_by_model': results_by_model,
        'model_results': model_results
    }

def find_existing_iter0_results(
    benchmark: str,
    models_count: int,
    num_groups: int,
    group_size: int,
    target_models: List[str],
    current_experiment_dir: Path
) -> Dict[str, str]:
    """
    Find existing iteration 0 results that can be reused
    
    Args:
        benchmark: Benchmark name (e.g., 'crossner_conll2003')
        models_count: Number of models (for models{n} folder)
        num_groups: Number of groups for grouping
        group_size: Size of each group
        target_models: List of model names to find results for
        current_experiment_dir: Current experiment directory to avoid self-reference
    
    Returns:
        Dictionary mapping {model_name: path_to_existing_json}
    """
    existing_results = {}
    
    # Search pattern: experiment_results/{benchmark}/models{models_count}/*/g{num_groups}_s{group_size}_grp*_iter0*
    base_results_dir = Path('experiment_results') / benchmark / f"models{models_count}"
    
    if not base_results_dir.exists():
        print(f"No base results directory found: {base_results_dir}")
        return existing_results
    
    # Search across all supervisor model directories
    for supervisor_dir in base_results_dir.iterdir():
        if not supervisor_dir.is_dir():
            continue
            
        # Look for iteration 0 experiment directories with matching group settings
        pattern = f"g{num_groups}_s{group_size}_grp*_iter0*"
        for exp_dir in supervisor_dir.glob(pattern):
            if not exp_dir.is_dir():
                continue
                
            # Skip if this is the current experiment directory
            if exp_dir.resolve() == current_experiment_dir.resolve():
                continue
                
            model_results_dir = exp_dir / 'model_results'
            if not model_results_dir.exists():
                continue
                
            # Check for each target model
            for model_name in target_models:
                if model_name in existing_results:
                    continue  # Already found this model
                    
                model_safe_name = model_name.replace(':', '_').replace('/', '_').replace(' ', '_')
                model_json_path = model_results_dir / f"{model_safe_name}.json"
                
                if model_json_path.exists():
                    # Validate the JSON file
                    if validate_model_result_file(model_json_path, model_name):
                        existing_results[model_name] = str(model_json_path)
                        print(f"Found existing iter0 result for {model_name}: {model_json_path}")
    
    return existing_results

def validate_model_result_file(json_path: Path, expected_model_name: str) -> bool:
    """
    Validate that a model result JSON file is valid and complete
    
    Args:
        json_path: Path to JSON file
        expected_model_name: Expected model name
        
    Returns:
        True if file is valid, False otherwise
    """
    try:
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # Check essential fields
        required_fields = ['model_name', 'avg_metrics', 'detailed_results']
        for field in required_fields:
            if field not in data:
                print(f"Missing required field '{field}' in {json_path}")
                return False
        
        # Check if model name matches (allow for normalization differences)
        file_model_name = data['model_name']
        if normalize_model_name(file_model_name) != normalize_model_name(expected_model_name):
            print(f"Model name mismatch in {json_path}: expected {expected_model_name}, got {file_model_name}")
            return False
        
        # Check if avg_metrics has essential metrics
        avg_metrics = data['avg_metrics']
        if not isinstance(avg_metrics, dict):
            print(f"Invalid avg_metrics format in {json_path}")
            return False
            
        essential_metrics = ['precision', 'recall', 'f1']
        for metric in essential_metrics:
            if metric not in avg_metrics:
                print(f"Missing metric '{metric}' in {json_path}")
                return False
        
        # Check if detailed_results is a list and not empty
        detailed_results = data['detailed_results']
        if not isinstance(detailed_results, list) or len(detailed_results) == 0:
            print(f"Invalid or empty detailed_results in {json_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating {json_path}: {e}")
        return False

def copy_iter0_model_results(
    existing_results: Dict[str, str],
    target_model_results_dir: Path
) -> List[str]:
    """
    Copy existing iteration 0 model results to target directory
    
    Args:
        existing_results: Dictionary mapping {model_name: source_json_path}
        target_model_results_dir: Target directory to copy results to
        
    Returns:
        List of successfully copied model names
    """
    successfully_copied = []
    
    # Ensure target directory exists
    target_model_results_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, source_path in existing_results.items():
        try:
            source_file = Path(source_path)
            if not source_file.exists():
                print(f"Source file no longer exists: {source_path}")
                continue
            
            # Create target filename
            model_safe_name = model_name.replace(':', '_').replace('/', '_').replace(' ', '_')
            target_file = target_model_results_dir / f"{model_safe_name}.json"
            
            # Copy the file
            shutil.copy2(source_file, target_file)
            successfully_copied.append(model_name)
            print(f"Copied iter0 result: {model_name} -> {target_file}")
            
        except Exception as e:
            print(f"Failed to copy result for {model_name}: {e}")
    
    return successfully_copied

def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name for comparison (moved from utils_annotator.py if needed)
    
    Args:
        model_name: Original model name
        
    Returns:
        Normalized model name for comparison
    """
    normalized = model_name.lower()
    normalized = normalized.replace(":", "_").replace("-", "_").replace("/", "_")
    normalized = re.sub(r'[^a-z0-9_]', '', normalized)
    return normalized

def load_existing_model_result(json_path: str, model_name: str) -> Dict[str, Any]:
    """
    Load existing model result from JSON file and ensure it has correct iteration_number
    
    Args:
        json_path: Path to existing JSON file
        model_name: Model name for validation
        
    Returns:
        Loaded model result dictionary with iteration_number set to 0
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # Ensure iteration_number is 0 for reused results
        result['iteration_number'] = 0
        result['reused_from_existing'] = True
        result['original_source_path'] = json_path
        
        # Update supervisor instructions summary for iteration 0
        if 'supervisor_instructions_summary' not in result:
            result['supervisor_instructions_summary'] = {
                'common_instructions_count': 0,
                'model_instructions_count': 0,
                'total_instructions_applied': 0
            }
        
        # Ensure avg_metrics has iteration_number
        if 'avg_metrics' in result:
            result['avg_metrics']['iteration_number'] = 0
        
        print(f"Successfully loaded existing result for {model_name} from {json_path}")
        return result
        
    except Exception as e:
        print(f"Error loading existing result for {model_name} from {json_path}: {e}")
        raise
