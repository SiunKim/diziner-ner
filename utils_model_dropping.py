"""
Utilities for model dropping functionality in iterative experiments
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def extract_model_pairwise_strict_f1(base_dir: str) -> Dict[str, float]:
    """
    Extract per-model average strict span F1 from pairwise agreements
    
    Args:
        base_dir (str): Base directory path containing pairwise agreements file
        
    Returns:
        Dict[str, float]: Dictionary with model names as keys and average strict span F1 as values
    """
    pairwise_agreements_file = Path(base_dir) / 'agreement_analysis' / 'main_results_analysis' / 'pairwise_agreements.json'
    
    if not pairwise_agreements_file.exists():
        return {}
    
    try:
        with open(pairwise_agreements_file, 'r', encoding='utf-8') as f:
            pairwise_agreements = json.load(f)
        
        model_strict_sums = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        
        for pair_key, agreement_record in pairwise_agreements.items():
            model1 = agreement_record.get('model1')
            model2 = agreement_record.get('model2')
            avg_agreement = agreement_record.get('avg_agreement', {})
            strict_span_f1 = avg_agreement.get('strict_span_f1', 0.0)
            
            if not model1 or not model2 or strict_span_f1 is None:
                continue
            
            for model in [model1, model2]:
                model_strict_sums[model]['sum'] += float(strict_span_f1)
                model_strict_sums[model]['count'] += 1
        
        model_strict_f1 = {}
        for model, aggregation in model_strict_sums.items():
            if aggregation['count'] > 0:
                model_strict_f1[model] = aggregation['sum'] / aggregation['count']
            else:
                model_strict_f1[model] = 0.0
        
        return model_strict_f1
        
    except Exception:
        return {}


def should_drop_worst_model(iteration_results: Dict[int, Dict], 
                           current_models: List[str], 
                           iteration: int,
                           min_models: int = 4,
                           threshold: float = 0.1) -> Optional[str]:
    """
    Determine if worst performing model should be dropped based on previous iteration results
    
    Args:
        iteration_results: Dictionary of iteration results
        current_models: List of currently active models
        iteration: Current iteration number
        min_models: Minimum number of models to maintain
        threshold: Minimum difference between worst and average to trigger drop
        
    Returns:
        Model name to drop or None if no model should be dropped
    """
    # Check basic conditions
    if iteration < 2:  # Start dropping from iteration 2
        return None
    
    if len(current_models) <= min_models:
        return None
    
    # Get previous iteration results
    previous_iteration = iteration - 1
    if previous_iteration not in iteration_results:
        return None
    
    previous_result = iteration_results[previous_iteration]
    annotation_results = previous_result.get('annotation_results', {})
    experiment_directory = annotation_results.get('experiment_directory')
    
    if not experiment_directory:
        return None
    
    # Extract pairwise strict F1 scores
    model_strict_f1 = extract_model_pairwise_strict_f1(experiment_directory)
    
    if not model_strict_f1:
        return None
    
    # Filter to only current models
    current_model_f1 = {model: f1 for model, f1 in model_strict_f1.items() 
                       if model in current_models}
    
    if len(current_model_f1) < 2:
        return None
    
    # Find worst model and calculate statistics
    f1_values = list(current_model_f1.values())
    average_f1 = np.mean(f1_values)
    
    worst_model = min(current_model_f1.items(), key=lambda x: x[1])
    worst_model_name, worst_f1 = worst_model
    
    # Check if difference exceeds threshold
    f1_difference = average_f1 - worst_f1
    
    if f1_difference >= threshold:
        return worst_model_name
    
    return None


def log_model_dropping_decision(iteration: int, 
                               current_models: List[str],
                               model_to_drop: Optional[str],
                               model_f1_scores: Dict[str, float] = None,
                               threshold: float = 0.05):
    """
    Log model dropping decision with relevant statistics
    
    Args:
        iteration: Current iteration number
        current_models: List of currently active models
        model_to_drop: Model to be dropped (or None)
        model_f1_scores: F1 scores for decision context
        threshold: Threshold used for decision
    """
    print(f"\n{'='*50}")
    print(f"MODEL DROPPING ANALYSIS - ITERATION {iteration}")
    print(f"{'='*50}")
    print(f"Current models: {len(current_models)}")
    print(f"Models: {current_models}")
    
    if model_f1_scores:
        f1_values = list(model_f1_scores.values())
        average_f1 = np.mean(f1_values)
        worst_model_name = min(model_f1_scores.items(), key=lambda x: x[1])[0]
        worst_f1 = model_f1_scores[worst_model_name]
        f1_difference = average_f1 - worst_f1
        
        print(f"F1 Statistics:")
        print(f"  Average F1: {average_f1:.4f}")
        print(f"  Worst model: {worst_model_name} (F1: {worst_f1:.4f})")
        print(f"  Difference: {f1_difference:.4f} (threshold: {threshold})")
    
    if model_to_drop:
        print(f"DECISION: Dropping {model_to_drop}")
        remaining_models = [m for m in current_models if m != model_to_drop]
        print(f"Remaining models ({len(remaining_models)}): {remaining_models}")
    else:
        print("DECISION: No model dropped")
        if len(current_models) <= 4:
            print("  Reason: Minimum model count reached")
        elif model_f1_scores:
            print(f"  Reason: F1 difference ({f1_difference:.4f}) below threshold ({threshold})")
        else:
            print("  Reason: No F1 data available")
    
    print(f"{'='*50}")
