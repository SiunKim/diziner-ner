import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from base_annotator import NERAgent
from utils_annotator import (
    convert_bio_to_entities, 
    convert_entities_to_bio, 
    calculate_ner_metrics,
    calculate_token_accuracy,
    aggregate_strict_span_metrics,
    VERBOSE,
    get_validation_failure_reason,
    validate_annotation_result,
    normalize_model_name
)

from utils_experiments import (
    ANALYZE_CONFUSING_CASES
    )
from debug import DEBUG

VERBOSE = 1


def run_model_annotation(model_name: str,
                         test_samples: List[Dict],
                         ner_scheme: Dict[str, str],
                         model_source_map: Optional[Dict[str, str]] = None,
                         final_task_goal: Optional[str] = None,
                         supervisor_results_path: Optional[str] = None,
                         iteration_number: int = 0,
                         llm_infer_by_openrouter: bool = False,
                         skip_final_goal_update: bool = False,
                         max_annotation_retries: int = 3) -> Dict[str, Any]:
    """Run NER annotation with HF support"""
    if DEBUG:
        print(f"DEBUG: run_model_annotation called")
        print(f"  model_name: {model_name}")
        print(f"  iteration_number: {iteration_number}")
        print(f"  supervisor_results_path: {supervisor_results_path}")
        print(f"  supervisor_results_path exists: {os.path.exists(supervisor_results_path) if supervisor_results_path else 'N/A'}")

    if VERBOSE >= 1:
        print(f"\n{'='*60}")
        print(f"Testing Model: {model_name} (Iteration {iteration_number})")
        if llm_infer_by_openrouter:
            print("Inference method: OpenRouter")
        print(f"{'='*60}")
        
        # Show model source
        if model_source_map and model_name in model_source_map:
            source = model_source_map[model_name]
            if source == "ollama":
                print("Model source: Ollama" + (" -> OpenRouter" if llm_infer_by_openrouter else ""))
            else:
                print(f"Model source: Hugging Face ({source})")
        
        if ANALYZE_CONFUSING_CASES:
            print("Confusing case analysis: ENABLED")
        if final_task_goal:
            print("Task goal provided: YES")
        if supervisor_results_path and iteration_number > 0:
            print(f"Supervisor instructions: {supervisor_results_path}")
        print(f"Max annotation retries: {max_annotation_retries}")
    
    # Load supervisor instructions
    supervisor_common_instructions = ""
    supervisor_model_instructions = ""
    if supervisor_results_path and iteration_number > 0:
        supervisor_common_instructions, supervisor_model_instructions = load_supervisor_instructions_from_file(
            supervisor_results_path, model_name
        )
        if DEBUG:
            print(f"DEBUG: Loaded supervisor instructions - common: {len(supervisor_common_instructions)} chars, model: {len(supervisor_model_instructions)} chars")
    
    # Initialize NER agent with model source mapping
    agent = NERAgent(
        model_name=model_name,
        ner_scheme=ner_scheme,
        model_source_map=model_source_map,
        final_task_goal=final_task_goal,
        supervisor_common_instructions=supervisor_common_instructions,
        supervisor_model_instructions=supervisor_model_instructions,
        iteration_number=iteration_number,
        llm_infer_by_openrouter=llm_infer_by_openrouter,
        skip_final_goal_update=skip_final_goal_update,
        verbose=VERBOSE
    )

    results = []
    all_confusing_cases = []
    sample_times = []
    retry_stats = {'total_retries': 0, 'samples_requiring_retries': 0, 'validation_failures': 0}
    
    # For aggregated metrics calculation
    all_predicted_entities = []
    all_gold_entities = []
    all_predicted_labels = []
    all_gold_labels = []
    
    experiment_start_time = time.time()
    
    for i, sample in enumerate(test_samples):
        if VERBOSE >= 1:
            print(f"Processing sample {i+1}/{len(test_samples)}")
        
        sample_result = process_single_sample_with_retry(
            agent, sample, i, max_annotation_retries
        )
        
        retry_attempt = sample_result.get('retry_attempt', 0)
        if retry_attempt > 0:
            retry_stats['samples_requiring_retries'] += 1
            retry_stats['total_retries'] += retry_attempt
        
        if 'error' in sample_result:
            retry_stats['validation_failures'] += 1
        
        # Collect data for aggregated metrics
        if 'error' not in sample_result:
            predicted_entities = sample_result.get('predicted_entities', [])
            gold_entities = sample_result.get('gold_entities', [])
            predicted_labels = sample_result.get('predicted_labels', [])
            gold_labels = sample_result.get('labels', [])
            
            all_predicted_entities.append(predicted_entities)
            all_gold_entities.append(gold_entities)
            all_predicted_labels.extend(predicted_labels)
            all_gold_labels.extend(gold_labels)
        
        confusing_cases = sample_result.get('confusing_cases', [])
        if confusing_cases:
            all_confusing_cases.extend(confusing_cases)
        
        sample_times.append(sample_result.get('processing_time_seconds', 0.0))
        results.append(sample_result)
        
        if VERBOSE == 1:
            # Get legacy metrics for display compatibility
            legacy_metrics = sample_result.get('metrics', {'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
            confusing_info = f", Confusing: {len(confusing_cases)}" if ANALYZE_CONFUSING_CASES else ""
            retry_info = f", Retry: {retry_attempt}" if retry_attempt > 0 else ""
            error_info = " [ERROR]" if 'error' in sample_result else ""
            
            gold_count = len(sample_result.get('gold_entities', []))
            pred_count = len(sample_result.get('predicted_entities', []))
            f1_score = legacy_metrics.get('f1', 0.0)
            processing_time = sample_result.get('processing_time_seconds', 0.0)
            print(f"  Gold: {gold_count}, Predicted: {pred_count}, F1: {f1_score:.3f}{confusing_info}{retry_info}, Time: {processing_time:.2f}s{error_info}")
        elif VERBOSE >= 2:
            legacy_metrics = sample_result.get('metrics', {'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
            print(f"  Gold entities: {len(sample_result.get('gold_entities', []))}")
            print(f"  Predicted entities: {len(sample_result.get('predicted_entities', []))}")
            print(f"  F1 Score: {legacy_metrics.get('f1', 0.0):.3f}")
            print(f"  Processing time: {sample_result.get('processing_time_seconds', 0.0):.2f} seconds")
            if retry_attempt > 0:
                print(f"  Retry attempts: {retry_attempt}")
            if 'error' in sample_result:
                print(f"  Error: {sample_result['error']}")
            if ANALYZE_CONFUSING_CASES:
                print(f"  Confusing cases: {len(confusing_cases)}")
            if iteration_number > 0:
                print(f"  Supervisor instructions applied: {len(supervisor_common_instructions)} common, {len(supervisor_model_instructions)} model-specific")
    
    # Record total experiment time
    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time
    
    # Calculate timing statistics
    if sample_times:
        avg_time_per_sample = sum(sample_times) / len(sample_times)
        min_time_per_sample = min(sample_times)
        max_time_per_sample = max(sample_times)
        total_processing_time = sum(sample_times)
    else:
        avg_time_per_sample = 0.0
        min_time_per_sample = 0.0
        max_time_per_sample = 0.0
        total_processing_time = 0.0
    
    # Calculate new aggregated strict span metrics
    entity_types = list(ner_scheme.keys())
    aggregated_metrics = aggregate_strict_span_metrics(
        all_predicted_entities, all_gold_entities, entity_types
    )
    
    # Calculate token accuracy
    token_accuracy = calculate_token_accuracy(all_predicted_labels, all_gold_labels)
    
    # Create avg_metrics with backward compatibility
    # Use micro metrics as the main metrics for backward compatibility
    micro_metrics = aggregated_metrics['micro']
    macro_metrics = aggregated_metrics['macro']
    per_type_metrics = aggregated_metrics['per_type']
    
    avg_metrics = {
        # Main metrics (backward compatible) - use micro metrics
        'precision': micro_metrics['precision'],
        'recall': micro_metrics['recall'], 
        'f1': micro_metrics['f1'],
        'token_accuracy': token_accuracy,
        
        # NEW: Detailed strict span metrics
        'strict_span_metrics': {
            'micro': micro_metrics,
            'macro': macro_metrics,
            'per_type': per_type_metrics
        },
        
        # Existing fields for backward compatibility
        'total_samples': len(test_samples),
        'successful_samples': len([r for r in results if 'error' not in r]),
        'avg_time_per_sample_seconds': avg_time_per_sample,
        'min_time_per_sample_seconds': min_time_per_sample,
        'max_time_per_sample_seconds': max_time_per_sample,
        'total_processing_time_seconds': total_processing_time,
        'total_experiment_time_seconds': total_experiment_time,
        'iteration_number': iteration_number,
        'supervisor_instructions_applied': {
            'common_instructions_count': len(supervisor_common_instructions),
            'model_instructions_count': len(supervisor_model_instructions)
        },
        'retry_statistics': retry_stats,
        
        # Additional aggregated metrics info
        'total_entities_gold': aggregated_metrics['total_gold'],
        'total_entities_predicted': aggregated_metrics['total_predicted'],
        'entity_type_count': len(entity_types)
    }

    # Add confusing case statistics
    if ANALYZE_CONFUSING_CASES:
        avg_metrics['total_confusing_cases'] = len(all_confusing_cases)
        avg_metrics['avg_confusing_cases_per_sample'] = len(all_confusing_cases) / len(test_samples)
    
    # Calculate per-type averages (legacy format for backward compatibility)
    per_type_avg = {}
    for entity_type, metrics in per_type_metrics.items():
        per_type_avg[entity_type] = {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'avg_support': metrics['support']
        }
    
    avg_metrics['per_type_avg'] = per_type_avg
    
    # Enhanced timing summary with retry information
    if VERBOSE >= 1:
        iteration_info = f" (Iteration {iteration_number})" if iteration_number > 0 else ""
        supervisor_info = f" with {len(supervisor_common_instructions)+len(supervisor_model_instructions)} supervisor instructions" if iteration_number > 0 else ""
        retry_info = f", Retries: {retry_stats['samples_requiring_retries']}/{len(test_samples)} samples ({retry_stats['total_retries']} total attempts)" if retry_stats['total_retries'] > 0 else ""
        
        print(f"\n--- SUMMARY for {model_name}{iteration_info} ---")
        print(f"Total time: {total_experiment_time:.2f}s, Avg per sample: {avg_time_per_sample:.2f}s{supervisor_info}{retry_info}")
        print(f"Strict Span Metrics - Micro F1: {micro_metrics['f1']:.3f}, Macro F1: {macro_metrics['f1']:.3f}")
        if retry_stats['validation_failures'] > 0:
            print(f"Validation failures: {retry_stats['validation_failures']}/{len(test_samples)} samples")

    final_result = {
        'model_name': model_name,
        'avg_metrics': avg_metrics,
        'all_confusing_cases': all_confusing_cases,
        'detailed_results': results,
        'experiment_timestamp': datetime.now().isoformat(),
        'iteration_number': iteration_number,
        'supervisor_results_path': supervisor_results_path,
        'supervisor_instructions_summary': {
            'common_instructions_count': len(supervisor_common_instructions),
            'model_instructions_count': len(supervisor_model_instructions),
            'total_instructions_applied': len(supervisor_common_instructions) + len(supervisor_model_instructions)
        }
    }
    
    if llm_infer_by_openrouter and hasattr(agent.client, 'get_usage_stats'):
        final_result['openrouter_usage'] = agent.client.get_usage_stats()
        if hasattr(agent.client, 'get_error_statistics'):
            final_result['openrouter_errors'] = agent.client.get_error_statistics()
    
    return final_result
    


def process_single_sample_with_retry(agent: NERAgent, sample: Dict,
                                     sample_index: int, 
                                   max_retries: int = 3) -> Dict[str, Any]:
    """
    Process a single sample with retry logic for validation failures
    
    Args:
        agent: NER agent instance
        sample: Sample data
        sample_index: Index of the sample
        max_retries: Maximum number of retry attempts
        
    Returns:
        Valid sample result or error result
    """
    expected_token_count = len(sample['tokens'])
    
    for attempt in range(max_retries + 1):
        sample_start_time = time.time()
        
        try:
            # Clear conversation history for each retry to avoid context contamination
            agent._clear_conversation_history()
            
            if VERBOSE >= 2 and attempt > 0:
                print(f"    Retry attempt {attempt}/{max_retries}")
            
            # Extract entities using model with confusing case analysis
            model_result = agent.extract_entities(
                sample['text'], 
                analyze_confusing_cases=ANALYZE_CONFUSING_CASES,
                return_final_prompt=True
            )
            predicted_entities = model_result['entities']
            confusing_cases = model_result.get('confusing_cases', [])
            
            # Convert gold standard BIO tags to entity format
            gold_entities = convert_bio_to_entities(sample['tokens'], sample['labels'])
            
            # Calculate metrics for this sample using legacy function for compatibility
            legacy_metrics = calculate_ner_metrics(predicted_entities, gold_entities)
            
            # Convert predicted entities back to BIO format
            valid_entity_types = set(agent.ner_scheme.keys())
            predicted_labels = convert_entities_to_bio(
                                    sample['tokens'], 
                                    predicted_entities, 
                                    sample['text'],
                                    valid_entity_types=valid_entity_types
                                )
            
            # Record sample processing time
            sample_end_time = time.time()
            sample_duration = sample_end_time - sample_start_time
            
            # Create sample result
            sample_result = {
                'sample_id': sample_index,
                'text': sample['text'],
                'tokens': sample['tokens'],
                'labels': sample['labels'],
                'predicted_labels': predicted_labels,
                'gold_entities': gold_entities,
                'predicted_entities': predicted_entities,
                'confusing_cases': confusing_cases,
                'metrics': legacy_metrics,  # Keep legacy format for individual samples
                'model_response': model_result,
                'final_prompt': model_result.get('final_prompt', ''),
                'processing_time_seconds': sample_duration,
                'iteration_number': getattr(agent, 'iteration_number', 0),
                'supervisor_instructions_applied': {
                    'common_instructions_count': len(getattr(agent, 'supervisor_common_instructions', [])),
                    'model_instructions_count': len(getattr(agent, 'supervisor_model_instructions', []))
                },
                'retry_attempt': attempt  # Track which attempt succeeded
            }
            
            # Validate the result
            if validate_annotation_result(sample_result, expected_token_count):
                if VERBOSE >= 2 and attempt > 0:
                    print(f"    Validation passed on attempt {attempt + 1}")
                return sample_result
            else:
                if attempt < max_retries:
                    if VERBOSE >= 1:
                        # Get more specific failure reason from validation
                        failure_reason = get_validation_failure_reason(sample_result, expected_token_count)
                        print(f"    RETRY: {failure_reason} (attempt {attempt + 1}/{max_retries + 1})")
                    continue
                else:
                    # All retries exhausted, return error result
                    if VERBOSE >= 1:
                        print(f"    All {max_retries + 1} attempts failed validation")
                    
                    error_result = {
                        'sample_id': sample_index,
                        'text': sample['text'],
                        'tokens': sample.get('tokens', []),
                        'labels': sample.get('labels', []),
                        'predicted_labels': ['O'] * len(sample['tokens']),  # Fallback to all O labels
                        'gold_entities': gold_entities,
                        'predicted_entities': [],
                        'confusing_cases': [],
                        'error': f'Validation failed after {max_retries + 1} attempts: label/token mismatch',
                        'metrics': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                        'processing_time_seconds': sample_duration,
                        'iteration_number': getattr(agent, 'iteration_number', 0),
                        'supervisor_instructions_applied': {
                            'common_instructions_count': len(getattr(agent, 'supervisor_common_instructions', [])),
                            'model_instructions_count': len(getattr(agent, 'supervisor_model_instructions', []))
                        },
                        'retry_attempts_made': max_retries + 1
                    }
                    return error_result
                    
        except Exception as e:
            sample_end_time = time.time()
            sample_duration = sample_end_time - sample_start_time
            
            if attempt < max_retries:
                if VERBOSE >= 1:
                    print(f"    RETRY: Exception occurred - {str(e)} (attempt {attempt + 1}/{max_retries + 1})")
                continue
            else:
                # All retries exhausted due to exceptions
                if VERBOSE >= 1:
                    print(f"    All {max_retries + 1} attempts failed with exceptions")
                
                error_result = {
                    'sample_id': sample_index,
                    'text': sample['text'],
                    'tokens': sample.get('tokens', []),
                    'labels': sample.get('labels', []),
                    'predicted_labels': ['O'] * len(sample['tokens']),  # Fallback to all O labels
                    'gold_entities': convert_bio_to_entities(sample['tokens'], sample['labels']),
                    'predicted_entities': [],
                    'confusing_cases': [],
                    'error': f'Exception after {max_retries + 1} attempts: {str(e)}',
                    'metrics': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                    'processing_time_seconds': sample_duration,
                    'iteration_number': getattr(agent, 'iteration_number', 0),
                    'supervisor_instructions_applied': {
                        'common_instructions_count': len(getattr(agent, 'supervisor_common_instructions', [])),
                        'model_instructions_count': len(getattr(agent, 'supervisor_model_instructions', []))
                    },
                    'retry_attempts_made': max_retries + 1
                }
                return error_result
    
    # This should never be reached, but just in case
    raise Exception("Unexpected error in retry logic")


def format_hierarchical_instructions(hierarchical_instructions: List[Dict]) -> str:
    """
    Format hierarchical instruction structure into readable text format
    
    Args:
        hierarchical_instructions: List of hierarchical instruction objects
        
    Returns:
        Formatted instruction text with proper indentation and examples
    """
    formatted_lines = []
    
    for main_instruction in hierarchical_instructions:
        if not isinstance(main_instruction, dict):
            continue
            
        # Add main instruction
        main_text = main_instruction.get("instruction_text", "")
        instruction_number = main_instruction.get("instruction_number", "")
        
        if main_text:
            formatted_lines.append(f"{instruction_number}. {main_text}")
        
        # Add sub-instructions with indentation
        sub_instructions = main_instruction.get("sub_instructions", [])
        for sub_instr in sub_instructions:
            if isinstance(sub_instr, dict):
                sub_text = sub_instr.get("instruction_text", "")
                if sub_text:
                    # Add sub-instruction with tab indentation
                    formatted_lines.append(f"   - {sub_text}")
                    
                    # Add examples if present
                    examples = sub_instr.get("examples", [])
                    if examples:
                        formatted_lines.append("       * Examples:")
                        for i, example in enumerate(examples):
                            if isinstance(example, dict):
                                text = example.get("text", "")
                                annotation = example.get("correct_annotation", "")
                                explanation = example.get("explanation", "")
                                
                                formatted_lines.append(f"         {i+1}. \"{text}\" -> {annotation}")
                                if explanation:
                                    formatted_lines.append(f"            Rationale: {explanation}")
    
    return "\n".join(formatted_lines)


def format_model_specific_instructions(model_instructions: List[Dict]) -> str:
    """
    Format model-specific instructions into readable text format
    
    Args:
        model_instructions: List of model-specific instruction objects
        
    Returns:
        Formatted instruction text
    """
    formatted_lines = []
    
    for i, instr in enumerate(model_instructions, 1):
        if isinstance(instr, dict):
            text = instr.get("instruction_text", "")
            if text:
                formatted_lines.append(f"{i}. {text}")
                
                # Add examples if present
                examples = instr.get("examples", [])
                if examples:
                    formatted_lines.append("   * Examples:")
                    for j, example in enumerate(examples):
                        if isinstance(example, dict):
                            ex_text = example.get("text", "")
                            annotation = example.get("correct_annotation", "")
                            
                            formatted_lines.append(f"     {j+1}. \"{ex_text}\" -> {annotation}")
    
    return "\n".join(formatted_lines)

def load_supervisor_instructions_from_file(supervisor_path: str,
                                           model_name: str) -> tuple[str, str]:
    """
    Load supervisor instructions for specific model from results file
    MODIFIED: Return formatted strings instead of dictionaries
    """
    if not supervisor_path or not os.path.exists(supervisor_path):
        if VERBOSE >= 1:
            print(f"Supervisor file not found: {supervisor_path}")
        return "", ""
    
    try:
        with open(supervisor_path, 'r', encoding='utf-8') as f:
            supervisor_results = json.load(f)
        
        # Extract enhanced_guidelines from 4-phase structure
        enhanced_guidelines = supervisor_results.get("enhanced_guidelines", {})
        
        if not enhanced_guidelines:
            if VERBOSE >= 1:
                print(f"No enhanced_guidelines found in supervisor file")
            return "", ""
        
        # Format hierarchical common instructions
        hierarchical_common = enhanced_guidelines.get("hierarchical_common_instructions", [])
        common_instructions_text = format_hierarchical_instructions(hierarchical_common)
        
        # Extract and format model-specific instructions
        prioritized_model_instructions = enhanced_guidelines.get("prioritized_model_instructions", {})
        
        model_instructions_text = ""
        target_model = normalize_model_name(model_name)
        
        # Try exact match first, then :mv suffix
        if model_name in prioritized_model_instructions:
            model_instructions_raw = prioritized_model_instructions[model_name]
            if VERBOSE >= 2:
                print(f"Exact model match found for '{model_name}'")
        elif f"{model_name}:mv" in prioritized_model_instructions:
            model_instructions_raw = prioritized_model_instructions[f"{model_name}:mv"]
            if VERBOSE >= 2:
                print(f"Model match with :mv suffix found for '{model_name}'")
        else:
            # Try fuzzy matching
            matched_key = None
            for key in prioritized_model_instructions.keys():
                if normalize_model_name(key) == target_model:
                    matched_key = key
                    break
            
            if matched_key:
                model_instructions_raw = prioritized_model_instructions[matched_key]
                if VERBOSE >= 2:
                    print(f"Fuzzy model match: '{model_name}' -> '{matched_key}'")
            else:
                model_instructions_raw = []
                if VERBOSE >= 1:
                    print(f"No model-specific instructions found for '{model_name}'")
                    print(f"Available models: {list(prioritized_model_instructions.keys())}")
        
        if model_instructions_raw:
            model_instructions_text = format_model_specific_instructions(model_instructions_raw)
        
        if VERBOSE >= 1:
            common_count = len(hierarchical_common)
            model_count = len(model_instructions_raw) if model_instructions_raw else 0
            print(f"Loaded supervisor instructions (4-phase): {common_count} hierarchical common, {model_count} model-specific for {model_name}")
            
        if DEBUG:
            print(f"DEBUG: load_supervisor_instructions_from_file result")
            print(f"  supervisor_path: {supervisor_path}")
            print(f"  model_name: {model_name}")
            print(f"  common_instructions_text length: {len(common_instructions_text)}")
            print(f"  model_instructions_text length: {len(model_instructions_text)}")
            if common_instructions_text:
                print(f"  common_instructions preview: {common_instructions_text[:200]}...")
            if model_instructions_text:
                print(f"  model_instructions preview: {model_instructions_text[:200]}...")
            try:
                with open(supervisor_path, 'r', encoding='utf-8') as f:
                    supervisor_results = json.load(f)
                enhanced_guidelines = supervisor_results.get("enhanced_guidelines", {})
                print(f"  enhanced_guidelines keys: {list(enhanced_guidelines.keys())}")
                if "hierarchical_common_instructions" in enhanced_guidelines:
                    print(f"  hierarchical_common_instructions count: {len(enhanced_guidelines['hierarchical_common_instructions'])}")
                if "prioritized_model_instructions" in enhanced_guidelines:
                    model_keys = list(enhanced_guidelines['prioritized_model_instructions'].keys())
                    print(f"  available model keys: {model_keys}")
                    print(f"  target model: {model_name}")
                    print(f"  exact match found: {model_name in model_keys}")
                    print(f"  :mv suffix match found: {f'{model_name}:mv' in model_keys}")
            except Exception as e:
                print(f"  ERROR parsing supervisor file: {e}")
        
        return common_instructions_text, model_instructions_text
        
    except Exception as e:
        if VERBOSE >= 1:
            print(f"Error loading supervisor instructions from {supervisor_path}: {e}")
        return "", ""

def collect_openrouter_statistics(results_by_model: Dict[str, Any]) -> Dict[str, Any]:
    """collect_openrouter_statistics"""
    total_stats = {
        'model_statistics': {},
        'error_summary': {
            'timeout_errors': 0,
            'rate_limit_errors': 0,
            'client_errors': 0,
            'server_errors': 0,
            'unknown_errors': 0
        },
        'performance_summary': {
            'fastest_model': None,
            'slowest_model': None,
            'most_errors': None,
            'most_reliable': None
        }
    }
    
    model_performances = {}
    
    for model_name, result in results_by_model.items():
        openrouter_usage = result.get('openrouter_usage', {})
        if not openrouter_usage:
            continue
        model_stats = {
            'total_calls': openrouter_usage.get('api_calls_made', 0),
            'total_cost': openrouter_usage.get('total_cost_usd', 0.0),
            'current_model_tier': openrouter_usage.get('current_model_tier', 'unknown'),
            'avg_cost_per_call': openrouter_usage.get('average_cost_per_call', 0.0)
        }
        
        if hasattr(result.get('client'), 'get_error_statistics'):
            error_stats = result['client'].get_error_statistics()
            model_stats.update(error_stats)
            
            # 전체 에러 합계에 추가
            for error_type, count in error_stats.get('error_breakdown', {}).items():
                total_stats['error_summary'][error_type] += count
        
        total_stats['model_statistics'][model_name] = model_stats
        
        avg_time = result.get('avg_metrics', {}).get('avg_time_per_sample_seconds', 0)
        error_rate = model_stats.get('error_rate', 0)
        model_performances[model_name] = {
            'avg_time': avg_time,
            'error_rate': error_rate,
            'total_calls': model_stats['total_calls']
        }
    
    if model_performances:
        fastest = min(model_performances.items(), key=lambda x: x[1]['avg_time'])
        slowest = max(model_performances.items(), key=lambda x: x[1]['avg_time'])
        most_errors = max(model_performances.items(), key=lambda x: x[1]['error_rate'])
        most_reliable = min(model_performances.items(), key=lambda x: x[1]['error_rate'])
        
        total_stats['performance_summary'] = {
            'fastest_model': f"{fastest[0]} ({fastest[1]['avg_time']:.2f}s avg)",
            'slowest_model': f"{slowest[0]} ({slowest[1]['avg_time']:.2f}s avg)",
            'most_errors': f"{most_errors[0]} ({most_errors[1]['error_rate']:.3f} error rate)",
            'most_reliable': f"{most_reliable[0]} ({most_reliable[1]['error_rate']:.3f} error rate)"
        }
    
    return total_stats

def print_openrouter_summary(openrouter_stats: Dict[str, Any]):
    """print_openrouter_summary"""
    print(f"\n{'='*80}")
    print("OPENROUTER USAGE SUMMARY")
    print(f"{'='*80}")
    
    error_summary = openrouter_stats['error_summary']
    total_errors = sum(error_summary.values())
    if total_errors > 0:
        print("\nError Breakdown:")
        for error_type, count in error_summary.items():
            if count > 0:
                print(f"  {error_type}: {count}")
    else:
        print("\nNo errors reported")
    
    perf_summary = openrouter_stats['performance_summary']
    print(f"\nPerformance Summary:")
    for metric, value in perf_summary.items():
        if value:
            print(f"  {metric.replace('_', ' ').title()}: {value}")
    
    print(f"\nPer-Model Statistics:")
    for model_name, stats in openrouter_stats['model_statistics'].items():
        calls = stats.get('total_calls', 0)
        cost = stats.get('total_cost', 0.0)
        tier = stats.get('current_model_tier', 'unknown')
        error_rate = stats.get('error_rate', 0)
        
        print(f"  {model_name}:")
        print(f"    Calls: {calls}, Cost: ${cost:.4f}, Tier: {tier}, Error Rate: {error_rate:.3f}")