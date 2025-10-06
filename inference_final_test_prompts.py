"""
Test Inference Runner for final-test-prompts

This module processes prompt templates from final-test-prompts directory,
runs NER inference using updated model names, and saves results alongside templates.
"""
import random 
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from tqdm import tqdm

# Import existing modules
from base_annotator import NERAgent
from utils_annotator import (
    aggregate_strict_span_metrics,
    calculate_token_accuracy,
    convert_bio_to_entities,
    convert_entities_to_bio
)
from utils_experiments import load_config
# from openrouter_client import OpenRouterClient

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Debug mode - set to True for detailed output
DEBUG = False

# Test configuration
TEST_SAMPLE_SIZE = None  # Number of samples to use from 300 (set to None for all 300 samples)
TEST_NUMBER = None  # Number of prompt templates to test (set to None for all templates)

# Skip configuration
SKIP_BENCHMARKS = [
    # 'mitner_restaurant',
    # 'FabNER',
    # 'HarveyNER',
    # 'mitner_movie',
    # 'ACE05',
    # 'GENIA',
    # 'MultiNERD',
    # 'crossner_conll2003',
    # 'crossner_ai',
    # 'crossner_literature',
    # 'crossner_music',
    # 'crossner_politics',
    # 'crossner_science',
    # 'anatem',
    # 'bc2gm',
    # 'bc4chemd',
    # 'bc5cdr',
    # 'Broad Twitter',
    # 'OntoNotes'
    ]  # e.g., ['ACE05', 'crossner_conll2003'] - empty list means process all benchmarks
SKIP_MODEL_PATTERNS = [
    # 'gpt-oss',
    # 'gemma',
    # 'qwen',
    # 'llama',
    # 'nemotron',
    # 'grok',
    # 'sonoma',
    # 'glm',
    # 'gemini',
    # 'mistral',
    # 'hynyuan',
    # 'deepseek',
    # 'lunaris',
    # 'phi',
    ]  # e.g., ['sonosky', 'gpt-4'] - empty list means process all models

# Maximum number of parallel workers for inference
MAX_WORKERS = 50

# Model name mapping from old names to new names
MODEL_NAME_MAPPING = {
    "gpt-oss": "gpt-oss:20b",
    "deepseek-r1_8b": "deepseek-r1:8b",
    "gemma3": "gemma3:12b",
    "qwen3_14b": "qwen3:14b",
    "qwen3-14b": "qwen3:14b",
    "llama3.1_8b": "llama3.1:8b",
    "phi4": "phi4:14b",
    "mistral-small3.2": "mistral-small3.2:24b",
    "Randomblock1_nemotron": "Randomblock1/nemotron-nano:8b",
    "glm4": "glm4:32b",
    "sonoma": "x-ai/grok-3-mini"
}

# Default inference parameters
INFERENCE_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 8000,
    "use_openrouter": True,
    "max_retries": 3,
    "timeout_seconds": 120
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PromptTemplateInfo:
    """Information about a prompt template file"""
    file_path: Path
    benchmark: str
    model_name: str
    iteration: str
    condition_path: str
    relative_path: str

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def debug_print(message: str):
    """Print message only if DEBUG is True"""
    if DEBUG:
        print(f"[DEBUG] {message}")

def scan_prompt_templates(base_dir: str = "final-test-prompts_diff_annr_numbers") -> List[PromptTemplateInfo]:
    """
    Recursively scan for all prompt template files in final-test-prompts_diff_annr_numbers directory
    
    Args:
        base_dir: Base directory to scan
        
    Returns:
        List of PromptTemplateInfo objects
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    
    prompt_templates = []
    
    debug_print(f"Scanning directory: {base_dir}")
    
    # Find all *_prompt_template.txt files
    for txt_file in base_path.rglob("*_prompt_template.txt"):
        try:
            # Extract metadata from file path and name
            relative_path = txt_file.relative_to(base_path)
            parts = relative_path.parts
            
            debug_print(f"Processing file: {relative_path}")
            
            # Expected structure: benchmark/condition/prompts/model_prompt_template.txt
            if len(parts) >= 3 and parts[-2] == "prompts":
                benchmark = parts[0]
                condition_path = "/".join(parts[1:-2]) if len(parts) > 3 else parts[-3]
                
                # Extract model name from filename
                filename = txt_file.stem  # Remove .txt extension
                if filename.endswith("_prompt_template"):
                    model_part = filename[:-len("_prompt_template")]
                    
                    # Extract iteration if present
                    if "_iter" in model_part:
                        model_name = model_part.split("_iter")[0]
                        iteration = model_part.split("_iter")[1].split("_")[0]
                    else:
                        model_name = model_part
                        iteration = "0"
                    
                    template_info = PromptTemplateInfo(
                        file_path=txt_file,
                        benchmark=benchmark,
                        model_name=model_name,
                        iteration=iteration,
                        condition_path=condition_path,
                        relative_path=str(relative_path)
                    )
                    prompt_templates.append(template_info)
                    
                    debug_print(f"  -> benchmark={benchmark}, model={model_name}, iter={iteration}")
                    
        except Exception as e:
            print(f"Warning: Failed to process {txt_file}: {e}")
            print(f"DEBUG TEMPLATE ERROR: {template_info.relative_path}")
            print(f"DEBUG MODEL: {model_name}")
            print(f"DEBUG ERROR TYPE: {type(e).__name__}")
            print(f"DEBUG ERROR DETAILS: {str(e)}")
    
    print(f"Found {len(prompt_templates)} prompt template files")
    return prompt_templates

def should_skip_template(template_info: PromptTemplateInfo) -> tuple[bool, str]:
    """
    Check if template should be skipped based on benchmark or model patterns
    
    Args:
        template_info: Template information
        
    Returns:
        Tuple of (should_skip, reason)
    """
    # Check benchmark skip list
    if SKIP_BENCHMARKS and template_info.benchmark in SKIP_BENCHMARKS:
        return True, f"Benchmark '{template_info.benchmark}' in skip list"
    
    # Check model pattern skip list
    if SKIP_MODEL_PATTERNS:
        for pattern in SKIP_MODEL_PATTERNS:
            if pattern in template_info.model_name:
                return True, f"Model '{template_info.model_name}' matches skip pattern '{pattern}'"
    
    return False, ""

def load_test_dataset(benchmark: str) -> tuple[List[Dict], Dict[str, Any]]:
    """
    Load test dataset for given benchmark
    
    Args:
        benchmark: Benchmark name (e.g., 'crossner_conll2003')
        
    Returns:
        Tuple of (test_samples, ner_scheme)
    """
    debug_print(f"Loading test dataset for benchmark: {benchmark}")
    
    config_path = f'experiment_settings/{benchmark}_default_config.json'
    config = load_config(config_path)
    
    # Get original dataset path and modify for test data
    original_dataset_path = config['experiment']['dataset_path']
    test_dataset_path = original_dataset_path.replace('_ner_dataset.pkl', '_sampled_test_300.pkl')
    
    debug_print(f"Test dataset path: {test_dataset_path}")
    
    if not Path(test_dataset_path).exists():
        raise FileNotFoundError(f"Test dataset not found: {test_dataset_path}")
    
    # Load test data (should be a direct list of ~300 samples)
    with open(test_dataset_path, 'rb') as f:
        test_samples = pickle.load(f)
    
    if not isinstance(test_samples, list):
        raise ValueError(f"Expected list of samples, got {type(test_samples)}")
    
    # Limit samples if TEST_SAMPLE_SIZE is set
    if TEST_SAMPLE_SIZE is not None and TEST_SAMPLE_SIZE < len(test_samples):
        test_samples = test_samples[:TEST_SAMPLE_SIZE]
        debug_print(f"Limited to {TEST_SAMPLE_SIZE} samples")
    elif TEST_SAMPLE_SIZE is None:
        debug_print(f"Using all {len(test_samples)} samples (TEST_SAMPLE_SIZE=None)")
    
    ner_scheme = config['ner_scheme']
    
    print(f"Loaded {len(test_samples)} test samples for {benchmark}")
    return test_samples, ner_scheme

def map_model_name(old_name: str) -> str:
    """
    Convert old model name to new format
    
    Args:
        old_name: Original model name from prompt template
        
    Returns:
        Updated model name for inference
    """
    # Check for exact matches first
    if old_name in MODEL_NAME_MAPPING:
        mapped = MODEL_NAME_MAPPING[old_name]
        debug_print(f"Model mapping: {old_name} -> {mapped}")
        return mapped
    
    # Check for partial matches (handle cases where old_name might have suffixes)
    for old_key, new_name in MODEL_NAME_MAPPING.items():
        if old_name.startswith(old_key):
            debug_print(f"Partial model mapping: {old_name} -> {new_name}")
            return new_name
    
    debug_print(f"No mapping found for model '{old_name}', using as-is")
    return old_name

def load_prompt_template(template_path: Path) -> str:
    """
    Load and validate prompt template
    
    Args:
        template_path: Path to prompt template file
        
    Returns:
        Prompt template string
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    if "{document}" not in template:
        raise ValueError(f"Prompt template missing {{document}} placeholder: {template_path}")
    
    debug_print(f"Loaded prompt template: {len(template)} characters")
    return template

def run_inference_for_template(template_info: PromptTemplateInfo, 
                             test_samples: List[Dict],
                             ner_scheme: Dict[str, Any],
                             model_name: str) -> Dict[str, Any]:
    """
    Run NER inference for a single prompt template with progress bar
    
    Args:
        template_info: Information about prompt template
        test_samples: Test samples for inference
        ner_scheme: NER scheme configuration
        model_name: Mapped model name for inference
        
    Returns:
        Results dictionary with performance metrics and sample results
    """
    start_time = time.time()
    
    try:
        # Load prompt template
        prompt_template = load_prompt_template(template_info.file_path)
        debug_print(f"Processing template: {template_info.relative_path}")
        
        # Initialize NER agent with OpenRouter
        agent = NERAgent(
            model_name=model_name,
            ner_scheme=ner_scheme,
            llm_infer_by_openrouter=True,
            verbose=0  # Minimal output for parallel processing
        )
        debug_print(f"Initialized NER agent for model: {model_name}")
        
        # Process each test sample with progress bar
        sample_results = []
        all_predicted_entities = []
        all_gold_entities = []
        all_predicted_labels = []
        all_gold_labels = []
        
        # Create progress bar for this template
        template_desc = f"{template_info.benchmark}/{template_info.model_name}"
        pbar = tqdm(test_samples, desc=template_desc, leave=False, disable=not DEBUG)
        
        for i, sample in enumerate(pbar):
            sample_start_time = time.time()
            
            try:
                # Apply prompt template using simple string replace (safer than .format())
                prompt = prompt_template.replace("{document}", sample['text'])
                debug_print(f"  Processing sample {i+1}/{len(test_samples)}")
                debug_print(f"  Prompt length: {len(prompt)} characters")
                
                # Clear conversation history for each sample
                agent._clear_conversation_history()
                
                # Run inference
                debug_print(f"  Running inference for sample {i+1}")
                response = agent._generate_with_context(prompt)
                debug_print(f"  Got response: {len(response)} characters")
                debug_print(f"  Response preview: {response[:200]}...")
                
                # Parse response and extract entities
                debug_print(f"  Parsing JSON response for sample {i+1}")
                try:
                    parsed_response, predicted_positions = agent._parse_json_response(response, prompt)
                    debug_print(f"  JSON parsing successful for sample {i+1}")
                except Exception as parse_error:
                    debug_print(f"  JSON parsing failed for sample {i+1}: {parse_error}")
                    debug_print(f"  Raw response: {response}")
                    raise Exception(f"JSON parsing failed: {parse_error}")
                
                predicted_entities = []
                
                if "entities" in parsed_response:
                    debug_print(f"  Found {len(parsed_response['entities'])} entities in response")
                    for entity in parsed_response["entities"]:
                        entity_text = entity.get("text", "")
                        entity_type = entity.get("type", "")
                        
                        if entity_type in ner_scheme:
                            # Find actual positions
                            debug_print(f"    Processing entity: '{entity_text}' ({entity_type})")
                            positions = agent._find_entity_positions(sample['text'], entity_text)
                            debug_print(f"    Found {len(positions)} position matches")
                            for start_pos, end_pos in positions:
                                predicted_entities.append({
                                    "text": entity_text,
                                    "type": entity_type,
                                    "start_pos": start_pos,
                                    "end_pos": end_pos,
                                    "confidence": entity.get("confidence", "medium")
                                })
                        else:
                            debug_print(f"    Skipping entity with invalid type: '{entity_type}'")
                else:
                    debug_print(f"  No 'entities' key found in parsed response")
                
                # Convert gold standard BIO tags to entities
                debug_print(f"  Converting gold standard BIO tags for sample {i+1}")
                gold_entities = convert_bio_to_entities(sample['tokens'], sample['labels'])
                debug_print(f"  Gold entities: {len(gold_entities)}")
                
                # Convert predicted entities to BIO labels for token accuracy
                debug_print(f"  Converting predicted entities to BIO labels for sample {i+1}")
                predicted_labels = convert_entities_to_bio(
                    sample['tokens'], predicted_entities, sample['text'],
                    valid_entity_types=set(ner_scheme.keys())
                )
                debug_print(f"  Predicted labels: {len(predicted_labels)}, Gold labels: {len(sample['labels'])}")
                
                sample_end_time = time.time()
                
                # Store sample result
                sample_result = {
                    "sample_id": i,
                    "text": sample['text'],
                    "tokens": sample['tokens'],
                    "gold_labels": sample['labels'],
                    "predicted_labels": predicted_labels,
                    "gold_entities": gold_entities,
                    "predicted_entities": predicted_entities,
                    "processing_time": sample_end_time - sample_start_time
                }
                sample_results.append(sample_result)
                
                # Collect for aggregate metrics
                all_predicted_entities.append(predicted_entities)
                all_gold_entities.append(gold_entities)
                all_predicted_labels.extend(predicted_labels)
                all_gold_labels.extend(sample['labels'])
                
                # Update progress bar
                if not DEBUG:
                    pbar.set_postfix({"sample": f"{i+1}/{len(test_samples)}"})
                
            except Exception as e:
                debug_print(f"  ERROR processing sample {i}: {str(e)}")
                debug_print(f"  Error type: {type(e).__name__}")
                debug_print(f"  Sample text preview: {sample['text'][:100]}...")
                debug_print(f"  Sample tokens count: {len(sample.get('tokens', []))}")
                debug_print(f"  Sample labels count: {len(sample.get('labels', []))}")
                
                if DEBUG:
                    import traceback
                    debug_print(f"  Full traceback:")
                    traceback.print_exc()
                    
                    # Stop execution in debug mode for easier debugging
                    print(f"\nSTOPPING EXECUTION FOR DEBUGGING - Sample {i} failed")
                    print(f"Error: {str(e)}")
                    print(f"Template: {template_info.relative_path}")
                    print(f"Model: {model_name}")
                    raise e
                
                # Add error sample with fallback values
                sample_result = {
                    "sample_id": i,
                    "text": sample['text'],
                    "tokens": sample.get('tokens', []),
                    "gold_labels": sample.get('labels', []),
                    "predicted_labels": ['O'] * len(sample.get('tokens', [])),
                    "gold_entities": convert_bio_to_entities(sample['tokens'], sample['labels']) if sample.get('tokens') and sample.get('labels') else [],
                    "predicted_entities": [],
                    "error": str(e),
                    "processing_time": 0.0
                }
                sample_results.append(sample_result)
                
                # Add empty results for aggregate metrics
                all_predicted_entities.append([])
                if sample.get('tokens') and sample.get('labels'):
                    all_gold_entities.append(convert_bio_to_entities(sample['tokens'], sample['labels']))
                    all_gold_labels.extend(sample['labels'])
                else:
                    all_gold_entities.append([])
                all_predicted_labels.extend(['O'] * len(sample.get('tokens', [])))
        
        pbar.close()
        
        # Calculate aggregate performance metrics
        entity_types = list(ner_scheme.keys())
        aggregated_metrics = aggregate_strict_span_metrics(
            all_predicted_entities, all_gold_entities, entity_types
        )
        
        # Calculate token accuracy
        token_accuracy = calculate_token_accuracy(all_predicted_labels, all_gold_labels)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        debug_print(f"Completed inference: {total_time:.2f}s total")
        
        # Compile results
        results = {
            "prompt_template_path": str(template_info.file_path),
            "model_name": model_name,
            "original_model_name": template_info.model_name,
            "benchmark": template_info.benchmark,
            "condition_path": template_info.condition_path,
            "iteration": template_info.iteration,
            "test_samples_count": len(test_samples),
            "test_samples_used": len(test_samples),  # After limiting
            "inference_metadata": {
                "timestamp": datetime.now().isoformat(),
                "inference_config": INFERENCE_CONFIG,
                "processing_time_seconds": total_time,
                "avg_time_per_sample": total_time / len(test_samples) if test_samples else 0,
                "successful_samples": len([r for r in sample_results if 'error' not in r]),
                "debug_mode": DEBUG,
                "test_sample_size_limit": TEST_SAMPLE_SIZE,
            },
            "performance_metrics": {
                "micro": aggregated_metrics['micro'],
                "macro": aggregated_metrics['macro'],
                "per_type": aggregated_metrics['per_type'],
                "token_accuracy": token_accuracy,
                "total_entities_predicted": aggregated_metrics['total_predicted'],
                "total_entities_gold": aggregated_metrics['total_gold']
            },
            "sample_results": sample_results
        }
        
        return results
        
    except Exception as e:
        debug_print(f"Failed to process template: {e}")
        return {
            "prompt_template_path": str(template_info.file_path),
            "model_name": model_name,
            "original_model_name": template_info.model_name,
            "benchmark": template_info.benchmark,
            "error": str(e),
            "inference_metadata": {
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": time.time() - start_time,
                "failed": True,
                "debug_mode": DEBUG
            }
        }

def save_inference_results(results: Dict[str, Any], template_path: Path):
    """
    Save inference results as JSON file next to prompt template
    Handle failed results with special naming convention
    
    Args:
        results: Results dictionary
        template_path: Path to original prompt template
    """
    # Check if this is a failed result
    is_failed = 'error' in results
    
    if is_failed:
        # Create output filename for failed results: model_failed.json, model_failed2.json, etc.
        base_filename = template_path.stem.replace("_prompt_template", "_failed")
        counter = 1
        
        while True:
            if counter == 1:
                output_filename = f"{base_filename}.json"
            else:
                output_filename = f"{base_filename}{counter}.json"
            
            output_path = template_path.parent / output_filename
            
            if not output_path.exists():
                break
            counter += 1
    else:
        # Create output filename for successful results: model_inference_results.json
        output_filename = template_path.stem.replace("_prompt_template", "_inference_results") + ".json"
        output_path = template_path.parent / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    debug_print(f"Saved results to: {output_path}")
    return output_path

def process_single_template(template_info: PromptTemplateInfo, 
                          test_datasets: Dict[str, tuple]) -> tuple[PromptTemplateInfo, bool, str]:
    """
    Worker function to process a single template (for parallel execution)
    
    Args:
        template_info: Template information
        test_datasets: Dictionary mapping benchmark -> (test_samples, ner_scheme)
        
    Returns:
        Tuple of (template_info, success, message)
    """
    try:
        # Get test data for this benchmark
        if template_info.benchmark not in test_datasets:
            return template_info, False, f"No test data for benchmark: {template_info.benchmark}"
        
        test_samples, ner_scheme = test_datasets[template_info.benchmark]
        
        # Map model name
        # print(f"template_info.model_name: {template_info.model_name}")
        model_name = map_model_name(template_info.model_name)
        
        # Check if results already exist
        result_filename = template_info.file_path.stem.replace("_prompt_template", "_inference_results") + ".json"
        result_path = template_info.file_path.parent / result_filename
        
        if result_path.exists():
            return template_info, True, f"Results already exist: {result_path}"
        
        # Run inference
        results = run_inference_for_template(template_info, test_samples, ner_scheme, model_name)
        
        # Save results
        output_path = save_inference_results(results, template_info.file_path)
        
        if 'error' in results:
            return template_info, False, f"Inference failed: {results['error']}"
        else:
            f1_score = results.get('performance_metrics', {}).get('micro', {}).get('f1', 0.0)
            return template_info, True, f"Success - F1: {f1_score:.3f} -> {output_path.name}"
    
    except Exception as e:
        return template_info, False, f"Exception: {str(e)}"

def main_test_inference(base_dir: str = "final-test-prompts_diff_annr_numbers", 
                       max_workers: int = MAX_WORKERS):
    """
    Main function to orchestrate test inference process
    
    Args:
        base_dir: Base directory containing prompt templates
        max_workers: Maximum number of parallel workers
    """
    print(f"Starting test inference process...")
    print(f"Base directory: {base_dir}")
    print(f"Max workers: {max_workers}")
    print(f"Debug mode: {DEBUG}")
    
    # Display configuration with clear indication of full processing
    if TEST_SAMPLE_SIZE is None:
        print(f"Test sample size: ALL (300 samples per benchmark)")
    else:
        print(f"Test sample size: {TEST_SAMPLE_SIZE}")
    
    if TEST_NUMBER is None:
        print(f"Test number limit: ALL TEMPLATES")
    else:
        print(f"Test number limit: {TEST_NUMBER}")
    
    # Show full processing mode
    if TEST_SAMPLE_SIZE is None and TEST_NUMBER is None:
        print("*** FULL PROCESSING MODE: All templates and all test samples ***")
    
    start_time = time.time()
    
    # 1. Scan for prompt templates
    print(f"\n{'='*60}")
    print("SCANNING PROMPT TEMPLATES")
    print(f"{'='*60}")
    
    prompt_templates = scan_prompt_templates(base_dir)
    random.shuffle(prompt_templates)
    
    if not prompt_templates:
        print("No prompt templates found!")
        return
    
    # Filter out skipped templates
    original_count = len(prompt_templates)
    filtered_templates = []
    skipped_by_filter = 0
    
    for template in prompt_templates:
        should_skip, reason = should_skip_template(template)
        if should_skip:
            skipped_by_filter += 1
            if DEBUG:
                print(f"Skipping template: {template.relative_path} - {reason}")
        else:
            filtered_templates.append(template)
    
    prompt_templates = filtered_templates
    
    if skipped_by_filter > 0:
        print(f"Filtered out {skipped_by_filter} templates by skip rules")
        print(f"Remaining templates: {len(prompt_templates)}")
    
    if not prompt_templates:
        print("No templates remain after filtering!")
        return
    
    # Limit number of templates if TEST_NUMBER is set
    if TEST_NUMBER is not None and TEST_NUMBER < len(prompt_templates):
        prompt_templates = prompt_templates[:TEST_NUMBER]
        print(f"Limited to {TEST_NUMBER} templates for testing")
    elif TEST_NUMBER is None:
        print(f"Processing all {len(prompt_templates)} templates (TEST_NUMBER=None)")
    
    # Group by benchmark
    benchmarks = set(template.benchmark for template in prompt_templates)
    print(f"Found templates for benchmarks: {sorted(benchmarks)}")
    
    # 2. Load test datasets for all benchmarks
    print(f"\n{'='*60}")
    print("LOADING TEST DATASETS")
    print(f"{'='*60}")
    
    test_datasets = {}
    for benchmark in benchmarks:
        try:
            test_samples, ner_scheme = load_test_dataset(benchmark)
            test_datasets[benchmark] = (test_samples, ner_scheme)
        except Exception as e:
            print(f"Failed to load test data for {benchmark}: {e}")
    
    if not test_datasets:
        print("No test datasets loaded successfully!")
        return
    
    print(f"Loaded test datasets for: {list(test_datasets.keys())}")
    
    # 3. Process templates in parallel
    print(f"\n{'='*60}")
    print("RUNNING INFERENCE")
    print(f"{'='*60}")
    print(f"Processing {len(prompt_templates)} prompt templates...")
    
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Create overall progress bar
    overall_pbar = tqdm(total=len(prompt_templates), desc="Overall Progress", position=0)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_template = {
            executor.submit(process_single_template, template, test_datasets): template
            for template in prompt_templates
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_template), 1):
            template_info = future_to_template[future]
            
            try:
                template_info, success, message = future.result()
                
                if success:
                    if "already exist" in message:
                        skipped_count += 1
                        status = "SKIP"
                    else:
                        successful_count += 1
                        status = "OK"
                else:
                    failed_count += 1
                    status = "FAIL"
                
                progress = f"[{i}/{len(prompt_templates)}]"
                model_info = f"{template_info.benchmark}/{template_info.model_name}"
                
                if DEBUG:
                    print(f"{progress} {status:4} {model_info:30} - {message}")
                
                # Update overall progress bar
                overall_pbar.set_postfix({
                    "OK": successful_count,
                    "SKIP": skipped_count, 
                    "FAIL": failed_count
                })
                overall_pbar.update(1)
                
            except Exception as e:
                failed_count += 1
                if DEBUG:
                    print(f"[{i}/{len(prompt_templates)}] ERROR {template_info.model_name:30} - Exception: {str(e)}")
                overall_pbar.update(1)
    
    overall_pbar.close()
    
    # 4. Final summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total templates processed: {len(prompt_templates)}")
    print(f"Successful: {successful_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Total processing time: {total_time:.1f} seconds")
    print(f"Average time per template: {total_time/len(prompt_templates):.1f} seconds")
    
    if successful_count > 0:
        print(f"\nInference results saved as *_inference_results.json files alongside prompt templates")

if __name__ == "__main__":
    # Run the main inference process
    main_test_inference()
