import json
from pathlib import Path
import re
from typing import Dict, Any, List

# Global verbosity setting
# 0: Minimal output (only final summary)
# 1: Model-level progress (default)
# 2: Basic sample-level output (simplified)
VERBOSE = 2


def get_validation_failure_reason(sample_result: Dict[str, Any], expected_token_count: int) -> str:
    """
    Get specific reason why validation failed for better logging
    
    Args:
        sample_result: Result dictionary from model annotation
        expected_token_count: Expected number of tokens
        
    Returns:
        Human-readable failure reason
    """
    try:
        if 'predicted_labels' not in sample_result:
            return "No predicted_labels found"
        
        predicted_labels = sample_result['predicted_labels']
        
        if not predicted_labels:
            return "Empty predicted_labels"
        
        if len(predicted_labels) != expected_token_count:
            return f"Token/label mismatch ({len(predicted_labels)} labels vs {expected_token_count} tokens)"
        
        if 'predicted_entities' not in sample_result:
            return "No predicted_entities found"
        
        metrics = sample_result.get('metrics', {})
        if not isinstance(metrics, dict):
            return "Invalid metrics structure"
        
        return "Unknown validation failure"
        
    except Exception as e:
        return f"Validation error: {str(e)}"

def validate_annotation_result(sample_result: Dict[str, Any], expected_token_count: int) -> bool:
    """
    Validate that annotation result has correct structure and token alignment
    
    Args:
        sample_result: Result dictionary from model annotation
        expected_token_count: Expected number of tokens
        
    Returns:
        True if valid, False if needs retry
    """
    try:
        # Check if we have predicted_labels
        if 'predicted_labels' not in sample_result:
            if VERBOSE >= 2:
                print(f"    Validation failed: No predicted_labels found")
            return False
        
        predicted_labels = sample_result['predicted_labels']
        
        # Check if predicted_labels is empty or None
        if not predicted_labels:
            if VERBOSE >= 2:
                print(f"    Validation failed: Empty predicted_labels")
            return False
        
        # Check length mismatch
        if len(predicted_labels) != expected_token_count:
            if VERBOSE >= 2:
                print(f"    Validation failed: Label/token length mismatch: {len(predicted_labels)} vs {expected_token_count}")
            return False
        
        # Check if we have predicted_entities
        if 'predicted_entities' not in sample_result:
            if VERBOSE >= 2:
                print(f"    Validation failed: No predicted_entities found")
            return False
        
        # Check if metrics are reasonable (not all zeros due to error)
        metrics = sample_result.get('metrics', {})
        if not isinstance(metrics, dict):
            if VERBOSE >= 2:
                print(f"    Validation failed: Invalid metrics structure")
            return False
        
        return True
        
    except Exception as e:
        if VERBOSE >= 2:
            print(f"    Validation failed with exception: {str(e)}")
        return False

def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name for fuzzy matching
    
    Args:
        model_name: Original model name
        
    Returns:
        Normalized model name for comparison
    """
    # Remove common variations and normalize
    normalized = model_name.lower()
    normalized = normalized.replace(":", "_").replace("-", "_").replace("/", "_")
    normalized = re.sub(r'[^a-z0-9_]', '', normalized)
    return normalized

def get_default_critical_instructions() -> List[str]:
    """Get default critical instructions for NER task"""
    return [
        "**Completeness**: Scan the document systematically from beginning to end. Do NOT miss any entities.",
        "**Accuracy**: Extract the exact text spans as they appear in the document - no modifications.",
        "**Semantic Precision**: Carefully distinguish between different entity types based on context and meaning.",
        "**Example Guidance**: Use the provided positive examples as guides for what TO include, and negative examples for what NOT to include.",
        "**Relationship Awareness**: Consider how entities relate to each other and their surrounding context.",
        "**Order**: Report entities in the order they appear in the document (left to right, top to bottom).",
    ]

def get_default_analysis_instructions() -> List[str]:
    """Get default analysis instructions for confusing case identification"""
    return [
        "Re-examine each annotated entity and consider alternative interpretations",
        "Look for text spans you didn't annotate but could potentially be entities",
        "Consider boundary variations (longer/shorter spans for the same entity)",
        "Think about cases where context could lead to different interpretations",
        "Only include cases where the alternative interpretation is genuinely reasonable",
        "Consider the semantic context and surrounding words",
        "Reference the provided examples to guide your analysis",
        "**IMPORTANT**: For \"text_possible\", only use exact text spans that actually exist in the original document - do not create new text, modify existing text, or add explanations"
    ]

def convert_bio_to_entities(tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
    """
    Convert BIO-tagged tokens to entity list with positions
    
    Args:
        tokens: List of tokens
        labels: List of BIO labels (e.g., ['B-PER', 'I-PER', 'O'])
    
    Returns:
        List of entity dictionaries with text, type, start_pos, end_pos
    """
    entities = []
    current_entity = None
    char_pos = 0
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith('B-'):
            # Start new entity
            if current_entity:
                entities.append(current_entity)
            
            entity_type = label[2:]  # Remove 'B-' prefix
            current_entity = {
                'text': token,
                'type': entity_type,
                'start_pos': char_pos,
                'end_pos': char_pos + len(token)
            }
        
        elif label.startswith('I-') and current_entity:
            # Continue current entity
            current_entity['text'] += ' ' + token
            current_entity['end_pos'] = char_pos + len(token)
        
        else:  # 'O' or end of entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        
        # Update character position (add space except for last token)
        char_pos += len(token)
        if i < len(tokens) - 1:
            char_pos += 1  # Add space
    
    # Add final entity if exists
    if current_entity:
        entities.append(current_entity)
    
    return entities

def convert_entities_to_bio(tokens: List[str], entities: List[Dict[str, Any]], 
                           text: str, valid_entity_types: set = None) -> List[str]:
    """
    Convert entity list back to BIO format labels
    
    Args:
        tokens: List of tokens
        entities: List of entity dictionaries with start_pos, end_pos, type
        text: Original text string
        valid_entity_types: Set of valid entity types to filter
    
    Returns:
        List of BIO labels corresponding to tokens
    """
    bio_labels = ["O"] * len(tokens)
    
    if not entities:
        return bio_labels
    
    # Filter entities by valid types if specified
    valid_entities = []
    if valid_entity_types is not None:
        for entity in entities:
            if entity.get('type') in valid_entity_types:
                valid_entities.append(entity)
    else:
        valid_entities = entities
        
    # Create character position to token index mapping
    char_to_token = {}
    char_pos = 0
    
    for token_idx, token in enumerate(tokens):
        token_start = char_pos
        token_end = char_pos + len(token)
        
        # Map each character position to token index
        for char_idx in range(token_start, token_end):
            if char_idx < len(text):
                char_to_token[char_idx] = token_idx
        
        char_pos = token_end
        if token_idx < len(tokens) - 1:
            char_pos += 1  # Add space between tokens
    
    # Convert entities to BIO labels
    for entity in valid_entities:
        entity_start = entity['start_pos']
        entity_end = entity['end_pos']
        entity_type = entity['type']
        
        # Find tokens that overlap with this entity
        entity_token_indices = set()
        for char_idx in range(entity_start, entity_end):
            if char_idx in char_to_token:
                entity_token_indices.add(char_to_token[char_idx])
        
        # Sort token indices and assign BIO labels
        sorted_token_indices = sorted(entity_token_indices)
        
        for i, token_idx in enumerate(sorted_token_indices):
            if i == 0:
                bio_labels[token_idx] = f"B-{entity_type}"
            else:
                bio_labels[token_idx] = f"I-{entity_type}"
    
    return bio_labels

def calculate_strict_span_metrics(predicted_entities: List[Dict], gold_entities: List[Dict], 
                                 entity_types: List[str] = None) -> Dict[str, Any]:
    """
    Calculate precision, recall, and F1-score using strict span matching
    Strict span matching: (start_pos, end_pos, entity_type) must match exactly
    
    Args:
        predicted_entities: List of predicted entity dictionaries
        gold_entities: List of gold standard entity dictionaries
        entity_types: List of entity types to consider (if None, use all found types)
    
    Returns:
        Dictionary containing micro, macro, and per-type metrics
    """
    # Create sets of (start_pos, end_pos, type) tuples for strict matching
    pred_spans = set()
    for entity in predicted_entities:
        start = entity.get('start_pos', -1)
        end = entity.get('end_pos', -1)
        etype = entity.get('type', '')
        if start != -1 and end != -1 and etype:
            pred_spans.add((start, end, etype))
    
    gold_spans = set()
    for entity in gold_entities:
        start = entity.get('start_pos', -1)
        end = entity.get('end_pos', -1)
        etype = entity.get('type', '')
        if start != -1 and end != -1 and etype:
            gold_spans.add((start, end, etype))
    
    # Get all entity types if not specified
    if entity_types is None:
        entity_types = sorted(set(span[2] for span in pred_spans | gold_spans))
    
    # Calculate per-type metrics
    per_type_metrics = {}
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    
    for etype in entity_types:
        # Filter spans for this entity type
        pred_type_spans = {span for span in pred_spans if span[2] == etype}
        gold_type_spans = {span for span in gold_spans if span[2] == etype}
        
        # Calculate TP, FP, FN for this type
        tp = len(pred_type_spans & gold_type_spans)
        fp = len(pred_type_spans - gold_type_spans)
        fn = len(gold_type_spans - pred_type_spans)
        
        # Per-type metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_type_metrics[etype] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': len(gold_type_spans),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        
        # Accumulate for micro metrics
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    
    # Calculate micro metrics (aggregate across all types)
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # Calculate macro metrics (average across types)
    if per_type_metrics:
        macro_precision = sum(metrics['precision'] for metrics in per_type_metrics.values()) / len(per_type_metrics)
        macro_recall = sum(metrics['recall'] for metrics in per_type_metrics.values()) / len(per_type_metrics)
        macro_f1 = sum(metrics['f1'] for metrics in per_type_metrics.values()) / len(per_type_metrics)
    else:
        macro_precision = macro_recall = macro_f1 = 0.0
    
    return {
        'micro': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1,
            'tp': micro_tp,
            'fp': micro_fp,
            'fn': micro_fn
        },
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'per_type': per_type_metrics,
        'total_predicted': len(pred_spans),
        'total_gold': len(gold_spans)
    }

def calculate_token_accuracy(predicted_labels: List[str], gold_labels: List[str]) -> float:
    """
    Calculate token-level accuracy
    
    Args:
        predicted_labels: List of predicted BIO labels
        gold_labels: List of gold BIO labels
        
    Returns:
        Token accuracy as float
    """
    if len(predicted_labels) != len(gold_labels):
        return 0.0
    
    if len(predicted_labels) == 0:
        return 1.0
    
    correct = sum(1 for p, g in zip(predicted_labels, gold_labels) if p == g)
    return correct / len(predicted_labels)

def aggregate_strict_span_metrics(all_predicted_entities: List[List[Dict]], 
                                  all_gold_entities: List[List[Dict]],
                                  entity_types: List[str] = None) -> Dict[str, Any]:
    """
    Aggregate strict span metrics across multiple samples
    
    Args:
        all_predicted_entities: List of predicted entity lists (one per sample)
        all_gold_entities: List of gold entity lists (one per sample)
        entity_types: List of entity types to consider
        
    Returns:
        Aggregated metrics dictionary
    """
    # Flatten all entities into single lists
    all_predicted = []
    all_gold = []
    
    for pred_entities in all_predicted_entities:
        all_predicted.extend(pred_entities)
    
    for gold_entities in all_gold_entities:
        all_gold.extend(gold_entities)
    
    # Calculate metrics on the aggregated data
    return calculate_strict_span_metrics(all_predicted, all_gold, entity_types)

def calculate_ner_metrics(predicted_entities: List[Dict], gold_entities: List[Dict]) -> Dict[str, float]:
    """
    Legacy function for backward compatibility
    Returns simplified metrics structure matching the old interface
    
    Args:
        predicted_entities: List of predicted entity dictionaries
        gold_entities: List of gold standard entity dictionaries
    
    Returns:
        Dictionary containing simplified metrics for compatibility
    """
    # Use the new strict span metrics function
    detailed_metrics = calculate_strict_span_metrics(predicted_entities, gold_entities)
    
    # Return in the old format for compatibility
    micro_metrics = detailed_metrics['micro']
    
    # Create per_type_metrics in old format
    per_type_old_format = {}
    for etype, metrics in detailed_metrics['per_type'].items():
        per_type_old_format[etype] = {
            'accuracy': metrics['recall'],  # In old format, accuracy was same as recall
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'support': metrics['support']
        }
    
    return {
        'precision': micro_metrics['precision'],
        'recall': micro_metrics['recall'],
        'f1': micro_metrics['f1'],
        'true_positives': micro_metrics['tp'],
        'false_positives': micro_metrics['fp'],
        'false_negatives': micro_metrics['fn'],
        'per_type_metrics': per_type_old_format
    }

def save_prompt_templates_from_experiment(experiment_dir: Path):
    """
    Save prompt templates for all models by extracting from their result files
    
    Args:
        experiment_dir: Path to experiment directory containing model_results
    """
    model_results_dir = experiment_dir / "model_results"
    if not model_results_dir.exists():
        print(f"No model_results directory found in {experiment_dir}")
        return
    
    prompts_dir = experiment_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    
    saved_count = 0
    
    for json_file in model_results_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract required data
            model_name = data.get('model_name', json_file.stem)
            iteration_number = data.get('iteration_number', 0)
            detailed_results = data.get('detailed_results', [])
            
            if not detailed_results:
                print(f"No detailed_results for {model_name}")
                continue
            
            text = detailed_results[0].get('text', '')
            final_prompt = detailed_results[0].get('final_prompt', '')
            
            if not text or not final_prompt:
                print(f"Missing text or final_prompt for {model_name}")
                continue
            
            # Create f-string template
            final_prompt_fstring = final_prompt.replace(text, "{document}")
            
            # Create safe filename
            safe_model_name = model_name.replace(":", "_").replace("/", "_")
            template_filename = f"{safe_model_name}_iter{iteration_number}_prompt_template.txt"
            template_path = prompts_dir / template_filename
            
            # Save template
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(final_prompt_fstring)
            
            print(f"Saved prompt template: {template_filename}")
            saved_count += 1
            
        except Exception as e:
            print(f"Failed to process {json_file.name}: {e}")
    
    print(f"Saved {saved_count} prompt templates in {prompts_dir}")
