import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from collections import Counter
from datetime import datetime

VERBOSE = 1

class NERErrorAnalyzer:
    """
    Analyze NER prediction errors by comparing individual model predictions 
    with majority voting results and categorize errors into 4 types:
    1. O -> Ent (False Positive): O labeled as Entity
    2. Ent -> O (False Negative): Entity labeled as O  
    3. Ent -> Ent (Wrong Type): Entity labeled as different entity type
    4. Span Error: Correct entity type but wrong span boundaries
    
    Counts errors both at token level and sample level.
    """
    def __init__(self, results_data: Dict[str, Any]):
        """
        Initialize analyzer with experiment results
        MODIFIED: Added gold standard support
        
        Args:
            results_data: Results from experiments_250818_grouping.py
        """
        self.results_data = results_data
        self.supervised_by_gold_standard = False  # NEW: Track analysis mode
        
        # Extract entity types from experiment info
        experiment_info = results_data.get('experiment_info', {})
        ner_scheme = experiment_info.get('ner_scheme', {})
        
        if ner_scheme:
            self.entity_types = list(ner_scheme.keys())
            self.ner_scheme = ner_scheme
            if VERBOSE >= 1:
                print(f"Entity types from experiment: {self.entity_types}")
        else:
            # Fallback to default CoNLL entity types
            self.entity_types = ['PER', 'ORG', 'LOC', 'MISC']
            self.ner_scheme = {}
            if VERBOSE >= 1:
                print(f"Using default entity types: {self.entity_types}")
        
        self.error_categories = ['O_to_Ent', 'Ent_to_O', 'Ent_to_Ent', 'Span_Error']
        
        # Create detailed error categories
        self.detailed_error_categories = []
        
        # O_to_Ent categories (by entity type)
        for ent_type in self.entity_types:
            self.detailed_error_categories.append(f'O_to_{ent_type}')
        
        # Ent_to_O categories (by entity type)
        for ent_type in self.entity_types:
            self.detailed_error_categories.append(f'{ent_type}_to_O')
        
        # Ent_to_Ent categories (by entity type transitions)
        all_types = self.entity_types
        for from_type in all_types:
            for to_type in all_types:
                if from_type != to_type:
                    self.detailed_error_categories.append(f'{from_type}_to_{to_type}')
        
        # Span Error at the end
        self.detailed_error_categories.append('Span_Error')
        
        # Extract test samples and model results
        self.test_samples = results_data.get('test_samples', [])
        self.results_by_model = results_data.get('results_by_model', {})
        
        # Initialize error storage
        self.model_errors = {}
        self.majority_voting_labels = []
        self.gold_standard_labels = []  # NEW: For gold standard mode
        
        if VERBOSE >= 1:
            print(f"Initialized NER Error Analyzer")
            print(f"Total samples: {len(self.test_samples)}")
            print(f"Models found: {list(self.results_by_model.keys())}")
            print(f"NER scheme loaded: {bool(self.ner_scheme)}")

    def prepare_gold_standard_reference(self) -> List[List[str]]:
        """
        Prepare gold standard labels as reference
        NEW: For gold standard mode
        
        Returns:
            List of gold standard labels for each sample
        """
        if VERBOSE >= 1:
            print("Preparing gold standard reference labels...")
        
        gold_labels = []
        
        for sample_idx, sample in enumerate(self.test_samples):
            gold_sample_labels = sample.get('labels', [])
            tokens = sample.get('tokens', [])
            
            if not gold_sample_labels:
                # No gold labels - use O labels
                gold_labels.append(['O'] * len(tokens))
                continue
            
            # Ensure gold labels match token count
            if len(gold_sample_labels) != len(tokens):
                if VERBOSE >= 2:
                    print(f"  Warning: Gold label count mismatch for sample {sample_idx}")
                    print(f"    Tokens: {len(tokens)}, Gold labels: {len(gold_sample_labels)}")
                # Pad or truncate
                if len(gold_sample_labels) < len(tokens):
                    gold_sample_labels = gold_sample_labels + ['O'] * (len(tokens) - len(gold_sample_labels))
                else:
                    gold_sample_labels = gold_sample_labels[:len(tokens)]
            
            gold_labels.append(gold_sample_labels)
        
        self.gold_standard_labels = gold_labels
        if VERBOSE >= 1:
            valid_samples = sum(1 for labels in gold_labels if any(label != 'O' for label in labels))
            print(f"Prepared gold standard labels for {len(gold_labels)} samples ({valid_samples} with entities)")
        return gold_labels

    def extract_entity_positions(self, labels: List[str]) -> Set[Tuple[int, int, str]]:
        """
        Extract entity positions and types from BIO labels
        
        Args:
            labels: List of BIO format labels
            
        Returns:
            Set of (start_idx, end_idx, entity_type) tuples
        """
        entities = set()
        current_entity = None
        start_idx = None
        
        for i, label in enumerate(labels):
            if label.startswith('B-'):
                # Start of new entity
                if current_entity is not None:
                    # Close previous entity
                    entities.add((start_idx, i-1, current_entity))
                current_entity = label[2:]  # Remove 'B-' prefix
                start_idx = i
            elif label.startswith('I-'):
                # Continuation of entity
                entity_type = label[2:]  # Remove 'I-' prefix
                if current_entity != entity_type:
                    # Type mismatch - close previous and start new
                    if current_entity is not None:
                        entities.add((start_idx, i-1, current_entity))
                    current_entity = entity_type
                    start_idx = i
            else:  # 'O' or other
                # End of entity
                if current_entity is not None:
                    entities.add((start_idx, i-1, current_entity))
                    current_entity = None
                    start_idx = None
        
        # Handle entity at end of sequence
        if current_entity is not None:
            entities.add((start_idx, len(labels)-1, current_entity))
        
        return entities

    def compute_majority_voting(self) -> List[List[str]]:
        """
        Compute majority voting labels for each sample across all models
        
        Returns:
            List of majority voting labels for each sample
        """
        if VERBOSE >= 1:
            print("Computing majority voting labels...")
            
        majority_labels = []
        model_names = list(self.results_by_model.keys())
        
        for sample_idx, sample in enumerate(self.test_samples):
            # Get predictions from all models for this sample
            model_predictions = []
            
            for model_name in model_names:
                model_result = self.results_by_model[model_name]
                if 'detailed_results' in model_result:
                    sample_result = model_result['detailed_results'][sample_idx]
                    if 'predicted_labels' in sample_result:
                        predicted_labels = sample_result['predicted_labels']
                        model_predictions.append(predicted_labels)
            
            if not model_predictions:
                # No valid predictions - use O labels
                sample_tokens = sample['tokens']
                majority_labels.append(['O'] * len(sample_tokens))
                continue
            
            # Ensure all predictions have same length
            max_length = max(len(pred) for pred in model_predictions)
            for pred in model_predictions:
                while len(pred) < max_length:
                    pred.append('O')
            
            # Compute majority vote for each position
            sample_majority = []
            for token_idx in range(max_length):
                token_votes = [pred[token_idx] for pred in model_predictions if token_idx < len(pred)]
                if token_votes:
                    # Get most common label
                    vote_counts = Counter(token_votes)
                    majority_label = vote_counts.most_common(1)[0][0]
                    sample_majority.append(majority_label)
                else:
                    sample_majority.append('O')
            
            majority_labels.append(sample_majority)
        
        self.majority_voting_labels = majority_labels
        if VERBOSE >= 1:
            print(f"Computed majority voting for {len(majority_labels)} samples")
        return majority_labels

    def categorize_error(self, pred_entities: Set[Tuple[int, int, str]], 
                        maj_entities: Set[Tuple[int, int, str]], 
                        pred_labels: List[str], 
                        maj_labels: List[str],
                        sample_idx: int) -> Dict[str, List[Dict]]:
        """
        Categorize errors between predicted and majority voting entities
        """
        errors = {category: [] for category in self.detailed_error_categories}
        
        # Get sample tokens for context
        sample_tokens = self.test_samples[sample_idx]['tokens'] if sample_idx < len(self.test_samples) else []
        
        # Helper function to normalize entity type (OTHER -> O 변환)
        def normalize_entity_type(ent_type: str) -> str:
            """Convert OTHER or undefined types to O for error analysis"""
            if ent_type not in self.entity_types:
                if VERBOSE >= 2:
                    print(f"  Converting undefined entity type '{ent_type}' to 'O' for analysis")
                return 'O'
            return ent_type
        
        # 1. O -> Ent errors (False Positives)
        for start, end, ent_type in pred_entities:
            # Check if this predicted entity overlaps with any majority entity
            overlaps = False
            for maj_start, maj_end, maj_type in maj_entities:
                if not (end < maj_start or start > maj_end):  # Has overlap
                    overlaps = True
                    break
            
            if not overlaps:
                # Check if majority labels in this span are mostly 'O'
                maj_span_labels = maj_labels[start:end+1]
                o_count = sum(1 for label in maj_span_labels if label == 'O')
                if o_count > len(maj_span_labels) / 2:  # Majority are 'O'
                    # Normalize entity type
                    normalized_ent_type = normalize_entity_type(ent_type)
                    
                    if normalized_ent_type == 'O':
                        # 미정의 타입은 O로 처리되므로 O->O는 에러가 아님 (스킵)
                        continue
                    
                    error_category = f'O_to_{normalized_ent_type}'
                    
                    token_span = sample_tokens[start:end+1] if start < len(sample_tokens) else []
                    errors[error_category].append({
                        'sample_idx': sample_idx,
                        'span': (start, end),
                        'predicted_type': ent_type,  # 원본 타입 보존
                        'normalized_type': normalized_ent_type,  # 정규화된 타입 추가
                        'majority_labels': maj_span_labels,
                        'tokens_span': token_span,
                        'token_count': end - start + 1,
                        'context_tokens': sample_tokens if sample_tokens else []
                    })
        
        # 2. Ent -> O errors (False Negatives)
        for start, end, ent_type in maj_entities:
            # Check if this majority entity overlaps with any predicted entity
            overlaps = False
            for pred_start, pred_end, pred_type in pred_entities:
                if not (end < pred_start or start > pred_end):  # Has overlap
                    overlaps = True
                    break
            
            if not overlaps:
                # Check if predicted labels in this span are mostly 'O'
                pred_span_labels = pred_labels[start:end+1]
                o_count = sum(1 for label in pred_span_labels if label == 'O')
                if o_count > len(pred_span_labels) / 2:  # Majority are 'O'
                    # Normalize entity type
                    normalized_ent_type = normalize_entity_type(ent_type)
                    
                    if normalized_ent_type == 'O':
                        # 미정의 타입은 O로 처리되므로 O->O는 에러가 아님 (스킵)
                        continue
                    
                    error_category = f'{normalized_ent_type}_to_O'
                    
                    token_span = sample_tokens[start:end+1] if start < len(sample_tokens) else []
                    errors[error_category].append({
                        'sample_idx': sample_idx,
                        'span': (start, end),
                        'majority_type': ent_type,  # 원본 타입 보존
                        'normalized_type': normalized_ent_type,  # 정규화된 타입 추가
                        'predicted_labels': pred_span_labels,
                        'tokens_span': token_span,
                        'token_count': end - start + 1,
                        'context_tokens': sample_tokens[max(0, start-3):min(len(sample_tokens), end+4)] if sample_tokens else []
                    })
        
        # 3. Ent -> Ent errors (Wrong Type)
        for pred_start, pred_end, pred_type in pred_entities:
            for maj_start, maj_end, maj_type in maj_entities:
                # Check for overlap
                if not (pred_end < maj_start or pred_start > maj_end):
                    # Normalize both types
                    normalized_pred_type = normalize_entity_type(pred_type)
                    normalized_maj_type = normalize_entity_type(maj_type)
                    
                    if normalized_pred_type != normalized_maj_type:
                        # Skip O->O transitions
                        if normalized_pred_type == 'O' and normalized_maj_type == 'O':
                            continue
                        
                        # Create error category
                        error_category = f'{normalized_maj_type}_to_{normalized_pred_type}'
                        
                        overlap_start = max(pred_start, maj_start)
                        overlap_end = min(pred_end, maj_end)
                        overlap_tokens = sample_tokens[overlap_start:overlap_end+1] if overlap_start < len(sample_tokens) else []
                        errors[error_category].append({
                            'sample_idx': sample_idx,
                            'predicted_span': (pred_start, pred_end),
                            'majority_span': (maj_start, maj_end),
                            'predicted_type': pred_type,  # 원본 타입 보존
                            'majority_type': maj_type,  # 원본 타입 보존
                            'normalized_predicted_type': normalized_pred_type,  # 정규화된 타입
                            'normalized_majority_type': normalized_maj_type,  # 정규화된 타입
                            'overlap_start': overlap_start,
                            'overlap_end': overlap_end,
                            'overlap_tokens': overlap_tokens,
                            'token_count': overlap_end - overlap_start + 1,
                            'context_tokens': sample_tokens[max(0, overlap_start-3):min(len(sample_tokens), overlap_end+4)] if sample_tokens else []
                        })
                    elif (pred_start, pred_end) != (maj_start, maj_end):
                        # 4. Same type but different span
                        span_start = min(pred_start, maj_start)
                        span_end = max(pred_end, maj_end)
                        span_tokens = sample_tokens[span_start:span_end+1] if span_start < len(sample_tokens) else []
                        errors['Span_Error'].append({
                            'sample_idx': sample_idx,
                            'predicted_span': (pred_start, pred_end),
                            'majority_span': (maj_start, maj_end),
                            'entity_type': pred_type,  # 원본 타입 보존
                            'normalized_entity_type': normalized_pred_type,  # 정규화된 타입
                            'span_diff_start': pred_start - maj_start,
                            'span_diff_end': pred_end - maj_end,
                            'affected_tokens': span_tokens,
                            'token_count': span_end - span_start + 1,
                            'context_tokens': sample_tokens[max(0, span_start-3):min(len(sample_tokens), span_end+4)] if sample_tokens else []
                        })
        
        return errors
    # def categorize_error(self, pred_entities: Set[Tuple[int, int, str]], 
    #                     maj_entities: Set[Tuple[int, int, str]], 
    #                     pred_labels: List[str], 
    #                     maj_labels: List[str],
    #                     sample_idx: int) -> Dict[str, List[Dict]]:
    #     """
    #     Categorize errors between predicted and majority voting entities
        
    #     Args:
    #         pred_entities: Predicted entities as (start, end, type) tuples
    #         maj_entities: Majority voting entities as (start, end, type) tuples  
    #         pred_labels: Predicted BIO labels
    #         maj_labels: Majority voting BIO labels
    #         sample_idx: Index of the current sample
            
    #     Returns:
    #         Dictionary with error categories and their instances
    #     """
    #     errors = {category: [] for category in self.detailed_error_categories}
        
    #     # Get sample tokens for context
    #     sample_tokens = self.test_samples[sample_idx]['tokens'] if sample_idx < len(self.test_samples) else []
        
    #     def normalize_entity_type(ent_type: str) -> str:
    #         """Convert OTHER or undefined types to O for error analysis"""
    #         if ent_type not in self.entity_types:
    #             if VERBOSE >= 2:
    #                 print(f"  Converting undefined entity type '{ent_type}' to 'O' for analysis")
    #             return 'O'
    #         return ent_type

    #     # 1. O -> Ent errors (False Positives) - now categorized by entity type
    #     for start, end, ent_type in pred_entities:
    #         if ent_type not in self.entity_types:
    #             if VERBOSE >= 2:
    #                 print(f"  Skipping undefined entity type in error analysis: '{ent_type}'")
    #             continue

    #         # Check if this predicted entity overlaps with any majority entity
    #         overlaps = False
    #         for maj_start, maj_end, maj_type in maj_entities:
    #             if not (end < maj_start or start > maj_end):  # Has overlap
    #                 overlaps = True
    #                 break
            
    #         if not overlaps:
    #             # Check if majority labels in this span are mostly 'O'
    #             maj_span_labels = maj_labels[start:end+1]
    #             o_count = sum(1 for label in maj_span_labels if label == 'O')
    #             if o_count > len(maj_span_labels) / 2:  # Majority are 'O'
    #                 error_category = f'O_to_{ent_type}'
                    
    #                 token_span = sample_tokens[start:end+1] if start < len(sample_tokens) else []
    #                 errors[error_category].append({
    #                     'sample_idx': sample_idx,
    #                     'span': (start, end),
    #                     'predicted_type': ent_type,
    #                     'majority_labels': maj_span_labels,
    #                     'tokens_span': token_span,
    #                     'token_count': end - start + 1,
    #                     'context_tokens': sample_tokens if sample_tokens else []
    #                 })
        
    #     # 2. Ent -> O errors (False Negatives) - now categorized by entity type
    #     for start, end, ent_type in maj_entities:
    #         # Check if this majority entity overlaps with any predicted entity
    #         overlaps = False
    #         for pred_start, pred_end, pred_type in pred_entities:
    #             if not (end < pred_start or start > pred_end):  # Has overlap
    #                 overlaps = True
    #                 break
            
    #         if not overlaps:
    #             # Check if predicted labels in this span are mostly 'O'
    #             pred_span_labels = pred_labels[start:end+1]
    #             o_count = sum(1 for label in pred_span_labels if label == 'O')
    #             if o_count > len(pred_span_labels) / 2:  # Majority are 'O'
    #                 # Check if entity type is in our defined types, otherwise use 'OTHER'
    #                 if ent_type in self.entity_types:
    #                     error_category = f'{ent_type}_to_O'
    #                 else:
    #                     error_category = 'OTHER_to_O'
    #                     if VERBOSE >= 2:
    #                         print(f"  Warning: Unexpected entity type '{ent_type}' found, categorizing as 'OTHER'")
                    
    #                 token_span = sample_tokens[start:end+1] if start < len(sample_tokens) else []
    #                 errors[error_category].append({
    #                     'sample_idx': sample_idx,
    #                     'span': (start, end),
    #                     'majority_type': ent_type,
    #                     'predicted_labels': pred_span_labels,
    #                     'tokens_span': token_span,
    #                     'token_count': end - start + 1,
    #                     'context_tokens': sample_tokens[max(0, start-3):min(len(sample_tokens), end+4)] if sample_tokens else []
    #                 })
        
    #     # 3. Ent -> Ent errors (Wrong Type) - now categorized by specific type transitions
    #     for pred_start, pred_end, pred_type in pred_entities:
    #         for maj_start, maj_end, maj_type in maj_entities:
    #             # Check for overlap
    #             if not (pred_end < maj_start or pred_start > maj_end):
    #                 if pred_type != maj_type:
    #                     # Different entity types - categorize by specific transition
    #                     maj_type_safe = maj_type if maj_type in self.entity_types else 'OTHER'
    #                     pred_type_safe = pred_type if pred_type in self.entity_types else 'OTHER'
    #                     error_category = f'{maj_type_safe}_to_{pred_type_safe}'
                        
    #                     if VERBOSE >= 2 and (maj_type != maj_type_safe or pred_type != pred_type_safe):
    #                         print(f"  Warning: Unexpected entity type transition '{maj_type}' → '{pred_type}', categorizing as '{error_category}'")
                        
    #                     overlap_start = max(pred_start, maj_start)
    #                     overlap_end = min(pred_end, maj_end)
    #                     overlap_tokens = sample_tokens[overlap_start:overlap_end+1] if overlap_start < len(sample_tokens) else []
    #                     errors[error_category].append({
    #                         'sample_idx': sample_idx,
    #                         'predicted_span': (pred_start, pred_end),
    #                         'majority_span': (maj_start, maj_end),
    #                         'predicted_type': pred_type,
    #                         'majority_type': maj_type,
    #                         'overlap_start': overlap_start,
    #                         'overlap_end': overlap_end,
    #                         'overlap_tokens': overlap_tokens,
    #                         'token_count': overlap_end - overlap_start + 1,
    #                         'context_tokens': sample_tokens[max(0, overlap_start-3):min(len(sample_tokens), overlap_end+4)] if sample_tokens else []
    #                     })
    #                 elif (pred_start, pred_end) != (maj_start, maj_end):
    #                     # 4. Same type but different span
    #                     span_start = min(pred_start, maj_start)
    #                     span_end = max(pred_end, maj_end)
    #                     span_tokens = sample_tokens[span_start:span_end+1] if span_start < len(sample_tokens) else []
    #                     errors['Span_Error'].append({
    #                         'sample_idx': sample_idx,
    #                         'predicted_span': (pred_start, pred_end),
    #                         'majority_span': (maj_start, maj_end),
    #                         'entity_type': pred_type,
    #                         'span_diff_start': pred_start - maj_start,
    #                         'span_diff_end': pred_end - maj_end,
    #                         'affected_tokens': span_tokens,
    #                         'token_count': span_end - span_start + 1,
    #                         'context_tokens': sample_tokens[max(0, span_start-3):min(len(sample_tokens), span_end+4)] if sample_tokens else []
    #                     })
        
    #     return errors

    def analyze_model_errors(self) -> Dict[str, Dict]:
        """
        Analyze errors for each model compared to reference (MV or Gold Standard)
        MODIFIED: Support both MV and gold standard modes
        
        Returns:
            Dictionary with error analysis for each model
        """
        reference_mode = "gold standard" if self.supervised_by_gold_standard else "majority voting"
        if VERBOSE >= 1:
            print(f"Analyzing model errors against {reference_mode}...")
        
        # Prepare reference labels
        if self.supervised_by_gold_standard:
            if not self.gold_standard_labels:
                self.prepare_gold_standard_reference()
            reference_labels = self.gold_standard_labels
        else:
            if not self.majority_voting_labels:
                self.compute_majority_voting()
            reference_labels = self.majority_voting_labels
        
        model_error_analysis = {}
        
        for model_name, model_result in self.results_by_model.items():
            if VERBOSE >= 1:
                print(f"Analyzing errors for model: {model_name} vs {reference_mode}")
            
            if 'detailed_results' not in model_result:
                if VERBOSE >= 1:
                    print(f"  No detailed results found for {model_name}")
                continue
            
            model_errors = {
                'token_level_errors': {category: Counter() for category in self.detailed_error_categories},
                'sample_level_errors': {category: Counter() for category in self.detailed_error_categories},
                'error_samples': {category: [] for category in self.detailed_error_categories},
                'error_instances': {category: [] for category in self.detailed_error_categories},
                'total_token_errors': Counter(),
                'total_sample_errors': Counter(),
                'sample_error_details': [],
                'reference_mode': reference_mode  # NEW: Track reference mode
            }
            
            detailed_results = model_result['detailed_results']
            
            for sample_idx, sample_result in enumerate(detailed_results):
                if 'error' in sample_result:
                    # Skip samples with processing errors
                    continue
                
                if sample_idx >= len(reference_labels):
                    continue
                
                predicted_labels = sample_result.get('predicted_labels', [])
                ref_labels = reference_labels[sample_idx]
                
                # Extract entities
                pred_entities = self.extract_entity_positions(predicted_labels)
                ref_entities = self.extract_entity_positions(ref_labels)
                
                # Categorize errors for this sample
                sample_errors = self.categorize_error(
                    pred_entities, ref_entities, predicted_labels, ref_labels, sample_idx
                )
                
                # Count errors by category and entity type (same logic as before)
                sample_error_summary = {'sample_idx': sample_idx, 'errors': {}}
                sample_has_errors = False
                
                for error_category, error_instances in sample_errors.items():
                    if error_instances:
                        sample_has_errors = True
                        model_errors['error_samples'][error_category].append(sample_idx)
                        model_errors['error_instances'][error_category].extend(error_instances)
                        sample_error_summary['errors'][error_category] = len(error_instances)
                        
                        # Count at sample level (1 per sample regardless of error count)
                        model_errors['sample_level_errors'][error_category]['count'] += 1
                        
                        # Count at token level (sum of all token counts in errors)
                        total_tokens_in_category = sum(inst.get('token_count', 1) for inst in error_instances)
                        model_errors['token_level_errors'][error_category]['count'] += total_tokens_in_category
                        
                        # For detailed O_to_Ent categories, also update general counters
                        if error_category.startswith('O_to_'):
                            entity_type = error_instances[0].get('predicted_type', 'OTHER')
                            if entity_type in self.entity_types:
                                model_errors['total_token_errors'][entity_type] += total_tokens_in_category
                                model_errors['total_sample_errors'][entity_type] += 1
                            else:
                                model_errors['total_token_errors']['OTHER'] += total_tokens_in_category
                                model_errors['total_sample_errors']['OTHER'] += 1
                
                if sample_has_errors:
                    model_errors['sample_error_details'].append(sample_error_summary)
            
            model_error_analysis[model_name] = model_errors
            
            # Print summary for this model
            if VERBOSE >= 1:
                total_token_errors = sum(sum(counter.values()) for counter in model_errors['token_level_errors'].values())
                total_sample_errors = sum(sum(counter.values()) for counter in model_errors['sample_level_errors'].values())
                print(f"  Total token-level errors: {total_token_errors}")
                print(f"  Total sample-level errors: {total_sample_errors}")
                if VERBOSE >= 2:
                    for category in self.detailed_error_categories:
                        token_count = model_errors['token_level_errors'][category].get('count', 0)
                        sample_count = model_errors['sample_level_errors'][category].get('count', 0)
                        if token_count > 0 or sample_count > 0:
                            print(f"  {category}: {token_count} tokens, {sample_count} samples")
        
        self.model_errors = model_error_analysis
        return model_error_analysis

    def generate_model_documents(self, output_dir: str = "error_analysis_results",
                            source_file_path: str = None,
                            supervised_by_gold_standard: bool = False) -> Dict[str, str]:
        """
        Generate per-model markdown reports.
        MODIFIED: Added mode-specific file naming and content
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_documents: Dict[str, str] = {}
        reference_mode = "Gold Standard" if supervised_by_gold_standard else "Majority Voting"
        mode_suffix = "_goldstd" if supervised_by_gold_standard else "_mv"

        for model_name, model_data in self.model_errors.items():
            clean_model_name = sanitize_model_name(model_name)
            doc_file = output_path / f"model_error_analysis_{clean_model_name}{mode_suffix}.md"

            # Skip if file already exists
            if doc_file.exists():
                if VERBOSE >= 1:
                    print(f"Skip existing: {doc_file}")
                model_documents[model_name] = str(doc_file)
                continue

            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write(f"# NER Error Analysis Report: {model_name}\n")
                f.write(f"**Reference Mode**: {reference_mode}\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")

                # Overall Statistics
                f.write("## Overall Error Statistics\n\n")
                total_token_errors = sum(sum(counter.values()) for counter in model_data['token_level_errors'].values())
                total_sample_errors = sum(sum(counter.values()) for counter in model_data['sample_level_errors'].values())
                f.write(f"- **Reference Mode**: {reference_mode}\n")
                f.write(f"- **Total Token-Level Errors**: {total_token_errors}\n")
                f.write(f"- **Total Sample-Level Errors**: {total_sample_errors}\n")
                f.write(f"- **Samples Analyzed**: {len(self.test_samples)}\n")
                f.write(f"- **Error Rate (Samples)**: {total_sample_errors/len(self.test_samples)*100:.2f}%\n\n")

                # Error Breakdown by Category
                f.write("## Error Breakdown by Category\n\n")
                f.write("| Error Category | Token-Level Count | Sample-Level Count | Sample Indices |\n")
                f.write("|---|---|---|---|\n")
                for category in self.detailed_error_categories:
                    token_count = model_data['token_level_errors'][category].get('count', 0)
                    sample_count = model_data['sample_level_errors'][category].get('count', 0)
                    if token_count > 0 or sample_count > 0:  # Only show categories with errors
                        error_samples = model_data['error_samples'][category]
                        sample_indices = ', '.join(map(str, error_samples[:10]))
                        if len(error_samples) > 10:
                            sample_indices += f" (+{len(error_samples)-10} more)"
                        f.write(f"| {category} | {token_count} | {sample_count} | {sample_indices} |\n")
                f.write("\n")

                # Entity Type Summary
                f.write("## Entity Type Error Summary (O → Entity)\n\n")
                f.write("| Entity Type | Token-Level Count | Sample-Level Count |\n")
                f.write("|---|---|---|\n")
                all_entity_types = self.entity_types + ['OTHER']
                for entity_type in all_entity_types:
                    tcnt = model_data['total_token_errors'].get(entity_type, 0)
                    scnt = model_data['total_sample_errors'].get(entity_type, 0)
                    if tcnt > 0 or scnt > 0:
                        f.write(f"| {entity_type} | {tcnt} | {scnt} |\n")
                f.write("\n")

                # Representative Examples
                f.write("## Representative Error Examples\n\n")
                for category in self.detailed_error_categories:
                    error_instances = model_data['error_instances'][category]
                    if not error_instances:
                        continue
                    error_sample_count = len(set(inst['sample_idx'] for inst in error_instances))
                    error_percentage = (error_sample_count / len(self.test_samples)) * 100
                    f.write(f"### {category} ({len(error_instances)} cases, {error_percentage:.1f}% of samples)\n\n")

                    examples_to_show = min(5, len(error_instances))
                    for i, instance in enumerate(error_instances[:examples_to_show]):
                        f.write(f"**Example {i+1}** (Sample {instance['sample_idx']}):\n")
                        
                        # Show tokens
                        if 'tokens_span' in instance and instance['tokens_span']:
                            f.write(f"- **Tokens**: {' '.join(instance['tokens_span'])}\n")
                        elif 'overlap_tokens' in instance and instance['overlap_tokens']:
                            f.write(f"- **Tokens**: {' '.join(instance['overlap_tokens'])}\n")
                        elif 'affected_tokens' in instance and instance['affected_tokens']:
                            f.write(f"- **Tokens**: {' '.join(instance['affected_tokens'])}\n")
                        
                        # Show context
                        if 'context_tokens' in instance and instance['context_tokens']:
                            f.write(f"- **Context**: {' '.join(instance['context_tokens'])}\n")

                        # Show error details based on category
                        if category.startswith('O_to_'):
                            explanation = " (model incorrectly predicted entity)" if i == 0 else ""
                            f.write(f"- **{reference_mode} → Predicted Label**: O → {instance.get('predicted_type', 'N/A')}{explanation}\n")
                        elif category.endswith('_to_O'):
                            explanation = " (model missed entity)" if i == 0 else ""
                            ref_type = instance.get('majority_type', instance.get('gold_type', 'N/A'))
                            f.write(f"- **{reference_mode} → Predicted Label**: {ref_type} → O{explanation}\n")
                        elif '_to_' in category and category != 'Span_Error':
                            explanation = " (model confused entity type)" if i == 0 else ""
                            ref_type = instance.get('majority_type', instance.get('gold_type', 'N/A'))
                            pred_type = instance.get('predicted_type', 'N/A')
                            f.write(f"- **{reference_mode} → Predicted Label**: {ref_type} → {pred_type}{explanation}\n")
                        elif category == 'Span_Error':
                            maj_span = instance.get('majority_span', instance.get('gold_span', (0, 0)))
                            pred_span = instance.get('predicted_span', (0, 0))
                            maj_start, maj_end = maj_span
                            pred_start, pred_end = pred_span
                            instance_sample_idx = instance['sample_idx']
                            instance_tokens = self.test_samples[instance_sample_idx]['tokens'] if instance_sample_idx < len(self.test_samples) else []
                            maj_tokens = instance_tokens[maj_start:maj_end+1] if maj_start < len(instance_tokens) else []
                            pred_tokens = instance_tokens[pred_start:pred_end+1] if pred_start < len(instance_tokens) else []
                            f.write(f"- **Entity Type**: {instance.get('entity_type', 'N/A')}\n")
                            explanation = " (model detected wrong span)" if i == 0 else ""
                            f.write(f"- **{reference_mode} → Predicted Span**: \"{' '.join(maj_tokens)}\" → \"{' '.join(pred_tokens)}\"{explanation}\n")
                        f.write("\n")

                    if len(error_instances) > examples_to_show:
                        f.write(f"*... and {len(error_instances) - examples_to_show} more examples*\n\n")

            model_documents[model_name] = str(doc_file)
            if VERBOSE >= 1:
                print(f"Generated model document: {doc_file}")

        return model_documents
    # def analyze_model_errors(self) -> Dict[str, Dict]:
    #     """
    #     Analyze errors for each model compared to majority voting
        
    #     Returns:
    #         Dictionary with error analysis for each model
    #     """
    #     if VERBOSE >= 1:
    #         print("Analyzing model errors...")
        
    #     if not self.majority_voting_labels:
    #         self.compute_majority_voting()
        
    #     model_error_analysis = {}
        
    #     for model_name, model_result in self.results_by_model.items():
    #         if VERBOSE >= 1:
    #             print(f"Analyzing errors for model: {model_name}")
            
    #         if 'detailed_results' not in model_result:
    #             if VERBOSE >= 1:
    #                 print(f"  No detailed results found for {model_name}")
    #             continue
            
    #         model_errors = {
    #             'token_level_errors': {category: Counter() for category in self.detailed_error_categories},
    #             'sample_level_errors': {category: Counter() for category in self.detailed_error_categories},
    #             'error_samples': {category: [] for category in self.detailed_error_categories},
    #             'error_instances': {category: [] for category in self.detailed_error_categories},
    #             'total_token_errors': Counter(),
    #             'total_sample_errors': Counter(),
    #             'sample_error_details': []
    #         }
            
    #         detailed_results = model_result['detailed_results']
            
    #         for sample_idx, sample_result in enumerate(detailed_results):
    #             if 'error' in sample_result:
    #                 # Skip samples with processing errors
    #                 continue
                
    #             predicted_labels = sample_result.get('predicted_labels', [])
    #             majority_labels = self.majority_voting_labels[sample_idx]
                
    #             # Extract entities
    #             pred_entities = self.extract_entity_positions(predicted_labels)
    #             maj_entities = self.extract_entity_positions(majority_labels)
                
    #             # Categorize errors for this sample
    #             sample_errors = self.categorize_error(
    #                 pred_entities, maj_entities, predicted_labels, majority_labels, sample_idx
    #             )
                
    #             # Count errors by category and entity type
    #             sample_error_summary = {'sample_idx': sample_idx, 'errors': {}}
    #             sample_has_errors = False
                
    #             for error_category, error_instances in sample_errors.items():
    #                 if error_instances:
    #                     sample_has_errors = True
    #                     model_errors['error_samples'][error_category].append(sample_idx)
    #                     model_errors['error_instances'][error_category].extend(error_instances)
    #                     sample_error_summary['errors'][error_category] = len(error_instances)
                        
    #                     # Count at sample level (1 per sample regardless of error count)
    #                     model_errors['sample_level_errors'][error_category]['count'] += 1
                        
    #                     # Count at token level (sum of all token counts in errors)
    #                     total_tokens_in_category = sum(inst.get('token_count', 1) for inst in error_instances)
    #                     model_errors['token_level_errors'][error_category]['count'] += total_tokens_in_category
                        
    #                     # For detailed O_to_Ent categories, also update general counters
    #                     if error_category.startswith('O_to_'):
    #                         entity_type = error_instances[0]['predicted_type']  # Get from first instance
    #                         if entity_type in self.entity_types:
    #                             model_errors['total_token_errors'][entity_type] += total_tokens_in_category
    #                             model_errors['total_sample_errors'][entity_type] += 1
    #                         else:
    #                             model_errors['total_token_errors']['OTHER'] += total_tokens_in_category
    #                             model_errors['total_sample_errors']['OTHER'] += 1
                
    #             if sample_has_errors:
    #                 model_errors['sample_error_details'].append(sample_error_summary)
            
    #         model_error_analysis[model_name] = model_errors
            
    #         # Print summary for this model
    #         if VERBOSE >= 1:
    #             total_token_errors = sum(sum(counter.values()) for counter in model_errors['token_level_errors'].values())
    #             total_sample_errors = sum(sum(counter.values()) for counter in model_errors['sample_level_errors'].values())
    #             print(f"  Total token-level errors: {total_token_errors}")
    #             print(f"  Total sample-level errors: {total_sample_errors}")
    #             if VERBOSE >= 2:
    #                 for category in self.detailed_error_categories:
    #                     token_count = model_errors['token_level_errors'][category].get('count', 0)
    #                     sample_count = model_errors['sample_level_errors'][category].get('count', 0)
    #                     print(f"  {category}: {token_count} tokens, {sample_count} samples")
        
    #     self.model_errors = model_error_analysis
    #     return model_error_analysis

    # def generate_model_documents(self, output_dir: str = "error_analysis_results",
    #                             source_file_path: str = None) -> Dict[str, str]:
    #     """
    #     Generate per-model markdown reports.
    #     - No date/time subfolders.
    #     - Skip generation if the same filename already exists.
    #     """
    #     output_path = Path(output_dir)  # do not append subdir from source_file_path
    #     output_path.mkdir(parents=True, exist_ok=True)

    #     model_documents: Dict[str, str] = {}

    #     for model_name, model_data in self.model_errors.items():
    #         clean_model_name = sanitize_model_name(model_name)
    #         doc_file = output_path / f"model_error_analysis_{clean_model_name}.md"

    #         # Skip if file already exists
    #         if doc_file.exists():
    #             if VERBOSE >= 1:
    #                 print(f"Skip existing: {doc_file}")
    #             model_documents[model_name] = str(doc_file)
    #             continue

    #         with open(doc_file, 'w', encoding='utf-8') as f:
    #             f.write(f"# NER Error Analysis Report: {model_name}\n\n")
    #             f.write(f"Generated: {datetime.now().isoformat()}\n\n")

    #             # Overall Statistics
    #             f.write("## Overall Error Statistics\n\n")
    #             total_token_errors = sum(sum(counter.values()) for counter in model_data['token_level_errors'].values())
    #             total_sample_errors = sum(sum(counter.values()) for counter in model_data['sample_level_errors'].values())
    #             f.write(f"- **Total Token-Level Errors**: {total_token_errors}\n")
    #             f.write(f"- **Total Sample-Level Errors**: {total_sample_errors}\n")
    #             f.write(f"- **Samples Analyzed**: {len(self.test_samples)}\n")
    #             f.write(f"- **Error Rate (Samples)**: {total_sample_errors/len(self.test_samples)*100:.2f}%\n\n")

    #             # Error Breakdown by Category
    #             f.write("## Error Breakdown by Category\n\n")
    #             f.write("| Error Category | Token-Level Count | Sample-Level Count | Sample Indices |\n")
    #             f.write("|---|---|---|---|\n")
    #             for category in self.detailed_error_categories:
    #                 token_count = model_data['token_level_errors'][category].get('count', 0)
    #                 sample_count = model_data['sample_level_errors'][category].get('count', 0)
    #                 error_samples = model_data['error_samples'][category]
    #                 sample_indices = ', '.join(map(str, error_samples[:10]))
    #                 if len(error_samples) > 10:
    #                     sample_indices += f" (+{len(error_samples)-10} more)"
    #                 f.write(f"| {category} | {token_count} | {sample_count} | {sample_indices} |\n")
    #             f.write("\n")

    #             # Entity Type Summary
    #             f.write("## Entity Type Error Summary (O → Entity)\n\n")
    #             f.write("| Entity Type | Token-Level Count | Sample-Level Count |\n")
    #             f.write("|---|---|---|\n")
    #             all_entity_types = self.entity_types + ['OTHER']
    #             for entity_type in all_entity_types:
    #                 tcnt = model_data['total_token_errors'].get(entity_type, 0)
    #                 scnt = model_data['total_sample_errors'].get(entity_type, 0)
    #                 if tcnt > 0 or scnt > 0:
    #                     f.write(f"| {entity_type} | {tcnt} | {scnt} |\n")
    #             f.write("\n")

    #             # Representative Examples
    #             f.write("## Representative Error Examples\n\n")
    #             for category in self.detailed_error_categories:
    #                 error_instances = model_data['error_instances'][category]
    #                 if not error_instances:
    #                     continue
    #                 error_sample_count = len(set(inst['sample_idx'] for inst in error_instances))
    #                 error_percentage = (error_sample_count / len(self.test_samples)) * 100
    #                 f.write(f"### {category} ({len(error_instances)} cases, {error_percentage:.1f}% of samples)\n\n")

    #                 examples_to_show = min(5, len(error_instances))
    #                 for i, instance in enumerate(error_instances[:examples_to_show]):
    #                     f.write(f"**Example {i+1}** (Sample {instance['sample_idx']}):\n")
    #                     if 'tokens_span' in instance and instance['tokens_span']:
    #                         f.write(f"- **Tokens**: {' '.join(instance['tokens_span'])}\n")
    #                     elif 'overlap_tokens' in instance and instance['overlap_tokens']:
    #                         f.write(f"- **Tokens**: {' '.join(instance['overlap_tokens'])}\n")
    #                     elif 'affected_tokens' in instance and instance['affected_tokens']:
    #                         f.write(f"- **Tokens**: {' '.join(instance['affected_tokens'])}\n")
    #                     if 'context_tokens' in instance and instance['context_tokens']:
    #                         f.write(f"- **Context**: {' '.join(instance['context_tokens'])}\n")

    #                     if category.startswith('O_to_'):
    #                         explanation = " (model incorrectly predicted entity)" if i == 0 else ""
    #                         f.write(f"- **Major Voting → Predicted Label**: O → {instance.get('predicted_type', 'N/A')}{explanation}\n")
    #                     elif category.endswith('_to_O'):
    #                         explanation = " (model missed entity)" if i == 0 else ""
    #                         f.write(f"- **Major Voting → Predicted Label**: {instance.get('majority_type', 'N/A')} → O{explanation}\n")
    #                     elif '_to_' in category and category != 'Span_Error':
    #                         explanation = " (model confused entity type)" if i == 0 else ""
    #                         f.write(f"- **Major Voting → Predicted Label**: {instance.get('majority_type', 'N/A')} → {instance.get('predicted_type', 'N/A')}{explanation}\n")
    #                     elif category == 'Span_Error':
    #                         maj_start, maj_end = instance.get('majority_span', (0, 0))
    #                         pred_start, pred_end = instance.get('predicted_span', (0, 0))
    #                         instance_sample_idx = instance['sample_idx']
    #                         instance_tokens = self.test_samples[instance_sample_idx]['tokens'] if instance_sample_idx < len(self.test_samples) else []
    #                         maj_tokens = instance_tokens[maj_start:maj_end+1] if maj_start < len(instance_tokens) else []
    #                         pred_tokens = instance_tokens[pred_start:pred_end+1] if pred_start < len(instance_tokens) else []
    #                         f.write(f"- **Entity Type**: {instance.get('entity_type', 'N/A')}\n")
    #                         explanation = " (model detected wrong span)" if i == 0 else ""
    #                         f.write(f"- **Major Voting → Predicted Span**: \"{' '.join(maj_tokens)}\" → \"{' '.join(pred_tokens)}\"{explanation}\n")
    #                     f.write("\n")

    #                 if len(error_instances) > examples_to_show:
    #                     f.write(f"*... and {len(error_instances) - examples_to_show} more examples*\n\n")

    #         model_documents[model_name] = str(doc_file)
    #         if VERBOSE >= 1:
    #             print(f"Generated model document: {doc_file}")

    #     return model_documents

def analyze_ner_errors_from_file(results_file_path: str, 
                                output_dir: str = "error_analysis_results", 
                                verbose: int = 1,
                                supervised_by_gold_standard: bool = False) -> Dict[str, str]:
    """
    Load experiment results and perform error analysis
    MODIFIED: Added gold standard support
    
    Args:
        results_file_path: Path to experiment results JSON file
        output_dir: Directory to save analysis results
        verbose: Verbosity level (0: minimal, 1: normal, 2: detailed)
        supervised_by_gold_standard: Use gold standard instead of majority voting
        
    Returns:
        Dictionary mapping model names to their document file paths
    """
    global VERBOSE
    VERBOSE = verbose
    
    analysis_mode = "Gold Standard Error Analysis" if supervised_by_gold_standard else "Majority Vote Error Analysis"
    
    if VERBOSE >= 1:
        print(f"Loading results from: {results_file_path}")
        print(f"Analysis mode: {analysis_mode}")
    
    # Load results
    if results_file_path.endswith('.pkl'):
        with open(results_file_path, 'rb') as f:
            results_data = pickle.load(f)
        # Handle nested structure from pickle files
        if 'experiment_results' in results_data:
            results_data = results_data['experiment_results']
    else:
        with open(results_file_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    
    # Check if we have enough models for error analysis
    results_by_model = results_data.get('results_by_model', {})
    if len(results_by_model) < 2 and not supervised_by_gold_standard:
        if VERBOSE >= 1:
            print(f"Warning: Only {len(results_by_model)} model(s) found. Need at least 2 models for majority voting error analysis.")
        return {}
    elif len(results_by_model) < 1 and supervised_by_gold_standard:
        if VERBOSE >= 1:
            print(f"Warning: No models found for gold standard error analysis.")
        return {}
    
    # Validate gold standard availability if needed
    if supervised_by_gold_standard:
        test_samples = results_data.get('test_samples', [])
        valid_samples = sum(1 for sample in test_samples 
                           if 'labels' in sample and sample['labels'])
        if valid_samples == 0:
            if VERBOSE >= 1:
                print("Warning: No gold labels found in test samples. Cannot perform gold standard error analysis.")
            return {}
        if VERBOSE >= 1:
            print(f"Found {valid_samples}/{len(test_samples)} samples with gold labels")
    
    # Early-exit: if every expected md already exists, skip analysis
    output_path = Path(output_dir)
    model_names = list(results_by_model.keys())
    
    # Add mode suffix to filenames
    mode_suffix = "_goldstd" if supervised_by_gold_standard else "_mv"
    expected_files = {m: output_path / f"model_error_analysis_{sanitize_model_name(m)}{mode_suffix}.md"
                      for m in model_names}
    
    if model_names and all(p.exists() for p in expected_files.values()):
        if VERBOSE >= 1:
            print(f"All model markdown files already exist for {analysis_mode}. Skip error analysis.")
        return {m: str(p) for m, p in expected_files.items()}

    # Create analyzer with gold standard mode
    analyzer = NERErrorAnalyzer(results_data)
    analyzer.supervised_by_gold_standard = supervised_by_gold_standard
    
    if supervised_by_gold_standard:
        # Use gold standard as reference
        analyzer.prepare_gold_standard_reference()
    else:
        # Use majority voting as reference (existing)
        analyzer.compute_majority_voting()
    
    analyzer.analyze_model_errors()
    model_documents = analyzer.generate_model_documents(output_dir, results_file_path, supervised_by_gold_standard)

    return model_documents
# def analyze_ner_errors_from_file(results_file_path: str, output_dir: str = "error_analysis_results", verbose: int = 1) -> Dict[str, str]:
#     """
#     Load experiment results and perform error analysis
    
#     Args:
#         results_file_path: Path to experiment results JSON file
#         output_dir: Directory to save analysis results
#         verbose: Verbosity level (0: minimal, 1: normal, 2: detailed)
        
#     Returns:
#         Dictionary mapping model names to their document file paths
#     """
#     global VERBOSE
#     VERBOSE = verbose
    
#     if VERBOSE >= 1:
#         print(f"Loading results from: {results_file_path}")
    
#     # Load results
#     if results_file_path.endswith('.pkl'):
#         with open(results_file_path, 'rb') as f:
#             results_data = pickle.load(f)
#         # Handle nested structure from pickle files
#         if 'experiment_results' in results_data:
#             results_data = results_data['experiment_results']
#     else:
#         with open(results_file_path, 'r', encoding='utf-8') as f:
#             results_data = json.load(f)
    
#     # Check if we have enough models for error analysis
#     results_by_model = results_data.get('results_by_model', {})
#     if len(results_by_model) < 2:
#         if VERBOSE >= 1:
#             print(f"Warning: Only {len(results_by_model)} model(s) found. Need at least 2 models for error analysis.")
#         return {}
    
#     # Early-exit: if every expected md already exists, skip analysis
#     output_path = Path(output_dir)  # do not append date/time subdir
#     model_names = list(results_by_model.keys())
#     expected_files = {m: output_path / f"model_error_analysis_{m.replace(':','_').replace('/','_')}.md"
#                       for m in model_names}
#     if model_names and all(p.exists() for p in expected_files.values()):
#         if VERBOSE >= 1:
#             print("All model markdown files already exist. Skip error analysis.")
#         return {m: str(p) for m, p in expected_files.items()}

#     # proceed as before
#     analyzer = NERErrorAnalyzer(results_data)
#     analyzer.compute_majority_voting()
#     analyzer.analyze_model_errors()
#     model_documents = analyzer.generate_model_documents(output_dir, results_file_path)

#     return model_documents

def set_verbosity(level: int):
    """
    Set global verbosity level
    
    Args:
        level: Verbosity level (0: minimal, 1: normal, 2: detailed)
    """
    global VERBOSE
    VERBOSE = level
    if VERBOSE >= 1:
        print(f"Verbosity set to level {level}")

def run_error_analysis_on_existing_results(
    results_file_path: str, 
    output_dir: str = None,
    verbose: int = 1
) -> Dict[str, str]:
    """
    Convenience function to run error analysis on existing experiment results
    
    Args:
        results_file_path: Path to existing experiment results (JSON or pickle)
        output_dir: Custom output directory (if None, will create based on input file)
        verbose: Verbosity level
        
    Returns:
        Dictionary mapping model names to their document file paths
    """
    # Auto-generate output directory if not provided
    if output_dir is None:
        results_path = Path(results_file_path)
        if results_path.parent.name == "comprehensive_results":
            # For comprehensive results, create in same directory
            output_dir = results_path.parent / "error_analysis"
        else:
            # For other results, create relative to file
            output_dir = results_path.parent / "error_analysis"
    
    return analyze_ner_errors_from_file(
        results_file_path=results_file_path,
        output_dir=str(output_dir),
        verbose=verbose
    )

def sanitize_model_name(name: str) -> str:
    # simple and consistent file-safe name
    return name.replace(':', '_').replace('/', '_')

# def main():
#     """
#     Example usage of the NER error analysis pipeline
#     """
#     # Set verbosity level
#     set_verbosity(1)
    
#     # Example: analyze results from experiment file
#     results_file = "experiment_results/pickled_results/grouped_experiment_results_20groups_50size_20250818_172008.pkl"
    
#     if Path(results_file).exists():
#         print(f"Analyzing results from: {results_file}")
#         model_documents = analyze_ner_errors_from_file(results_file, verbose=2)
#         print(f"Analysis complete! Generated {len(model_documents)} model documents:")
#         for model_name, doc_path in model_documents.items():
#             print(f"  {model_name}: {doc_path}")
#     else:
#         print(f"Results file not found: {results_file}")
#         print("Please provide a valid path to experiment results file")
        
#         # List available result files
#         results_dir = Path("experiment_results")
#         if results_dir.exists():
#             result_files = list(results_dir.glob("*.json")) + list(results_dir.glob("*.pkl"))
#             if result_files:
#                 print("\nAvailable result files:")
#                 for i, file_path in enumerate(result_files[-10:]):  # Show last 10
#                     print(f"  {i+1}. {file_path}")


# if __name__ == "__main__":
#     main()