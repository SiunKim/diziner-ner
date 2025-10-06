# disagreement_documentation.py 최종 수정 버전

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime

from disagreement_analysis_in_pipeline import (
    load_experiment_results,
)

class HotspotDocumentationGenerator:
    """
    Generate comprehensive documentation for disagreement hotspots
    including error analysis and model reasoning
    MODIFIED: Support both MV and gold standard modes
    """
    
    DEFAULT_ENTITY_TYPES = ['PER', 'ORG', 'LOC', 'MISC']
    CONTEXT_TOKENS_BEFORE = 3
    CONTEXT_TOKENS_AFTER = 3
    MAX_EXAMPLES_PER_TYPE = 10
    MAX_CONFUSING_CASES_PER_MODEL = 2
    TOKEN_RELATEDNESS_DISTANCE = 2
    RANDOM_SEED = 42
    
    def __init__(self, results_data: Dict[str, Any], disagreement_data: Dict[str, Any]):
        """Initialize documentation generator - MODIFIED to handle gold standard mode"""
        self.results_data = results_data
        self.disagreement_data = disagreement_data
        
        self.experiment_results = results_data.get('experiment_results', results_data)
        self.test_samples = self.experiment_results.get('test_samples', [])
        self.results_by_model = self.experiment_results.get('results_by_model', {})
        
        agreement_results = results_data.get('agreement_analysis_results', {})
        mv_results = agreement_results.get('majority_voting_results', {})
        self.consensus_results = mv_results.get('consensus_results', [])
        
        self.hotspots = disagreement_data.get('disagreement_analysis', {}).get('hotspots', [])
        self.coalition_analysis = disagreement_data.get('coalition_analysis', {})
        self.weights_used = disagreement_data.get('disagreement_analysis', {}).get('weights_used', {})
        
        # Check if we're in gold standard mode
        disagreement_analysis_data = disagreement_data.get('disagreement_analysis', {})
        dataset_summary = disagreement_analysis_data.get('dataset_summary', {})
        self.use_gold_standard = dataset_summary.get('use_gold_standard', False)
        
        experiment_info = self.experiment_results.get('experiment_info', {})
        ner_scheme = experiment_info.get('ner_scheme', {})
        self.entity_types = list(ner_scheme.keys()) if ner_scheme else self.DEFAULT_ENTITY_TYPES
        
        # Fix rest models calculation (skip for gold standard mode)
        all_models = list(self.results_by_model.keys())
        if not self.use_gold_standard:
            coalition_models = self.coalition_analysis.get('coalition_models', [])
            rest_models = [m for m in all_models if m not in coalition_models]
            self.coalition_analysis['rest_models'] = rest_models
        else:
            self.coalition_analysis['coalition_models'] = []
            self.coalition_analysis['rest_models'] = all_models
        
        random.seed(self.RANDOM_SEED)
        
        analysis_mode = "Gold Standard" if self.use_gold_standard else "Majority Vote"
        print(f"Initialized documentation generator ({analysis_mode} mode)")
        print(f"Hotspots: {len(self.hotspots)}")
        print(f"Models: {list(self.results_by_model.keys())}")
        print(f"Entity types: {self.entity_types}")
        if not self.use_gold_standard:
            coalition_models = self.coalition_analysis.get('coalition_models', [])
            rest_models = self.coalition_analysis.get('rest_models', [])
            print(f"Coalition models: {coalition_models}")
            print(f"Rest models: {rest_models}")
        else:
            print("Analysis mode: All models vs Gold Standard")

    def _get_sample_data(self, sentence_id: int) -> Tuple[List[str], List[str]]:
        """Get sample tokens and reference labels (MV or Gold) - MODIFIED"""
        tokens = []
        reference_labels = []
        
        if sentence_id < len(self.test_samples):
            tokens = self.test_samples[sentence_id]['tokens']
        elif sentence_id < len(self.consensus_results):
            tokens = self.consensus_results[sentence_id].get('tokens', [])
        
        if self.use_gold_standard:
            if sentence_id < len(self.test_samples):
                reference_labels = self.test_samples[sentence_id].get('labels', [])
        else:
            if sentence_id < len(self.consensus_results):
                consensus_data = self.consensus_results[sentence_id]
                model_labels_dict = consensus_data.get('model_labels', {})
                
                if model_labels_dict:
                    for token_idx in range(len(tokens)):
                        token_votes = [labels[token_idx] for labels in model_labels_dict.values() 
                                     if token_idx < len(labels)]
                        if token_votes:
                            vote_counts = Counter(token_votes)
                            majority_label = vote_counts.most_common(1)[0][0]
                            reference_labels.append(majority_label)
                        else:
                            reference_labels.append('O')
                else:
                    reference_labels = consensus_data.get('consensus_labels', [])
        
        return tokens, reference_labels

    def _extract_entities_from_bio(self, labels: List[str]) -> Set[Tuple[int, int, str]]:
        """Extract entities from BIO labels"""
        entities = set()
        current_entity = None
        start_idx = None
        
        for i, label in enumerate(labels):
            if label.startswith('B-'):
                if current_entity is not None:
                    entities.add((start_idx, i-1, current_entity))
                current_entity = label[2:]
                start_idx = i
            elif label.startswith('I-'):
                entity_type = label[2:]
                if current_entity != entity_type:
                    if current_entity is not None:
                        entities.add((start_idx, i-1, current_entity))
                    current_entity = entity_type
                    start_idx = i
            else:
                if current_entity is not None:
                    entities.add((start_idx, i-1, current_entity))
                    current_entity = None
                    start_idx = None
        
        if current_entity is not None:
            entities.add((start_idx, len(labels)-1, current_entity))
        
        return entities

    def _bio_to_entity_type(self, label: str) -> str:
        """Convert BIO label to entity type"""
        if label.startswith('B-') or label.startswith('I-'):
            return label[2:]
        return 'O'

    def _compute_coalition_consensus(self, coalition_models: List[str], 
                                   model_labels: Dict[str, List[str]], 
                                   span: Tuple[int, int]) -> List[str]:
        """Compute coalition consensus through majority voting"""
        start, end = span
        consensus_labels = []
        
        for token_idx in range(start, end + 1):
            token_votes = []
            for model in coalition_models:
                if model in model_labels and token_idx < len(model_labels[model]):
                    token_votes.append(model_labels[model][token_idx])
            
            if token_votes:
                vote_counts = Counter(token_votes)
                consensus_label = vote_counts.most_common(1)[0][0]
                consensus_labels.append(consensus_label)
            else:
                consensus_labels.append('O')
        
        return consensus_labels

    def _find_most_different_models(self, coalition_models: List[str], 
                                  model_labels: Dict[str, List[str]], 
                                  span: Tuple[int, int]) -> Tuple[str, str]:
        """Find the two coalition models with most differences in the span"""
        start, end = span
        max_diff = 0
        best_pair = (coalition_models[0], coalition_models[1] if len(coalition_models) > 1 else coalition_models[0])
        
        for i, model_a in enumerate(coalition_models):
            for j, model_b in enumerate(coalition_models[i+1:], i+1):
                if model_a not in model_labels or model_b not in model_labels:
                    continue
                
                diff_count = sum(1 for token_idx in range(start, end + 1)
                               if (token_idx < len(model_labels[model_a]) and 
                                   token_idx < len(model_labels[model_b]) and
                                   model_labels[model_a][token_idx] != model_labels[model_b][token_idx]))
                
                if (diff_count > max_diff or 
                    (diff_count == max_diff and 
                     self.weights_used.get(model_a, 0) + self.weights_used.get(model_b, 0) > 
                     self.weights_used.get(best_pair[0], 0) + self.weights_used.get(best_pair[1], 0))):
                    max_diff = diff_count
                    best_pair = (model_a, model_b)
        
        return best_pair

    def _categorize_span_errors(self, labels_a: List[str], labels_b: List[str], 
                              span: Tuple[int, int], hotspot_info: Dict = None, 
                              sample_tokens: List[str] = None) -> Dict[str, Any]:
        """Categorize errors between two label sequences within a span"""
        errors = {
            'O_to_Ent': defaultdict(list),
            'Ent_to_O': defaultdict(list),
            'Ent_to_Ent': defaultdict(list),
            'Span_Error': []
        }
        
        start, end = span
        
        entities_a = self._extract_entities_from_bio(labels_a[start:end+1])
        entities_b = self._extract_entities_from_bio(labels_b[start:end+1])
        
        entities_a = {(s+start, e+start, t) for s, e, t in entities_a}
        entities_b = {(s+start, e+start, t) for s, e, t in entities_b}
        
        for token_idx in range(start, end + 1):
            if token_idx >= len(labels_a) or token_idx >= len(labels_b):
                continue
                
            label_a, label_b = labels_a[token_idx], labels_b[token_idx]
            if label_a == label_b:
                continue
            
            type_a = self._bio_to_entity_type(label_a)
            type_b = self._bio_to_entity_type(label_b)
            
            error_data = {
                'token_idx': token_idx,
                'reference_label': label_a,
                'comparison_label': label_b,
                'hotspot': hotspot_info,
                'sample_tokens': sample_tokens
            }
            
            if type_a == 'O' and type_b != 'O':
                ent_type = type_b if type_b in self.entity_types else 'OTHER'
                error_data['entity_type'] = type_b
                errors['O_to_Ent'][f'O_to_{ent_type}'].append(error_data)
            elif type_a != 'O' and type_b == 'O':
                ent_type = type_a if type_a in self.entity_types else 'OTHER'
                error_data['entity_type'] = type_a
                errors['Ent_to_O'][f'{ent_type}_to_O'].append(error_data)
            elif type_a != 'O' and type_b != 'O' and type_a != type_b:
                type_a_safe = type_a if type_a in self.entity_types else 'OTHER'
                type_b_safe = type_b if type_b in self.entity_types else 'OTHER'
                error_data.update({
                    'reference_type': type_a,
                    'comparison_type': type_b
                })
                errors['Ent_to_Ent'][f'{type_a_safe}_to_{type_b_safe}'].append(error_data)
        
        for ent_a in entities_a:
            start_a, end_a, type_a = ent_a
            for ent_b in entities_b:
                start_b, end_b, type_b = ent_b
                
                if (type_a == type_b and 
                    not (end_a < start_b or start_a > end_b) and
                    (start_a, end_a) != (start_b, end_b)):
                    
                    errors['Span_Error'].append({
                        'reference_span': (start_a, end_a),
                        'comparison_span': (start_b, end_b),
                        'entity_type': type_a,
                        'span_diff_start': start_b - start_a,
                        'span_diff_end': end_b - end_a,
                        'hotspot': hotspot_info,
                        'sample_tokens': sample_tokens
                    })
        
        return errors

    def analyze_hotspot_errors(self, hotspot: Dict) -> Dict[str, Any]:
        """Analyze errors for a single hotspot based on its type - MODIFIED for gold standard"""
        sentence_id = hotspot['sentence_id']
        start, end = hotspot['start'], hotspot['end']
        disagreement_type = hotspot['disagreement_type']
        
        sample_labels = {}
        for model_name, model_result in self.results_by_model.items():
            if 'detailed_results' in model_result and sentence_id < len(model_result['detailed_results']):
                sample_result = model_result['detailed_results'][sentence_id]
                if 'predicted_labels' in sample_result:
                    sample_labels[model_name] = sample_result['predicted_labels']
        
        _, reference_labels = self._get_sample_data(sentence_id)
        span = (start, end)
        
        error_analysis = {
            'hotspot_info': hotspot,
            'disagreement_type': disagreement_type,
            'error_comparisons': [],
            'confusing_cases': {},
            'analysis_mode': 'gold_standard' if self.use_gold_standard else 'majority_vote'
        }
        
        if self.use_gold_standard:
            # Gold standard mode: compare each model against gold standard
            for model_name in sample_labels.keys():
                comparison = self._analyze_error_comparison(
                    'model_vs_gold', 'gold_standard', model_name, 
                    sample_labels, span, hotspot, reference_labels
                )
                if comparison:
                    error_analysis['error_comparisons'].append(comparison)
        else:
            # Original MV mode logic
            coalition_models = self.coalition_analysis.get('coalition_models', [])
            rest_models = self.coalition_analysis.get('rest_models', [])
            
            if disagreement_type in ["True Elite Split", "Complex (Elite Split + Systematic Bias)"]:
                if len(coalition_models) >= 2:
                    model_a, model_b = self._find_most_different_models(coalition_models, sample_labels, span)
                    comparison = self._analyze_error_comparison('elite_split', model_a, model_b, 
                                                              sample_labels, span, hotspot)
                    if comparison:
                        error_analysis['error_comparisons'].append(comparison)
            
            if disagreement_type in ["Systematic Bias", "Complex (Elite Split + Systematic Bias)", "Minor Disagreement"]:
                comp_type = 'systematic_bias' if disagreement_type != "Minor Disagreement" else 'coalition_vs_rest'
                for rest_model in rest_models:
                    if rest_model in sample_labels:
                        comparison = self._analyze_error_comparison(comp_type, 'coalition_consensus', 
                                                                  rest_model, sample_labels, span, hotspot, reference_labels)
                        if comparison:
                            error_analysis['error_comparisons'].append(comparison)
        
        all_models = set()
        for comparison in error_analysis['error_comparisons']:
            if comparison['reference_model'] not in ['coalition_consensus', 'gold_standard']:
                all_models.add(comparison['reference_model'])
            if comparison['comparison_model'] not in ['coalition_consensus', 'gold_standard']:
                all_models.add(comparison['comparison_model'])
        
        for model in all_models:
            confusing_cases = self._extract_model_confusing_cases(model, sentence_id, span)
            if confusing_cases:
                error_analysis['confusing_cases'][model] = confusing_cases
        
        return error_analysis

    def _analyze_error_comparison(self, comparison_type: str, model_a: str, model_b: str, 
                                sample_labels: Dict[str, List[str]], span: Tuple[int, int], 
                                hotspot: Dict, mv_labels: List[str] = None) -> Optional[Dict[str, Any]]:
        """Unified error comparison logic - MODIFIED for gold standard support"""
        start, end = span
        sample_tokens, reference_labels = self._get_sample_data(hotspot['sentence_id'])
        
        if model_a == 'gold_standard':
            full_labels_a = reference_labels.copy() if reference_labels else []
        elif model_a == 'coalition_consensus':
            coalition_models = self.coalition_analysis.get('coalition_models', [])
            coalition_consensus = self._compute_coalition_consensus(coalition_models, sample_labels, span)
            
            full_labels_a = mv_labels.copy() if mv_labels else reference_labels.copy() if reference_labels else []
            for i, label in enumerate(coalition_consensus):
                if start + i < len(full_labels_a):
                    full_labels_a[start + i] = label
        else:
            full_labels_a = sample_labels.get(model_a, [])
        
        full_labels_b = sample_labels.get(model_b, [])
        
        if not full_labels_a or not full_labels_b:
            return None
            
        errors = self._categorize_span_errors(
            full_labels_a, full_labels_b, span,
            hotspot_info=hotspot, sample_tokens=sample_tokens
        )
        
        return {
            'comparison_type': comparison_type,
            'reference_model': model_a,
            'comparison_model': model_b,
            'errors': errors
        }

    def _extract_model_confusing_cases(self, model_name: str, sample_idx: int, 
                                     hotspot_span: Tuple[int, int]) -> List[Dict]:
        """Extract confusing cases from model results that overlap with hotspot"""
        if model_name not in self.results_by_model:
            return []
        
        model_result = self.results_by_model[model_name]
        if 'detailed_results' not in model_result or sample_idx >= len(model_result['detailed_results']):
            return []
        
        sample_result = model_result['detailed_results'][sample_idx]
        confusing_cases = sample_result.get('confusing_cases', [])
        
        if not confusing_cases:
            return []
        
        sample_tokens, _ = self._get_sample_data(sample_idx)
        if not sample_tokens:
            return []
            
        hotspot_start, hotspot_end = hotspot_span
        hotspot_tokens = sample_tokens[hotspot_start:hotspot_end+1]
        hotspot_text = ' '.join(hotspot_tokens)
        
        return [case for case in confusing_cases 
                if case.get('text_original', '') and case.get('text_original', '') in hotspot_text]

    def _write_documentation_header(self, f):
        """Write documentation header - MODIFIED for mode awareness"""
        analysis_mode = "Gold Standard" if self.use_gold_standard else "Majority Vote"
        
        f.write(f"# NER Disagreement Hotspot Analysis Report ({analysis_mode} Mode)\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Analysis Overview\n\n")
        f.write(f"This report analyzes {len(self.hotspots)} disagreement hotspots identified through disagreement analysis.\n")
        
        if self.use_gold_standard:
            f.write(f"The analysis compares model predictions against gold standard labels within hotspot spans.\n\n")
        else:
            f.write(f"The analysis compares model predictions within hotspot spans using majority vote as reference.\n\n")
        
        f.write("The errors are categorized into four main types:\n\n")
        
        reference_type = "gold standard" if self.use_gold_standard else "reference (majority vote)"
        f.write(f"- **O_to_Ent**: Incorrectly predicting entity where {reference_type} has O\n")
        f.write(f"- **Ent_to_O**: Missing entity that {reference_type} has as entity\n")
        f.write(f"- **Ent_to_Ent**: Predicting wrong entity type compared to {reference_type}\n")
        f.write(f"- **Span_Error**: Correct entity type but wrong span boundaries compared to {reference_type}\n\n")
        f.write("---\n\n")
    
    def _compute_error_statistics(self, analyzed_hotspots: List[Dict]) -> Dict[str, int]:
        """Compute error statistics across all hotspot analyses"""
        stats = defaultdict(int)
        
        for analysis in analyzed_hotspots:
            for comparison in analysis['error_comparisons']:
                errors = comparison['errors']
                
                for error_type, error_list in errors['O_to_Ent'].items():
                    stats[error_type] += len(error_list)
                
                for error_type, error_list in errors['Ent_to_O'].items():
                    stats[error_type] += len(error_list)
                
                for error_type, error_list in errors['Ent_to_Ent'].items():
                    stats[error_type] += len(error_list)
                
                stats['Span_Error'] += len(errors['Span_Error'])
        
        return stats

    def _write_overall_statistics(self, f, analyzed_hotspots: List[Dict]):
        """Write overall statistics section"""
        f.write("## 1. Overall Hotspot Statistics\n\n")
        
        # Count by disagreement type
        type_counts = Counter(analysis['disagreement_type'] for analysis in analyzed_hotspots)
        f.write("### Hotspot Distribution by Type\n\n")
        f.write("| Disagreement Type | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        
        total_hotspots = len(analyzed_hotspots)
        for disagreement_type, count in type_counts.most_common():
            percentage = (count / total_hotspots) * 100
            f.write(f"| {disagreement_type} | {count} | {percentage:.1f}% |\n")
        
        f.write(f"\n**Total Hotspots**: {total_hotspots}\n\n")
        
        # Add error type statistics
        f.write("### Error Type Distribution (All Hotspots)\n\n")
        error_stats = self._compute_error_statistics(analyzed_hotspots)
        
        f.write("| Error Type | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        
        total_errors = sum(error_stats.values())
        for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / total_errors) * 100 if total_errors > 0 else 0
                f.write(f"| {error_type} | {count} | {percentage:.1f}% |\n")
        
        f.write(f"\n**Total Errors**: {total_errors}\n\n")
        
        # Model information
        f.write("### Model Information\n\n")
        f.write(f"**Coalition Models**: {', '.join(self.coalition_analysis.get('coalition_models', []))}\n")
        f.write(f"**Rest Models**: {', '.join(self.coalition_analysis.get('rest_models', []))}\n")
        f.write(f"**Entity Types**: {', '.join(self.entity_types)}\n\n")
        f.write("---\n\n")
    
    def _have_same_label_pattern(self, error1: Dict, error2: Dict) -> bool:
        """Check if two errors have the same label change pattern"""
        if 'entity_type' in error1 and 'entity_type' in error2:
            return error1['entity_type'] == error2['entity_type']
        elif 'reference_type' in error1 and 'reference_type' in error2:
            return (error1['reference_type'] == error2['reference_type'] and
                   error1['comparison_type'] == error2['comparison_type'])
        return False
    
    def _group_consecutive_errors_and_patterns(self, error_list: List[Dict]) -> List[List[Dict]]:
        """Group consecutive errors with same patterns"""
        if not error_list:
            return []
        
        sorted_errors = sorted(error_list, key=lambda x: (
            x.get('hotspot', {}).get('sentence_id', 0),
            x.get('token_idx', 0),
            x.get('disagreement_type', ''),
            x.get('comparison_type', '')
        ))
        
        grouped_spans = []
        current_span = [sorted_errors[0]]
        
        for i in range(1, len(sorted_errors)):
            current_error = sorted_errors[i]
            prev_error = sorted_errors[i-1]
            
            same_sample = (current_error.get('hotspot', {}).get('sentence_id') == 
                          prev_error.get('hotspot', {}).get('sentence_id'))
            
            consecutive_or_related = abs(current_error.get('token_idx', 0) - prev_error.get('token_idx', 0)) <= self.TOKEN_RELATEDNESS_DISTANCE
            
            same_context = (
                current_error.get('disagreement_type') == prev_error.get('disagreement_type') and
                current_error.get('comparison_type') == prev_error.get('comparison_type')
            )
            
            same_pattern = self._have_same_label_pattern(current_error, prev_error)
            
            if same_sample and consecutive_or_related and same_context and same_pattern:
                current_span.append(current_error)
            else:
                grouped_spans.append(current_span)
                current_span = [current_error]
        
        grouped_spans.append(current_span)
        return grouped_spans
    
    def _write_span_error_example(self, f, error: Dict, example_num: int):
        """Write span error example"""
        f.write(f"*Example {example_num}*:\n")
        
        hotspot = error['hotspot']
        sample_tokens = error['sample_tokens']
        ref_span = error['reference_span']
        comp_span = error['comparison_span']
        entity_type = error['entity_type']
        disagreement_type = error.get('disagreement_type', 'Unknown')
        
        ref_start, ref_end = ref_span
        comp_start, comp_end = comp_span
        
        if ref_start < len(sample_tokens) and comp_start < len(sample_tokens):
            ref_tokens = sample_tokens[ref_start:ref_end+1]
            comp_tokens = sample_tokens[comp_start:comp_end+1]
            
            f.write(f"- **Entity Type**: {entity_type}\n")
            f.write(f"- **Reference Span**: \"{' '.join(ref_tokens)}\" (tokens {ref_start}-{ref_end})\n")
            f.write(f"- **Comparison Span**: \"{' '.join(comp_tokens)}\" (tokens {comp_start}-{comp_end})\n")
            
            # Show full context
            context_start = max(0, min(ref_start, comp_start) - self.CONTEXT_TOKENS_BEFORE)
            context_end = min(len(sample_tokens), max(ref_end, comp_end) + 1 + self.CONTEXT_TOKENS_AFTER)
            context_tokens = sample_tokens[context_start:context_end]
            f.write(f"- **Full Context**: {' '.join(context_tokens)}\n")
        
        f.write(f"- **Hotspot Type**: {disagreement_type}\n")
        
        span_diff_start = error['span_diff_start']
        span_diff_end = error['span_diff_end']
        f.write(f"- **Span Differences**: Start {span_diff_start:+d}, End {span_diff_end:+d}\n")
        f.write("\n")
    
    def _write_error_section(self, f, main_category: str, error_dict: Dict):
        """Write error section for each category"""
        f.write(f"## 2. {main_category} Errors\n\n")
        
        if main_category == 'Span_Error':
            error_list = error_dict['Span_Error']
            f.write(f"**Total Span errors**: {len(error_list)}\n\n")
            
            # Random sampling for span errors
            examples_to_show = min(self.MAX_EXAMPLES_PER_TYPE, len(error_list))
            if len(error_list) > self.MAX_EXAMPLES_PER_TYPE:
                selected_errors = random.sample(error_list, examples_to_show)
            else:
                selected_errors = error_list
                
            for i, error in enumerate(selected_errors):
                self._write_span_error_example(f, error, i+1)
            
            if len(error_list) > examples_to_show:
                f.write(f"*... and {len(error_list) - examples_to_show} more examples (randomly sampled {examples_to_show})*\n\n")
        else:
            total_errors = sum(len(errors) for errors in error_dict.values())
            f.write(f"**Total {main_category} errors**: {total_errors}\n\n")
            
            for error_type, error_list in sorted(error_dict.items(), key=lambda x: len(x[1]), reverse=True):
                if not error_list:
                    continue
                    
                grouped_spans = self._group_consecutive_errors_and_patterns(error_list)
                f.write(f"**{error_type}**: {len(error_list)} cases ({len(grouped_spans)} mentions)\n\n")
                
                # Random sampling for grouped spans
                examples_to_show = min(self.MAX_EXAMPLES_PER_TYPE, len(grouped_spans))
                if len(grouped_spans) > self.MAX_EXAMPLES_PER_TYPE:
                    selected_spans = random.sample(grouped_spans, examples_to_show)
                else:
                    selected_spans = grouped_spans
                    
                for i, span_errors in enumerate(selected_spans):
                    self._write_error_example(f, span_errors, i+1)
                
                if len(grouped_spans) > examples_to_show:
                    f.write(f"*... and {len(grouped_spans) - examples_to_show} more examples (randomly sampled {examples_to_show})*\n\n")
        
        f.write("---\n\n")

    def _write_context_with_highlighting(self, f, tokens: List[str], start_idx: int, end_idx: int):
        """Write context with highlighted span"""
        context_start = max(0, start_idx - self.CONTEXT_TOKENS_BEFORE)
        context_end = min(len(tokens), end_idx + 1 + self.CONTEXT_TOKENS_AFTER)
        context_tokens = tokens[context_start:context_end]
        
        highlighted_context = []
        for i, ctx_token in enumerate(context_tokens):
            ctx_idx = context_start + i
            if start_idx <= ctx_idx <= end_idx:
                highlighted_context.append(f"**{ctx_token}**")
            else:
                highlighted_context.append(ctx_token)
        
        f.write(f"- **Context**: {' '.join(highlighted_context)}\n")

    
    def _analyze_label_disagreement_in_span(self, span_errors: List[Dict], mv_labels: List[str]) -> Dict[str, Any]:
        """Analyze label disagreement for a span"""
        if not span_errors:
            return {}
            
        first_error = span_errors[0]
        sentence_id = first_error['hotspot']['sentence_id']
        
        # Get all model predictions for this sample
        all_model_labels = {}
        for model_name, model_result in self.results_by_model.items():
            if ('detailed_results' in model_result and 
                sentence_id < len(model_result['detailed_results'])):
                sample_result = model_result['detailed_results'][sentence_id]
                if 'predicted_labels' in sample_result:
                    all_model_labels[model_name] = sample_result['predicted_labels']
        
        # Analyze token positions in the span
        token_indices = [error['token_idx'] for error in span_errors]
        start_idx, end_idx = min(token_indices), max(token_indices)
        
        # Group models by their predictions vs MV
        mv_agreement_models = []
        disagreement_models = defaultdict(list)
        mv_pattern = []
        
        for model_name, model_labels in all_model_labels.items():
            agrees_with_mv = True
            model_pattern = []
            
            for token_idx in range(start_idx, end_idx + 1):
                if token_idx < len(model_labels) and token_idx < len(mv_labels):
                    model_type = self._bio_to_entity_type(model_labels[token_idx])
                    mv_type = self._bio_to_entity_type(mv_labels[token_idx])
                    
                    model_pattern.append(model_type)
                    if not mv_pattern or len(mv_pattern) <= token_idx - start_idx:
                        mv_pattern.append(mv_type)
                    
                    if model_type != mv_type:
                        agrees_with_mv = False
            
            if agrees_with_mv:
                mv_agreement_models.append(model_name)
            else:
                pattern_key = ' → '.join(model_pattern)
                disagreement_models[pattern_key].append(model_name)
        
        return {
            'mv_agreement_models': mv_agreement_models,
            'disagreement_models': disagreement_models,
            'mv_pattern': ' → '.join(mv_pattern),
            'span_range': (start_idx, end_idx)
        }
    
    def _write_error_example(self, f, span_errors: List[Dict], example_num: int):
        """Write unified error example"""
        if not span_errors:
            return
            
        f.write(f"*Example {example_num}*:\n")
        
        first_error = span_errors[0]
        hotspot = first_error['hotspot']
        sample_tokens = first_error['sample_tokens']
        sentence_id = hotspot['sentence_id']
        disagreement_type = first_error.get('disagreement_type', 'Unknown')
        
        _, mv_labels = self._get_sample_data(sentence_id)
        disagreement_analysis = self._analyze_label_disagreement_in_span(span_errors, mv_labels)
        
        token_indices = [error['token_idx'] for error in span_errors]
        start_idx, end_idx = min(token_indices), max(token_indices)
        
        # Show token span with context
        if start_idx < len(sample_tokens) and end_idx < len(sample_tokens):
            span_tokens = sample_tokens[start_idx:end_idx+1]
            f.write(f"- **Token Span**: \"{' '.join(span_tokens)}\" (positions {start_idx}-{end_idx})\n")
            self._write_context_with_highlighting(f, sample_tokens, start_idx, end_idx)
        
        f.write(f"- **Hotspot Type**: {disagreement_type}\n")
        
        # Show label change pattern
        mv_pattern = disagreement_analysis.get('mv_pattern', '')
        if mv_pattern:
            disagreement_models = disagreement_analysis.get('disagreement_models', {})
            if disagreement_models:
                most_common_pattern = max(disagreement_models.items(), key=lambda x: len(x[1]))
                minor_pattern = most_common_pattern[0]
                f.write(f"- **Label Change Pattern**: {mv_pattern} (MV) →  {minor_pattern} (Minor Opinion)\n")
            else:
                f.write(f"- **Label Change Pattern**: {mv_pattern} (MV)\n")
        
        # Show model groups
        mv_models = disagreement_analysis.get('mv_agreement_models', [])
        disagreement_models = disagreement_analysis.get('disagreement_models', {})
        
        if mv_models:
            f.write(f"- **Models Supporting MV**: {', '.join(mv_models)}\n")
        
        for pattern, models in disagreement_models.items():
            if models:
                f.write(f"- **Models Supporting {pattern}**: {', '.join(models)}\n")
        
        # Show relevant confusing cases
        self._write_filtered_confusing_cases(f, span_errors, disagreement_analysis)
        f.write("\n")
    
    
    def _write_filtered_confusing_cases(self, f, span_errors: List[Dict], disagreement_analysis: Dict):
        """Write confusing cases, filtering out same-type transitions"""
        sentence_id = span_errors[0]['hotspot']['sentence_id']
        span_range = disagreement_analysis.get('span_range', (0, 0))
        
        # Collect all relevant models
        all_relevant_models = set(disagreement_analysis.get('mv_agreement_models', []))
        for models in disagreement_analysis.get('disagreement_models', {}).values():
            all_relevant_models.update(models)
        
        confusing_cases_found = False
        
        for model in all_relevant_models:
            confusing_cases = self._extract_model_confusing_cases(model, sentence_id, span_range)
            
            # Filter out same-type transitions
            filtered_cases = [case for case in confusing_cases 
                            if case.get('type_original', '') != case.get('type_possible', '')]
            
            if filtered_cases:
                if not confusing_cases_found:
                    f.write(f"- **Model Reasoning**:\n")
                    confusing_cases_found = True
                
                f.write(f"  - *{model}*: ")
                for i, case in enumerate(filtered_cases[:self.MAX_CONFUSING_CASES_PER_MODEL]):
                    if i > 0:
                        f.write("; ")
                    original_type = case.get('type_original', 'Unknown')
                    possible_type = case.get('type_possible', 'Unknown')
                    reasoning = case.get('reasoning', 'No reasoning provided')
                    f.write(f"{original_type}→{possible_type} ({reasoning})")
                f.write("\n")
    
    def _aggregate_errors_by_type(self, analyzed_hotspots: List[Dict]) -> Dict[str, Any]:
        """Aggregate all errors by type across hotspots"""
        aggregated_errors = {
            'O_to_Ent': defaultdict(list),
            'Ent_to_O': defaultdict(list),
            'Ent_to_Ent': defaultdict(list),
            'Span_Error': []
        }
        
        for analysis in analyzed_hotspots:
            disagreement_type = analysis['disagreement_type']
            
            for comparison in analysis['error_comparisons']:
                errors = comparison['errors']
                
                # Add context to each error
                for error_type, error_list in errors['O_to_Ent'].items():
                    for error in error_list:
                        error.update({'disagreement_type': disagreement_type, 'comparison_type': comparison['comparison_type']})
                        aggregated_errors['O_to_Ent'][error_type].append(error)
                
                for error_type, error_list in errors['Ent_to_O'].items():
                    for error in error_list:
                        error.update({'disagreement_type': disagreement_type, 'comparison_type': comparison['comparison_type']})
                        aggregated_errors['Ent_to_O'][error_type].append(error)
                
                for error_type, error_list in errors['Ent_to_Ent'].items():
                    for error in error_list:
                        error.update({'disagreement_type': disagreement_type, 'comparison_type': comparison['comparison_type']})
                        aggregated_errors['Ent_to_Ent'][error_type].append(error)
                
                for error in errors['Span_Error']:
                    error.update({'disagreement_type': disagreement_type, 'comparison_type': comparison['comparison_type']})
                    aggregated_errors['Span_Error'].append(error)
        
        return aggregated_errors

    def generate_hotspot_documentation(self, output_dir: str) -> str:
        """Generate comprehensive hotspot documentation"""
        output_path = Path(output_dir) / 'hotspot_docs'
        output_path.mkdir(parents=True, exist_ok=True)
        
        doc_file = output_path / "hotspot_disagreement_analysis.md"
        
        # Analyze all hotspots
        analyzed_hotspots = [self.analyze_hotspot_errors(hotspot) for hotspot in self.hotspots]
        
        # Generate documentation
        with open(doc_file, 'w', encoding='utf-8') as f:
            self._write_documentation_header(f)
            self._write_overall_statistics(f, analyzed_hotspots)
            
            # Aggregate errors and write sections
            aggregated_errors = self._aggregate_errors_by_type(analyzed_hotspots)
            
            for main_category in ['O_to_Ent', 'Ent_to_O', 'Ent_to_Ent', 'Span_Error']:
                if main_category == 'Span_Error':
                    if aggregated_errors[main_category]:
                        self._write_error_section(f, main_category, {'Span_Error': aggregated_errors[main_category]})
                else:
                    if aggregated_errors[main_category]:
                        self._write_error_section(f, main_category, aggregated_errors[main_category])
        
        print(f"Generated hotspot documentation: {doc_file}")
        return str(doc_file)

# Main documentation generation function
def generate_hotspot_documentation_from_results(results_file_path: str,
                                               disagreement_data: Dict[str, Any],
                                               output_dir: str = None) -> str:
    """
    Generate hotspot documentation from experiment results and disagreement analysis
    WORKS FOR BOTH MV AND GOLD STANDARD MODES
    """
    results_data = load_experiment_results(results_file_path)
    
    if output_dir is None:
        output_dir = disagreement_data.get('output_directory', './hotspot_documentation')
    
    doc_generator = HotspotDocumentationGenerator(results_data, disagreement_data)
    doc_file = doc_generator.generate_hotspot_documentation(str(output_dir))
    
    return doc_file