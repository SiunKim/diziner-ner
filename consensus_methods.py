"""
Consensus methods for NER annotation aggregation
Updated to use IO-level majority voting and F1-only weight calculation
"""
import numpy as np
import random
from typing import List, Dict, Any
from collections import defaultdict
from agreement_metrics import AgreementMetrics


def print_sample_comparison_with_consensus(consensus_results: List[Dict[str, Any]], 
                                         test_samples: List[Dict[str, Any]],
                                         valid_models: List[str], 
                                         num_samples: int = 3):
    """
    Print detailed sample annotation comparison with consensus results in matrix format
    Tokens as rows, models as columns
    
    Args:
        consensus_results: List of consensus results for each sample
        test_samples: Original test samples with gold labels
        valid_models: List of valid model names
        num_samples: Number of random samples to display
    """
    if not consensus_results or len(consensus_results) < num_samples:
        return
    
    # Randomly select samples
    random.seed(42)  # For reproducible results
    selected_samples = random.sample(consensus_results, min(num_samples, len(consensus_results)))
    
    print(f"\n{'='*120}")
    print(f"SAMPLE ANNOTATION COMPARISON ({num_samples} Random Examples)")
    print(f"{'='*120}")
    
    for i, sample_result in enumerate(selected_samples, 1):
        sample_idx = sample_result.get('sample_idx', i-1)
        tokens = sample_result.get('tokens', [])
        consensus_labels = sample_result.get('consensus_labels', [])
        model_labels = sample_result.get('model_labels', {})
        
        # Get gold standard labels
        gold_labels = []
        if sample_idx < len(test_samples):
            gold_labels = test_samples[sample_idx].get('labels', [])
        
        print(f"\n--- SAMPLE {i} (Index: {sample_idx}) ---")
        
        # Matrix format: tokens as rows, models as columns
        if len(tokens) > 0:
            # Create header with model names
            header = "Token".ljust(15) + "Gold".ljust(8)
            for model_name in valid_models:
                model_display = model_name[:10] + ".." if len(model_name) > 10 else model_name
                header += model_display.ljust(12)
            header += "Consensus".ljust(12) + "Match"
            print(header)
            print("-" * len(header))
            
            # Print each token as a row
            for token_idx, token in enumerate(tokens):
                # Token column
                token_display = token[:13] + ".." if len(token) > 13 else token
                row = token_display.ljust(15)
                
                # Gold label column
                gold_label = gold_labels[token_idx] if token_idx < len(gold_labels) else 'O'
                row += gold_label.ljust(8)
                
                # Model prediction columns
                model_predictions = []
                for model_name in valid_models:
                    if model_name in model_labels:
                        model_pred = model_labels[model_name]
                        pred_label = model_pred[token_idx] if token_idx < len(model_pred) else 'O'
                    else:
                        pred_label = 'O'
                    model_predictions.append(pred_label)
                    row += pred_label.ljust(12)
                
                # Consensus column
                consensus_label = consensus_labels[token_idx] if token_idx < len(consensus_labels) else 'O'
                row += consensus_label.ljust(12)
                
                # Match indicator
                match_symbol = "✓" if consensus_label == gold_label else "✗"
                row += match_symbol
                
                print(row)
            
            print("-" * len(header))
        
        print(f"\n{'='*60}")
    
    print(f"{'='*120}")

def analyze_voting_pattern(model_labels: Dict[str, List[str]], 
                          consensus_labels: List[str], 
                          valid_models: List[str]) -> List[Dict[str, Any]]:
    """
    analyze_voting_pattern
    """
    if not model_labels or not consensus_labels:
        return []
    
    num_tokens = len(consensus_labels)
    analysis_results = []
    
    for token_idx in range(num_tokens):
        votes = {}
        for model_name in valid_models:
            if model_name in model_labels:
                model_pred = model_labels[model_name]
                if token_idx < len(model_pred):
                    votes[model_name] = model_pred[token_idx]
                else:
                    votes[model_name] = 'O'
        
        analysis_results.append({
            'token_idx': token_idx,
            'votes': votes,
            'consensus': consensus_labels[token_idx] if token_idx < len(consensus_labels) else 'O'
        })
    
    return analysis_results

class ConsensusMethods:
    """Methods for creating consensus annotations from multiple annotators"""
    @staticmethod
    def bio_to_io(bio_tags: List[str]) -> List[str]:
        """
        Convert BIO tags to IO tags (remove B- prefix, keep only I- and O)
        
        Args:
            bio_tags: List of BIO tags
            
        Returns:
            List of IO tags
        """
        io_tags = []
        for tag in bio_tags:
            if tag.startswith('B-'):
                io_tags.append('I-' + tag[2:])  # Convert B-TYPE to I-TYPE
            else:
                io_tags.append(tag)  # Keep I-TYPE and O as is
        return io_tags
    
    @staticmethod
    def io_to_bio(io_tags: List[str]) -> List[str]:
        """
        Convert IO tags back to BIO tags
        
        Args:
            io_tags: List of IO tags
            
        Returns:
            List of BIO tags
        """
        bio_tags = []
        prev_tag = 'O'
        
        for tag in io_tags:
            if tag == 'O':
                bio_tags.append('O')
                prev_tag = 'O'
            elif tag.startswith('I-'):
                entity_type = tag[2:]
                if prev_tag == 'O' or (prev_tag.startswith('I-') and prev_tag[2:] != entity_type):
                    # Start of new entity
                    bio_tags.append('B-' + entity_type)
                else:
                    # Continuation of entity
                    bio_tags.append('I-' + entity_type)
                prev_tag = tag
            else:
                # Handle unexpected format
                bio_tags.append(tag)
                prev_tag = tag
        
        return bio_tags
    
    @staticmethod
    def bio_tags_to_entities(bio_tags: List[str], tokens: List[str], text: str) -> List[Dict[str, Any]]:
        """
        Convert BIO tags back to entity format
        
        Args:
            bio_tags: List of BIO tags
            tokens: List of tokens
            text: Original text
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        current_entity = None
        
        # Create token to character mapping
        char_pos = 0
        token_positions = []
        
        for token in tokens:
            # Skip leading whitespace
            while char_pos < len(text) and text[char_pos].isspace():
                char_pos += 1
            
            start_pos = char_pos
            end_pos = char_pos + len(token)
            token_positions.append((start_pos, end_pos))
            char_pos = end_pos
        
        for i, tag in enumerate(bio_tags):
            if tag.startswith('B-'):
                # Start of new entity
                if current_entity:
                    # Finish previous entity
                    entities.append(current_entity)
                
                entity_type = tag[2:]
                start_pos, _ = token_positions[i]
                current_entity = {
                    'text': tokens[i],
                    'label': entity_type,  # Use 'label' instead of 'type' for consistency
                    'start': start_pos,
                    'end': token_positions[i][1],
                    'token_indices': [i]
                }
                
            elif tag.startswith('I-') and current_entity:
                # Continue current entity
                entity_type = tag[2:]
                if current_entity['label'] == entity_type:
                    # Extend current entity
                    current_entity['text'] += ' ' + tokens[i]
                    current_entity['end'] = token_positions[i][1]
                    current_entity['token_indices'].append(i)
                else:
                    # Type mismatch, start new entity
                    entities.append(current_entity)
                    start_pos, _ = token_positions[i]
                    current_entity = {
                        'text': tokens[i],
                        'label': entity_type,
                        'start': start_pos,
                        'end': token_positions[i][1],
                        'token_indices': [i]
                    }
            else:
                # O tag or I- without current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add last entity if exists
        if current_entity:
            entities.append(current_entity)
        
        # Clean up entities - remove token_indices for output
        for entity in entities:
            entity.pop('token_indices', None)
        
        return entities
    
    @staticmethod
    def majority_voting_consensus(all_model_labels: Dict[str, List[str]], 
                                model_weights: Dict[str, float] = None) -> List[str]:
        """
        Create consensus using weighted majority voting at IO level, then convert back to BIO
        
        Args:
            all_model_labels: Dictionary mapping model names to their predicted BIO labels
            model_weights: Dictionary mapping model names to their reliability weights
            
        Returns:
            List of consensus BIO labels
        """
        if not all_model_labels:
            return []
        
        # Get sequence length from first model
        first_model_labels = next(iter(all_model_labels.values()))
        seq_length = len(first_model_labels)
        
        if seq_length == 0:
            return []
        
        # Default weights to 1.0 if not provided
        if model_weights is None:
            model_weights = {model_name: 1.0 for model_name in all_model_labels.keys()}
        
        # Convert all BIO labels to IO labels
        all_model_io_labels = {}
        for model_name, bio_labels in all_model_labels.items():
            # Verify sequence length and adjust if needed
            if len(bio_labels) != seq_length:
                print(f"Warning: Model {model_name} has different sequence length: {len(bio_labels)} vs {seq_length}")
                # Truncate or pad to match
                if len(bio_labels) > seq_length:
                    bio_labels = bio_labels[:seq_length]
                else:
                    bio_labels = bio_labels + ['O'] * (seq_length - len(bio_labels))
            
            io_labels = ConsensusMethods.bio_to_io(bio_labels)
            all_model_io_labels[model_name] = io_labels
        
        # Perform weighted majority voting for each token position using IO labels
        consensus_io_labels = []
        for token_idx in range(seq_length):
            # Collect weighted votes
            label_weights = defaultdict(float)
            total_weight = 0.0
            
            for model_name, io_labels in all_model_io_labels.items():
                if token_idx < len(io_labels):
                    label = io_labels[token_idx]
                else:
                    label = 'O'  # Default for missing labels
                
                weight = model_weights.get(model_name, 1.0)
                label_weights[label] += weight
                total_weight += weight
            
            # Find label with highest weighted vote
            if not label_weights:
                consensus_io_labels.append('O')
                continue
            
            # Get the label with maximum weight
            best_label = max(label_weights.items(), key=lambda x: x[1])
            best_label_name, best_weight = best_label
            
            # Check for ties and handle them conservatively
            max_weight = best_weight
            tied_labels = [label for label, weight in label_weights.items() if abs(weight - max_weight) < 1e-9]
            
            if len(tied_labels) > 1:
                # In case of tie, choose 'O' if it's among tied labels, otherwise choose conservatively
                if 'O' in tied_labels:
                    consensus_io_labels.append('O')
                else:
                    # Choose the first label alphabetically for consistency
                    consensus_io_labels.append(sorted(tied_labels)[0])
            else:
                # Use the label with highest weight (min_votes requirement removed)
                consensus_io_labels.append(best_label_name)
        
        # Convert consensus IO labels back to BIO labels
        consensus_bio_labels = ConsensusMethods.io_to_bio(consensus_io_labels)
        
        return consensus_bio_labels

    @staticmethod
    def calculate_model_reliability_weights(all_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate model reliability weights using weighted micro strict span F1 only
        
        Args:
            all_results: Combined results from experiments
            
        Returns:
            Dictionary mapping model names to their reliability weights (sum=1.0)
        """
        results_by_model = all_results.get('results_by_model', {})
        test_samples = all_results.get('test_samples', [])
        
        # Get valid models
        valid_models = []
        for model_name, model_result in results_by_model.items():
            if ('error' not in model_result and 
                'detailed_results' in model_result and 
                len(model_result['detailed_results']) >= len(test_samples)):
                valid_models.append(model_name)
        
        if len(valid_models) < 2:
            # If less than 2 models, assign equal weights
            if len(valid_models) == 1:
                return {valid_models[0]: 1.0}
            else:
                return {}
        
        print(f"Calculating F1-based model reliability weights for {len(valid_models)} models...")
        
        # Calculate pairwise F1 scores for all model pairs using global concatenation
        model_f1_scores = defaultdict(list)
        
        for i, model1 in enumerate(valid_models):
            for j, model2 in enumerate(valid_models):
                if i >= j:  # Skip self-comparison and duplicates
                    continue
                
                # Get detailed results for both models
                model1_results = results_by_model[model1]['detailed_results']
                model2_results = results_by_model[model2]['detailed_results']
                
                # Calculate global agreement using concatenated labels
                agreement_result = AgreementMetrics.calculate_pairwise_agreement(
                    model1_results, model2_results, test_samples
                )
                
                # Use weighted micro strict span F1
                micro_f1 = agreement_result['strict_span_f1']['f1']
                
                model_f1_scores[model1].append(micro_f1)
                model_f1_scores[model2].append(micro_f1)
        
        # Calculate F1-based weights for each model
        model_weights = {}
        for model_name in valid_models:
            # Calculate mean F1 score
            mean_f1 = np.mean(model_f1_scores[model_name]) if model_f1_scores[model_name] else 0.0
            
            # Use F1 directly as weight with minimum threshold
            f1_weight = max(0.1, mean_f1)
            
            model_weights[model_name] = f1_weight
        
        # Normalize weights so they sum to 1.0
        total_weight = sum(model_weights.values())
        if total_weight > 0:
            for model_name in model_weights:
                model_weights[model_name] = model_weights[model_name] / total_weight
        else:
            # Fallback to equal weights if all weights are zero
            equal_weight = 1.0 / len(valid_models)
            model_weights = {model_name: equal_weight for model_name in valid_models}
        
        print(f"F1-based model reliability weights (sum={sum(model_weights.values()):.3f}): {model_weights}")
        return model_weights

    @staticmethod
    def calculate_consensus_agreement(model_entities: Dict[str, List[Dict[str, Any]]],
                                    consensus_labels: List[str],
                                    tokens: List[str],
                                    text: str) -> Dict[str, Dict[str, Any]]:
        """
        Calculate agreement between each model's entities and consensus labels
        
        Args:
            model_entities: Dictionary mapping model names to their entities
            consensus_labels: Consensus labels (BIO format)
            tokens: List of tokens
            text: Original text
            
        Returns:
            Dictionary mapping model names to agreement metrics with consensus
        """
        agreement_results = {}
        
        # Convert consensus labels to entities for comparison
        consensus_entities = ConsensusMethods.bio_tags_to_entities(consensus_labels, tokens, text)
        
        for model_name, entities in model_entities.items():
            agreement = AgreementMetrics.strict_span_f1(entities, consensus_entities)
            agreement_results[model_name] = {
                'strict_span_f1': agreement,
                'token_macro_f1': {'macro_f1': 0.0},  # Placeholder for compatibility
                'cohens_kappa': {'kappa': 0.0}  # Placeholder for compatibility
            }
        
        return agreement_results
    
    @staticmethod
    def create_majority_voting_result(all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create majority voting consensus for all samples using IO-level voting
        
        Args:
            all_results: Combined results from experiments
            
        Returns:
            Dictionary with consensus results and agreement metrics
        """
        if 'results_by_model' not in all_results:
            raise ValueError("Expected 'results_by_model' in input data")
        
        results_by_model = all_results['results_by_model']
        test_samples = all_results.get('test_samples', [])
        
        if not test_samples:
            return {
                'method': 'majority_voting',
                'consensus_results': [],
                'consensus_entities': [],
                'avg_consensus_agreements': {},
                'sample_agreements': [],
                'summary': {
                    'total_samples': 0,
                    'models_included': []
                }
            }
        
        consensus_results = []
        consensus_entities = []
        sample_agreements = []
        model_consensus_agreements = defaultdict(list)
        
        # Get list of models that have valid results
        valid_models = []
        for model_name, model_result in results_by_model.items():
            if ('error' not in model_result and 
                'detailed_results' in model_result and 
                len(model_result['detailed_results']) >= len(test_samples)):
                valid_models.append(model_name)
        
        if not valid_models:
            print("Warning: No valid models found for majority voting")
            return {
                'method': 'majority_voting',
                'consensus_results': [],
                'consensus_entities': [],
                'avg_consensus_agreements': {},
                'sample_agreements': [],
                'summary': {
                    'total_samples': 0,
                    'models_included': []
                }
            }
        
        # Calculate model reliability weights using F1-only method
        model_weights = ConsensusMethods.calculate_model_reliability_weights(all_results)
        
        print(f"Processing {len(test_samples)} samples with {len(valid_models)} models: {valid_models}")
        print(f"Using F1-based model weights: {model_weights}")
        
        # Process each sample
        for sample_idx, sample in enumerate(test_samples):
            tokens = sample.get('tokens', [])
            text = sample.get('text', '')
            
            if not tokens:
                print(f"Warning: Sample {sample_idx} has no tokens, skipping")
                continue
            
            # Collect predicted labels from valid models for this sample
            sample_model_labels = {}
            sample_model_entities = {}
            
            for model_name in valid_models:
                model_result = results_by_model[model_name]
                if sample_idx < len(model_result['detailed_results']):
                    sample_result = model_result['detailed_results'][sample_idx]
                    predicted_labels = sample_result.get('predicted_labels', [])
                    predicted_entities = sample_result.get('predicted_entities', [])
                    
                    # Ensure labels match token length
                    if len(predicted_labels) == len(tokens):
                        sample_model_labels[model_name] = predicted_labels
                        sample_model_entities[model_name] = predicted_entities
                    else:
                        print(f"Warning: Model {model_name} sample {sample_idx} has label/token length mismatch: {len(predicted_labels)} vs {len(tokens)}")
            
            # Skip if insufficient models have valid predictions for this sample
            if len(sample_model_labels) < 2:
                print(f"Warning: Sample {sample_idx} has insufficient valid models ({len(sample_model_labels)}), skipping")
                continue
            
            # Create consensus using weighted majority voting on IO-level labels
            consensus_labels = ConsensusMethods.majority_voting_consensus(
                sample_model_labels, 
                model_weights=model_weights
            )
            
            # Convert consensus labels to entities for comparison
            consensus_sample_entities = ConsensusMethods.bio_tags_to_entities(consensus_labels, tokens, text)
            
            # Calculate agreement with consensus using entities (for compatibility)
            consensus_agreements = ConsensusMethods.calculate_consensus_agreement(
                sample_model_entities, consensus_labels, tokens, text
            )
            
            # Store results
            sample_result = {
                'sample_idx': sample_idx,
                'text': text,
                'tokens': tokens,
                'model_labels': sample_model_labels,
                'model_entities': sample_model_entities,
                'consensus_labels': consensus_labels,
                'consensus_entities': consensus_sample_entities,
                'consensus_agreements': consensus_agreements
            }
            consensus_results.append(sample_result)
            consensus_entities.append(consensus_sample_entities)
            sample_agreements.append(consensus_agreements)
            
            # Aggregate model agreements
            for model_name, agreement in consensus_agreements.items():
                model_consensus_agreements[model_name].append(agreement)
        
        # Calculate average agreement with consensus for each model
        avg_consensus_agreements = {}
        for model_name, agreements in model_consensus_agreements.items():
            if agreements:
                avg_agreement = {
                    'strict_span_f1': np.mean([a['strict_span_f1']['f1'] for a in agreements]),
                    'token_macro_f1': np.mean([a['token_macro_f1']['macro_f1'] for a in agreements]),
                    'cohens_kappa': np.mean([a['cohens_kappa']['kappa'] for a in agreements])
                }
                avg_consensus_agreements[model_name] = avg_agreement
        
        print(f"IO-level majority voting completed: {len(consensus_results)} samples processed")
        
        # 샘플 비교 출력 추가
        if consensus_results and valid_models:
            print_sample_comparison_with_consensus(consensus_results, test_samples, valid_models, num_samples=3)
        
        return {
            'method': 'majority_voting',
            'consensus_results': consensus_results,
            'consensus_entities': consensus_entities,
            'avg_consensus_agreements': avg_consensus_agreements,
            'sample_agreements': sample_agreements,
            'summary': {
                'total_samples': len(consensus_results),
                'models_included': valid_models
            }
        }

    @staticmethod
    def create_elite_majority_voting_result(all_results: Dict[str, Any],
                                            coverage_threshold: float = 0.5,
                                            min_models: int = 1,
                                            verbose: int = 1) -> Dict[str, Any]:
        """
        Create consensus using only elite models whose cumulative reliability weight
        reaches the coverage_threshold (default 0.5). Falls back to top 'min_models'
        if threshold would select fewer than that.
        """
        results_by_model = all_results.get('results_by_model', {})
        test_samples = all_results.get('test_samples', [])
        if not results_by_model or not test_samples:
            return {'method': 'elite_majority_voting',
                    'consensus_results': [],
                    'summary': {'total_samples': 0, 'models_included': []}}

        # 1) Valid models (same criterion as standard consensus)
        valid_models = []
        for model_name, model_result in results_by_model.items():
            if ('error' not in model_result and
                'detailed_results' in model_result and
                len(model_result['detailed_results']) >= len(test_samples)):
                valid_models.append(model_name)
        if len(valid_models) == 0:
            return {'method': 'elite_majority_voting',
                    'consensus_results': [],
                    'summary': {'total_samples': 0, 'models_included': []}}

        # 2) Weights and elite cut by cumulative coverage
        model_weights = ConsensusMethods.calculate_model_reliability_weights(all_results)
        ranked = sorted(valid_models, key=lambda m: model_weights.get(m, 0.0), reverse=True)

        elite, cumw = [], 0.0
        for m in ranked:
            if cumw >= coverage_threshold and len(elite) >= min_models:
                break
            elite.append(m)
            cumw += model_weights.get(m, 0.0)

        # Safety: ensure at least min_models when weights are all tiny
        if len(elite) < min_models and len(ranked) >= min_models:
            elite = ranked[:min_models]

        if verbose and elite:
            print(f"[Elite MV] Selected {len(elite)} models with cum. weight={cumw:.3f} (threshold={coverage_threshold})")

        # 3) Build per-sample label matrix limited to elite set
        consensus_results = []
        for sample_idx, sample in enumerate(test_samples):
            tokens = sample.get('tokens', [])
            text = sample.get('text', '')
            if not tokens:
                continue

            model_labels = {}
            for m in elite:
                dr = results_by_model[m]['detailed_results'][sample_idx]
                if 'error' in dr:
                    continue
                labels = dr.get('predicted_labels', [])
                # Truncate or pad to match
                if len(labels) != len(tokens):
                    if len(labels) > len(tokens):
                        labels = labels[:len(tokens)]
                    else:
                        labels = labels + ['O'] * (len(tokens) - len(labels))
                model_labels[m] = labels

            # 4) Elite-only weighted MV (reuse existing MV core)
            elite_weights = {m: model_weights.get(m, 1.0) for m in model_labels.keys()}
            consensus_labels = ConsensusMethods.majority_voting_consensus(model_labels, model_weights=elite_weights)

            consensus_results.append({
                'sample_idx': sample_idx,
                'tokens': tokens,
                'text_len': len(text),
                'model_labels': model_labels,
                'consensus_labels': consensus_labels
            })

        return {
            'method': 'elite_majority_voting',
            'consensus_results': consensus_results,
            'summary': {
                'total_samples': len(consensus_results),
                'models_included': elite
            }
        }

