"""
Agreement metrics for NER annotation comparison
Updated to use global concatenation instead of sample-wise averaging
"""
from typing import List, Dict, Tuple, Set, Any
from sklearn.metrics import cohen_kappa_score, f1_score


class AgreementMetrics:
    """Calculate various agreement metrics between NER annotators using global concatenation"""
    
    @staticmethod
    def _concatenate_bio_tags_across_samples(
        model1_results: List[Dict[str, Any]], 
        model2_results: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """
        Concatenate BIO tags from all samples for two models
        
        Args:
            model1_results: First model's detailed results for all samples
            model2_results: Second model's detailed results for all samples
            
        Returns:
            Tuple of concatenated BIO tags for model1 and model2
        """
        model1_all_tags = []
        model2_all_tags = []
        
        for result1, result2 in zip(model1_results, model2_results):
            if 'error' in result1 or 'error' in result2:
                continue
                
            tags1 = result1.get('predicted_labels', [])
            tags2 = result2.get('predicted_labels', [])
            
            if len(tags1) == len(tags2) and len(tags1) > 0:
                model1_all_tags.extend(tags1)
                model2_all_tags.extend(tags2)
        
        return model1_all_tags, model2_all_tags
    
    @staticmethod
    def _concatenate_entity_spans_across_samples(
        model1_results: List[Dict[str, Any]], 
        model2_results: List[Dict[str, Any]],
        test_samples: List[Dict[str, Any]]
    ) -> Tuple[Set[Tuple[int, int, str]], Set[Tuple[int, int, str]]]:
        """
        Concatenate entity spans from all samples with global character positions
        
        Args:
            model1_results: First model's detailed results for all samples
            model2_results: Second model's detailed results for all samples
            test_samples: Original test samples for character offset calculation
            
        Returns:
            Tuple of sets containing global entity spans for model1 and model2
        """
        model1_spans = set()
        model2_spans = set()
        global_char_offset = 0
        
        for idx, (result1, result2) in enumerate(zip(model1_results, model2_results)):
            if 'error' in result1 or 'error' in result2:
                continue
                
            if idx >= len(test_samples):
                continue
                
            sample = test_samples[idx]
            text = sample.get('text', '')
            
            # Get entities from both models
            entities1 = result1.get('predicted_entities', [])
            entities2 = result2.get('predicted_entities', [])
            
            # Convert to global spans
            for entity in entities1:
                start = entity.get('start_pos', -1)
                end = entity.get('end_pos', -1)
                entity_type = entity.get('type', '')
                if start != -1 and end != -1:
                    global_start = global_char_offset + start
                    global_end = global_char_offset + end
                    model1_spans.add((global_start, global_end, entity_type))
            
            for entity in entities2:
                start = entity.get('start_pos', -1)
                end = entity.get('end_pos', -1)
                entity_type = entity.get('type', '')
                if start != -1 and end != -1:
                    global_start = global_char_offset + start
                    global_end = global_char_offset + end
                    model2_spans.add((global_start, global_end, entity_type))
            
            # Update global offset (add text length + separator)
            global_char_offset += len(text) + 1  # +1 for separator
        
        return model1_spans, model2_spans
    
    @staticmethod
    def extract_spans(entities: List[Dict[str, Any]]) -> Set[Tuple[int, int, str]]:
        """
        Extract entity spans as (start, end, type) tuples
        
        Args:
            entities: List of entity dictionaries with start_pos, end_pos, type
            
        Returns:
            Set of (start_pos, end_pos, type) tuples
        """
        spans = set()
        for entity in entities:
            start = entity.get('start_pos', -1)
            end = entity.get('end_pos', -1)
            entity_type = entity.get('type', '')
            if start != -1 and end != -1:
                spans.add((start, end, entity_type))
        return spans
    
    @staticmethod
    def strict_span_f1(entities1: List[Dict[str, Any]], 
                      entities2: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate strict span F1 score between two sets of entities
        Requires exact boundary and type match
        
        Args:
            entities1: First annotator's entities
            entities2: Second annotator's entities
            
        Returns:
            Dictionary with precision, recall, f1 scores
        """
        spans1 = AgreementMetrics.extract_spans(entities1)
        spans2 = AgreementMetrics.extract_spans(entities2)
        
        if len(spans1) == 0 and len(spans2) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if len(spans1) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if len(spans2) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        intersection = spans1.intersection(spans2)
        
        precision = len(intersection) / len(spans2) if len(spans2) > 0 else 0.0
        recall = len(intersection) / len(spans1) if len(spans1) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'intersection_count': len(intersection),
            'spans1_count': len(spans1),
            'spans2_count': len(spans2)
        }
    
    @staticmethod
    def strict_span_f1_global(
        model1_results: List[Dict[str, Any]], 
        model2_results: List[Dict[str, Any]],
        test_samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate strict span F1 using global concatenation across all samples
        
        Args:
            model1_results: First model's detailed results for all samples
            model2_results: Second model's detailed results for all samples
            test_samples: Original test samples
            
        Returns:
            Dictionary with global precision, recall, f1 scores
        """
        spans1, spans2 = AgreementMetrics._concatenate_entity_spans_across_samples(
            model1_results, model2_results, test_samples
        )
        
        if len(spans1) == 0 and len(spans2) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if len(spans1) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if len(spans2) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        intersection = spans1.intersection(spans2)
        
        precision = len(intersection) / len(spans2) if len(spans2) > 0 else 0.0
        recall = len(intersection) / len(spans1) if len(spans1) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'intersection_count': len(intersection),
            'spans1_count': len(spans1),
            'spans2_count': len(spans2)
        }
    
    @staticmethod
    def entities_to_bio_tags(entities: List[Dict[str, Any]], 
                           tokens: List[str], 
                           text: str) -> List[str]:
        """
        Convert entities to BIO tags for token-level comparison
        
        Args:
            entities: List of entity dictionaries
            tokens: List of tokens
            text: Original text
            
        Returns:
            List of BIO tags for each token
        """
        bio_tags = ['O'] * len(tokens)
        
        # Create character to token mapping
        char_to_token = {}
        char_pos = 0
        
        for token_idx, token in enumerate(tokens):
            # Skip leading whitespace
            while char_pos < len(text) and text[char_pos].isspace():
                char_pos += 1
            
            # Map characters of this token
            token_start = char_pos
            token_end = char_pos + len(token)
            
            for char_idx in range(token_start, min(token_end, len(text))):
                char_to_token[char_idx] = token_idx
            
            char_pos = token_end
        
        # Convert entities to BIO tags
        for entity in entities:
            start_pos = entity.get('start_pos', -1)
            end_pos = entity.get('end_pos', -1)
            entity_type = entity.get('type', '')
            
            if start_pos == -1 or end_pos == -1:
                continue
            
            # Find tokens that overlap with this entity
            entity_tokens = set()
            for char_idx in range(start_pos, min(end_pos, len(text))):
                if char_idx in char_to_token:
                    entity_tokens.add(char_to_token[char_idx])
            
            # Assign BIO tags
            entity_tokens = sorted(entity_tokens)
            for i, token_idx in enumerate(entity_tokens):
                if i == 0:
                    bio_tags[token_idx] = f'B-{entity_type}'
                else:
                    bio_tags[token_idx] = f'I-{entity_type}'
        
        return bio_tags
    
    @staticmethod
    def token_level_macro_f1(
        model1_results: List[Dict[str, Any]], 
        model2_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate token-level macro F1 score using global concatenation
        
        Args:
            model1_results: First model's detailed results for all samples
            model2_results: Second model's detailed results for all samples
            
        Returns:
            Dictionary with macro F1 and per-class F1 scores
        """
        bio_tags1, bio_tags2 = AgreementMetrics._concatenate_bio_tags_across_samples(
            model1_results, model2_results
        )
        
        if not bio_tags1 or not bio_tags2:
            return {'macro_f1': 0.0, 'per_class_f1': {}}
        
        # Get all unique labels
        all_labels = sorted(set(bio_tags1 + bio_tags2))
        
        if len(all_labels) <= 1:
            return {'macro_f1': 1.0 if bio_tags1 == bio_tags2 else 0.0, 'per_class_f1': {}}
        
        # Calculate macro F1
        macro_f1 = f1_score(bio_tags1, bio_tags2, labels=all_labels, average='macro', zero_division=0)
        
        # Calculate per-class F1
        per_class_f1 = f1_score(bio_tags1, bio_tags2, labels=all_labels, average=None, zero_division=0)
        per_class_dict = {label: score for label, score in zip(all_labels, per_class_f1)}
        
        return {
            'macro_f1': macro_f1,
            'per_class_f1': per_class_dict,
            'token_count': len(bio_tags1),
            'agreement_count': sum(1 for t1, t2 in zip(bio_tags1, bio_tags2) if t1 == t2)
        }
    
    @staticmethod
    def cohens_kappa(
        model1_results: List[Dict[str, Any]], 
        model2_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate Cohen's kappa for token-level agreement using global concatenation
        
        Args:
            model1_results: First model's detailed results for all samples
            model2_results: Second model's detailed results for all samples
            
        Returns:
            Dictionary with kappa score and interpretation
        """
        bio_tags1, bio_tags2 = AgreementMetrics._concatenate_bio_tags_across_samples(
            model1_results, model2_results
        )
        
        if not bio_tags1 or not bio_tags2:
            return {'kappa': 0.0, 'interpretation': 'Poor', 'token_count': 0, 'unique_labels': 0}
        
        if len(set(bio_tags1 + bio_tags2)) <= 1:
            # Perfect agreement or all same label
            kappa = 1.0
        else:
            kappa = cohen_kappa_score(bio_tags1, bio_tags2)
        
        # Interpretation of kappa values
        if kappa < 0:
            interpretation = "Poor"
        elif kappa < 0.2:
            interpretation = "Slight"
        elif kappa < 0.4:
            interpretation = "Fair"
        elif kappa < 0.6:
            interpretation = "Moderate"
        elif kappa < 0.8:
            interpretation = "Substantial"
        else:
            interpretation = "Almost Perfect"
        
        return {
            'kappa': kappa,
            'interpretation': interpretation,
            'token_count': len(bio_tags1),
            'unique_labels': len(set(bio_tags1 + bio_tags2))
        }
    
    @staticmethod
    def calculate_pairwise_agreement(
        model1_results: List[Dict[str, Any]],
        model2_results: List[Dict[str, Any]],
        test_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate all agreement metrics between two models using global concatenation
        
        Args:
            model1_results: First model's detailed results for all samples
            model2_results: Second model's detailed results for all samples
            test_samples: Original test samples
            
        Returns:
            Dictionary with all agreement metrics
        """
        strict_f1 = AgreementMetrics.strict_span_f1_global(model1_results, model2_results, test_samples)
        token_f1 = AgreementMetrics.token_level_macro_f1(model1_results, model2_results)
        kappa = AgreementMetrics.cohens_kappa(model1_results, model2_results)
        
        return {
            'strict_span_f1': strict_f1,
            'token_macro_f1': token_f1,
            'cohens_kappa': kappa
        }