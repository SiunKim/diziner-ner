from typing import Dict, List, Any

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

def calculate_ner_metrics(predicted_entities: List[Dict], gold_entities: List[Dict]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1-score for NER predictions
    
    Args:
        predicted_entities: List of predicted entity dictionaries
        gold_entities: List of gold standard entity dictionaries
    
    Returns:
        Dictionary containing precision, recall, F1, and per-type metrics
    """
    # Convert to sets of (text, type) tuples for exact matching
    pred_set = set((entity['text'].lower().strip(), entity['type']) for entity in predicted_entities)
    gold_set = set((entity['text'].lower().strip(), entity['type']) for entity in gold_entities)
    
    # Calculate overall metrics
    true_positives = len(pred_set & gold_set)
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)
    
    accuracy = true_positives / len(gold_set) if len(gold_set) > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate per-type metrics
    entity_types = set(entity['type'] for entity in gold_entities) | set(entity['type'] for entity in predicted_entities)
    per_type_metrics = {}
    
    for entity_type in entity_types:
        pred_type_set = set(text for text, etype in pred_set if etype == entity_type)
        gold_type_set = set(text for text, etype in gold_set if etype == entity_type)
        
        tp_type = len(pred_type_set & gold_type_set)
        fp_type = len(pred_type_set - gold_type_set)
        fn_type = len(gold_type_set - pred_type_set)
        
        prec_type = tp_type / (tp_type + fp_type) if (tp_type + fp_type) > 0 else 0.0
        rec_type = tp_type / (tp_type + fn_type) if (tp_type + fn_type) > 0 else 0.0
        f1_type = 2 * prec_type * rec_type / (prec_type + rec_type) if (prec_type + rec_type) > 0 else 0.0
        
        per_type_metrics[entity_type] = {
            'accuracy': accuracy,
            'precision': prec_type,
            'recall': rec_type,
            'f1': f1_type,
            'support': len(gold_type_set)
        }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'per_type_metrics': per_type_metrics
    }

def convert_entities_to_bio(tokens, entities, text, valid_entity_types=None):
    """
    Convert entity list back to BIO format labels
    
    Args:
        tokens: List of tokens
        entities: List of entity dictionaries with start_pos, end_pos, type
        text: Original text string
    
    Returns:
        List of BIO labels corresponding to tokens
    """
    bio_labels = ["O"] * len(tokens)
    
    if not entities:
        return bio_labels
    
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
    for entity in entities:
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
