import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable, Optional, Set

import numpy as np
import pandas as pd

from sklearn.metrics import cohen_kappa_score

# =====================
# Global configuration
# =====================

HOTSPOT_PERCENTILE = 90     # Top percentile tokens selected as hotspots (GLOBAL)
COALITION_CUTOFF = 0.5      # Cumulative weight for top annotator group
USE_BOUNDARY_VARIANT = True # True: start/inside based, False: entity vs O based
MIN_BLOCK_LEN = 1
MERGE_WITHIN_GAP = True
ELITE_INTERNAL_PERCENTILE = 90    # Elite internal disagreement threshold
TV_BETWEEN_PERCENTILE = 90        # TV between groups threshold

OUTDIR = "./disagreement_results"
VERBOSE = True  # Global verbosity control

# =====================
# Utility functions
# =====================

def vprint(*args, **kwargs):
    """Verbose print - only prints if VERBOSE is True"""
    if VERBOSE:
        print(*args, **kwargs)

def set_verbosity(verbose: bool):
    """Set global verbosity"""
    global VERBOSE
    VERBOSE = verbose


# =====================
# BIO helpers
# =====================

def label_type(y: str) -> str:
    """Extract entity type from BIO label"""
    if y == "O":
        return "O"
    if "-" in y:
        return y.split("-", 1)[1]
    return "O"

def is_B(y: str) -> bool:
    return y.startswith("B-")

def is_I(y: str) -> bool:
    return y.startswith("I-")

# =====================
# BIO span extraction functions
# =====================
def extract_spans_from_bio(labels: List[str]) -> Set[Tuple[int, int, str]]:
    """
    Extract entity spans from BIO labels
    
    Args:
        labels: List of BIO labels
        
    Returns:
        Set of tuples (start, end, entity_type) where end is exclusive
    """
    spans = set()
    current_start = None
    current_type = None
    
    for i, label in enumerate(labels):
        if label.startswith('B-'):
            # End previous span if exists
            if current_start is not None:
                spans.add((current_start, i, current_type))
            
            # Start new span
            current_start = i
            current_type = label[2:]  # Remove 'B-' prefix
            
        elif label.startswith('I-'):
            entity_type = label[2:]  # Remove 'I-' prefix
            
            # Continue span only if type matches
            if current_start is not None and current_type == entity_type:
                continue
            else:
                # End previous span if exists and start new one
                if current_start is not None:
                    spans.add((current_start, i, current_type))
                current_start = i
                current_type = entity_type
                
        else:  # 'O' label
            # End current span if exists
            if current_start is not None:
                spans.add((current_start, i, current_type))
                current_start = None
                current_type = None
    
    # End final span if exists
    if current_start is not None:
        spans.add((current_start, len(labels), current_type))
    
    return spans


def calculate_strict_span_f1(labels1: List[str], labels2: List[str]) -> float:
    """
    Calculate strict span F1 score between two label sequences
    
    Args:
        labels1: First label sequence
        labels2: Second label sequence
        
    Returns:
        Strict span F1 score (0.0 to 1.0)
    """
    if len(labels1) != len(labels2):
        raise ValueError("Label sequences must have the same length")
    
    spans1 = extract_spans_from_bio(labels1)
    spans2 = extract_spans_from_bio(labels2)
    
    # Calculate precision, recall, F1
    if len(spans1) == 0 and len(spans2) == 0:
        return 1.0  # Perfect agreement on no entities
    
    if len(spans1) == 0 or len(spans2) == 0:
        return 0.0  # One has entities, other doesn't
    
    # Count exact matches
    exact_matches = len(spans1.intersection(spans2))
    
    precision = exact_matches / len(spans2) if len(spans2) > 0 else 0.0
    recall = exact_matches / len(spans1) if len(spans1) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def token_label_dist(labels_by_annot: List[str], weights: List[float]) -> Dict[str, float]:
    """Calculate weighted label distribution for a token"""
    p = defaultdict(float)
    for y, w in zip(labels_by_annot, weights):
        p[y] += w
    return dict(p)

def token_type_dist(p_label: Dict[str, float]) -> Dict[str, float]:
    """Convert label distribution to type distribution"""
    q = defaultdict(float)
    for y, prob in p_label.items():
        q[label_type(y)] += prob
    return dict(q)

def start_prob(p_label: Dict[str, float]) -> float:
    """Probability of B- tags"""
    return sum(prob for y, prob in p_label.items() if is_B(y))

def inside_prob(p_label: Dict[str, float]) -> float:
    """Probability of I- tags"""
    return sum(prob for y, prob in p_label.items() if is_I(y))


# =====================
# Enhanced test cases
# =====================
# def build_test_cases():
#     """Build diverse test cases covering various disagreement scenarios"""
#     weights = {"A1":0.35,"A2":0.30,"A3":0.20,"A4":0.15}
#     tests = {}
    
#     # Case 1: Basic entity recognition disagreement
#     tokens1 = "John works at OpenAI in San Francisco .".split()
#     tests["basic_disagree"] = {
#         "tokens": tokens1,
#         "weights": weights,
#         "labels": {
#             "A1":["B-PER","O","O","B-ORG","O","B-LOC","I-LOC","O"],
#             "A2":["B-PER","O","O","B-PER","O","B-LOC","I-LOC","O"],  # OpenAI as PER
#             "A3":["B-PER","O","O","B-ORG","O","B-LOC","I-LOC","O"],
#             "A4":["B-PER","O","O","B-ORG","O","B-LOC","I-LOC","O"],
#         }
#     }
    
#     # Case 2: True elite split (high elite_internal)
#     tokens2 = "Dr Smith founded Apple in Cupertino California .".split()
#     tests["true_elite_split"] = {
#         "tokens": tokens2,
#         "weights": weights,
#         "labels": {
#             "A1":["B-TITLE","B-PER","O","B-ORG","O","B-LOC","B-LOC","O"],  # Dr as TITLE
#             "A2":["B-PER","I-PER","O","B-ORG","O","B-LOC","B-LOC","O"],    # Dr Smith as one PER
#             "A3":["B-PER","I-PER","O","B-ORG","O","B-LOC","I-LOC","O"],    # Cupertino California as one LOC
#             "A4":["B-PER","I-PER","O","B-COMPANY","O","B-CITY","B-STATE","O"], # Different granularity
#         }
#     }
    
#     # Case 3: Systematic bias (high TV_between, low elite_internal)
#     tokens3 = "Google Search helps users find information online .".split()
#     tests["systematic_bias"] = {
#         "tokens": tokens3,
#         "weights": weights,
#         "labels": {
#             "A1":["B-ORG","B-PROD","O","O","O","O","O","O"],     # Elite: consistent ORG+PROD
#             "A2":["B-ORG","B-PROD","O","O","O","O","O","O"],     # Elite: same as A1
#             "A3":["B-COMPANY","B-SERVICE","O","O","O","O","O","O"], # Rest: more granular
#             "A4":["B-TECH","B-TOOL","O","O","O","O","O","O"],       # Rest: different perspective
#         }
#     }
    
#     # Case 4: Boundary disagreement
#     tokens4 = "The New York based startup raised funds .".split()
#     tests["boundary_disagree"] = {
#         "tokens": tokens4,
#         "weights": weights,
#         "labels": {
#             "A1":["O","B-LOC","I-LOC","O","O","O","O","O"],
#             "A2":["O","B-LOC","I-LOC","O","O","O","O","O"],
#             "A3":["O","O","B-LOC","O","O","O","O","O"],  # Different boundary
#             "A4":["O","O","O","O","O","O","O","O"],     # No entity
#         }
#     }
    
#     # Case 5: Mixed patterns
#     tokens5 = "Tesla CEO Elon Musk announced new Model Y .".split()
#     tests["mixed_patterns"] = {
#         "tokens": tokens5,
#         "weights": weights,
#         "labels": {
#             "A1":["B-ORG","O","B-PER","I-PER","O","O","B-PROD","I-PROD","O"],
#             "A2":["B-ORG","B-TITLE","B-PER","I-PER","O","O","B-PROD","I-PROD","O"], # CEO as TITLE
#             "A3":["B-COMPANY","O","B-PER","I-PER","O","B-ADJ","B-MODEL","I-MODEL","O"], # Different types
#             "A4":["B-STOCK","O","B-PERSON","I-PERSON","O","O","B-CAR","I-CAR","O"], # More specific
#         }
#     }
    
#     # Case 6: Perfect agreement (control)
#     tokens6 = "Barack Obama was born in Hawaii .".split()
#     tests["perfect_agreement"] = {
#         "tokens": tokens6,
#         "weights": weights,
#         "labels": {
#             "A1":["B-PER","I-PER","O","O","O","B-LOC","O"],
#             "A2":["B-PER","I-PER","O","O","O","B-LOC","O"],
#             "A3":["B-PER","I-PER","O","O","O","B-LOC","O"],
#             "A4":["B-PER","I-PER","O","O","O","B-LOC","O"],
#         }
#     }
    
#     # Case 7: Medical entities
#     tokens7 = "Patient has COVID-19 and diabetes mellitus type 2 .".split()
#     tests["medical_complexity"] = {
#         "tokens": tokens7,
#         "weights": weights,
#         "labels": {
#             "A1":["O","O","B-DISEASE","O","B-DISEASE","I-DISEASE","I-DISEASE","I-DISEASE","O"],
#             "A2":["O","O","B-VIRUS","O","B-CONDITION","I-CONDITION","I-CONDITION","I-CONDITION","O"], # Elite split
#             "A3":["O","O","B-COVID","O","B-DIABETES","O","O","B-TYPE","O"], # Rest: granular
#             "A4":["O","O","B-ILLNESS","O","B-CHRONIC","I-CHRONIC","I-CHRONIC","I-CHRONIC","O"], # Rest: different
#         }
#     }
    
#     return tests

def calculate_cohen_kappa_bio(labels1: List[str], labels2: List[str]) -> float:
    """
    Calculate Cohen's Kappa for BIO labels between two sequences
    
    Args:
        labels1: First label sequence
        labels2: Second label sequence
        
    Returns:
        Cohen's Kappa score (can be negative, typically -1 to 1)
    """
    if len(labels1) != len(labels2):
        raise ValueError("Label sequences must have the same length")
    
    if len(labels1) == 0:
        return 1.0  # Perfect agreement on empty sequence
    
    try:
        kappa = cohen_kappa_score(labels1, labels2)
        # Handle NaN case (perfect agreement with single class)
        if np.isnan(kappa):
            # Check if they are identical
            if labels1 == labels2:
                return 1.0
            else:
                return 0.0
        return kappa
    except Exception as e:
        vprint(f"Warning: Cohen's Kappa calculation failed: {e}")
        # Fallback to simple agreement
        agreements = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == l2)
        return agreements / len(labels1) if len(labels1) > 0 else 0.0

def calculate_hybrid_weights(labels_by_model: Dict[str, List[str]], 
                           span_weight: float = 0.5,
                           kappa_weight: float = 0.5) -> Dict[str, float]:
    """
    Calculate hybrid weights combining Span F1 and Token-level Cohen's Kappa
    
    Args:
        labels_by_model: Dictionary mapping model names to label sequences
        span_weight: Weight for span F1 component (default: 0.5)
        kappa_weight: Weight for Cohen's Kappa component (default: 0.5)
        
    Returns:
        Dictionary mapping model names to normalized weights
    """
    model_names = list(labels_by_model.keys())
    n_models = len(model_names)
    
    if n_models <= 1:
        return {name: 1.0 for name in model_names}
    
    if abs(span_weight + kappa_weight - 1.0) > 1e-6:
        raise ValueError(f"span_weight and kappa_weight must sum to 1.0, got {span_weight + kappa_weight}")
    
    # Calculate Span F1 scores
    span_f1_scores = {}
    for i, model1 in enumerate(model_names):
        total_f1 = 0.0
        count = 0
        
        for j, model2 in enumerate(model_names):
            if i != j:
                f1 = calculate_strict_span_f1(
                    labels_by_model[model1], 
                    labels_by_model[model2]
                )
                total_f1 += f1
                count += 1
        
        span_f1_scores[model1] = total_f1 / count if count > 0 else 0.0
    
    # Calculate Cohen's Kappa scores
    kappa_scores = {}
    for i, model1 in enumerate(model_names):
        total_kappa = 0.0
        count = 0
        
        for j, model2 in enumerate(model_names):
            if i != j:
                kappa = calculate_cohen_kappa_bio(
                    labels_by_model[model1], 
                    labels_by_model[model2]
                )
                total_kappa += kappa
                count += 1
        
        kappa_scores[model1] = total_kappa / count if count > 0 else 0.0
    
    # Handle negative Kappa scores by shifting all scores to be non-negative
    min_kappa = min(kappa_scores.values())
    if min_kappa < 0:
        kappa_adjusted = {model: score - min_kappa for model, score in kappa_scores.items()}
    else:
        kappa_adjusted = kappa_scores.copy()
    
    # Normalize each component to sum to 1 (like probability distributions)
    span_total = sum(span_f1_scores.values())
    kappa_total = sum(kappa_adjusted.values())
    
    if span_total > 0:
        span_normalized = {model: score / span_total for model, score in span_f1_scores.items()}
    else:
        span_normalized = {model: 1.0 / n_models for model in model_names}
    
    if kappa_total > 0:
        kappa_normalized = {model: score / kappa_total for model, score in kappa_adjusted.items()}
    else:
        kappa_normalized = {model: 1.0 / n_models for model in model_names}
    
    # Combine scores with weights
    hybrid_scores = {}
    for model in model_names:
        hybrid_scores[model] = (
            span_weight * span_normalized[model] + 
            kappa_weight * kappa_normalized[model]
        )
    
    # Final weights are already normalized (sum to 1)
    weights = hybrid_scores
    
    # Print detailed information
    vprint(f"Hybrid weight calculation (Span F1: {span_weight}, Kappa: {kappa_weight}):")
    vprint(f"{'Model':<15} {'Span F1':<10} {'Kappa':<10} {'Hybrid':<10} {'Weight':<10}")
    vprint("-" * 65)
    
    for model in model_names:
        vprint(f"{model:<15} {span_f1_scores[model]:<10.3f} {kappa_scores[model]:<10.3f} "
               f"{hybrid_scores[model]:<10.3f} {weights[model]:<10.3f}")
    
    return weights

def calculate_auto_weights_original_span_only(labels_by_model: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Original span-only weight calculation (renamed for clarity)
    This is the same as the current calculate_auto_weights() function
    """
    return calculate_auto_weights(labels_by_model)

def calculate_auto_weights_kappa_only(labels_by_model: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calculate weights based only on Cohen's Kappa scores
    
    Args:
        labels_by_model: Dictionary mapping model names to label sequences
        
    Returns:
        Dictionary mapping model names to normalized weights
    """
    model_names = list(labels_by_model.keys())
    n_models = len(model_names)
    
    if n_models <= 1:
        return {name: 1.0 for name in model_names}
    
    # Calculate Cohen's Kappa scores
    kappa_scores = {}
    for i, model1 in enumerate(model_names):
        total_kappa = 0.0
        count = 0
        
        for j, model2 in enumerate(model_names):
            if i != j:
                kappa = calculate_cohen_kappa_bio(
                    labels_by_model[model1], 
                    labels_by_model[model2]
                )
                total_kappa += kappa
                count += 1
        
        kappa_scores[model1] = total_kappa / count if count > 0 else 0.0
    
    # Since Cohen's Kappa can be negative, we need to handle this for weights
    # Option 1: Shift all scores to be non-negative
    min_kappa = min(kappa_scores.values())
    if min_kappa < 0:
        adjusted_scores = {model: score - min_kappa for model, score in kappa_scores.items()}
    else:
        adjusted_scores = kappa_scores.copy()
    
    # Normalize to sum to 1
    total_score = sum(adjusted_scores.values())
    if total_score > 0:
        weights = {model: score / total_score for model, score in adjusted_scores.items()}
    else:
        # Fallback to uniform weights
        weights = {model: 1.0 / n_models for model in model_names}
    
    vprint(f"Kappa-only weight calculation:")
    for model, weight in weights.items():
        vprint(f"  {model}: {weight:.3f} (avg kappa: {kappa_scores[model]:.3f})")
    
    return weights

# Modified main weight calculation function
def calculate_auto_weights_enhanced(labels_by_model: Dict[str, List[str]], 
                                  method: str = "hybrid",
                                  span_weight: float = 0.7,
                                  kappa_weight: float = 0.3) -> Dict[str, float]:
    """
    Enhanced weight calculation with multiple methods
    
    Args:
        labels_by_model: Dictionary mapping model names to label sequences
        method: Weight calculation method ('hybrid', 'span_only', 'kappa_only', 'token_agreement')
        span_weight: Weight for span F1 component in hybrid method
        kappa_weight: Weight for Cohen's Kappa component in hybrid method
        
    Returns:
        Dictionary mapping model names to normalized weights
    """
    if method == "hybrid":
        return calculate_hybrid_weights(labels_by_model, span_weight, kappa_weight)
    elif method == "span_only":
        return calculate_auto_weights_original_span_only(labels_by_model)
    elif method == "kappa_only":
        return calculate_auto_weights_kappa_only(labels_by_model)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hybrid', 'span_only', 'kappa_only', or 'token_agreement'")

# Update the main calculate_auto_weights function to use hybrid by default
def calculate_auto_weights(labels_by_model: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calculate weights using hybrid method (Span F1 + Cohen's Kappa) by default
    This replaces the original function to maintain backward compatibility
    """
    return calculate_hybrid_weights(labels_by_model, span_weight=0.5, kappa_weight=0.5)

