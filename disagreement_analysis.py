# disagreement_analysis.py 최종 수정 버전

import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from utils_disagreement import (
    start_prob,
    inside_prob,
    vprint,
    token_label_dist,
    token_type_dist,
    label_type,
    calculate_auto_weights,
)

# =====================
# Default configuration constants
# =====================

DEFAULT_HOTSPOT_PERCENTILE = 80
DEFAULT_COALITION_CUTOFF = 0.5
DEFAULT_USE_BOUNDARY_VARIANT = True
DEFAULT_MIN_BLOCK_LEN = 1
DEFAULT_MERGE_WITHIN_GAP = True
DEFAULT_ELITE_INTERNAL_PERCENTILE = 90
DEFAULT_TV_BETWEEN_PERCENTILE = 90

VERBOSE = True

# =====================
# Core functions (unchanged)
# =====================

def D_bio(p_label: Dict[str, float]) -> float:
    """BIO label disagreement probability"""
    s2 = sum(v*v for v in p_label.values())
    return 1.0 - s2

def D_type(p_type: Dict[str, float]) -> float:
    """Entity type disagreement (conditional on non-O)"""
    m = 1.0 - p_type.get("O", 0.0)
    if m <= 1e-12:
        return 0.0
    s2 = 0.0
    for c, v in p_type.items():
        if c == "O":
            continue
        pc = v / m
        s2 += pc*pc
    return 1.0 - s2

def U_boundary(p_label: Dict[str, float], use_boundary_variant: bool = DEFAULT_USE_BOUNDARY_VARIANT) -> float:
    """Boundary uncertainty"""
    if use_boundary_variant:
        qs = start_prob(p_label)
        qi = inside_prob(p_label)
        return max(4*qs*(1-qs), 4*qi*(1-qi))
    else:
        q = 1.0 - p_label.get("O", 0.0)
        return 4*q*(1-q)

def coalition_indices(weights: List[float], cutoff: float = DEFAULT_COALITION_CUTOFF) -> List[int]:
    """Select top annotators forming a coalition"""
    idx = list(range(len(weights)))
    idx.sort(key=lambda i: weights[i], reverse=True)
    total = 0.0
    chosen = []
    for i in idx:
        chosen.append(i)
        total += weights[i]
        if total >= cutoff:
            break
    return chosen

def comprehensive_coalition_analysis(labels_by_annot: List[str], weights: List[float],
                                   coalition_cutoff: float = DEFAULT_COALITION_CUTOFF) -> Dict[str, float]:
    """Comprehensive analysis of coalition vs rest and internal disagreements"""
    C = coalition_indices(weights, coalition_cutoff)
    R = [i for i in range(len(weights)) if i not in C]
    
    wC = sum(weights[i] for i in C)
    wR = sum(weights[i] for i in R) if R else 0.0
    
    pC, pR = defaultdict(float), defaultdict(float)
    for i, y in enumerate(labels_by_annot):
        if i in C:
            pC[y] += weights[i]
        else:
            pR[y] += weights[i]
    
    if wC > 0:
        for y in list(pC.keys()):
            pC[y] /= wC
    if wR > 0:
        for y in list(pR.keys()):
            pR[y] /= wR
    
    all_labels = set(pC.keys()) | set(pR.keys())
    tv_between = 0.5 * sum(abs(pC.get(y,0) - pR.get(y,0)) for y in all_labels) if wR > 0 else 0.0
    
    if len(C) > 1:
        coalition_labels = [labels_by_annot[i] for i in C]
        coalition_weights = [weights[i] for i in C]
        coalition_weights = [w/wC for w in coalition_weights]
        p_coalition_internal = token_label_dist(coalition_labels, coalition_weights)
        elite_internal = D_bio(p_coalition_internal)
    else:
        elite_internal = 0.0
    
    if len(R) > 1 and wR > 0:
        rest_labels = [labels_by_annot[i] for i in R]
        rest_weights = [weights[i] for i in R]
        rest_weights = [w/wR for w in rest_weights]
        p_rest_internal = token_label_dist(rest_labels, rest_weights)
        rest_internal = D_bio(p_rest_internal)
    else:
        rest_internal = 0.0
    
    elite_dominance = (tv_between - elite_internal) if tv_between > 0 else 0.0
    
    return {
        "TV_between": tv_between,
        "elite_internal": elite_internal,
        "rest_internal": rest_internal,
        "elite_dominance": elite_dominance,
        "coalition_size": len(C),
        "coalition_weight": wC
    }

def analyze_token_disagreement(labels_by_annot: List[str], weights: List[float], 
                               ann_names: List[str], coalition_cutoff: float = DEFAULT_COALITION_CUTOFF) -> Dict:
    """Analyze disagreement patterns for a single token"""
    p_label = token_label_dist(labels_by_annot, weights)
    p_type = token_type_dist(p_label)
    
    coalition_analysis = comprehensive_coalition_analysis(labels_by_annot, weights, coalition_cutoff)
    C = coalition_indices(weights, coalition_cutoff)
    
    maj_label = max(p_label.items(), key=lambda kv: kv[1])[0]
    maj_type = label_type(maj_label)
    
    annot_analysis = []
    for i, (ann, label) in enumerate(zip(ann_names, labels_by_annot)):
        annot_analysis.append({
            "annotator": ann,
            "label": label,
            "type": label_type(label),
            "weight": weights[i],
            "is_coalition": i in C,
            "agrees_with_mv": label == maj_label
        })
    
    return {
        "majority_label": maj_label,
        "majority_type": maj_type,
        "annotator_details": annot_analysis,
        "coalition_analysis": coalition_analysis,
        "num_disagreeing_annotators": sum(1 for ann in annot_analysis if not ann["agrees_with_mv"])
    }

def analyze_token_vs_gold(labels_by_annot: List[str], gold_label: str,
                         ann_names: List[str], weights: Dict[str, float]) -> Dict:
    """Analyze disagreement patterns for a single token against gold standard"""
    p_label = token_label_dist(labels_by_annot, [weights[a] for a in ann_names])
    
    maj_label = max(p_label.items(), key=lambda kv: kv[1])[0]
    maj_type = label_type(maj_label)
    gold_type = label_type(gold_label)
    
    annot_analysis = []
    for i, (ann, label) in enumerate(zip(ann_names, labels_by_annot)):
        annot_analysis.append({
            "annotator": ann,
            "label": label,
            "type": label_type(label),
            "weight": weights[ann],
            "agrees_with_gold": label == gold_label,
            "agrees_with_mv": label == maj_label
        })
    
    return {
        "gold_label": gold_label,
        "gold_type": gold_type,
        "majority_label": maj_label,
        "majority_type": maj_type,
        "annotator_details": annot_analysis,
        "num_models_agree_gold": sum(1 for ann in annot_analysis if ann["agrees_with_gold"]),
        "num_models_agree_mv": sum(1 for ann in annot_analysis if ann["agrees_with_mv"])
    }

def analyze_annotator_bias(all_token_data: List[Dict], ann_names: List[str]) -> pd.DataFrame:
    """Analyze bias patterns for each annotator compared to majority vote"""
    bias_data = []
    
    for ann in ann_names:
        total_tokens = len(all_token_data)
        agreements = sum(1 for token in all_token_data 
                        for ann_data in token["annotator_details"] 
                        if ann_data["annotator"] == ann and ann_data["agrees_with_mv"])
        
        coalition_membership = sum(1 for token in all_token_data 
                                 for ann_data in token["annotator_details"] 
                                 if ann_data["annotator"] == ann and ann_data["is_coalition"])
        
        type_stats = defaultdict(lambda: {"ann_count": 0, "mv_count": 0})
        
        for token in all_token_data:
            mv_type = token["majority_type"]
            ann_data = next(a for a in token["annotator_details"] if a["annotator"] == ann)
            ann_type = ann_data["type"]
            
            type_stats[mv_type]["mv_count"] += 1
            type_stats[ann_type]["ann_count"] += 1
        
        entity_over_tagging = sum(stats["ann_count"] for typ, stats in type_stats.items() if typ != "O") - \
                             sum(stats["mv_count"] for typ, stats in type_stats.items() if typ != "O")
        
        bias_data.append({
            "annotator": ann,
            "agreement_rate": agreements / total_tokens,
            "coalition_rate": coalition_membership / total_tokens,
            "entity_over_tagging": entity_over_tagging,
        })
    
    return pd.DataFrame(bias_data)

def analyze_model_vs_gold_bias(all_token_data: List[Dict], ann_names: List[str]) -> pd.DataFrame:
    """Analyze bias patterns for each model compared to gold standard"""
    bias_data = []
    
    for ann in ann_names:
        total_tokens = len(all_token_data)
        gold_agreements = sum(1 for token in all_token_data 
                            for ann_data in token["annotator_details"] 
                            if ann_data["annotator"] == ann and ann_data["agrees_with_gold"])
        
        mv_agreements = sum(1 for token in all_token_data 
                          for ann_data in token["annotator_details"] 
                          if ann_data["annotator"] == ann and ann_data["agrees_with_mv"])
        
        type_stats = defaultdict(lambda: {"ann_count": 0, "gold_count": 0})
        
        for token in all_token_data:
            gold_type = token["gold_type"]
            ann_data = next(a for a in token["annotator_details"] if a["annotator"] == ann)
            ann_type = ann_data["type"]
            
            type_stats[gold_type]["gold_count"] += 1
            type_stats[ann_type]["ann_count"] += 1
        
        entity_over_tagging = sum(stats["ann_count"] for typ, stats in type_stats.items() if typ != "O") - \
                             sum(stats["gold_count"] for typ, stats in type_stats.items() if typ != "O")
        
        bias_data.append({
            "annotator": ann,
            "gold_agreement_rate": gold_agreements / total_tokens,
            "mv_agreement_rate": mv_agreements / total_tokens,
            "entity_over_tagging_vs_gold": entity_over_tagging,
        })
    
    return pd.DataFrame(bias_data)

def compute_sentence_metrics(tokens: List[str],
                             ann_labels: Dict[str, List[str]],
                             weights_by_annot: Dict[str, float],
                             coalition_cutoff: float = DEFAULT_COALITION_CUTOFF,
                             use_boundary_variant: bool = DEFAULT_USE_BOUNDARY_VARIANT) -> Tuple[pd.DataFrame, List[Dict]]:
    """Compute disagreement metrics for each token"""
    ann_names = list(ann_labels.keys())
    W = [weights_by_annot[a] for a in ann_names]
    rows = []
    token_analyses = []
    
    for t in range(len(tokens)):
        labels_t = [ann_labels[a][t] for a in ann_names]
        p_lab = token_label_dist(labels_t, W)
        p_typ = token_type_dist(p_lab)
        
        coalition_analysis = comprehensive_coalition_analysis(labels_t, W, coalition_cutoff)
        
        row = {
            "tok_idx": t,
            "token": tokens[t],
            "maj_label": max(p_lab.items(), key=lambda kv: kv[1])[0],
            "D_bio": D_bio(p_lab),
            "D_type": D_type(p_typ),
            "U": U_boundary(p_lab, use_boundary_variant),
            "TV_between": coalition_analysis["TV_between"],
            "elite_internal": coalition_analysis["elite_internal"],
            "rest_internal": coalition_analysis["rest_internal"],
            "elite_dominance": coalition_analysis["elite_dominance"],
        }
        rows.append(row)
        
        token_analysis = analyze_token_disagreement(labels_t, W, ann_names, coalition_cutoff)
        token_analysis["tok_idx"] = t
        token_analysis["token"] = tokens[t]
        token_analyses.append(token_analysis)
    
    df = pd.DataFrame(rows)
    df["U_star"] = df[["D_bio","D_type","U"]].max(axis=1)
    return df, token_analyses

def compute_sentence_metrics_with_gold(tokens: List[str],
                                     ann_labels: Dict[str, List[str]],
                                     gold_labels: List[str],
                                     weights_by_annot: Dict[str, float],
                                     use_boundary_variant: bool = DEFAULT_USE_BOUNDARY_VARIANT) -> Tuple[pd.DataFrame, List[Dict]]:
    """Compute disagreement metrics for each token using gold standard as reference"""
    ann_names = list(ann_labels.keys())
    rows = []
    token_analyses = []
    
    for t in range(len(tokens)):
        if t >= len(gold_labels):
            continue
            
        labels_t = [ann_labels[a][t] for a in ann_names if t < len(ann_labels[a])]
        gold_label = gold_labels[t]
        
        if len(labels_t) != len(ann_names):
            continue
        
        p_lab = token_label_dist(labels_t, [weights_by_annot[a] for a in ann_names])
        p_typ = token_type_dist(p_lab)
        
        row = {
            "tok_idx": t,
            "token": tokens[t],
            "gold_label": gold_label,
            "maj_label": max(p_lab.items(), key=lambda kv: kv[1])[0],
            "D_bio": D_bio(p_lab),
            "D_type": D_type(p_typ),
            "U": U_boundary(p_lab, use_boundary_variant),
            "TV_between": 0.0,
            "elite_internal": 0.0,
            "rest_internal": 0.0,
            "elite_dominance": 0.0,
        }
        rows.append(row)
        
        token_analysis = analyze_token_vs_gold(labels_t, gold_label, ann_names, weights_by_annot)
        token_analysis["tok_idx"] = t
        token_analysis["token"] = tokens[t]
        token_analyses.append(token_analysis)
    
    df = pd.DataFrame(rows)
    df["U_star"] = df[["D_bio","D_type","U"]].max(axis=1)
    return df, token_analyses

def extract_hotspot_blocks(df: pd.DataFrame, global_threshold: float,
                          min_block_len: int = DEFAULT_MIN_BLOCK_LEN,
                          merge_within_gap: bool = DEFAULT_MERGE_WITHIN_GAP) -> List[Dict]:
    """Extract contiguous hotspot blocks using global threshold"""
    mask = (df["U_star"].values >= global_threshold)
    blocks, n, i = [], len(df), 0
    
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start, j = i, i
        while j+1<n and mask[j+1]:
            j += 1
        if merge_within_gap:
            k = j+1
            if k<n and not mask[k]:
                if k+1<n and mask[k+1]:
                    j = k+1
                    while j+1<n and mask[j+1]:
                        j += 1
        if j-start+1 >= min_block_len:
            blocks.append((start,j))
        i = j+1
    
    out = []
    for (s,e) in blocks:
        sub = df.iloc[s:e+1]
        out.append({
            "start": int(s), "end": int(e),
            "span_tokens": " ".join(sub["token"].tolist()),
            "U_star_mean": float(sub["U_star"].mean()),
            "TV_between_max": float(sub["TV_between"].max()),
            "elite_internal_max": float(sub["elite_internal"].max()),
            "elite_dominance_max": float(sub["elite_dominance"].max()),
        })
    
    return out

def analyze_ner_disagreement_dataset(sentence_data: List[Dict],
                                   weights: Optional[Dict[str, float]] = None,
                                   output_dir: str = "./disagreement_analysis",
                                   save_results: bool = True,
                                   hotspot_percentile: float = DEFAULT_HOTSPOT_PERCENTILE,
                                   coalition_cutoff: float = DEFAULT_COALITION_CUTOFF,
                                   use_boundary_variant: bool = DEFAULT_USE_BOUNDARY_VARIANT,
                                   min_block_len: int = DEFAULT_MIN_BLOCK_LEN,
                                   merge_within_gap: bool = DEFAULT_MERGE_WITHIN_GAP,
                                   elite_internal_percentile: float = DEFAULT_ELITE_INTERNAL_PERCENTILE,
                                   tv_between_percentile: float = DEFAULT_TV_BETWEEN_PERCENTILE,
                                   use_gold_standard: bool = False,
                                   gold_standard_key: str = 'gold_labels') -> Dict:
    """
    Analyze NER disagreement patterns across a complete dataset
    MODIFIED: Added gold standard support
    """
    analysis_mode = "Gold Standard" if use_gold_standard else "Majority Vote"
    vprint(f"Analyzing dataset of {len(sentence_data)} sentences using {analysis_mode} reference")
    vprint(f"Configuration: hotspot_percentile={hotspot_percentile}, coalition_cutoff={coalition_cutoff}")
    
    if use_gold_standard:
        valid_samples = sum(1 for sent_data in sentence_data 
                           if gold_standard_key in sent_data and sent_data[gold_standard_key])
        if valid_samples == 0:
            raise ValueError(f"No gold standard labels found in sentence data under key '{gold_standard_key}'")
        vprint(f"Found {valid_samples}/{len(sentence_data)} sentences with gold standard labels")
    
    if weights is None and not use_gold_standard:
        all_labels_by_models = defaultdict(list)
        for sent_data in sentence_data:
            for model, labels in sent_data['labels_by_models'].items():
                all_labels_by_models[model].extend(labels)
        weights = calculate_auto_weights(dict(all_labels_by_models))
    elif use_gold_standard:
        model_names = list(sentence_data[0]['labels_by_models'].keys())
        weights = {model: 1.0 for model in model_names}
    
    all_token_analyses = []
    all_u_star_values = []
    all_tv_between_values = []
    all_elite_internal_values = []
    all_token_metrics = []
    
    for i, sent_data in enumerate(sentence_data):
        if use_gold_standard:
            df, token_analyses = compute_sentence_metrics_with_gold(
                tokens=sent_data['tokens'],
                ann_labels=sent_data['labels_by_models'],
                gold_labels=sent_data.get(gold_standard_key, []),
                weights_by_annot=weights,
                use_boundary_variant=use_boundary_variant
            )
        else:
            df, token_analyses = compute_sentence_metrics(
                tokens=sent_data['tokens'],
                ann_labels=sent_data['labels_by_models'],
                weights_by_annot=weights,
                coalition_cutoff=coalition_cutoff,
                use_boundary_variant=use_boundary_variant
            )
        
        df['sentence_id'] = i
        for token_analysis in token_analyses:
            token_analysis['sentence_id'] = i
        
        all_token_metrics.append(df)
        all_token_analyses.extend(token_analyses)
        all_u_star_values.extend(df['U_star'].values)
        
        if not use_gold_standard:
            all_tv_between_values.extend(df['TV_between'].values)
            all_elite_internal_values.extend(df['elite_internal'].values)
    
    combined_df = pd.concat(all_token_metrics, ignore_index=True)
    
    global_u_star_threshold = np.percentile(all_u_star_values, hotspot_percentile)
    
    if use_gold_standard:
        vprint(f"Global threshold (Gold Standard mode):")
        vprint(f"  U* threshold (percentile {hotspot_percentile}): {global_u_star_threshold:.4f}")
        global_tv_between_threshold = 0.0
        global_elite_internal_threshold = 0.0
    else:
        global_tv_between_threshold = np.percentile(all_tv_between_values, tv_between_percentile)
        global_elite_internal_threshold = np.percentile(all_elite_internal_values, elite_internal_percentile)
        
        vprint(f"Global thresholds (Majority Vote mode):")
        vprint(f"  U* threshold (percentile {hotspot_percentile}): {global_u_star_threshold:.4f}")
        vprint(f"  TV_between threshold (percentile {tv_between_percentile}): {global_tv_between_threshold:.4f}")
        vprint(f"  Elite_internal threshold (percentile {elite_internal_percentile}): {global_elite_internal_threshold:.4f}")
    
    all_hotspots = []
    for i, sent_data in enumerate(sentence_data):
        sent_df = combined_df[combined_df['sentence_id'] == i].copy()
        sent_df = sent_df.reset_index(drop=True)
        
        blocks = extract_hotspot_blocks(sent_df, global_u_star_threshold, min_block_len, merge_within_gap)
        
        for block in blocks:
            start, end = block["start"], block["end"]
            sub = sent_df.iloc[start:end+1]
            
            block["sentence_id"] = i
            
            if use_gold_standard:
                block["disagreement_type"] = "Model vs Gold Standard"
                block["true_elite_split"] = False
                block["systematic_bias"] = False
            else:
                has_elite_split = (sub["elite_internal"] >= global_elite_internal_threshold).any()
                has_systematic_bias = (sub["TV_between"] >= global_tv_between_threshold).any()
                
                block["true_elite_split"] = has_elite_split
                block["systematic_bias"] = has_systematic_bias
                
                if has_elite_split and has_systematic_bias:
                    block["disagreement_type"] = "Complex (Elite Split + Systematic Bias)"
                elif has_elite_split:
                    block["disagreement_type"] = "True Elite Split"
                elif has_systematic_bias:
                    block["disagreement_type"] = "Systematic Bias"
                else:
                    block["disagreement_type"] = "Minor Disagreement"
                    
            all_hotspots.append(block)
    
    model_names = list(weights.keys())
    if use_gold_standard:
        global_bias_df = analyze_model_vs_gold_bias(all_token_analyses, model_names)
    else:
        global_bias_df = analyze_annotator_bias(all_token_analyses, model_names)
    
    dataset_summary = {
        "analysis_mode": analysis_mode,
        "use_gold_standard": use_gold_standard,
        "num_sentences": len(sentence_data),
        "total_tokens": len(combined_df),
        "num_models": len(model_names),
        "avg_D_bio": combined_df['D_bio'].mean(),
        "avg_D_type": combined_df['D_type'].mean(),
        "avg_U": combined_df['U'].mean(),
        "avg_U_star": combined_df['U_star'].mean(),
        "total_hotspots": len(all_hotspots),
        "sentences_with_hotspots": len(set(h['sentence_id'] for h in all_hotspots)),
        "hotspot_percentile_used": hotspot_percentile,
    }
    
    if not use_gold_standard:
        dataset_summary.update({
            "avg_TV_between": combined_df['TV_between'].mean(),
            "avg_elite_internal": combined_df['elite_internal'].mean(),
            "avg_elite_dominance": combined_df['elite_dominance'].mean(),
            "coalition_cutoff_used": coalition_cutoff,
            "true_elite_splits": sum(1 for h in all_hotspots if h['true_elite_split']),
            "systematic_biases": sum(1 for h in all_hotspots if h['systematic_bias']),
            "complex_disagreements": sum(1 for h in all_hotspots if h['true_elite_split'] and h['systematic_bias']),
        })
    
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        
        combined_df.to_csv(os.path.join(output_dir, "dataset_token_metrics.csv"), index=False)
        
        if all_hotspots:
            hotspots_df = pd.DataFrame(all_hotspots)
            hotspots_df.to_csv(os.path.join(output_dir, "dataset_hotspots.csv"), index=False)
        
        global_bias_df.to_csv(os.path.join(output_dir, "model_bias_analysis.csv"), index=False)
        
        summary_df = pd.DataFrame([dataset_summary])
        summary_df.to_csv(os.path.join(output_dir, "dataset_summary.csv"), index=False)
        
        vprint(f"Results saved to {output_dir}/")
    
    vprint(f"\n{'='*50}")
    vprint(f"DATASET DISAGREEMENT ANALYSIS SUMMARY ({analysis_mode}):")
    vprint(f"{'='*50}")
    vprint(f"Sentences: {dataset_summary['num_sentences']}")
    vprint(f"Total tokens: {dataset_summary['total_tokens']}")
    vprint(f"Models: {', '.join(model_names)}")
    vprint(f"Average disagreement metrics:")
    vprint(f"  D_bio: {dataset_summary['avg_D_bio']:.3f}")
    vprint(f"  D_type: {dataset_summary['avg_D_type']:.3f}")
    vprint(f"  U_boundary: {dataset_summary['avg_U']:.3f}")
    vprint(f"  U_star: {dataset_summary['avg_U_star']:.3f}")
    
    if not use_gold_standard:
        vprint(f"  TV_between: {dataset_summary['avg_TV_between']:.3f}")
        vprint(f"  Elite_internal: {dataset_summary['avg_elite_internal']:.3f}")
        vprint(f"  Elite_dominance: {dataset_summary['avg_elite_dominance']:.3f}")
    
    vprint(f"Total hotspots: {dataset_summary['total_hotspots']}")
    vprint(f"Sentences with hotspots: {dataset_summary['sentences_with_hotspots']}")
    
    if all_hotspots:
        disagreement_type_counts = Counter(h["disagreement_type"] for h in all_hotspots)
        total_hotspots = len(all_hotspots)
        
        vprint(f"\nHotspot classification distribution:")
        for disagreement_type, count in disagreement_type_counts.most_common():
            percentage = 100 * count / total_hotspots
            vprint(f"  {disagreement_type}: {count} ({percentage:.1f}%)")
    
    if VERBOSE and not global_bias_df.empty:
        vprint(f"\n{'='*40}")
        vprint(f"MODEL BIAS ANALYSIS ({analysis_mode}):")
        vprint(f"{'='*40}")
        print(global_bias_df.to_string(index=False, float_format='{:.3f}'.format))
    
    return {
        "dataset_summary": dataset_summary,
        "token_metrics": combined_df,
        "hotspots": all_hotspots,
        "bias_analysis": global_bias_df,
        "global_thresholds": {
            "u_star": global_u_star_threshold,
            "tv_between": global_tv_between_threshold if not use_gold_standard else None,
            "elite_internal": global_elite_internal_threshold if not use_gold_standard else None
        },
        "weights_used": weights,
        "config_used": {
            "hotspot_percentile": hotspot_percentile,
            "coalition_cutoff": coalition_cutoff if not use_gold_standard else None,
            "use_boundary_variant": use_boundary_variant,
            "min_block_len": min_block_len,
            "merge_within_gap": merge_within_gap,
            "elite_internal_percentile": elite_internal_percentile if not use_gold_standard else None,
            "tv_between_percentile": tv_between_percentile if not use_gold_standard else None,
            "use_gold_standard": use_gold_standard,
            "gold_standard_key": gold_standard_key if use_gold_standard else None
        }
    }