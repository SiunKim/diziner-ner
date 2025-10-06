# disagreement_analysis_in_pipeline.py 최종 수정 버전

import os
import pickle
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

from disagreement_analysis import (
    coalition_indices, analyze_ner_disagreement_dataset
)

# 제거: from gold_standard_documentation import generate_gold_standard_documentation_from_results

def load_experiment_results(result_file_path: str) -> Dict[str, Any]:
    """Load experiment results from pickle or JSON file"""
    result_path = Path(result_file_path)
    
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_file_path}")
    
    if result_path.suffix == '.pkl':
        with open(result_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded pickle results from: {result_path}")
    elif result_path.suffix == '.json':
        with open(result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded JSON results from: {result_path}")
    else:
        raise ValueError(f"Unsupported file format: {result_path.suffix}")
    
    return results

def extract_experiment_metadata(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from experiment results for output naming"""
    if 'experiment_results' in results:
        experiment_data = results['experiment_results']
    else:
        experiment_data = results
    
    experiment_info = experiment_data.get('experiment_info', {})
    
    metadata = {
        'dataset_name': Path(experiment_info.get('dataset', 'unknown')).stem.replace('_ner_dataset', ''),
        'num_groups': experiment_info.get('num_groups', 'unknown'),
        'group_size': experiment_info.get('group_size', 'unknown'),
        'total_samples': experiment_info.get('total_samples', 'unknown'),
        'models_tested': experiment_info.get('models_tested', []),
        'timestamp': experiment_info.get('timestamp', 'unknown'),
        'experiment_type': experiment_info.get('experiment_type', 'unknown')
    }
    
    return metadata

def create_output_directory(base_dir: str, metadata: Dict[str, Any]) -> Path:
    """Create organized output directory based on experiment metadata"""
    folder_name = (
        f"{metadata['dataset_name']}_"
        f"{metadata['num_groups']}groups_"
        f"{metadata['group_size']}size_"
        f"{len(metadata['models_tested'])}models"
    )
    
    output_dir = Path(base_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    return output_dir

def convert_experiment_to_disagreement_format(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert experiment results to format expected by disagreement analysis"""
    if 'experiment_results' in results:
        experiment_data = results['experiment_results']
    else:
        experiment_data = results
    
    test_samples = experiment_data.get('test_samples', [])
    results_by_model = experiment_data.get('results_by_model', {})
    
    print(f"Converting {len(test_samples)} samples from {len(results_by_model)} models")
    
    sentence_data = []
    
    for sample_idx, sample in enumerate(test_samples):
        tokens = sample['tokens']
        
        labels_by_models = {}
        
        for model_name, model_result in results_by_model.items():
            if 'error' in model_result:
                print(f"Skipping model {model_name} due to error: {model_result['error']}")
                continue
            
            detailed_results = model_result.get('detailed_results', [])
            
            if sample_idx < len(detailed_results):
                sample_result = detailed_results[sample_idx]
                predicted_labels = sample_result.get('predicted_labels', [])
                
                if len(predicted_labels) == len(tokens):
                    labels_by_models[model_name] = predicted_labels
                else:
                    print(f"Warning: Label count mismatch for {model_name}, sample {sample_idx}")
                    print(f"  Tokens: {len(tokens)}, Labels: {len(predicted_labels)}")
        
        if len(labels_by_models) >= 2:
            sentence_data.append({
                'tokens': tokens,
                'labels_by_models': labels_by_models,
                'sample_id': sample_idx,
                'original_text': sample.get('text', ''),
                'gold_labels': sample.get('labels', [])
            })
        else:
            print(f"Skipping sample {sample_idx}: insufficient model results ({len(labels_by_models)} models)")
    
    print(f"Successfully converted {len(sentence_data)} samples for disagreement analysis")
    return sentence_data

def save_experiment_metadata(output_dir: Path, metadata: Dict[str, Any], 
                           disagreement_config: Dict[str, Any]) -> None:
    """Save experiment and disagreement analysis metadata"""
    metadata_info = {
        'experiment_metadata': metadata,
        'disagreement_analysis_config': disagreement_config,
        'analysis_timestamp': datetime.now().isoformat(),
        'analysis_version': 'unified_disagreement_analysis_v2.0'
    }
    
    metadata_file = output_dir / 'analysis_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_info, f, indent=2, ensure_ascii=False)
    
    print(f"Saved metadata to: {metadata_file}")

def analyze_coalition_composition(sentence_data: List[Dict[str, Any]], 
                                weights: Dict[str, float]) -> Dict[str, Any]:
    """Analyze coalition composition across all tokens to identify coalition and rest models"""
    model_names = list(weights.keys())
    coalition_membership_counts = defaultdict(int)
    total_tokens = 0
    
    for sent_data in sentence_data:
        tokens = sent_data['tokens']
        labels_by_models = sent_data['labels_by_models']
        
        for t in range(len(tokens)):
            labels_t = [labels_by_models[model][t] for model in model_names]
            W = [weights[model] for model in model_names]
            
            C = coalition_indices(W, 0.5)
            
            for i in C:
                coalition_membership_counts[model_names[i]] += 1
            
            total_tokens += 1
    
    coalition_rates = {model: count / total_tokens for model, count in coalition_membership_counts.items()}
    
    sorted_models = sorted(coalition_rates.items(), key=lambda x: x[1], reverse=True)
    coalition_models = [model for model, rate in sorted_models if rate > 0.5]
    rest_models = [model for model, rate in sorted_models if rate <= 0.5]
    
    return {
        'coalition_rates': coalition_rates,
        'coalition_models': coalition_models,
        'rest_models': rest_models,
        'total_tokens_analyzed': total_tokens
    }

def create_individual_plots(disagreement_results: Dict[str, Any], 
                          coalition_analysis: Dict[str, Any],
                          output_dir: Path) -> None:
    """Create individual visualization plots for disagreement analysis results"""
    token_metrics = disagreement_results['token_metrics']
    hotspots = disagreement_results['hotspots']
    dataset_summary = disagreement_results['dataset_summary']
    bias_df = disagreement_results['bias_analysis']
    
    plots_dir = output_dir / 'individual_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. D_bio distribution
    plt.figure(figsize=(5, 3))
    plt.hist(token_metrics['D_bio'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of D_bio (Label Disagreement)', fontsize=14, fontweight='bold')
    plt.xlabel('D_bio Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(plots_dir / 'D_bio_distribution.png', dpi=600, bbox_inches='tight')
    # plt.savefig(plots_dir / 'D_bio_distribution.pdf', bbox_inches='tight')
    # plt.savefig(plots_dir / 'D_bio_distribution.tiff', dpi=600, bbox_inches='tight')
    # plt.show()
    
    # 2. TV_between vs Elite_internal scatter (skip for gold standard mode)
    if not dataset_summary.get('use_gold_standard', False):
        plt.figure(figsize=(7, 5))

        np.random.seed(42)
        jitter_x = np.random.normal(0, 0.02, len(token_metrics))
        jitter_y = np.random.normal(0, 0.02, len(token_metrics))

        scatter = plt.scatter(token_metrics['TV_between'] + jitter_x, 
                            token_metrics['elite_internal'] + jitter_y,
                            alpha=0.4, c=token_metrics['U_star'], cmap='viridis', s=20)
        plt.colorbar(scatter, label='U_star Score')
        plt.title('Coalition vs Elite Internal Disagreement', fontsize=14, fontweight='bold')
        plt.xlabel('TV_between (Coalition vs Rest)')
        plt.ylabel('Elite_internal (Within Coalition)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        # plt.savefig(plots_dir / 'coalition_vs_elite_scatter.tiff', dpi=600, bbox_inches='tight')
        # plt.savefig(plots_dir / 'coalition_vs_elite_scatter.pdf', bbox_inches='tight')
        # plt.show()

        print(f"Coalition Models: {', '.join(coalition_analysis['coalition_models'])}")
        print(f"Rest Models: {', '.join(coalition_analysis['rest_models'])}")
    else:
        print("Gold Standard mode: Coalition analysis plots skipped")
    
    # 3. Hotspot types distribution
    if hotspots:
        plt.figure(figsize=(5, 4))
        hotspot_types = [h['disagreement_type'] for h in hotspots]
        type_counts = Counter(hotspot_types)
        
        types, counts = zip(*type_counts.most_common())
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple'][:len(types)]
        bars = plt.bar(range(len(types)), counts, color=colors, alpha=0.7, edgecolor='black')
        
        plt.title('Hotspot Types Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Disagreement Type')
        plt.ylabel('Count')
        plt.ylim(0, max(counts) * 1.2)
        plt.xticks(range(len(types)), [t.replace("(", "\n(") for t in types])
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        # plt.savefig(plots_dir / 'hotspot_types_distribution.tiff', dpi=600, bbox_inches='tight')
        # plt.savefig(plots_dir / 'hotspot_types_distribution.pdf', bbox_inches='tight')
        # plt.show()
    
    # 4. U_star distribution
    u_star_threshold = disagreement_results['global_thresholds']['u_star']
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    n, bins, patches = plt.hist(token_metrics['U_star'], bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(u_star_threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Hotspot Threshold (Top 20%): {u_star_threshold:.3f}')
    plt.title('Distribution of U_star Scores', fontsize=12, fontweight='bold')
    plt.xlabel('U_star Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    u_star_values = sorted(token_metrics['U_star'], reverse=True)
    cumulative_pct = np.arange(1, len(u_star_values) + 1) / len(u_star_values) * 100
    
    plt.plot(u_star_values, cumulative_pct, 'b-', linewidth=2, label='Cumulative Distribution')
    plt.axvline(u_star_threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Top 20% Threshold: {u_star_threshold:.3f}')
    plt.axhline(20, color='red', linestyle=':', alpha=0.7, label='20% Line')
    
    plt.title('Cumulative Distribution of U_star Scores', fontsize=12, fontweight='bold')
    plt.xlabel('U_star Score (High to Low)')
    plt.ylabel('Cumulative Percentage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    # plt.savefig(plots_dir / 'U_star_distribution_analysis.tiff', dpi=600, bbox_inches='tight')
    # plt.savefig(plots_dir / 'U_star_distribution_analysis.pdf', bbox_inches='tight')
    # plt.show()

    print(f"Saved individual plots to: {plots_dir}")

def create_disagreement_visualization(disagreement_results: Dict[str, Any], 
                                    coalition_analysis: Dict[str, Any],
                                    output_dir: Path) -> None:
    """Create individual visualization plots for disagreement analysis results"""
    create_individual_plots(disagreement_results, coalition_analysis, output_dir)
    print(f"Saved visualization plots to: {output_dir}")

def analyze_experiment_disagreement(result_file_path: str,
                                  base_output_dir: str = "./disagreement_analysis",
                                  weights: Optional[Dict[str, float]] = None,
                                  save_results: bool = True,
                                  create_visualizations: bool = True,
                                  use_experiment_structure: bool = True,
                                  supervised_by_gold_standard: bool = False,
                                  gold_standard_config: Dict[str, Any] = None,
                                  **disagreement_kwargs) -> Dict[str, Any]:
    """
    Complete pipeline to analyze disagreement from experiment results
    UNIFIED PIPELINE: Always runs disagreement analysis, with optional gold standard mode
    """
    analysis_mode = "Gold Standard Supervision" if supervised_by_gold_standard else "Disagreement Analysis"
    
    print(f"{'='*80}")
    print(f"NER EXPERIMENT UNIFIED ANALYSIS PIPELINE")
    print(f"Primary Mode: {analysis_mode}")
    print(f"{'='*80}")
    print(f"Input file: {result_file_path}")
    
    # Load experiment results
    results = load_experiment_results(result_file_path)
    
    # Extract metadata
    metadata = extract_experiment_metadata(results)
    print(f"Experiment metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Determine output directory - ALWAYS use disagreement_analysis folder
    if use_experiment_structure:
        if 'experiment_results' in results:
            experiment_data = results['experiment_results']
        else:
            experiment_data = results
            
        result_path = Path(result_file_path).resolve()
        if 'experiment_results' in result_path.parts:
            experiment_dir = result_path.parent
            if experiment_dir.name == 'agreement_analysis':
                experiment_dir = experiment_dir.parent
            
            output_dir = experiment_dir / 'disagreement_analysis'
        else:
            output_dir = create_output_directory(base_output_dir, metadata)
    else:
        output_dir = create_output_directory(base_output_dir, metadata)
    
    print(f"Analysis output directory: {output_dir}")
    
    # Convert to disagreement analysis format
    sentence_data = convert_experiment_to_disagreement_format(results)
    
    if not sentence_data:
        raise ValueError("No valid sentence data found for analysis")
    
    # Configure disagreement analysis
    disagreement_config = {
        'hotspot_percentile': disagreement_kwargs.get('hotspot_percentile', 80),
        'coalition_cutoff': disagreement_kwargs.get('coalition_cutoff', 0.5),
        'use_boundary_variant': disagreement_kwargs.get('use_boundary_variant', True),
        'min_block_len': disagreement_kwargs.get('min_block_len', 1),
        'merge_within_gap': disagreement_kwargs.get('merge_within_gap', True),
        'elite_internal_percentile': disagreement_kwargs.get('elite_internal_percentile', 90),
        'tv_between_percentile': disagreement_kwargs.get('tv_between_percentile', 90)
    }
    
    final_results = {}
    
    # RUN UNIFIED DISAGREEMENT ANALYSIS (using MV or Gold as reference)
    print(f"\n{'='*60}")
    if supervised_by_gold_standard:
        print("RUNNING DISAGREEMENT ANALYSIS (Gold Standard Reference)")
    else:
        print("RUNNING DISAGREEMENT ANALYSIS (Majority Vote Reference)")
    print(f"{'='*60}")
    
    disagreement_results = analyze_ner_disagreement_dataset(
        sentence_data=sentence_data,
        weights=weights,
        output_dir=str(output_dir) if save_results else None,
        save_results=save_results,
        use_gold_standard=supervised_by_gold_standard,
        gold_standard_key='gold_labels',
        **disagreement_config
    )
    
    # Analyze coalition composition (skip for gold standard mode)
    if not supervised_by_gold_standard:
        print(f"\nAnalyzing coalition composition...")
        coalition_analysis = analyze_coalition_composition(sentence_data, disagreement_results['weights_used'])
        
        print(f"\n{'='*60}")
        print("COALITION ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Coalition Models (>50% membership): {', '.join(coalition_analysis['coalition_models'])}")
        print(f"Rest Models (≤50% membership): {', '.join(coalition_analysis['rest_models'])}")
        print(f"\nCoalition membership rates:")
        for model, rate in sorted(coalition_analysis['coalition_rates'].items(), 
                                 key=lambda x: x[1], reverse=True):
            print(f"  {model}: {rate:.3f} ({rate*100:.1f}%)")
    else:
        coalition_analysis = {
            'coalition_rates': {model: 0.0 for model in disagreement_results['weights_used'].keys()},
            'coalition_models': [],
            'rest_models': list(disagreement_results['weights_used'].keys()),
            'total_tokens_analyzed': disagreement_results['dataset_summary']['total_tokens']
        }
    
    final_results['disagreement_analysis'] = disagreement_results
    final_results['coalition_analysis'] = coalition_analysis
    
    # GENERATE UNIFIED HOTSPOT DOCUMENTATION (works for both MV and Gold modes)
    print(f"\n{'='*60}")
    if supervised_by_gold_standard:
        print("GENERATING HOTSPOT DOCUMENTATION (Gold Standard Mode)")
    else:
        print("GENERATING HOTSPOT DOCUMENTATION (Majority Vote Mode)")
    print(f"{'='*60}")
    
    # Import and use disagreement documentation generator
    from disagreement_documentation import generate_hotspot_documentation_from_results
    
    doc_file = generate_hotspot_documentation_from_results(
        result_file_path, final_results, str(output_dir)
    )
    
    final_results['hotspot_documentation'] = doc_file
    print(f"Hotspot documentation generated: {doc_file}")
    
    # Save experiment metadata
    if save_results:
        enhanced_config = disagreement_config.copy()
        enhanced_config.update({
            'supervised_by_gold_standard': supervised_by_gold_standard,
            'gold_standard_config': gold_standard_config if supervised_by_gold_standard else None
        })
        save_experiment_metadata(output_dir, metadata, enhanced_config)
    
    # Create visualizations
    if create_visualizations:
        print(f"\n{'='*60}")
        print("CREATING VISUALIZATION PLOTS")
        print(f"{'='*60}")
        create_disagreement_visualization(disagreement_results, coalition_analysis, output_dir)
    
    # Prepare final results
    final_results.update({
        'experiment_metadata': metadata,
        'disagreement_config': disagreement_config,
        'output_directory': str(output_dir),
        'input_file': result_file_path,
        'sentence_data': sentence_data,
        'analysis_type': 'unified',
        'supervised_by_gold_standard': supervised_by_gold_standard,
        'gold_standard_config': gold_standard_config if supervised_by_gold_standard else None
    })
    
    # Save combined results
    if save_results:
        combined_results_file = output_dir / 'combined_disagreement_analysis.pkl'
        with open(combined_results_file, 'wb') as f:
            pickle.dump(final_results, f)
        
        print(f"Saved combined results to: {combined_results_file}")
    
    print(f"\n{'='*80}")
    print("UNIFIED ANALYSIS COMPLETED")
    print(f"{'='*80}")
    print(f"Analysis mode: {analysis_mode}")
    print(f"Output directory: {output_dir}")
    print(f"Hotspot documentation: {doc_file}")
    print(f"Total hotspots found: {disagreement_results['dataset_summary']['total_hotspots']}")
    print(f"Sentences with disagreements: {disagreement_results['dataset_summary']['sentences_with_hotspots']}")
    
    return final_results