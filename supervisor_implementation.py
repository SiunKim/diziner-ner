import os
import json
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from base_supervisor import GPT5AnnotationSupervisor
from utils_supervisor import set_verbosity
from utils_supervisor_from_cache import try_load_full_results, comprehensive_path
from config_apikey import config


def construct_output_path(model_name: str, base_output_dir: str = "supervisor_output") -> str:
    """Construct output path for supervisor results within experiment directory"""
    safe_model_name = model_name.replace(':', '_').replace('/', '_').replace('\\', '_')
    output_path = Path(base_output_dir) / safe_model_name
    return str(output_path)

def get_model_pricing(model_name: str) -> Tuple[float, float]:
    """Get input and output pricing for a model"""
    return config.get_model_pricing(model_name)

def run_supervisor_analysis(disagreement_doc_path: str,  # UNIFIED: only one doc path
                            error_analysis_dir: str,
                            analysis_type: str = 'disagreement',  # 'disagreement' or 'gold_standard'
                            ner_scheme: Dict[str, Any] = None, 
                            final_goal: str = "",
                            model_name: str = "gpt-5-2025-08-07", 
                            num_groups: int = 20, 
                            group_size: int = 50, 
                            num_models: int = 8,
                            iteration_number: int = 0, 
                            group_index: int = 0,
                            base_output_dir: str = "supervisor_output",
                            verbose: int = 1,
                            existing_instructions_path: str = None,
                            prompts_config_path: str = "prompts/instruction_supervision_0905.json",
                            skip_final_goal_update: bool = False,
                            use_cache: bool = True,
                            max_common_instructions: int = 5,
                            max_patterns: int = 10,
                            model_specific_for_all: bool = False,
                            max_model_specific_instructions: int = 3,
                            limit_instruction_changes: bool = False,
                            max_change_ratio: float = 0.2) -> Dict[str, Any]:
    """
    Run complete supervisor analysis - SIMPLIFIED: single documentation path for both modes
    """
    set_verbosity(verbose)
    
    if skip_final_goal_update:
        base_path = prompts_config_path.replace('.json', '_wihtoutfinaltaskgoal.json')
        if os.path.exists(base_path):
            effective_prompts_path = base_path
            print(f"Using no-final-goal prompts: {effective_prompts_path}")
        else:
            effective_prompts_path = prompts_config_path
            print(f"No-final-goal prompts not found, using default: {effective_prompts_path}")
    else:
        effective_prompts_path = prompts_config_path
    
    if analysis_type not in ['disagreement', 'gold_standard']:
        raise ValueError(f"analysis_type must be 'disagreement' or 'gold_standard', got: {analysis_type}")
    
    if not disagreement_doc_path:
        raise ValueError("disagreement_doc_path is required")
    
    if not os.path.exists(disagreement_doc_path):
        raise FileNotFoundError(f"Document path not found: {disagreement_doc_path}")
    
    if error_analysis_dir and not os.path.exists(error_analysis_dir):
        print(f"Warning: Error analysis directory not found: {error_analysis_dir}")
        error_analysis_dir = None
    
    # Configure prompts based on analysis type (optional specialized prompts)
    if analysis_type == 'gold_standard':
        gold_standard_prompts_path = prompts_config_path.replace('.json', '_gold_standard.json')
        if os.path.exists(gold_standard_prompts_path):
            effective_prompts_path = gold_standard_prompts_path
            print(f"Using gold standard supervision prompts: {effective_prompts_path}")
        else:
            effective_prompts_path = prompts_config_path
            print(f"!!!!!!!!Gold standard prompts not found!!!!!!!! using default: {effective_prompts_path}")
    else:
        effective_prompts_path = prompts_config_path
    
    if not os.path.exists(effective_prompts_path):
        print(f"Prompts configuration not found at {effective_prompts_path}. Please create it and rerun.")
        return {}
    
    output_dir = construct_output_path(model_name, base_output_dir)
    
    if use_cache:
        cached_final = try_load_full_results(output_dir)
        if cached_final is not None:
            if verbose:
                print(f"[cache-hit] Using final results at {comprehensive_path(output_dir)}")
            return cached_final

    input_cost, output_cost = get_model_pricing(model_name)
    
    supervisor = GPT5AnnotationSupervisor(
        model_name=model_name,
        input_cost_per_1k=input_cost,
        output_cost_per_1k=output_cost,
        prompts_config_path=effective_prompts_path,
        skip_final_goal_update=skip_final_goal_update,
        max_common_instructions=max_common_instructions,
        max_patterns=max_patterns,
        model_specific_for_all=model_specific_for_all,
        max_model_specific_instructions=max_model_specific_instructions,
        limit_instruction_changes=limit_instruction_changes,
        max_change_ratio=max_change_ratio,
    )

    # Run the 4-phase analysis - UNIFIED: same documentation for both modes
    doc_type = "gold standard enhanced disagreement" if analysis_type == 'gold_standard' else "disagreement"
    print(f"Running {doc_type} supervision analysis...")
    results = supervisor.run_complete_analysis(
        disagreement_doc_path=disagreement_doc_path,  # Same documentation path for both modes
        error_analysis_dir=error_analysis_dir,
        ner_scheme=ner_scheme,
        final_goal=final_goal, 
        existing_instructions_path=existing_instructions_path,
        decision_mode="gpt_autonomous",
        human_input=None,
        output_dir=output_dir
    )
    
    # Save metadata with analysis type info
    if results and not results.get('error'):
        output_path = Path(output_dir)
        metadata_file = output_path / 'supervisor_metadata.json'
        
        metadata = {
            'model_name': model_name,
            'analysis_type': analysis_type,
            'document_path': disagreement_doc_path,  # Same path for both modes
            'document_type': doc_type,
            'iteration_number': iteration_number,
            'group_index': group_index,
            'num_groups': num_groups,
            'group_size': group_size,
            'num_models': num_models,
            'pipeline_structure': '4_phase',
            'goal_update_behavior': {
                'skip_final_goal_update': skip_final_goal_update
            },
            'enhanced_parameters': {
                'max_common_instructions': max_common_instructions,
                'max_patterns': max_patterns,
                'model_specific_for_all': model_specific_for_all,
                'max_model_specific_instructions': max_model_specific_instructions,
                'limit_instruction_changes': limit_instruction_changes,
                'max_change_ratio': max_change_ratio,
            },
            'prompts_config_used': effective_prompts_path,
            'timestamp': results.get('metadata', {}).get('processing_timestamp'),
            'comprehensive_results_file': str(output_path / 'comprehensive_results.json'),
            'error_analysis_dir': error_analysis_dir
        }
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"Supervisor metadata saved to: {metadata_file}")
        except Exception as e:
            print(f"Warning: Could not save supervisor metadata: {e}")
    
    return results

def generate_experiment_suffix(max_common_instructions: int = 5,
                              max_patterns: int = 10,
                              model_specific_for_all: bool = False,
                              max_model_specific_instructions: int = 3,
                              limit_instruction_changes: bool = False,
                              max_change_ratio: float = 0.2,
                              drop_worst_annr: bool = False,
                              supervised_by_gold_standard: bool = False,
                              skip_final_goal_update: bool = False,
                              llm_family_config: Optional[str] = None) -> str:
    """Generate experiment folder suffix based on non-default parameters"""
    suffix_parts = []
    
    if supervised_by_gold_standard:
        suffix_parts.append("goldstd")
        
    if skip_final_goal_update:
        suffix_parts.append("skipgoal")
    
    if max_common_instructions != 5:
        suffix_parts.append(f"cin{max_common_instructions}")
    
    if max_patterns != 10:
        suffix_parts.append(f"mxpt{max_patterns}")
    
    if model_specific_for_all:
        suffix_parts.append("msifa")
    
    if max_model_specific_instructions != 3:
        suffix_parts.append(f"mxmsi{max_model_specific_instructions}")
    
    if limit_instruction_changes:
        ratio_str = str(int(max_change_ratio * 100))
        suffix_parts.append(f"lic{ratio_str}")
    
    if drop_worst_annr:
        suffix_parts.append("dwa")
    
    if llm_family_config:
        suffix_parts.append(f"fam{llm_family_config}")

    if suffix_parts:
        return "_" + "_".join(suffix_parts)
    return ""
