import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import openai
import tiktoken

from config_apikey import config
from utils_supervisor import (
    CostEstimate,
    PhaseResult,
    safe_json_serialize,
    extract_json_block,
    VERBOSE,
    load_disagreement_documentation,
    load_model_error_documents,
    load_elite_model_results,
    load_existing_instructions,
    handle_json_parsing_error,
    save_processing_summary,
    validate_instruction_format_4phase
)
from utils_supervisor_from_cache import (
    should_use_phase_cache,
    should_use_per_model_cache,
    save_per_model_result,
)

# =====================
# ENHANCED SUPERVISOR CLASS (4-Phase Structure with Dynamic Parameters)
# =====================
class GPT5AnnotationSupervisor:
    """
    GPT-5 based annotation supervisor for improving NER guidelines - Updated with Dynamic Parameters
    """
    def __init__(self, model_name: str = "gpt-5-2025-08-07",
                 input_cost_per_1k: float = 0.0025,
                 output_cost_per_1k: float = 0.01,
                 prompts_config_path: str = "instruction_supervision_0825.json",
                 skip_final_goal_update: bool = False,
                 # NEW PARAMETERS
                 max_common_instructions: int = 5,
                 max_patterns: int = 10,
                 model_specific_for_all: bool = False,
                 max_model_specific_instructions: int = 3,
                 limit_instruction_changes: bool = False,
                 max_change_ratio: float = 0.2):
        """Initialize the enhanced supervisor with dynamic parameters"""
        
        # Always load API key from config
        self.api_key = config.get_openai_api_key()
        
        # Debug information
        if not self.api_key:
            raise ValueError(
                "No OpenAI API key found. Please provide it via:\n"
                "1. openai_apikey.json file\n"
                "2. OPENAI_API_KEY environment variable"
            )
        
        # API key validation and cleaning
        self.api_key = self.api_key.strip()
        
        if not self.api_key.startswith('sk-'):
            raise ValueError(f"Invalid API key format. Expected format: sk-...")
        
        print(f"✅ API key loaded successfully from config")
        print(f"   Key prefix: {self.api_key[:10]}...")
        print(f"   Key length: {len(self.api_key)}")

        self.model_name = model_name
        self.input_cost_per_1k = input_cost_per_1k
        self.output_cost_per_1k = output_cost_per_1k
        self.prompts_config_path = prompts_config_path
        self.skip_final_goal_update = skip_final_goal_update
        self.client = openai.OpenAI(api_key=self.api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # NEW: Store dynamic parameters
        self.max_common_instructions = max_common_instructions
        self.max_patterns = max_patterns
        self.model_specific_for_all = model_specific_for_all
        self.max_model_specific_instructions = max_model_specific_instructions
        self.limit_instruction_changes = limit_instruction_changes
        self.max_change_ratio = max_change_ratio
        
        # Load prompts configuration
        self.prompts_config = self._load_prompts_config(prompts_config_path)
        
        # Initialize tracking
        self.total_cost = 0.0
        self.phase_results = []
        self.processing_start_time = None
        
        print(f"Initialized Enhanced GPT Annotation Supervisor with 4-Phase Structure")
        print(f"Model: {self.model_name}")
        print(f"Dynamic Parameters:")
        print(f"  - Max common instructions: {self.max_common_instructions}")
        print(f"  - Max patterns: {self.max_patterns}")
        print(f"  - Model-specific for all: {self.model_specific_for_all}")
        print(f"  - Max model-specific instructions: {self.max_model_specific_instructions}")
        print(f"  - Limit instruction changes: {self.limit_instruction_changes}")
        if self.limit_instruction_changes:
            print(f"  - Max change ratio: {self.max_change_ratio:.1%}")
        print(f"Pricing: ${self.input_cost_per_1k:.6f} input, ${self.output_cost_per_1k:.6f} output per 1K tokens")

    def estimate_cost(self, input_text: str, estimated_output_tokens: int = None) -> CostEstimate:
        """Estimate cost for API call with dynamic output estimation"""
        input_tokens = self.estimate_tokens(input_text)
        
        # Dynamic output estimation if not provided
        if estimated_output_tokens is None:
            # Estimate based on input complexity and typical response patterns
            estimated_output_tokens = max(500, min(input_tokens // 5, 32000))  # 5-20% of input, capped at 32k
        
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (estimated_output_tokens / 1000) * self.output_cost_per_1k
        total_cost = input_cost + output_cost
        
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=estimated_output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )

    def make_api_call(self, messages: List[Dict], 
                     phase_name: str = "", output_dir: str = "supervisor_output") -> Tuple[str, CostEstimate]:
        """Make API call with cost tracking and response saving"""
        input_text = "\n".join([msg.get("content", "") for msg in messages])
        estimated_cost = self.estimate_cost(input_text)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Log prompt if verbose
        if VERBOSE >= 2:
            print(f"\n{'='*60}")
            print(f"API CALL - {phase_name}")
            print(f"{'='*60}")
            print("PROMPT:")
            print(input_text)
            print(f"{'='*60}")
            print(f"\nAPI Call Estimate:")
            print(f"  Input tokens: {estimated_cost.input_tokens:,}")
            print(f"  Estimated cost: ${estimated_cost.total_cost:.4f}")
        
        try:
            # Remove max_tokens parameter to allow unlimited output
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                # max_tokens parameter removed
            )
            response_text = response.choices[0].message.content
            
            # Save response to file
            safe_phase_name = phase_name.lower().replace(' ', '_').replace(':', '')
            safe_phase_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', safe_phase_name)
            
            # Save raw response
            response_filename = output_path / f"response_{safe_phase_name}.txt"
            with open(response_filename, 'w', encoding='utf-8') as f:
                f.write(f"Phase: {phase_name}\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"{'='*60}\n")
                f.write("PROMPT:\n")
                f.write(f"{'='*60}\n")
                f.write(input_text)
                f.write(f"\n{'='*60}\n")
                f.write("RESPONSE:\n")
                f.write(f"{'='*60}\n")
                f.write(response_text)
                f.write(f"\n{'='*60}\n")
            
            # Log response if verbose
            if VERBOSE >= 2:
                print("RESPONSE:")
                print(response_text)
                print(f"{'='*60}")
            
            print(f"Response saved to: {response_filename}")
            
            # Update actual cost based on usage
            actual_input_tokens = response.usage.prompt_tokens
            actual_output_tokens = response.usage.completion_tokens
            
            actual_cost = CostEstimate(
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                input_cost=(actual_input_tokens / 1000) * self.input_cost_per_1k,
                output_cost=(actual_output_tokens / 1000) * self.output_cost_per_1k,
                total_cost=((actual_input_tokens / 1000) * self.input_cost_per_1k + 
                        (actual_output_tokens / 1000) * self.output_cost_per_1k)
            )
            
            self.total_cost += actual_cost.total_cost
            
            if VERBOSE >= 2:
                print(f"  Actual cost: ${actual_cost.total_cost:.4f}")
                print(f"  Actual output tokens: {actual_output_tokens:,}")
            
            return response_text, actual_cost
            
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"API CALL FAILED - {phase_name}")
            print(f"{'!'*60}")
            print(f"Error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response}")
            if messages:
                prompt_preview = messages[0].get("content", "")[:500]
                print(f"Prompt preview: {prompt_preview}...")
            print(f"{'!'*60}")
            raise
    
    def _load_prompts_config(self, config_path: str) -> Dict[str, Any]:
        """Load prompts configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Prompts config file not found at {config_path}")
            print("Using fallback inline prompts")
            return self._get_fallback_prompts()
    
    def _get_fallback_prompts(self) -> Dict[str, Any]:
        """Fallback prompts if config file is not found"""
        return {
            "phase_prompts": {
                "phase1_disagreement_pattern_analysis": {
                    "system_role": "You are an expert NER annotation supervisor.",
                    "task_description": ["Analyze disagreement patterns"],
                    "output_format": {"identified_patterns": []}
                }
            }
        }
    
    def _build_prompt(self, phase_name: str, context_data: Dict[str, Any]) -> str:
        """Build prompt from configuration and context data with dynamic parameter injection"""
        from utils_supervisor import PromptBuilder
        
        prompt_config = self.prompts_config["phase_prompts"].get(phase_name, {})
        enhanced_context = context_data.copy()
        enhanced_context.update({
            'max_patterns': self.max_patterns,
            'max_common_instructions': self.max_common_instructions,
            'max_model_specific_instructions': self.max_model_specific_instructions,
            'model_specific_for_all': self.model_specific_for_all,
            'limit_instruction_changes': self.limit_instruction_changes,
            'max_change_ratio': self.max_change_ratio,
        })
        if self.skip_final_goal_update:
            enhanced_context.pop('final_goal', None)
            enhanced_context.pop('updated_final_goal', None)
        
        # Phase-specific dynamic content
        if phase_name == "phase2_non_elite_model_analysis":
            enhanced_context.update({
                'model_selection_description': "all" if self.model_specific_for_all else "non-elite",
                'model_type': context_data.get('model_name', 'unknown'),
            })
        
        if phase_name == "phase4_hierarchical_guideline_organization":
            if self.limit_instruction_changes:
                change_instruction = (
                    f"CRITICAL: Limit the total scope of changes to existing instructions to no more than "
                    f"{self.max_change_ratio:.1%} of the total instruction content (measured at token level). "
                    f"This ensures continuity and prevents excessive disruption to established annotation practices."
                )
            else:
                change_instruction = "Focus on integration quality over change limitation."
            
            enhanced_context['change_limit_instruction'] = change_instruction
        
        if VERBOSE >= 2:
            print(f"\n=== PROMPT BUILDING DEBUG - {phase_name} ===")
            print(f"Dynamic parameters applied:")
            for key, value in enhanced_context.items():
                if key.startswith(('max_', 'model_specific_', 'limit_', 'change_')):
                    print(f"  {key}: {value}")
            print("=" * 50)
        
        return PromptBuilder.build_phase_prompt(prompt_config, enhanced_context)

    def _determine_elite_status(self, model_name: str, elite_model_results: List[str]) -> str:
        """Determine if a model is elite based on elite_model_results"""
        if model_name in elite_model_results:
            return "included_in_mv_coalition"
        else:
            return "excluded_from_mv_coalition"
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return len(text) // 4

    def _execute_phase(self, phase_name: str,
                   context_data: Dict[str, Any],
                   phase_display_name: str,
                   output_dir: str = "supervisor_output") -> Dict[str, Any]:
        """Generic phase execution with unified error handling"""
        # Build prompt with dynamic parameters
        prompt = self._build_prompt(phase_name, context_data)
        messages = [{"role": "user", "content": prompt}]

        # Light cache hook
        phase_key_map = {
            "phase1_disagreement_pattern_analysis": "phase1",
            "phase2_non_elite_model_analysis": "phase2",
            "phase3_instruction_generation_and_decision": "phase3",
            "phase4_hierarchical_guideline_organization": "phase4",
        }
        phase_key = phase_key_map.get(phase_name)
        if phase_key is not None:
            use_cache, cached, _ = should_use_phase_cache(
                output_dir=output_dir,
                phase_key=phase_key,
                prompt=prompt,
                require_prompt_match=False
            )
            if use_cache:
                cost_block = cached.get("cost", {
                    "input_tokens": 0, "output_tokens": 0,
                    "input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0
                })
                try:
                    self.total_cost += float(cost_block.get("total_cost", 0.0))
                except Exception:
                    pass

                return {
                    "result_data": cached.get("result_data"),
                    "cost": cost_block,
                    "raw_response": "[cache-hit]",
                    "prompt_used": cached.get("prompt_used", prompt),
                    "success": True,
                    "error_message": None
                }

        # Normal path (API call + parsing) - simplified, no special Phase 2 handling
        try:
            response, cost = self.make_api_call(messages, phase_name=phase_display_name, output_dir=output_dir)
            
            # Standard processing for all phases
            try:
                cleaned = extract_json_block(response)
                result_data = json.loads(cleaned)
                success = True
                error_msg = None
            except json.JSONDecodeError as e:
                result_data, success, error_msg = handle_json_parsing_error(
                    e, response, phase_display_name, output_dir
                )

        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"{phase_display_name.upper()} EXECUTION ERROR")
            print(f"{'!'*60}")
            print(f"Error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"{'!'*60}")

            result_data = {"execution_error": True, "error_details": str(e)}
            cost = CostEstimate(0, 0, 0, 0, 0)
            response = f"Execution failed: {str(e)}"
            success = False
            error_msg = str(e)

        return {
            "result_data": result_data,
            "cost": cost,
            "raw_response": response,
            "prompt_used": prompt,
            "success": success,
            "error_message": error_msg
        }


    # =====================
    # ENHANCED 4-PHASE METHODS
    # =====================

    def phase1_disagreement_pattern_analysis(self, disagreement_doc: str, ner_scheme: Dict,
                                             final_goal: str, elite_model_results: List[str],
                                             existing_common_instructions: List[Dict] = None,
                                             existing_model_instructions: Dict[str, List[Dict]] = None,
                                             output_dir: str = "supervisor_output") -> Dict[str, Any]:
        """Phase 1: Analyze disagreement patterns with dynamic parameters"""
        context_data = {
            "ner_scheme": ner_scheme,
            "disagreement_doc": disagreement_doc,
            "elite_model_results": elite_model_results,
            "existing_common_instructions": existing_common_instructions or [],
            "existing_model_instructions": existing_model_instructions or {}
        }
        if not self.skip_final_goal_update:
            context_data["final_goal"] = final_goal
        return self._execute_phase("phase1_disagreement_pattern_analysis", context_data,
                                "Phase 1: Disagreement Pattern Analysis", output_dir)
    
    def phase2_non_elite_model_analysis(
        self, phase1_results: Dict,
        model_error_docs: Dict[str, str],
        elite_model_results: List[str],
        existing_model_instructions: Dict[str, List[Dict]] = None,
        output_dir: str = "supervisor_output"
        ) -> Dict[str, Any]:
        """Phase 2: Analyze model-specific issues with simplified output structure"""
        existing_model_instructions = existing_model_instructions or {}
        elite_models = set(elite_model_results)

        per_model_results: Dict[str, Dict[str, Any]] = {}
        per_model_result_data: Dict[str, Any] = {}
        per_model_prompts: Dict[str, str] = {}
        per_model_raw: Dict[str, str] = {}
        failed_messages: list[str] = []

        total_input_tokens = 0
        total_output_tokens = 0
        total_input_cost = 0.0
        total_output_cost = 0.0

        for model_name, error_doc in model_error_docs.items():
            # Enhanced model selection logic
            if not self.model_specific_for_all and model_name in elite_models:
                if VERBOSE >= 1:
                    print(f"  Skipping elite model (model_specific_for_all=False): {model_name}")
                continue
            elif self.model_specific_for_all and model_name in elite_models:
                if VERBOSE >= 1:
                    print(f"  Analyzing elite model (model_specific_for_all=True): {model_name}")
            else:
                if VERBOSE >= 1:
                    print(f"  Analyzing non-elite model: {model_name}")

            context_data = {
                "phase1_results": phase1_results["result_data"],
                "model_name": model_name,
                "error_doc": error_doc,
                "elite_model_results": elite_model_results,
                "existing_model_specific_instructions": existing_model_instructions.get(model_name, [])
            }

            # Build prompt for this model, then try per-model cache
            model_prompt = self._build_prompt("phase2_non_elite_model_analysis", context_data)
            use_cache, cached_model, _ = should_use_per_model_cache(
                output_dir=output_dir,
                model_name=model_name,
                prompt=model_prompt,
                require_prompt_match=False,
                phase_key="phase2",
            )

            if use_cache:
                model_result = cached_model
                if VERBOSE >= 1:
                    print(f"    [cache-hit] phase2_per_model → {model_name}")
            else:
                model_result = self._execute_phase(
                    "phase2_non_elite_model_analysis",
                    context_data,
                    f"Phase 2: {model_name}",
                    output_dir
                )
                # Save per-model cache for future reuse
                save_per_model_result(output_dir=output_dir, model_name=model_name, result=model_result, phase_key="phase2")

            per_model_results[model_name] = model_result

            # Collect for aggregation - simplified structure
            model_data = model_result.get("result_data")
            if model_data and isinstance(model_data, dict):
                # Ensure we have the simplified structure
                simplified_data = {
                    "model_name": model_data.get("model_name", model_name),
                    "elite_or_not": model_name in elite_models,  # Simple boolean
                    "model_specific_patterns": model_data.get("model_specific_patterns", []),
                    "model_bias_summary": model_data.get("model_bias_summary", {}),
                    "instruction_candidate_needs": model_data.get("instruction_candidate_needs", [])
                }
                per_model_result_data[model_name] = simplified_data
            
            per_model_prompts[model_name] = model_result.get("prompt_used")
            per_model_raw[model_name] = model_result.get("raw_response")

            cin, cout, cin_cost, cout_cost = _cost_numbers(model_result.get("cost"))
            total_input_tokens  += cin
            total_output_tokens += cout
            total_input_cost    += cin_cost
            total_output_cost   += cout_cost

            if not model_result.get("success", True):
                em = model_result.get("error_message")
                failed_messages.append(f"{model_name}: {em}" if em else f"{model_name}: unknown error")

        aggregated_cost = CostEstimate(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            input_cost=total_input_cost,
            output_cost=total_output_cost,
            total_cost=total_input_cost + total_output_cost
        )

        all_success = len(failed_messages) == 0
        error_message = None if all_success else "; ".join(failed_messages)

        # Simplified result structure
        unified = {
            "result_data": per_model_result_data,  # Direct model results without wrapper
            "cost": aggregated_cost.to_dict(),
            "raw_response": per_model_raw,
            "prompt_used": per_model_prompts,
            "success": all_success,
            "error_message": error_message
        }

        return unified

    def phase3_instruction_generation_and_decision(
        self, phase1_results: Dict,
        phase2_results: Dict,
        final_goal: str,
        decision_mode: str = "gpt_autonomous",
        human_input: Dict = None,
        existing_common_instructions: List[Dict] = None,
        existing_model_instructions: Dict[str, List[Dict]] = None,
        output_dir: str = "supervisor_output"
        ) -> Dict[str, Any]:
        """Phase 3: Generate instructions with dynamic limits"""
        context_data = {
            "phase1_results": phase1_results["result_data"],
            "phase2_results": phase2_results["result_data"],
            "decision_mode": decision_mode,
            "human_input": human_input or {},
            "existing_common_instructions": existing_common_instructions or [],
            "existing_model_instructions": existing_model_instructions or {},
            "skip_final_goal_update": self.skip_final_goal_update
        }
        if not self.skip_final_goal_update:
            context_data["final_goal"] = final_goal
        
        return self._execute_phase("phase3_instruction_generation_and_decision",
                                context_data,
                                "Phase 3: Instruction Generation", output_dir)


    def phase4_hierarchical_guideline_organization(self, phase3_results: Dict, ner_scheme: Dict,
                                                   updated_final_goal: str = None,
                                                   existing_common_instructions: List[Dict] = None,
                                                   existing_model_instructions: Dict[str, List[Dict]] = None,
                                                   output_dir: str = "supervisor_output") -> Dict[str, Any]:
        """Phase 4: Organize instructions with preservation emphasis"""
        context_data = {
            "phase3_results": phase3_results["result_data"],
            "ner_scheme": ner_scheme,
            "existing_common_instructions": existing_common_instructions or [],
            "existing_model_instructions": existing_model_instructions or {}
        }
        
        # skip_final_goal_update=False일 때만 updated_final_goal 추가
        if not self.skip_final_goal_update and updated_final_goal:
            context_data["final_goal"] = updated_final_goal
        
        return self._execute_phase("phase4_hierarchical_guideline_organization", context_data, 
                                "Phase 4: Hierarchical Organization", output_dir)

    # [Rest of the methods remain the same but with enhanced metadata...]
    def run_complete_analysis(self, disagreement_doc_path: str,
                              error_analysis_dir: str, ner_scheme: Dict,
                              final_goal: str,
                              existing_instructions_path: str = None,
                              decision_mode: str = "gpt_autonomous",
                              human_input: Dict = None,
                              output_dir: str = "supervisor_output") -> Dict[str, Any]:
        """Run complete 4-phase analysis with enhanced parameter tracking"""
        self.processing_start_time = datetime.now()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("ENHANCED GPT-5 ANNOTATION SUPERVISOR PIPELINE (4-PHASE)")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Output directory: {output_path}")
        print(f"Decision mode: {decision_mode}")
        print(f"Skip final goal update: {self.skip_final_goal_update}")
        print(f"Enhanced Parameters:")
        print(f"  - Max common instructions: {self.max_common_instructions}")
        print(f"  - Max patterns: {self.max_patterns}")
        print(f"  - Model-specific for all: {self.model_specific_for_all}")
        print(f"  - Max model-specific instructions: {self.max_model_specific_instructions}")
        print(f"  - Limit instruction changes: {self.limit_instruction_changes}")
        if self.limit_instruction_changes:
            print(f"  - Max change ratio: {self.max_change_ratio:.1%}")
        
        # Load existing instructions if provided
        existing_common_instructions, existing_model_instructions = load_existing_instructions(existing_instructions_path)
        if existing_common_instructions or existing_model_instructions:
            print(f"Loaded existing instructions: {len(existing_common_instructions)} common, {sum(len(v) for v in existing_model_instructions.values())} model-specific")
        
        try:
            # Load input data
            print("\nLoading input data...")
            disagreement_doc = load_disagreement_documentation(disagreement_doc_path)
            model_error_docs = load_model_error_documents(error_analysis_dir)
            elite_model_results = load_elite_model_results(error_analysis_dir)
            
            print(f"Loaded disagreement doc: {len(disagreement_doc)} characters")
            print(f"Loaded {len(model_error_docs)} model error documents")
            print(f"Elite models: {elite_model_results}")
            
            # Initialize phase results dictionary
            phase_results_dict = {}
            all_phases_successful = True
            
            # PHASE 1: Enhanced Disagreement Pattern Analysis
            print(f"\n{'='*50}")
            print("PHASE 1: ENHANCED DISAGREEMENT PATTERN ANALYSIS")
            print(f"{'='*50}")
            print(f"Max patterns to identify: {self.max_patterns}")
            print(f"Max instruction principles: {self.max_common_instructions}")
            phase_start = datetime.now()
            
            current_results = self.phase1_disagreement_pattern_analysis(
                disagreement_doc, ner_scheme, final_goal, elite_model_results,
                existing_common_instructions, existing_model_instructions, output_dir
            )
            phase_results_dict['phase1_results'] = current_results
            
            phase_duration = (datetime.now() - phase_start).total_seconds()
            phase_cost = current_results["cost"]
            phase_cost = _as_cost_estimate(phase_cost)
            success = current_results.get("success", True)
            
            if not success:
                all_phases_successful = False
                print(f"Phase 1 failed: {current_results.get('error_message', 'Unknown error')}")
            
            self.phase_results.append(PhaseResult(
                phase_name="Phase 1: Enhanced Disagreement Pattern Analysis",
                phase_number=1,
                result_data=current_results.get("result_data"),
                cost_estimate=phase_cost,
                processing_time=phase_duration,
                success=success,
                error_message=current_results.get("error_message"),
                prompt_used=current_results.get("prompt_used"),
                raw_output=current_results.get("raw_response")
            ))
            
            with open(output_path / "phase1_disagreement_pattern_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(safe_json_serialize(current_results), f, indent=2, ensure_ascii=False)
            
            # PHASE 2: Enhanced Model Analysis
            print(f"\n{'='*50}")
            print("PHASE 2: ENHANCED MODEL ANALYSIS")
            print(f"{'='*50}")
            print(f"Target models: {'All models' if self.model_specific_for_all else 'Non-elite models only'}")
            print(f"Max model-specific instructions per model: {self.max_model_specific_instructions}")
            phase_start = datetime.now()

            phase1_results = phase_results_dict['phase1_results']
            current_results = self.phase2_non_elite_model_analysis(
                phase1_results, model_error_docs, elite_model_results,
                existing_model_instructions, output_dir
            )
            phase_results_dict['phase2_results'] = current_results
            
            phase_duration = (datetime.now() - phase_start).total_seconds()
            phase_cost = current_results["cost"]
            phase_cost = _as_cost_estimate(phase_cost)
            success = current_results.get("success", True)
            
            if not success:
                all_phases_successful = False
                print(f"Phase 2 failed: {current_results.get('error_message', 'Unknown error')}")
            
            self.phase_results.append(PhaseResult(
                phase_name="Phase 2: Enhanced Model Analysis",
                phase_number=2,
                result_data=current_results.get("result_data"),
                cost_estimate=phase_cost,
                processing_time=phase_duration,
                success=success,
                error_message=current_results.get("error_message"),
                prompt_used=current_results.get("prompt_used"),
                raw_output=current_results.get("raw_response")
            ))
            
            with open(output_path / "phase2_non_elite_model_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(safe_json_serialize(current_results), f, indent=2, ensure_ascii=False)
            
            # PHASE 3: Enhanced Instruction Generation
            print(f"\n{'='*50}")
            print("PHASE 3: ENHANCED INSTRUCTION GENERATION")
            print(f"{'='*50}")
            print(f"Max common instructions: {self.max_common_instructions}")
            print(f"Max model-specific instructions per model: {self.max_model_specific_instructions}")
            if self.skip_final_goal_update:
                print("Final goal updates DISABLED - preserving original goal")
            phase_start = datetime.now()
            
            phase1_results = phase_results_dict['phase1_results']
            phase2_results = phase_results_dict['phase2_results']
            current_results = self.phase3_instruction_generation_and_decision(
                phase1_results, phase2_results, final_goal, decision_mode, human_input,
                existing_common_instructions, existing_model_instructions, output_dir
            )
            phase_results_dict['phase3_results'] = current_results
            
            phase_duration = (datetime.now() - phase_start).total_seconds()
            phase_cost = current_results["cost"]
            phase_cost = _as_cost_estimate(phase_cost)
            success = current_results.get("success", True)
            
            if not success:
                all_phases_successful = False
                print(f"Phase 3 failed: {current_results.get('error_message', 'Unknown error')}")
            
            self.phase_results.append(PhaseResult(
                phase_name="Phase 3: Enhanced Instruction Generation",
                phase_number=3,
                result_data=current_results.get("result_data"),
                cost_estimate=phase_cost,
                processing_time=phase_duration,
                success=success,
                error_message=current_results.get("error_message"),
                prompt_used=current_results.get("prompt_used"),
                raw_output=current_results.get("raw_response")
            ))
            
            with open(output_path / "phase3_instruction_generation_and_decision.json", 'w', encoding='utf-8') as f:
                json.dump(safe_json_serialize(current_results), f, indent=2, ensure_ascii=False)
            
            # PHASE 4: Enhanced Hierarchical Organization
            print(f"\n{'='*50}")
            print("PHASE 4: ENHANCED HIERARCHICAL ORGANIZATION")
            print(f"{'='*50}")
            if self.limit_instruction_changes:
                print(f"Limit instruction changes: {self.max_change_ratio:.1%}")
            phase_start = datetime.now()
            
            phase3_results = phase_results_dict['phase3_results']
            
            # Handle goal update control
            if self.skip_final_goal_update:
                current_results = self.phase4_hierarchical_guideline_organization(
                    phase3_results, ner_scheme, None,
                    existing_common_instructions, existing_model_instructions, output_dir
                )
                effective_final_goal = final_goal
            else:
                updated_final_goal = phase3_results["result_data"].get(
                    "updated_final_goal", {}
                ).get("updated_final_goal_text", final_goal)
                current_results = self.phase4_hierarchical_guideline_organization(
                    phase3_results, ner_scheme, updated_final_goal,
                    existing_common_instructions, existing_model_instructions, output_dir
                )
                effective_final_goal = updated_final_goal
            
            current_results = self.phase4_hierarchical_guideline_organization(
                phase3_results, ner_scheme, updated_final_goal,
                existing_common_instructions, existing_model_instructions, output_dir
            )
            phase_results_dict['phase4_results'] = current_results
            
            phase_duration = (datetime.now() - phase_start).total_seconds()
            phase_cost = current_results["cost"]
            phase_cost = _as_cost_estimate(phase_cost)
            success = current_results.get("success", True)
            
            if not success:
                all_phases_successful = False
                print(f"Phase 4 failed: {current_results.get('error_message', 'Unknown error')}")
            
            self.phase_results.append(PhaseResult(
                phase_name="Phase 4: Enhanced Hierarchical Organization",
                phase_number=4,
                result_data=current_results.get("result_data"),
                cost_estimate=phase_cost,
                processing_time=phase_duration,
                success=success,
                error_message=current_results.get("error_message"),
                prompt_used=current_results.get("prompt_used"),
                raw_output=current_results.get("raw_response")
            ))
            
            with open(output_path / "phase4_hierarchical_guideline_organization.json", 'w', encoding='utf-8') as f:
                json.dump(safe_json_serialize(current_results), f, indent=2, ensure_ascii=False)
            
            # COMPILE ENHANCED FINAL RESULTS
            final_results = self.compile_final_results_4phase_enhanced(
                phase_results_dict['phase1_results'], 
                phase_results_dict['phase2_results'], 
                phase_results_dict['phase3_results'], 
                phase_results_dict['phase4_results'], 
                ner_scheme, 
                final_goal,
                updated_final_goal
            )
            
            # Save comprehensive results
            with open(output_path / "comprehensive_results.json", 'w', encoding='utf-8') as f:
                json.dump(safe_json_serialize(final_results), f, indent=2, ensure_ascii=False)
            
            # Save processing summary
            save_processing_summary(self.phase_results, self.processing_start_time, 
                                  self.total_cost, self.model_name, output_path)
            
            return final_results
            
        except Exception as e:
            print(f"\n{'!'*80}")
            print("CRITICAL ERROR IN ENHANCED ANALYSIS PIPELINE")
            print(f"{'!'*80}")
            print(f"Error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            if hasattr(e, '__traceback__'):
                import traceback
                print("Traceback:")
                traceback.print_exc()
            print(f"Partial results available: {len(self.phase_results)} phases completed")
            print(f"Total cost so far: ${self.total_cost:.4f}")
            print(f"{'!'*80}")
            
            # Save partial results if any phases completed
            if self.phase_results:
                try:
                    partial_results = {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "completed_phases": len(self.phase_results),
                        "phase_results": [result.to_dict() for result in self.phase_results],
                        "total_cost": self.total_cost,
                        "processing_duration": (datetime.now() - self.processing_start_time).total_seconds(),
                        "pipeline_status": "error",
                        "enhanced_parameters": {
                            "max_common_instructions": self.max_common_instructions,
                            "max_patterns": self.max_patterns,
                            "model_specific_for_all": self.model_specific_for_all,
                            "max_model_specific_instructions": self.max_model_specific_instructions,
                            "limit_instruction_changes": self.limit_instruction_changes,
                            "max_change_ratio": self.max_change_ratio,
                        }
                    }
                    
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    with open(output_path / "error_results.json", 'w', encoding='utf-8') as f:
                        json.dump(safe_json_serialize(partial_results), f, indent=2, ensure_ascii=False)
                    
                    print(f"Error results saved to: {output_path / 'error_results.json'}")
                    print("Note: comprehensive_results.json was NOT created due to critical error")
                except Exception as save_error:
                    print(f"Could not save partial results: {str(save_error)}")
            
            return {"error": str(e), "partial_results": [result.to_dict() for result in self.phase_results]}
    
    
    def compile_final_results_4phase_enhanced(self,
                                              phase1: Dict, phase2: Dict, phase3: Dict,
                                              phase4: Dict, original_ner_scheme: Dict,
                                              original_final_goal: str,
                                              updated_final_goal: str) -> Dict[str, Any]:
        """
        Compile all 4-phase results into final output with enhanced metadata
        """
        processing_duration = (datetime.now() - self.processing_start_time).total_seconds()
        # Extract final guidelines from Phase 4 with validation
        phase4_data = phase4["result_data"]
        phase3_data = phase3["result_data"]
        goal_was_updated = not (self.skip_final_goal_update)

        # Build enhanced guidelines from 4-phase structure
        enhanced_guidelines = {
            "final_goal_status": {
                "original_goal": original_final_goal,
                "goal_was_updated": goal_was_updated,
                "goal_update_skipped": self.skip_final_goal_update
            },
            "enhanced_ner_scheme": original_ner_scheme,  # No schema enhancement in 4-phase
            "hierarchical_common_instructions": phase4_data.get("hierarchical_common_instructions", []),
            "prioritized_model_instructions": phase4_data.get("prioritized_model_instructions", {}),
            "instruction_integration_summary": phase4_data.get("instruction_integration_analysis", {})
        }
        
        # Validate instruction format consistency
        validated_guidelines = validate_instruction_format_4phase(enhanced_guidelines)

        return {
            "metadata": {
                "processing_timestamp": datetime.now().isoformat(),
                "processing_duration_seconds": processing_duration,
                "total_cost_usd": self.total_cost,
                "model_used": self.model_name,
                "phase_count": 4,  # Always 4 phases now
                "pipeline_structure": "4_phase_enhanced",
                "goal_update_behavior": {
                    "skip_final_goal_update": self.skip_final_goal_update,
                    "goal_was_modified": goal_was_updated
                },
                # NEW: Enhanced parameter tracking
                "enhanced_parameters": {
                    "max_common_instructions": self.max_common_instructions,
                    "max_patterns": self.max_patterns,
                    "model_specific_for_all": self.model_specific_for_all,
                    "max_model_specific_instructions": self.max_model_specific_instructions,
                    "limit_instruction_changes": self.limit_instruction_changes,
                    "max_change_ratio": self.max_change_ratio,
                },
                "format_version": "4.0"  # NEW enhanced version
            },
            "input_data": {
                "original_ner_scheme": original_ner_scheme,
                "original_final_goal": original_final_goal
            },
            "enhanced_guidelines": validated_guidelines,
            "phase_results": {
                f"phase{i+1}": result.to_dict() for i, result in enumerate(self.phase_results)
            },
            "analysis_summary": {
                "patterns_identified": len(phase1["result_data"].get("identified_patterns", [])),
                "common_instructions_generated": len(validated_guidelines.get("hierarchical_common_instructions", [])),
                "model_specific_instructions": {
                    model: len(instructions) for model, instructions in 
                    validated_guidelines.get("prioritized_model_instructions", {}).items()
                },
                "contradictions_resolved": phase4["result_data"].get("instruction_integration_analysis", {}).get("contradictions_resolved", 0),
                "final_goal_preserved": not goal_was_updated,
                # NEW: Enhanced analysis metrics
                "models_analyzed_count": len(phase2["result_data"].get("model_analyses", {})),
                "all_models_analyzed": phase2["result_data"].get("all_models_analyzed", False),
                "existing_instruction_preservation_rate": phase4["result_data"].get("final_guideline_summary", {}).get("existing_instruction_preservation_rate", "Unknown")
            }
        }

def _cost_numbers(cost) -> tuple[int, int, float, float]:
    """Return (in_tokens, out_tokens, in_cost, out_cost) from dict or CostEstimate."""
    if cost is None:
        return 0, 0, 0.0, 0.0
    if isinstance(cost, dict):
        return (
            int(cost.get("input_tokens", 0)),
            int(cost.get("output_tokens", 0)),
            float(cost.get("input_cost", 0.0)),
            float(cost.get("output_cost", 0.0)),
        )
    return getattr(cost, "input_tokens", 0), getattr(cost, "output_tokens", 0), \
           float(getattr(cost, "input_cost", 0.0)), float(getattr(cost, "output_cost", 0.0))
           
def _as_cost_estimate(cost) -> CostEstimate:
    if cost is None:
        return CostEstimate(0, 0, 0.0, 0.0, 0.0)
    if isinstance(cost, dict):
        return CostEstimate(
            input_tokens=int(cost.get("input_tokens", 0)),
            output_tokens=int(cost.get("output_tokens", 0)),
            input_cost=float(cost.get("input_cost", 0.0)),
            output_cost=float(cost.get("output_cost", 0.0)),
            total_cost=float(cost.get("total_cost", 0.0)),
        )
    return cost

def process_phase2_response(response_text: str, model_name: str) -> Dict[str, Any]:
    """
    Process Phase 2 response, handling both single object and array responses
    """
    try:
        cleaned = extract_json_block(response_text)
        result_data = json.loads(cleaned)
        
        # Handle case where GPT returns an array instead of single object
        if isinstance(result_data, list):
            if len(result_data) == 0:
                raise ValueError("Empty array response from GPT")
            
            # Find the object for this specific model
            target_result = None
            for item in result_data:
                if isinstance(item, dict) and item.get("model_name") == model_name:
                    target_result = item
                    break
            
            if target_result is None:
                # If no exact match, take the first item and update model_name
                target_result = result_data[0]
                if isinstance(target_result, dict):
                    target_result["model_name"] = model_name
                    print(f"Warning: No exact model match found, using first result for {model_name}")
            
            result_data = target_result
        
        # Ensure result_data is a dict
        if not isinstance(result_data, dict):
            raise ValueError(f"Expected dict but got {type(result_data)}")
            
        # Validate required fields
        required_fields = ["model_name", "model_specific_patterns", "model_bias_summary", "instruction_candidate_needs"]
        for field in required_fields:
            if field not in result_data:
                print(f"Warning: Missing required field '{field}' in Phase 2 response for {model_name}")
                result_data[field] = [] if field.endswith(("patterns", "needs")) else {}
        
        return result_data, True, None
        
    except json.JSONDecodeError as e:
        return handle_json_parsing_error(e, response_text, f"Phase 2: {model_name}", "supervisor_output")
    except Exception as e:
        error_msg = f"Phase 2 response processing failed for {model_name}: {str(e)}"
        print(f"Error: {error_msg}")
        result_data = {
            "model_name": model_name,
            "processing_error": True, 
            "error_details": str(e),
            "model_specific_patterns": [],
            "model_bias_summary": {},
            "instruction_candidate_needs": []
        }
        return result_data, False, error_msg
