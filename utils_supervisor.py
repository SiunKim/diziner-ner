import os
import re
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
# import openai
# import tiktoken

# =====================
# GLOBAL CONFIGURATION
# =====================
# Verbosity Configuration
VERBOSE = 1  # 0: minimal, 1: normal, 2: detailed with prompts and outputs

# =====================
# DATA CLASSES
# =====================

@dataclass
class CostEstimate:
    """Cost estimation for API calls"""
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serializration"""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost
        }

@dataclass
class PhaseResult:
    """Result from a single phase"""
    phase_name: str
    phase_number: int
    result_data: Dict[str, Any]
    cost_estimate: CostEstimate
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    prompt_used: Optional[str] = None
    raw_output: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "phase_number": self.phase_number,
            "result_data": self.result_data,
            "cost_estimate": self.cost_estimate if isinstance(self.cost_estimate, dict) else self.cost_estimate.to_dict(),
            "processing_time": self.processing_time,
            "success": self.success,
            "error_message": self.error_message,
            "prompt_used": self.prompt_used,
            "raw_output": self.raw_output
        }


# =====================
# PROMPT BUILDER CLASS
# =====================

class PromptBuilder:
    """Utility class for building prompts from configuration and context"""
    
    @staticmethod
    def build_phase_prompt(prompt_config: Dict[str, Any], context_data: Dict[str, Any]) -> str:
        """Build prompt from configuration and context data with dynamic parameter substitution"""
        system_role = prompt_config.get("system_role", "You are an expert NER annotation supervisor.")
        task_description = prompt_config.get("task_description", [])
        output_format = prompt_config.get("output_format", {})
        
        # Extract dynamic parameters from context_data
        dynamic_params = {
            'max_patterns': context_data.get('max_patterns', 10),
            'max_common_instructions': context_data.get('max_common_instructions', 5),
            'max_model_specific_instructions': context_data.get('max_model_specific_instructions', 3),
            'model_specific_for_all': context_data.get('model_specific_for_all', False),
            'limit_instruction_changes': context_data.get('limit_instruction_changes', False),
            'max_change_ratio': context_data.get('max_change_ratio', 0.2),
            'model_name': context_data.get('model_name', 'unknown'),
            'model_type': context_data.get('model_type', 'unknown'),
            'model_selection_description': context_data.get('model_selection_description', 'non-elite'),
            'change_limit_instruction': context_data.get('change_limit_instruction', '')
        }
        
        # Apply dynamic parameter substitution to all text fields
        system_role = PromptBuilder._substitute_dynamic_params(system_role, dynamic_params)
        task_description = [PromptBuilder._substitute_dynamic_params(task, dynamic_params) 
                        for task in task_description]
        output_format = PromptBuilder._substitute_dynamic_params_in_dict(output_format, dynamic_params)
        
        # Build the prompt (기존 로직 유지)
        prompt_parts = [system_role, ""]
        prompt_parts.extend(PromptBuilder._build_context_sections(context_data))
        prompt_parts.extend([
            "## TASK:",
            *[f"{i+1}. {task}" for i, task in enumerate(task_description)],
            ""
        ])
        prompt_parts.extend([
            "## OUTPUT FORMAT (JSON):",
            json.dumps(output_format, indent=2)
        ])
        
        return "\n".join(prompt_parts)

    @staticmethod
    def _build_context_sections(context_data: Dict[str, Any]) -> List[str]:
        """Build context sections based on available data"""
        sections = []
        
        # Define context sections with their keys and titles
        context_sections = [
            ("ner_scheme", "CURRENT NER SCHEME", True),  # JSON format
            ("final_goal", "FINAL TASK GOAL", False),    # Text format
            ("existing_common_instructions", "EXISTING COMMON INSTRUCTIONS (from previous iteration)", True),
            ("existing_model_instructions", "EXISTING MODEL-SPECIFIC INSTRUCTIONS (from previous iteration)", True),
            ("disagreement_doc", "DISAGREEMENT ANALYSIS", False),
            ("elite_model_results", "ELITE MODEL IDENTIFICATION", True),
            ("phase1_results", "PHASE 1 RESULTS (Disagreement Patterns)", True),
            ("phase2_results", "PHASE 2 RESULTS (Non-Elite Model Patterns)", True),
            ("phase3_results", "PHASE 3 RESULTS (Generated Instructions)", True),
            ("decision_mode", "DECISION MODE", False),
            ("human_input", "HUMAN INPUT (for conflict resolution)", True)
        ]
        
        for key, title, is_json in context_sections:
            if key in context_data and context_data[key]:
                if key == "final_goal":
                    if key in context_data and context_data[key] and context_data[key].strip():
                        sections.extend([
                            f"## {title}:",
                            str(context_data[key]),
                            ""
                        ])
                else:
                    if key in context_data and context_data[key]:
                        sections.extend([
                            f"## {title}:",
                            json.dumps(safe_json_serialize(context_data[key]), indent=2) if is_json else str(context_data[key]),
                            ""
                        ])
        
        # Special handling for model-specific sections
        if "existing_model_specific_instructions" in context_data and context_data["existing_model_specific_instructions"]:
                sections.extend([
                    f"## EXISTING MODEL-SPECIFIC INSTRUCTIONS FOR {context_data['model_name']} (from previous iteration):",
                    json.dumps(safe_json_serialize(context_data["existing_model_specific_instructions"]), indent=2),
                    ""
                ])
        if "model_name" in context_data and "error_doc" in context_data:
            sections.extend([
                f"## MODEL ERROR ANALYSIS FOR {context_data['model_name']}:",
                context_data["error_doc"],
                ""
            ])
        
        return sections
    
    @staticmethod
    def _substitute_dynamic_params(text: str, params: Dict[str, Any]) -> str:
        """Substitute dynamic parameters in text"""
        if not isinstance(text, str):
            return text
            
        result = text
        for key, value in params.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        
        return result

    @staticmethod
    def _substitute_dynamic_params_in_dict(obj: Any, params: Dict[str, Any]) -> Any:
        """Recursively substitute dynamic parameters in dictionary/list structures"""
        if isinstance(obj, dict):
            return {k: PromptBuilder._substitute_dynamic_params_in_dict(v, params) 
                for k, v in obj.items()}
        elif isinstance(obj, list):
            return [PromptBuilder._substitute_dynamic_params_in_dict(item, params) 
                for item in obj]
        elif isinstance(obj, str):
            return PromptBuilder._substitute_dynamic_params(obj, params)
        else:
            return obj

def load_disagreement_documentation(doc_path: str) -> str:
    """Load disagreement documentation"""
    with open(doc_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_model_error_documents(error_analysis_dir: str) -> Dict[str, str]:
    """Load model-specific error analysis documents"""
    error_docs = {}
    error_dir = Path(error_analysis_dir)
    
    for doc_file in error_dir.glob("model_error_analysis_*.md"):
        model_name = doc_file.stem.replace("model_error_analysis_", "").replace("_", ":")
        with open(doc_file, 'r', encoding='utf-8') as f:
            error_docs[model_name] = f.read()
    
    return error_docs

def load_elite_model_results(error_analysis_dir: str) -> List[str]:
    """Load elite model identification results from disagreement analysis"""
    # Look for combined disagreement analysis pickle file
    disagreement_analysis_dir = error_analysis_dir.replace('error_analysis',
                                                           'disagreement_analysis')
    combined_file = Path(disagreement_analysis_dir) / "combined_disagreement_analysis.pkl"
    print(f"combined_file for load_elite_model_results: {combined_file}")
    
    if combined_file.exists():
        try:
            with open(combined_file, 'rb') as f:
                results = pickle.load(f)
            
            # Extract coalition models (elite models)
            coalition_models = results.get("coalition_analysis", {}).get("coalition_models", [])
            
            print(f"Loaded coalition analysis from pickle file:")
            print(f"  Elite models (coalition): {coalition_models}")
            
            return coalition_models
            
        except Exception as e:
            print(f"Warning: Could not load combined disagreement analysis from {combined_file}: {e}")
    else:
        print(f"Warning: Combined disagreement analysis file not found: {combined_file}")
    
    # Return empty list if no data found
    print("Returning empty elite model results - no coalition models identified")
    return []

def load_existing_instructions(instruction_file_path: str = None) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Load existing instructions from previous iteration
    
    Args:
        instruction_file_path: Path to previous iteration results
        
    Returns:
        Tuple of (common_instructions, model_specific_instructions)
    """
    if not instruction_file_path or not os.path.exists(instruction_file_path):
        return [], {}
    
    try:
        with open(instruction_file_path, 'r', encoding='utf-8') as f:
            previous_results = json.load(f)
        
        # Extract instructions from comprehensive results (4-phase structure)
        enhanced_guidelines = previous_results.get("enhanced_guidelines", {})
        
        # Try hierarchical structure first (Phase 4 output), fallback to Phase 3 structure
        common_instructions = enhanced_guidelines.get("hierarchical_common_instructions", [])
        if not common_instructions:
            common_instructions = enhanced_guidelines.get("finalized_common_instructions", [])
        
        model_instructions = enhanced_guidelines.get("prioritized_model_instructions", {})
        if not model_instructions:
            model_instructions = enhanced_guidelines.get("finalized_model_instructions", {})
        
        return common_instructions, model_instructions
        
    except Exception as e:
        print(f"Warning: Could not load existing instructions from {instruction_file_path}: {e}")
        return [], {}

def handle_json_parsing_error(e: Exception, response: str, phase_name: str, 
                             output_dir: str = "supervisor_output") -> Tuple[Dict[str, Any], bool, str]:
    """Centralized JSON parsing error handling with full output"""
    print(f"\n{'!'*60}")
    print(f"JSON PARSING ERROR - {phase_name}")
    print(f"{'!'*60}")
    print(f"JSON Error: {str(e)}")
    print(f"Full Response Length: {len(response)} characters")
    print(f"{'='*60}")
    print("FULL RESPONSE:")
    print(f"{'='*60}")
    print(response)  # Full response output
    print(f"{'='*60}")
    print("END OF RESPONSE")
    print(f"{'!'*60}")
    
    # Save to file
    error_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_phase_name = phase_name.lower().replace(' ', '_')
    safe_phase_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', safe_phase_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    error_filename = output_path / f"json_error_{safe_phase_name}_{error_timestamp}.txt"
    with open(error_filename, 'w', encoding='utf-8') as f:
        f.write(f"JSON Parsing Error Details\n")
        f.write(f"Phase: {phase_name}\n")
        f.write(f"Error: {str(e)}\n")
        f.write(f"Response Length: {len(response)}\n")
        f.write(f"{'='*60}\n")
        f.write("FULL RESPONSE:\n")
        f.write(f"{'='*60}\n")
        f.write(response)
        f.write(f"\n{'='*60}\n")
    print(f"Full error details saved to: {error_filename}")
    
    # Try to extract JSON from markdown code blocks
    if "```json" in response:
        try:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            if json_end != -1:
                json_content = response[json_start:json_end].strip()
                result_data = json.loads(json_content)
                print("Successfully extracted JSON from markdown code block!")
                return result_data, True, None
        except Exception as extract_error:
            print(f"Failed to extract JSON from code block: {extract_error}")
    
    result_data = {"raw_response": response, "parsing_error": True, "error_details": str(e)}
    error_msg = f"JSON parsing failed: {str(e)}"
    return result_data, False, error_msg

def validate_instruction_format_4phase(guidelines: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and standardize instruction format for 4-phase structure consistency
    """
    def validate_hierarchical_instruction(instruction):
        """Ensure each hierarchical instruction has required fields"""
        if isinstance(instruction, str):
            return {
                "level": "1",
                "instruction_number": "1",
                "instruction_text": instruction,
                "priority": "medium",
                "source": "auto-converted",
                "sub_instructions": []
            }
        elif isinstance(instruction, dict):
            standardized = {
                "level": instruction.get("level", "1"),
                "instruction_number": instruction.get("instruction_number", "1"),
                "instruction_text": instruction.get("instruction_text", str(instruction)),
                "priority": instruction.get("priority", "medium"),
                "source": instruction.get("source", "unknown"),
                "examples": instruction.get("examples", []),
                "sub_instructions": instruction.get("sub_instructions", [])
            }
            return standardized
        else:
            return {
                "level": "1",
                "instruction_number": "1", 
                "instruction_text": str(instruction),
                "priority": "medium",
                "source": "auto-converted",
                "sub_instructions": []
            }
    
    def validate_prioritized_instruction(instruction):
        """Ensure each prioritized model instruction has required fields"""
        if isinstance(instruction, str):
            return {
                "priority_rank": 1,
                "instruction_text": instruction,
                "priority": "medium",
                "source": "auto-converted",
                "examples": []
            }
        elif isinstance(instruction, dict):
            standardized = {
                "priority_rank": instruction.get("priority_rank", 1),
                "instruction_id": instruction.get("instruction_id", ""),
                "instruction_text": instruction.get("instruction_text", str(instruction)),
                "priority": instruction.get("priority", "medium"),
                "source": instruction.get("source", "unknown"),
                "examples": instruction.get("examples", []),
                "addresses_key_weaknesses": instruction.get("addresses_key_weaknesses", [])
            }
            return standardized
        else:
            return {
                "priority_rank": 1,
                "instruction_text": str(instruction),
                "priority": "medium",
                "source": "auto-converted",
                "examples": []
            }
    
    # Validate hierarchical common instructions
    hierarchical_instructions = guidelines.get("hierarchical_common_instructions", [])
    validated_hierarchical = [validate_hierarchical_instruction(instr) for instr in hierarchical_instructions]
    
    # Validate prioritized model instructions
    prioritized_instructions = guidelines.get("prioritized_model_instructions", {})
    validated_prioritized = {}
    for model_name, instructions in prioritized_instructions.items():
        if isinstance(instructions, list):
            validated_prioritized[model_name] = [validate_prioritized_instruction(instr) for instr in instructions]
        else:
            validated_prioritized[model_name] = []
    
    # Return validated guidelines
    validated_guidelines = guidelines.copy()
    validated_guidelines["hierarchical_common_instructions"] = validated_hierarchical
    validated_guidelines["prioritized_model_instructions"] = validated_prioritized
    
    if VERBOSE >= 1:
        print(f"4-Phase instruction format validation completed:")
        print(f"  Hierarchical common instructions: {len(validated_hierarchical)}")
        print(f"  Prioritized model instructions: {sum(len(v) for v in validated_prioritized.values())}")
    
    return validated_guidelines

def save_processing_summary(phase_results: List[PhaseResult], processing_start_time: datetime, 
                           total_cost: float, model_name: str, output_path: Path):
    """Save processing summary with costs and performance (4-phase structure)"""
    
    processing_duration = (datetime.now() - processing_start_time).total_seconds()
    
    summary = {
        "processing_summary": {
            "pipeline_structure": "4_phase",
            "start_time": processing_start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_duration_seconds": processing_duration,
            "total_cost_usd": total_cost,
            "model_used": model_name
        },
        "phase_breakdown": [result.to_dict() for result in phase_results],
        "cost_breakdown": {
            "total_input_tokens": sum(r.cost_estimate.input_tokens for r in phase_results),
            "total_output_tokens": sum(r.cost_estimate.output_tokens for r in phase_results),
            "total_input_cost": sum(r.cost_estimate.input_cost for r in phase_results),
            "total_output_cost": sum(r.cost_estimate.output_cost for r in phase_results),
            "total_cost": total_cost
        }
    }
    
    with open(output_path / "processing_summary.json", 'w', encoding='utf-8') as f:
        json.dump(safe_json_serialize(summary), f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("4-PHASE PROCESSING COMPLETED")
    print(f"{'='*80}")
    print(f"Total duration: {processing_duration:.1f} seconds")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Average cost per phase: ${total_cost/len(phase_results):.4f}")
    print(f"Results saved to: {output_path}")
    
    # Print detailed summary if verbose
    if VERBOSE >= 1:
        print(f"\nPhase Summary:")
        for result in phase_results:
            status = "✅" if result.success else "❌"
            print(f"  {status} Phase {result.phase_number}: {result.phase_name}")
            print(f"    Cost: ${result.cost_estimate.total_cost:.4f}, Time: {result.processing_time:.1f}s")
            if not result.success and result.error_message:
                print(f"    Error: {result.error_message}")

# =====================
# UTILITY FUNCTIONS
# =====================

def set_verbosity(level: int):
    """Set global verbosity level"""
    global VERBOSE
    VERBOSE = level
    print(f"Verbosity set to level {level}")

def safe_json_serialize(obj: Any) -> Any:
    """Safely serialize objects to JSON-compatible format"""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return {k: safe_json_serialize(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Fallback for non-serializable objects
        return str(obj)

def _strip_trailing_commas(s: str) -> str:
    """
    Remove trailing commas before } or ] more comprehensively
    Handles multi-line cases and nested structures
    """
    result = re.sub(r',\s*\n?\s*([}\]])', r'\1', s)
    result = re.sub(r',(\s*\n\s*)+([}\]])', r'\1\2', result)

    return result

def _convert_single_quoted_strings(s: str) -> str:
    # Convert only top-level single-quoted string tokens to JSON-compliant double-quoted strings.
    OUT, IN_DQ, IN_SQ = 0, 1, 2
    state = OUT
    out = []
    buf = []
    i = 0
    while i < len(s):
        ch = s[i]
        if state == OUT:
            if ch == '"':
                state = IN_DQ
                out.append(ch)
            elif ch == "'":
                state = IN_SQ
                buf = []
            else:
                out.append(ch)
        elif state == IN_DQ:
            if ch == '\\':
                out.append(ch)
                if i + 1 < len(s):
                    out.append(s[i+1])
                    i += 1
            elif ch == '"':
                state = OUT
                out.append(ch)
            else:
                out.append(ch)
        else:  # IN_SQ
            if ch == '\\':
                if i + 1 < len(s):
                    buf.append(s[i+1])
                    i += 1
                else:
                    buf.append('\\')
            elif ch == "'":
                # Close single-quoted token and escape " and \
                content = ''.join(buf)
                content = content.replace('\\', '\\\\').replace('"', '\\"')
                out.append(f'"{content}"')
                state = OUT
            else:
                buf.append(ch)
        i += 1
    if state == IN_SQ:
        content = ''.join(buf)
        content = content.replace('\\', '\\\\').replace('"', '\\"')
        out.append(f'"{content}"')
    return ''.join(out)

def _remove_unescaped_inner_quotes_in_value_strings(s: str) -> str:
    """
    Remove raw double quotes inside double-quoted VALUE strings only.
    Keys (strings immediately followed by :) are left untouched.
    """
    OUT, IN_STR = 0, 1
    state = OUT
    out = []
    i = 0
    start_idx = None  # start index of current string in out-buffer

    def next_nonspace_char(idx: int) -> str | None:
        while idx < len(s) and s[idx].isspace():
            idx += 1
        return s[idx] if idx < len(s) else None

    while i < len(s):
        ch = s[i]
        if state == OUT:
            if ch == '"':
                # entering a string; remember where it starts in output
                state = IN_STR
                start_idx = len(out)
                out.append(ch)
                i += 1
                continue
            out.append(ch)
            i += 1
        else:  # IN_STR
            if ch == '\\':
                # keep escape and next char as-is
                out.append(ch)
                if i + 1 < len(s):
                    out.append(s[i+1])
                    i += 2
                else:
                    i += 1
            elif ch == '"':
                # Tentatively treat as a closing quote
                # Look ahead to decide whether this string is a KEY (:) or a VALUE
                j = i + 1
                nn = next_nonspace_char(j)
                out.append('"')  # place the quote first; may remain as closer
                state = OUT
                i += 1

                if nn == ':':
                    # This was a KEY -> do nothing further
                    continue
                else:
                    # This is a VALUE string. But we may have seen an INNER quote earlier.
                    # We need to check whether this quote is a legitimate closer or an inner quote.
                    # If the next non-space char (nn) is one of ,}] or None, it's a real closer.
                    if nn in {',', '}', ']', None}:
                        # Legitimate closing quote for value; finalize
                        continue
                    else:
                        # This quote should be considered an INNER quote.
                        # Remove it we just appended, and stay inside the string.
                        out.pop()  # remove the '"' we appended
                        state = IN_STR
                        # Do not advance i further (already advanced), just continue
                        continue
            else:
                out.append(ch)
                i += 1

    return ''.join(out)

_CODEBLOCK_RE = re.compile(r"```(?:json|JSON)?(.*?)```", re.DOTALL)

def _extract_codeblock(text: str) -> str | None:
    m = _CODEBLOCK_RE.search(text)
    return m.group(1).strip() if m else None

def _find_first_json_like_segment(text: str) -> str | None:
    # Scan for first balanced {...} or [...] considering quotes and escapes.
    opens = ['{', '[']
    closes = {'{': '}', '[': ']'}
    start_idx = None
    start_ch = None
    # Find the first opening brace/bracket
    for idx, ch in enumerate(text):
        if ch in opens:
            start_idx = idx
            start_ch = ch
            break
    if start_idx is None:
        return None

    stack = [start_ch]
    IN_DQ = False
    escape = False
    for i in range(start_idx + 1, len(text)):
        ch = text[i]
        if IN_DQ:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                IN_DQ = False
        else:
            if ch == '"':
                IN_DQ = True
            elif ch in ['{', '[']:
                stack.append(ch)
            elif ch in ['}', ']']:
                if not stack:
                    break
                top = stack.pop()
                if closes[top] != ch:
                    # Mismatched bracket; abort segment search
                    return None
                if not stack:
                    # Balanced segment found
                    return text[start_idx:i+1].strip()
    # If unbalanced, return until end as a last-ditch candidate
    return text[start_idx:].strip()

def extract_and_loads_json_block(text: str):
    """
    Extract JSON from model output and return a parsed Python object.
    Strategy (improved order):
      1) Try raw text with json.loads
      2) Try trailing comma removal first (prioritized)
      3) Try fenced code block ```json ... ```
      4) Try first balanced JSON-like segment
      5) Apply single-quote conversion as fallback
      6) Apply combined sanitizers as final fallback
    Raises:
      ValueError with clear message if parsing ultimately fails.
    """
    # 1) Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) Try trailing comma removal first (prioritized for common case)
    try:
        fixed = _strip_trailing_commas(text.strip())
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 3) Code block extraction
    cb = _extract_codeblock(text)
    candidates = []
    if cb:
        candidates.append(cb)

    # 4) Balanced segment extraction
    seg = _find_first_json_like_segment(text)
    if seg and (not cb or seg != cb):
        candidates.append(seg)

    # 5) Try candidates as-is, then with sanitizers
    if not candidates:
        candidates = [text.strip()]

    last_err = None
    for cand in candidates:
        # As-is
        try:
            return json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = e
        
        # Trailing comma removal first
        try:
            fixed = _strip_trailing_commas(cand)
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            last_err = e
            
        # Single-quote conversion
        try:
            fixed = _convert_single_quoted_strings(cand)
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            last_err = e
            
        # Combined sanitization (trailing comma + quote conversion)
        try:
            fixed = _strip_trailing_commas(cand)
            fixed = _convert_single_quoted_strings(fixed)
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            last_err = e
            
        # Full sanitization pipeline (most comprehensive)
        try:
            fixed = _strip_trailing_commas(cand)
            fixed = _convert_single_quoted_strings(fixed)
            fixed = _remove_unescaped_inner_quotes_in_value_strings(fixed)
            fixed = _strip_trailing_commas(fixed)  # Final cleanup
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            last_err = e

    msg = "Failed to parse JSON after extraction and sanitization"
    if last_err is not None:
        msg += f": {last_err.msg} at pos {last_err.pos}"
    raise ValueError(msg)

def extract_json_block(text: str) -> str:
    """Extract clean JSON from model output by removing extra text/markdown."""
    # First try to find JSON code block
    codeblock_match = re.search(r"```(?:json|JSON)?(.*?)```", text, re.DOTALL)
    if codeblock_match:
        candidate = codeblock_match.group(1).strip()
        return candidate

    # Try to find JSON object or array
    json_patterns = [
        r"\{.*\}",  # Object
        r"\[.*\]"   # Array
    ]

    for pattern in json_patterns:
        json_match = re.search(pattern, text, re.DOTALL)
        if json_match:
            candidate = json_match.group(0).strip()
            return candidate

    # Clean up template variables in the whole text as fallback
    return text.strip()


def validate_supervisor_results(results: Dict[str, Any]) -> bool:
    """
    Validate supervisor results - UPDATED for 4-phase only structure
    """
    try:
        if not isinstance(results, dict):
            return False

        # Check for enhanced_guidelines (primary format)
        enhanced_guidelines = results.get("enhanced_guidelines")
        if enhanced_guidelines:
            # Check for 4-phase structure
            has_hierarchical = enhanced_guidelines.get("hierarchical_common_instructions", [])
            has_prioritized = enhanced_guidelines.get("prioritized_model_instructions", {})

            if has_hierarchical or has_prioritized:
                # Check metadata for completion markers
                metadata = results.get("metadata", {})
                phase_count = metadata.get("phase_count", 0)
                pipeline_structure = metadata.get("pipeline_structure", "")

                # Only accept 4-phase structure
                if phase_count == 4 and pipeline_structure == "4_phase":
                    return True

                # Backward compatibility: accept 5-phase if format_version < 3.0
                format_version = metadata.get("format_version", "1.0")
                if phase_count >= 4 and float(format_version.split('.')[0]) < 3:
                    return True

        # Check alternative format (iterative results)
        final_guidelines = results.get("final_guidelines")
        if final_guidelines:
            has_hierarchical = final_guidelines.get("hierarchical_common_instructions", [])
            has_prioritized = final_guidelines.get("prioritized_model_instructions", {})

            if has_hierarchical or has_prioritized:
                return True

        # Check for error indicators
        if results.get("error") or results.get("partial_results"):
            return False

        return False

    except Exception as e:
        print(f"Error validating supervisor results: {e}")
        return False
