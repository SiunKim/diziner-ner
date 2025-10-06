"""
NER Agent class for Named Entity Recognition tasks - MODIFIED for Supervisor Integration
"""
import json
import re
import os
from typing import Dict, List, Tuple, Any, Optional

from base_agent import BaseAgent
import llm_clients
from llm_clients import GLOBAL_MAX_TOKENS
from utils_annotator import (
    get_default_critical_instructions,
    get_default_analysis_instructions
    )
from debug import DEBUG

# Global inference parameters
GLOBAL_TEMPERATURE = 0.0
GLOBAL_TOP_P = 1.0
GLOBAL_REPEAT_PENALTY = 1.0
GLOBAL_FREQUENCY_PENALTY = 0.0
GLOBAL_PRESENCE_PENALTY = 0.0

class NERAgent(BaseAgent):
    """NER annotation agent using various LLM models with supervisor instructions support"""
    def __init__(self, model_name: str, ner_scheme: Dict[str, Any],
                model_source_map: Optional[Dict[str, str]] = None,
                final_task_goal: Optional[str] = None,
                critical_instructions: Optional[List[str]] = None,
                analysis_instructions: Optional[List[str]] = None,
                supervisor_common_instructions: Optional[str] = None,
                supervisor_model_instructions: Optional[str] = None,
                iteration_number: int = 0,
                api_key: Optional[str] = None,
                ollama_base_url: str = "http://localhost:11434",
                llm_infer_by_openrouter: bool = False,
                skip_final_goal_update: bool = False,
                verbose: int = 0):

        """
        Initialize NER Agent with supervisor instructions and task goal
        
        Args:
            model_name: Name of the model
            ner_scheme: Dictionary with entity types and definitions
            final_task_goal: Optional description of the final task goal and context
            critical_instructions: Optional list of critical instructions for NER task
            analysis_instructions: Optional list of instructions for confusing case analysis
            supervisor_common_instructions: Common instructions from supervisor (iteration > 0)
            supervisor_model_instructions: Model-specific instructions from supervisor (iteration > 0)
            iteration_number: Current iteration number (0 for baseline)
            api_key: API key for commercial models
            ollama_base_url: Base URL for Ollama server
            llm_infer_by_openrouter: Whether to use OpenRouter for inference
            verbose: Verbosity level (0: minimal, 1: progress, 2: detailed)
        """
        # Determine HF model path
        hf_model_path = None
        if model_source_map and model_name in model_source_map:
            source = model_source_map[model_name]
            if source != "ollama":
                hf_model_path = source

       # Initialize base agent with HF path
        super().__init__(
            model_name, 
            api_key, 
            ollama_base_url, 
            hf_model_path,
            use_openrouter=llm_infer_by_openrouter,
            verbose=verbose
        )
        
        # NER-specific attributes
        self.ner_scheme = ner_scheme
        self.final_task_goal = final_task_goal
        self.critical_instructions = critical_instructions or \
            get_default_critical_instructions()
        self.analysis_instructions = analysis_instructions or \
            get_default_analysis_instructions()

        # Store supervisor instructions as formatted strings
        self.supervisor_common_instructions = supervisor_common_instructions or ""
        self.supervisor_model_instructions = supervisor_model_instructions or ""
        self.iteration_number = iteration_number
        self.skip_final_goal_update = skip_final_goal_update
        
        if DEBUG:
            print(f"DEBUG: NERAgent init for {model_name}")
            print(f"  iteration_number: {iteration_number}")
            print(f"  supervisor_common_instructions length: {len(self.supervisor_common_instructions)}")
            print(f"  supervisor_model_instructions length: {len(self.supervisor_model_instructions)}")
            if self.supervisor_common_instructions:
                print(f"  common preview: {self.supervisor_common_instructions[:100]}...")
            if self.supervisor_model_instructions:
                print(f"  model preview: {self.supervisor_model_instructions[:100]}...")

        
        if llm_infer_by_openrouter:
            if hasattr(self.client, 'force_paid_model') and llm_clients.PREFER_PAID_MODELS:
                self.client.force_paid_model()
        
        if DEBUG:
            print(f"DEBUG NERAgent init: {model_name}, iter={iteration_number}")
        print(f"DEBUG: supervisor_common_instructions = {len(self.supervisor_common_instructions)} chars")
        print(f"DEBUG: supervisor_model_instructions = {len(self.supervisor_model_instructions)} chars")

    def _format_ner_scheme(self) -> str:
        """Format NER scheme description based on the input format"""
        scheme_parts = []
        
        for entity_type, definition in self.ner_scheme.items():
            if isinstance(definition, str):
                # Simple string format
                scheme_parts.append(f"- {entity_type}: {definition}")
            elif isinstance(definition, dict):
                # Detailed dictionary format with examples
                entity_desc = f"- {entity_type}: {definition.get('definition_en', 'No definition provided')}"
                
                # Add positive examples if available
                pos_examples = definition.get('positive_examples', [])
                if pos_examples:
                    examples_str = ", ".join([f'"{ex}"' for ex in pos_examples])
                    entity_desc += f"\n  ✓ Examples: {examples_str}"
                
                # Add negative examples if available
                neg_examples = definition.get('negative_examples', [])
                if neg_examples:
                    examples_str = ", ".join([f'"{ex}"' for ex in neg_examples])
                    entity_desc += f"\n  ✗ NOT examples: {examples_str}"
                
                scheme_parts.append(entity_desc)
            else:
                # Fallback for unexpected format
                scheme_parts.append(f"- {entity_type}: {str(definition)}")
        
        return "\n".join(scheme_parts)

    def _format_supervisor_instructions(self) -> str:
        """Format supervisor-generated instructions for prompt inclusion"""
        if not self.supervisor_common_instructions and not self.supervisor_model_instructions:
            return ""
        
        instruction_parts = []
        
        # Add common instructions
        if self.supervisor_common_instructions:
            instruction_parts.append("### Additional Instructions:")
            instruction_parts.append(self.supervisor_common_instructions)
            instruction_parts.append("")
        
        # Add model-specific instructions  
        if self.supervisor_model_instructions:
            instruction_parts.append(f"### Specific Guidelines for {self.model_name}:")
            instruction_parts.append(self.supervisor_model_instructions)
            instruction_parts.append("")
            
        result = "\n".join(instruction_parts)    
        if DEBUG:
            print(f"DEBUG: _format_supervisor_instructions for {self.model_name}")
            print(f"  iteration_number: {self.iteration_number}")
            print(f"  common_instructions available: {bool(self.supervisor_common_instructions)}")
            print(f"  model_instructions available: {bool(self.supervisor_model_instructions)}")
            print(f"  formatted result length: {len(result)}")
            if result:
                print(f"  formatted preview: {result[:200]}...")
        return result

    def load_supervisor_instructions(self, supervisor_results_path: str) -> Tuple[str, str]:
        """Load instructions from supervisor results file - UPDATED for 4-phase structure"""
        if not os.path.exists(supervisor_results_path):
            return "", ""
        
        try:
            with open(supervisor_results_path, 'r', encoding='utf-8') as f:
                supervisor_results = json.load(f)
            
            # Extract from enhanced_guidelines (4-phase structure)
            enhanced_guidelines = supervisor_results.get("enhanced_guidelines", {})
            
            # Try Phase 4 hierarchical structure first
            common_instructions = enhanced_guidelines.get("hierarchical_common_instructions", [])
            if not common_instructions:
                # Fallback to Phase 3 finalized structure
                common_instructions = enhanced_guidelines.get("finalized_common_instructions", [])
            
            # Get model-specific instructions - try Phase 4 prioritized structure first
            model_specific_all = enhanced_guidelines.get("prioritized_model_instructions", {})
            if not model_specific_all:
                # Fallback to Phase 3 finalized structure
                model_specific_all = enhanced_guidelines.get("finalized_model_instructions", {})
            
            model_instructions = model_specific_all.get(self.model_name, [])
            
            return common_instructions, model_instructions
            
        except Exception as e:
            print(f"Warning: Failed to load supervisor instructions from {supervisor_results_path}: {e}")
            return "", ""

    def _create_ner_prompt(self, document: str) -> str:
        """Create NER prompt with task goal and configurable instructions"""
        
        scheme_description = self._format_ner_scheme()
        
        # Task goal section
        task_goal_section = ""
        if self.final_task_goal and not self.skip_final_goal_update:
            task_goal_section = f"\n## Task Context and Goal\n{self.final_task_goal}\n"
        
        # NEW: Supervisor instructions section
        supervisor_instructions_section = ""
        if self.iteration_number > 0:
            if DEBUG:
                print(f"DEBUG: Creating supervisor instructions section for iteration {self.iteration_number}")
            supervisor_instructions_text = self._format_supervisor_instructions()
            if supervisor_instructions_text:
                supervisor_instructions_section = f"\n{supervisor_instructions_text}"
                if DEBUG:
                    print(f"DEBUG: Supervisor instructions section created, length = {len(supervisor_instructions_section)}")
            else:
                if DEBUG:
                    print(f"DEBUG: No supervisor instructions text generated")
        else:
            if DEBUG:
                print(f"DEBUG: Skipping supervisor instructions for baseline iteration")

        # Format critical instructions
        critical_instructions_text = "\n".join([f"{i+1}. {instruction}" 
                                            for i, instruction in enumerate(self.critical_instructions)])
        
        # Add iteration information if applicable
        iteration_info = ""
        if self.iteration_number > 0:
            iteration_info = f" (Iteration {self.iteration_number} - Enhanced Guidelines Applied)"
        
        prompt = f"""
## Task: Named Entity Recognition{iteration_info}

"You are performing Named Entity Recognition (NER) on the provided document."
{task_goal_section if task_goal_section else ""}

### Entity Types and Definitions:
{scheme_description}
{supervisor_instructions_section}
### Critical Instructions:
{critical_instructions_text}

### Output Format:
Return results in JSON format with the following exact structure:
{{
    "entities": [
        {{
            "text": "exact entity text as it appears",
            "type": "entity_type",
            "start_pos": character_start_position,
            "end_pos": character_end_position,
            "confidence": "high/medium/low"
        }}
    ]
}}

### Document to Analyze:
{document}

### Your Response (JSON only):
"""
        return prompt
    
    def _create_confusing_case_prompt(self, document: str,
                                    entities: List[Dict[str, Any]]) -> str:
        """Create comprehensive prompt for identifying confusing annotation cases with configurable instructions"""
        
        scheme_description = self._format_ner_scheme()
        
        entities_summary = "\n".join([
            f"- '{entity['text']}' as {entity['type']} at positions {entity['start_pos']}-{entity['end_pos']}"
            for entity in entities
        ])
        
        # Task goal section
        task_goal_section = ""
        if self.final_task_goal and not self.skip_final_goal_update:
            task_goal_section = f"\n## Task Context and Goal\n{self.final_task_goal}\n"
    
        # NEW: Supervisor instructions section for confusing case analysis
        supervisor_instructions_section = ""
        if self.iteration_number > 0:
            supervisor_instructions_text = self._format_supervisor_instructions()
            if supervisor_instructions_text:
                supervisor_instructions_section = f"\n{supervisor_instructions_text}"
        
        # Format analysis instructions
        analysis_instructions_text = "\n".join([f"{i+1}. {instruction}" 
                                            for i, instruction in enumerate(self.analysis_instructions)])
        
        prompt = f"""
## Task: Identify Confusing Annotation Cases

{task_goal_section if task_goal_section else ""}"You are analyzing Named Entity Recognition (NER) results for potential annotation ambiguities."

You previously annotated the following document for named entities. Now, please analyze these results for potentially confusing cases where the annotation could reasonably be done differently.

### Original Document:
{document}

### Entity Types and Definitions:
{scheme_description}
{supervisor_instructions_section}
### Your Previous NER Annotations:
{entities_summary}

### Analysis Instructions:
{analysis_instructions_text}

### Output Format:
Return results in JSON format with the following exact structure:
{{
    "confusing_entities": [
        {{
            "text_original": "original annotated text span",
            "type_original": "original entity type",
            "text_possible": "alternatively annotated text span",
            "type_possible": "alternative entity type or O",
            "reasoning": "brief explanation of why this case is confusing"
        }}
    ]
}}

### Your Analysis (JSON only):
"""
        return prompt

    def _create_json_retry_prompt(self) -> str:
        """Create prompt for requesting proper JSON format after parsing failure"""
        prompt = """
Your previous response could not be parsed as valid JSON. Please provide ONLY the JSON response in the exact format requested, without any additional text, explanations, or thinking processes.

Required JSON format:
{
    "entities": [
        {
            "text": "exact entity text as it appears",
            "type": "entity_type",
            "start_pos": character_start_position,
            "end_pos": character_end_position,
            "confidence": "high/medium/low"
        }
    ]
}

Respond with JSON only:
"""
        return prompt

    def _create_confusing_case_retry_prompt(self) -> str:
        """Create prompt for requesting proper JSON format for confusing cases after parsing failure"""
        prompt = """
Your previous response could not be parsed as valid JSON. Please provide ONLY the JSON response in the exact format requested, without any additional text, explanations, or thinking processes.

Required JSON format:
{
    "confusing_entities": [
        {
            "text_original": "original annotated text span",
            "type_original": "original entity type",
            "text_possible": "alternatively annotated text span",
            "type_possible": "alternative entity type or O",
            "reasoning": "brief explanation of why this case is confusing"
        }
    ]
}

Respond with JSON only:
"""
        return prompt

    def _extract_json_after_think_tag(self, response: str) -> str:
        """
        Extract JSON content after </think> tag or return original response
        
        Args:
            response: Raw response from the model
            
        Returns:
            Cleaned response for JSON parsing
        """
        # Look for </think> tag (case insensitive)
        think_tag_pattern = r'</think>\s*'
        match = re.search(think_tag_pattern, response, re.IGNORECASE)
        
        if match:
            # Extract everything after </think> tag and strip whitespace
            json_content = response[match.end():].strip()
            return json_content
        else:
            # No </think> tag found, return original response stripped
            return response.strip()

    def _parse_json_response(self, response: str, prompt: str,
                             max_retries: int = 3) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Parse JSON response and extract predicted positions with retry logic
        
        Args:
            response: Raw response from the model
            prompt: Original prompt for context in error messages
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (parsed_json, entities_with_predicted_positions)
        """
        original_response = response
        
        for attempt in range(max_retries + 1):
            # Extract content after </think> tag if present
            cleaned_response = self._extract_json_after_think_tag(response)
            
            # Try to extract JSON from cleaned response
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, cleaned_response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
            else:
                json_str = cleaned_response
            
            try:
                parsed = json.loads(json_str)
                
                # Extract entities with predicted positions
                predicted_positions = []
                if "entities" in parsed:
                    for entity in parsed["entities"]:
                        predicted_positions.append({
                            "text": entity.get("text", ""),
                            "type": entity.get("type", ""),
                            "predicted_start": entity.get("start_pos", -1),
                            "predicted_end": entity.get("end_pos", -1),
                            "confidence": entity.get("confidence", "medium")
                        })
                
                return parsed, predicted_positions
                
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                try:
                    parsed = json.loads(json_str)
                    
                    # Extract entities with predicted positions
                    predicted_positions = []
                    if "entities" in parsed:
                        for entity in parsed["entities"]:
                            predicted_positions.append({
                                "text": entity.get("text", ""),
                                "type": entity.get("type", ""),
                                "predicted_start": entity.get("start_pos", -1),
                                "predicted_end": entity.get("end_pos", -1),
                                "confidence": entity.get("confidence", "medium")
                            })
                    
                    return parsed, predicted_positions
                    
                except json.JSONDecodeError:
                    if attempt < max_retries:
                        retry_prompt = self._create_json_retry_prompt()
                        if self.verbose >= 2:
                            print(f"JSON parsing failed (attempt {attempt + 1}/{max_retries + 1}). Requesting proper format...")
                            print(f"prompt: {prompt}")
                            print(f"cleaned_response: {cleaned_response}")
                            print(f"json_str: {json_str}")
                        response = self._generate_with_context(retry_prompt)
                    else:
                        # Final attempt failed
                        raise Exception(f"Failed to parse JSON response after {max_retries + 1} attempts.\nFinal error: {str(e)}\nOriginal response: {original_response}")
        
        # This should never be reached, but just in case
        raise Exception("Unexpected error in JSON parsing with retries")

    def _parse_confusing_case_response(self, response: str, max_retries: int = 3) -> Dict[str, Any]:
        """Parse JSON response for confusing cases with retry logic"""
        original_response = response
        
        for attempt in range(max_retries + 1):
            # Extract content after </think> tag if present
            cleaned_response = self._extract_json_after_think_tag(response)
            
            # Try to extract JSON from cleaned response
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, cleaned_response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
            else:
                json_str = cleaned_response
            
            try:
                parsed = json.loads(json_str)
                return parsed
                
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                try:
                    parsed = json.loads(json_str)
                    return parsed
                    
                except json.JSONDecodeError:
                    if attempt < max_retries:
                        retry_prompt = self._create_confusing_case_retry_prompt()
                        if self.verbose >= 2:
                            print(f"Confusing case JSON parsing failed (attempt {attempt + 1}/{max_retries + 1}). Requesting proper format...")
                        response = self._generate_with_context(retry_prompt)
                    else:
                        # Final attempt failed - return empty result instead of crashing
                        print(f"Warning: Failed to parse confusing case response after {max_retries + 1} attempts: {str(e)}")
                        return {"confusing_entities": []}
        
        # This should never be reached, but just in case
        return {"confusing_entities": []}
    
    def _find_entity_positions(self, document: str, entity_text: str) -> List[Tuple[int, int]]:
        """
        Find all character-level start-end positions of entity in document
        Uses token-based matching for single tokens, string matching for multi-token entities
        """
        import re
        
        positions = []
        entity_text = entity_text.strip()
        
        if not entity_text:
            return positions
        
        # Check if entity is a single token (no whitespace)
        if ' ' not in entity_text:
            # Single token: use word boundary matching to avoid substring matches
            # This prevents 'I' from matching 'If', 'in', etc.
            pattern = r'\b' + re.escape(entity_text) + r'\b'
            
            for match in re.finditer(pattern, document):
                start_pos = match.start()
                end_pos = match.end()
                positions.append((start_pos, end_pos))
        
        else:
            # Multi-token entity: use original string matching approach
            # For phrases like "New York", "United States", etc.
            start = 0
            
            while True:
                pos = document.find(entity_text, start)
                if pos == -1:
                    break
                positions.append((pos, pos + len(entity_text)))
                start = pos + 1
        
        return positions
    
    def extract_entities(self, document: str, analyze_confusing_cases: bool = False,
                         return_final_prompt: bool = True) -> Dict[str, Any]:
        """
        Extract named entities from document with quality metrics and optional confusing case analysis
        
        Args:
            document: Input text document
            analyze_confusing_cases: Whether to analyze confusing annotation cases
            return_final_prompt: Whether to include final prompt in results
            
        Returns:
            Dictionary containing extracted entities, positions, quality metrics, and optional confusing cases
        """
        # Clear previous conversation history for new document
        self._clear_conversation_history()
        
        # Generate NER prompt
        prompt = self._create_ner_prompt(document)
        if DEBUG:
            print(f"\n{'='*80}")
            print(f"FINAL PROMPT FOR {self.model_name} (Iteration {self.iteration_number})")
            print(f"{'='*80}")
            print(prompt)
            print(f"{'='*80}")
            print(f"Prompt length: {len(prompt)} characters")
            print(f"Supervisor instructions: {len(self.supervisor_common_instructions)} common, {len(self.supervisor_model_instructions)} model-specific")
            print(f"{'='*80}\n")

        # Get LLM response using context-aware generation
        response = self._generate_with_context(prompt)
        
        # Parse JSON response and get predicted positions (with retry logic)
        parsed_response, predicted_positions = self._parse_json_response(response, prompt)
        
        # Process entities and add actual position information
        processed_entities = []
        filtered_entities_count = 0

        if "entities" in parsed_response:
            for entity in parsed_response["entities"]:
                entity_text = entity.get("text", "")
                entity_type = entity.get("type", "")
                confidence = entity.get("confidence", "medium")

                if entity_type not in self.ner_scheme:
                    filtered_entities_count += 1
                    if self.verbose >= 2:
                        print(f"Filtering undefined entity type: '{entity_type}' for text '{entity_text}'")
                    continue

                # Find actual character positions via string matching
                positions = self._find_entity_positions(document, entity_text)
                
                for start_pos, end_pos in positions:
                    processed_entities.append({
                        "text": entity_text,
                        "type": entity_type,
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                        "confidence": confidence,
                        "context": document[max(0, start_pos-50):end_pos+50]
                    })
        
        # Prepare result dictionary
        result = {
            "document": document,
            "entities": processed_entities,
            "ner_scheme": self.ner_scheme,
            "final_task_goal": self.final_task_goal,
            "model_used": self.model_name,
            # NEW: Add iteration and supervisor information
            "iteration_number": self.iteration_number,
            "supervisor_instructions_applied": {
                "common_instructions_count": len(self.supervisor_common_instructions),
                "model_instructions_count": len(self.supervisor_model_instructions)
            },
            "inference_settings": {
                "temperature": GLOBAL_TEMPERATURE,
                "top_p": GLOBAL_TOP_P,
                "repeat_penalty": GLOBAL_REPEAT_PENALTY,
                "frequency_penalty": GLOBAL_FREQUENCY_PENALTY,
                "presence_penalty": GLOBAL_PRESENCE_PENALTY,
                "max_tokens": GLOBAL_MAX_TOKENS
            },
            "raw_response": response
        }

        if return_final_prompt:
            result["final_prompt"] = prompt
        result.update({
            "filtered_entities_count": filtered_entities_count,
            "defined_entity_types": list(self.ner_scheme.keys())
        })

        # Analyze confusing cases if requested
        if analyze_confusing_cases and processed_entities:
            try:
                confusing_prompt = self._create_confusing_case_prompt(document, processed_entities)
                # Generate confusing case analysis using context-aware generation
                confusing_response = self._generate_with_context(confusing_prompt)
                confusing_cases = self._parse_confusing_case_response(confusing_response)
                
                result["confusing_cases"] = confusing_cases.get("confusing_entities", [])
                result["confusing_cases_raw_response"] = confusing_response
                
            except Exception as e:
                print(f"Warning: Failed to analyze confusing cases: {str(e)}")
                result["confusing_cases"] = []
                result["confusing_cases_error"] = str(e)
        
        elif analyze_confusing_cases:
            # No entities found, so no confusing cases to analyze
            result["confusing_cases"] = []
        
        return result
    
    def batch_extract(self, documents: List[str], analyze_confusing_cases: bool = False) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple documents with aggregated quality metrics and optional confusing case analysis
        
        Args:
            documents: List of input text documents
            analyze_confusing_cases: Whether to analyze confusing annotation cases
            
        Returns:
            List of dictionaries containing extracted entities, optional confusing cases, and overall metrics
        """
        results = []
        overall_metrics = {
            "total_documents": len(documents),
            "successful_extractions": 0,
            "failed_extractions": 0,
            "avg_entities_per_doc": 0.0,
        }
        
        if analyze_confusing_cases:
            overall_metrics["avg_confusing_cases_per_doc"] = 0.0
        
        total_entities = 0
        total_confusing_cases = 0
        
        for i, doc in enumerate(documents):
            if self.verbose >= 1:
                print(f"Processing document {i+1}/{len(documents)}")
            try:
                result = self.extract_entities(doc, analyze_confusing_cases=analyze_confusing_cases)
                results.append(result)
                overall_metrics["successful_extractions"] += 1
                
                # Aggregate metrics
                doc_entities = len(result["entities"])
                total_entities += doc_entities
                
                if analyze_confusing_cases:
                    total_confusing_cases += len(result.get("confusing_cases", []))
                
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Error processing document {i+1}: {str(e)}")
                results.append({
                    "document": doc,
                    "entities": [],
                    "error": str(e)
                })
                overall_metrics["failed_extractions"] += 1
        
        # Calculate overall averages
        successful_docs = overall_metrics["successful_extractions"]
        if successful_docs > 0:
            overall_metrics["avg_entities_per_doc"] = total_entities / successful_docs
            
            if analyze_confusing_cases:
                overall_metrics["avg_confusing_cases_per_doc"] = total_confusing_cases / successful_docs
        
        return results, overall_metrics
    
    def export_results(self, results: Dict[str, Any], output_file: str):
        """Export results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if self.verbose >= 1:
            print(f"Results exported to {output_file}")