# baseline_gpt5_ner.py
# Enhanced version with checkpointing and parallel processing
# - Saves intermediate results every 10 samples
# - Resumes from last checkpoint if interrupted
# - Processes 4 datasets in parallel
# - Cleans up checkpoint files after completion

import os
import json
import time
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# =========================
# Configuration
# =========================

# Where to look for datasets and experiment settings
BASE_DIR = "datasets"
BASE_CONFIG_DIR = "experiment_settings"  # <dataset_key>_default_config.json resides here

# Toggle to run only one sample per benchmark and print the prompt
DEBUG_RUN_ONE = False  # Set to False after you confirm

# Model & pricing (usage-based cost). Do NOT hardcode secrets.
MODEL_LIST = [
    # "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07"
]
EFFORT = "medium"
VERBOSITY = "medium"
# EFFORT = "minimal"
# VERBOSITY = "low"

PRICES = {
    "gpt-5-2025-08-07": {
        "input": 0.00125,
        "cached_input": 0.00013,
        "output": 0.01000
    },
    "gpt-5-mini-2025-08-07": {
        "input": 0.00025,
        "cached_input": 0.00003,
        "output": 0.00200
    }
}

DATASET_KEYS = [
    # single-pkl conventions
    #"OntoNotes", "AnatEM", "bc2gm", "bc4chemd", "bc5cdr", "Broad Twitter", 
    # "ACE05", 
    # "GENIA", 
    # "MultiNERD", 
    # "FabNER",
    # mitnerner & crossner multi-pkl (use file prefix as dataset key)
    "mit_restaurant",
    # "mit_movie",
    # "crossner_ai", "crossner_literature", "crossner_music", "crossner_science", "crossner_politics",
    # "crossner_conll2003"
    # "conllpp"            # excluded
    # "HarveyNER", "FindVehicle", 
]

# Use sampled test if available; else fall back to "test" split in main pkl
PREFER_SAMPLED = True
SAMPLED_SUFFIX = "_sampled_test_300.pkl"

# Output directory
OUTPUT_ROOT = "baseline_results_gpt5"

# Parallel processing configuration
MAX_PARALLEL_DATASETS = 4
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N samples

# =========================
# Thread-safe file operations
# =========================

file_lock = threading.Lock()

def safe_file_write(filepath: str, data: Any, is_json: bool = True):
    """Thread-safe file writing"""
    with file_lock:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if is_json:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(data))

def safe_file_read(filepath: str, is_json: bool = True) -> Any:
    """Thread-safe file reading"""
    with file_lock:
        if not os.path.exists(filepath):
            return None
        try:
            if is_json:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception:
            return None

def safe_file_delete(filepath: str):
    """Thread-safe file deletion"""
    with file_lock:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass

# =========================
# Minimal GPT-5 client (Responses API style)
# =========================

class GPT5Client:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        key = api_key or os.environ.get("OPENAI_API_KEY_ENV", None)
        if not key:
            raise RuntimeError("OPENAI_API_KEY_ENV not set in environment.")
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai package is required. pip install openai") from e
        self.client = OpenAI(api_key=key)
        self.model = model

    def ask(self, prompt: str, effort: str = EFFORT, verbosity: str = VERBOSITY) -> Tuple[str, Dict[str, Any]]:
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            reasoning={"effort": effort},
            text={"verbosity": verbosity}
        )
        text = self._extract_output_text(resp)
        usage = self._extract_usage(resp)
        return text, usage

    @staticmethod
    def _extract_output_text(resp) -> str:
        out = []
        for item in getattr(resp, "output", []):
            content = getattr(item, "content", None)
            if not content:
                continue
            for c in content:
                if getattr(c, "type", "") == "output_text":
                    out.append(c.text)
        return "".join(out).strip()

    @staticmethod
    def _extract_usage(resp) -> Dict[str, int]:
        usage = getattr(resp, "usage", None)
        if not usage:
            return {"input_tokens": 0, "output_tokens": 0}
        return {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0)
        }

def estimate_cost(usage: Dict[str, int], model: str) -> float:
    """USD cost estimate from token usage."""
    pricing = PRICES.get(model)
    if pricing is None:
        raise ValueError(f"Pricing not found for model='{model}'")
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    price_in = pricing["input"] * (in_tok / 1000)
    price_out = pricing["output"] * (out_tok / 1000)
    return round(price_in + price_out, 6)

# =========================
# Minimal NER utilities
# =========================

def bio_to_entities(tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
    """BIO -> entity spans with character positions (strict)."""
    entities = []
    current = None
    # precompute token char positions over joined text
    char_pos = 0
    token_positions = []
    for i, tok in enumerate(tokens):
        start = char_pos
        end = start + len(tok)
        token_positions.append((start, end))
        char_pos = end + (1 if i < len(tokens) - 1 else 0)

    for i, tag in enumerate(labels):
        if tag.startswith("B-"):
            if current:
                entities.append(current)
            et = tag[2:]
            s, e = token_positions[i]
            current = {"text": tokens[i], "type": et, "start_pos": s, "end_pos": e}
        elif tag.startswith("I-") and current:
            et = tag[2:]
            if et == current["type"]:
                _, e = token_positions[i]
                current["text"] += " " + tokens[i]
                current["end_pos"] = e
            else:
                entities.append(current)
                et2 = et
                s, e = token_positions[i]
                current = {"text": tokens[i], "type": et2, "start_pos": s, "end_pos": e}
        else:
            if current:
                entities.append(current)
                current = None
    if current:
        entities.append(current)
    return entities

def entities_to_bio(tokens: List[str], entities: List[Dict[str, Any]], text: Optional[str] = None,
                    valid_types: Optional[set] = None) -> List[str]:
    """Entities -> BIO aligned to tokens via char spans. Uses token char spans over ' '.join(tokens)."""
    bio = ["O"] * len(tokens)
    char_map = {}
    pos = 0
    for i, tok in enumerate(tokens):
        s = pos
        e = s + len(tok)
        for c in range(s, e):
            char_map[c] = i
        pos = e + (1 if i < len(tokens) - 1 else 0)

    if not entities:
        return bio

    for ent in entities:
        et = ent.get("type", "")
        if valid_types and et not in valid_types:
            continue
        s = ent.get("start_pos", -1)
        e = ent.get("end_pos", -1)
        if s < 0 or e <= s:
            continue
        token_idxs = sorted({char_map[c] for c in range(s, e) if c in char_map})
        for k, ti in enumerate(token_idxs):
            bio[ti] = ("B-" if k == 0 else "I-") + et
    return bio

def strict_span_micro_f1(all_pred_entities: List[List[Dict]], all_gold_entities: List[List[Dict]]) -> Dict[str, float]:
    """
    Strict span matching across full test set, micro-averaged.
    """
    pred = set()
    gold = set()
    for ents in all_pred_entities:
        for e in ents:
            key = (e.get("start_pos"), e.get("end_pos"), e.get("type"))
            if key[0] is not None and key[1] is not None and key[2]:
                pred.add(key)
    for ents in all_gold_entities:
        for e in ents:
            key = (e.get("start_pos"), e.get("end_pos"), e.get("type"))
            if key[0] is not None and key[1] is not None and key[2]:
                gold.add(key)
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

# =========================
# Config & Prompting
# =========================

def load_experiment_config(dataset_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Load ner_scheme and final_task_goal from experiment_settings/{dataset_key}_default_config.json
    Returns (ner_scheme, final_task_goal). If file missing or keys absent, returns fallbacks.
    """
    cfg_path = os.path.join(BASE_CONFIG_DIR, f"{dataset_key}_default_config.json")
    print(f"cfg_path: {cfg_path}")
    if not os.path.exists(cfg_path):
        return None, None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        ner_scheme = cfg.get("ner_scheme")
        final_task_goal = cfg.get("final_task_goal")
        return ner_scheme, final_task_goal
    except Exception:
        return None, None

def format_ner_scheme_for_prompt(ner_scheme: Dict[str, Any]) -> str:
    """
    Mirror base_annotator._format_ner_scheme style: each type line with definition and optional examples.
    """
    parts: List[str] = []
    for et, definition in ner_scheme.items():
        if isinstance(definition, str):
            parts.append(f"- {et}: {definition}")
        elif isinstance(definition, dict):
            desc = f"- {et}: {definition.get('definition_en', 'No definition provided')}"
            pos_ex = definition.get("positive_examples", [])
            neg_ex = definition.get("negative_examples", [])
            if pos_ex:
                desc += "\n  ✓ Examples: " + ", ".join([f'\"{ex}\"' for ex in pos_ex])
            if neg_ex:
                desc += "\n  ✗ NOT examples: " + ", ".join([f'\"{ex}\"' for ex in neg_ex])
            parts.append(desc)
        else:
            parts.append(f"- {et}: {str(definition)}")
    return "\n".join(parts)

def get_default_critical_instructions() -> List[str]:
    """
    Lightweight default critical instructions aligned to base_annotator defaults.
    """
    return [
        "Return JSON only. No extra text.",
        "Entity spans must be exact character offsets in the given document.",
        "Use only the allowed entity types from the scheme.",
        "Do not hallucinate entities not present in the text.",
        "Be strict about boundaries and avoid overlapping spans."
    ]

def build_baseline_prompt(
    ner_scheme: Optional[Dict[str, Any]],
    final_task_goal: Optional[str],
    document: str,
    iteration_number: int = 0,
    critical_instructions: Optional[List[str]] = None
) -> str:
    """
    Baseline NER prompt aligned to base_annotator._create_ner_prompt for iteration 0.
    """
    scheme_section = format_ner_scheme_for_prompt(ner_scheme) if ner_scheme else "- Use dataset-appropriate entity types (no explicit scheme found)."
    ci = critical_instructions or get_default_critical_instructions()
    ci_text = "\n".join([f"{i+1}. {ins}" for i, ins in enumerate(ci)])
    task_goal_section = f"\n## Task Context and Goal\n{final_task_goal}\n" if final_task_goal else ""

    prompt = f"""
## Task: Named Entity Recognition
You are performing Named Entity Recognition (NER) on the provided document.
{task_goal_section}

### Entity Types and Definitions:
{scheme_section}

### Critical Instructions:
{ci_text}

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
""".strip()
    return prompt

def scheme_types(ner_scheme: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """Extract flat list of entity types from ner_scheme keys."""
    if not ner_scheme:
        return None
    return list(ner_scheme.keys())

# =========================
# Dataset loading
# =========================

def find_main_pkl_path(base_dir: str, key: str) -> Optional[str]:
    """<folder>/<folder>_ner_dataset.pkl"""
    folder = os.path.join(base_dir, key)
    pkl = os.path.join(folder, f"{key}_ner_dataset.pkl")
    return pkl if os.path.exists(pkl) else None

def find_sampled_pkl_path(base_dir: str, key: str) -> Optional[str]:
    """
    For single-pkl datasets: <folder>/<folder>_sampled_test_300.pkl
    For multi-pkl keys (mit_*, crossner_*): <parent>/<key>_sampled_test_300.pkl under their group folder.
    """
    path1 = os.path.join(base_dir, key, f"{key}{SAMPLED_SUFFIX}")
    if os.path.exists(path1):
        return path1
    mit = os.path.join(base_dir, "mitnerner", f"{key}{SAMPLED_SUFFIX}")
    print(f"mit: {mit}")
    if os.path.exists(mit):
        return mit
    cr = os.path.join(base_dir, "crossner", f"{key}{SAMPLED_SUFFIX}")
    if os.path.exists(cr):
        return cr
    return None

def load_test_samples(base_dir: str, key: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns (samples, entity_types_from_labels)
    samples: list of dict with keys: text, tokens, labels
    entity_types_from_labels: inferred from labels in test set (used as fallback/type-validation)
    """
    if PREFER_SAMPLED:
        s_path = find_sampled_pkl_path(base_dir, key)
        print(f"base_dir: {base_dir}")
        print(f"s_path: {s_path}")
        if s_path:
            with open(s_path, "rb") as f:
                data = pickle.load(f)
            samples = data if isinstance(data, list) else []
            types = set()
            for ex in samples[:2000]:
                for lb in ex.get("labels", []):
                    if lb != "O" and "-" in lb:
                        types.add(lb.split("-", 1)[1])
            return samples, sorted(types)

    m_path = find_main_pkl_path(base_dir, key)
    if not m_path:
        print(f"base_dir: {base_dir}")
        raise FileNotFoundError(f"No dataset found for key={key}")
    with open(m_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "test" not in data:
        raise ValueError(f"Main PKL missing 'test' split for key={key}")
    samples = data["test"]
    types = set()
    for ex in samples[:2000]:
        for lb in ex.get("labels", []):
            if lb != "O" and "-" in lb:
                types.add(lb.split("-", 1)[1])
    return samples, sorted(types)

# =========================
# Checkpoint management
# =========================

def get_checkpoint_path(dataset_key: str, model: str) -> str:
    """Get checkpoint file path"""
    out_dir = os.path.join(OUTPUT_ROOT, model)
    return os.path.join(out_dir, f"{dataset_key}__checkpoint.json")

def load_checkpoint(dataset_key: str, model: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint data if exists"""
    checkpoint_path = get_checkpoint_path(dataset_key, model)
    return safe_file_read(checkpoint_path, is_json=True)

def save_checkpoint(dataset_key: str, model: str, checkpoint_data: Dict[str, Any]):
    """Save checkpoint data"""
    checkpoint_path = get_checkpoint_path(dataset_key, model)
    safe_file_write(checkpoint_path, checkpoint_data, is_json=True)

def remove_checkpoint(dataset_key: str, model: str):
    """Remove checkpoint file after completion"""
    checkpoint_path = get_checkpoint_path(dataset_key, model)
    safe_file_delete(checkpoint_path)

# =========================
# Runner with checkpointing
# =========================

def has_entities(labels: List[str]) -> bool:
    """Check if a sample has any entities (non-O labels)"""
    return any(label != "O" for label in labels)

def select_debug_samples(samples: List[Dict[str, Any]], n_samples: int = 5) -> List[Dict[str, Any]]:
    """
    Select n_samples that contain entities for debugging purposes.
    Returns samples with entities, up to n_samples count.
    """
    entity_samples = []
    for sample in samples:
        labels = sample.get("labels", [])
        if has_entities(labels):
            entity_samples.append(sample)
            if len(entity_samples) >= n_samples:
                break
    
    if len(entity_samples) < n_samples:
        print(f"Warning: Only found {len(entity_samples)} samples with entities out of requested {n_samples}")
    
    return entity_samples

def run_benchmark_on_gpt5(dataset_key: str, client: GPT5Client, model: str) -> Dict[str, Any]:
    """
    Runs baseline NER on the test set for a single dataset key with checkpointing.
    Modified to select 5 entity-containing samples when DEBUG_RUN_ONE = True.
    """
    ner_scheme_cfg, final_task_goal = load_experiment_config(dataset_key)
    print(f"ner_scheme_cfg: {ner_scheme_cfg}")
    print(f"final_task_goal: {final_task_goal}")


    # Resolve samples
    all_samples, inferred_types = load_test_samples(BASE_DIR, dataset_key)
    if len(all_samples) == 0:
        print(f"[{dataset_key}] No samples found. Skipping.")
        return {"dataset": dataset_key, "model": model, "total_samples": 0, "strict_span_micro": {"f1": 0.0}, "cost_usd_total": 0.0}

    # Select samples based on DEBUG_RUN_ONE setting
    if DEBUG_RUN_ONE:
        samples = select_debug_samples(all_samples, n_samples=5)
        n_to_run = len(samples)
        print(f"[{dataset_key}] DEBUG mode: Selected {n_to_run} samples with entities")
    else:
        samples = all_samples
        n_to_run = len(samples)

    # Output path with model subfolder + samplen
    out_dir = os.path.join(OUTPUT_ROOT, model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_key}__samplen{n_to_run}.json")

    # Determine type list for validation: prefer ner_scheme keys, fallback to inferred
    types_from_scheme = scheme_types(ner_scheme_cfg)
    type_list_for_validation = set(types_from_scheme) if types_from_scheme else set(inferred_types)

    print(f"\n=== {dataset_key} | total_samples={len(all_samples)} | running={n_to_run} | model={model} | effort={EFFORT}, verbosity={VERBOSITY} ===")

    # Load checkpoint if exists
    checkpoint = load_checkpoint(dataset_key, model)
    start_idx = 0
    records = []
    total_cost = 0.0
    
    if checkpoint:
        start_idx = checkpoint.get("last_processed_idx", 0) + 1
        records = checkpoint.get("records", [])
        total_cost = checkpoint.get("total_cost", 0.0)
        print(f"[{dataset_key}] Resuming from sample {start_idx}, previous cost: ${total_cost:.4f}")
    
    # Process samples from start_idx
    all_pred_entities: List[List[Dict[str, Any]]] = []
    all_gold_entities: List[List[Dict[str, Any]]] = []

    # Load existing predictions for evaluation
    for record in records:
        all_pred_entities.append(record.get("predicted_entities", []))
        all_gold_entities.append(record.get("gold_entities", []))

    for i in tqdm(range(start_idx, n_to_run), desc=f"Processing {dataset_key}"):
        ex = samples[i]  # Now using filtered samples
        text = ex["text"]
        tokens = ex["tokens"]
        gold_labels = ex["labels"]
        gold_entities = bio_to_entities(tokens, gold_labels)

        # Build prompt
        prompt = build_baseline_prompt(
            ner_scheme=ner_scheme_cfg,
            final_task_goal=final_task_goal,
            document=text,
            iteration_number=0,
            critical_instructions=get_default_critical_instructions()
        )

        if DEBUG_RUN_ONE and i == 0:
            print("\n--- Prompt Preview ---")
            print(prompt)
            print("----------------------")

        t0 = time.time()
        try:
            resp_text, usage = client.ask(prompt, effort=EFFORT, verbosity=VERBOSITY)
        except Exception as e:
            print(f"[{dataset_key}] Inference error on sample {i}: {e}")
            resp_text, usage = "{}", {"input_tokens": 0, "output_tokens": 0}
        infer_secs = time.time() - t0
        cost = estimate_cost(usage, model=model)
        total_cost += cost

        # Parse JSON strictly; try to salvage common issues
        pred_entities = []
        try:
            json_match = re.search(r"\{.*\}", resp_text, flags=re.DOTALL)
            js = json_match.group(0) if json_match else resp_text
            js = re.sub(r',\s*}', '}', js)
            js = re.sub(r',\s*]', ']', js)
            parsed = json.loads(js)
            for ent in parsed.get("entities", []):
                pred_entities.append({
                    "text": ent.get("text", ""),
                    "type": ent.get("type", ""),
                    "start_pos": int(ent.get("start_pos", -1)),
                    "end_pos": int(ent.get("end_pos", -1))
                })
        except Exception:
            pred_entities = []

        # Convert to BIO for compatibility
        pred_labels = entities_to_bio(tokens, pred_entities, valid_types=type_list_for_validation)

        # Per-sample strict span metrics
        per_sample_scores = strict_span_micro_f1([pred_entities], [gold_entities])

        # Build per-sample record including requested fields
        record = {
            "dataset": dataset_key,
            "sample_id": i,
            "prompt_used": prompt,                 # required
            "raw_response": resp_text,             # required
            "text": text,
            "tokens": tokens,
            "token_labels_gold": gold_labels,      # required
            "token_labels_pred": pred_labels,      # required
            "gold_entities": gold_entities,
            "predicted_entities": pred_entities,
            "metrics": {
                "strict_span_precision": per_sample_scores["precision"],
                "strict_span_recall": per_sample_scores["recall"],
                "strict_span_f1": per_sample_scores["f1"]
            },
            "model_name": model,
            "inference_time_seconds": infer_secs,
            "usage": usage,
            "cost_usd": cost,
        }
        # Optional trace fields in debug mode
        if DEBUG_RUN_ONE:
            record["final_task_goal"] = final_task_goal
            record["ner_scheme_keys"] = list(type_list_for_validation)

        records.append(record)
        all_pred_entities.append(pred_entities)
        all_gold_entities.append(gold_entities)

        # Save checkpoint every CHECKPOINT_INTERVAL samples
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_data = {
                "dataset": dataset_key,
                "model": model,
                "last_processed_idx": i,
                "total_cost": total_cost,
                "records": records
            }
            save_checkpoint(dataset_key, model, checkpoint_data)
            print(f"[{dataset_key}] Checkpoint saved at sample {i+1}")

    # Aggregate micro strict span F1 across test set
    agg = strict_span_micro_f1(all_pred_entities, all_gold_entities)

    # Compose final JSON document
    result_doc = {
        "dataset": dataset_key,
        "model": model,
        "debug_run_one": DEBUG_RUN_ONE,
        "effort": EFFORT,
        "verbosity": VERBOSITY,
        "total_samples_saved": n_to_run,
        "cost_usd_total": round(total_cost, 6),
        "strict_span_micro": {
            "precision": agg["precision"],
            "recall": agg["recall"],
            "f1": agg["f1"],
            "tp": agg["tp"], "fp": agg["fp"], "fn": agg["fn"]
        },
        "records": records
    }

    # Save final JSON file
    safe_file_write(out_path, result_doc, is_json=True)
    
    # Remove checkpoint file after successful completion
    remove_checkpoint(dataset_key, model)

    print(f"[{dataset_key} | {model}] micro-F1={agg['f1']:.3f}  cost=${total_cost:.4f}  -> {out_path}")
    return {
        "dataset": dataset_key,
        "model": model,
        "total_samples": n_to_run,
        "strict_span_micro": result_doc["strict_span_micro"],
        "cost_usd_total": result_doc["cost_usd_total"]
    }

# =========================
# Parallel processing wrapper
# =========================

def process_single_dataset_model(dataset_key: str, model: str) -> Optional[Dict[str, Any]]:
    """Wrapper function for processing a single dataset-model combination"""
    try:
        client = GPT5Client(
            api_key="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",
            model=model
        )
        return run_benchmark_on_gpt5(dataset_key, client, model=model)
    except FileNotFoundError as e:
        print(f"dataset_key: {dataset_key}")
        print(f"[SKIP] {dataset_key} ({model}): {e}")
        return None
    except Exception as e:
        print(f"[ERROR] {dataset_key} ({model}): {e}")
        return None

# =========================
# Orchestrator with parallel processing
# =========================
def main():
    all_summaries = []
    
    # Create all dataset-model combinations
    tasks = []
    for model in MODEL_LIST:
        for key in DATASET_KEYS:
            tasks.append((key, model))
    
    print(f"Processing {len(tasks)} dataset-model combinations with max {MAX_PARALLEL_DATASETS} parallel workers")
    
    # Process tasks in parallel
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_DATASETS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_dataset_model, dataset_key, model): (dataset_key, model)
            for dataset_key, model in tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            dataset_key, model = future_to_task[future]
            try:
                result = future.result()
                if result:
                    all_summaries.append(result)
            except Exception as exc:
                print(f'[ERROR] {dataset_key} ({model}) generated an exception: {exc}')

    # Build per-model overview
    overview: Dict[str, List[Dict[str, Any]]] = {m: [] for m in MODEL_LIST}
    for s in all_summaries:
        model = s.get("model", "unknown")
        overview.setdefault(model, []).append({
            "dataset": s["dataset"],
            "samples": s["total_samples"],
            "micro_f1": s["strict_span_micro"]["f1"],
            "cost_usd_total": s["cost_usd_total"]
        })

    # Save overview under each model folder
    for model, rows in overview.items():
        out_dir = os.path.join(OUTPUT_ROOT, model)
        os.makedirs(out_dir, exist_ok=True)
        overview_data = {
            "debug_run_one": DEBUG_RUN_ONE,
            "model": model,
            "effort": EFFORT,
            "verbosity": VERBOSITY,
            "report": rows
        }
        safe_file_write(os.path.join(out_dir, "baseline_gpt_overview.json"), overview_data, is_json=True)

    print("\n=== Baseline Overview by Model ===")
    for model, rows in overview.items():
        print(f"\n[{model}]")
        for r in rows:
            print(f"{r['dataset']:20s}  n={r['samples']:4d}  microF1={r['micro_f1']:.3f}  cost=${r['cost_usd_total']:.4f}")

if __name__ == "__main__":
    main()