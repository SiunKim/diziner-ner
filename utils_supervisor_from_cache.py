from __future__ import annotations
import json, re, hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from utils_supervisor import safe_json_serialize

# ---------- path helpers ----------

def phase_result_filename(phase_key: str) -> str:
    m = {
        "phase1": "phase1_disagreement_pattern_analysis.json",
        "phase2": "phase2_non_elite_model_analysis.json",
        "phase3": "phase3_instruction_generation_and_decision.json",
        "phase4": "phase4_hierarchical_guideline_organization.json",
    }
    return m[phase_key]

def phase_result_path(output_dir: str | Path, phase_key: str) -> Path:
    return Path(output_dir) / phase_result_filename(phase_key)

def comprehensive_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "comprehensive_results.json"

def safe_model_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "model"

# ---------- light validation & load/save ----------

def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def has_minimal_keys(d: Dict[str, Any], required_keys=None) -> bool:
    # Minimal validity for cache reuse
    required = required_keys or ["success", "result_data"]
    try:
        return bool(d) and all(k in d for k in required) and d.get("success") is True
    except Exception:
        return False

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _prompt_matches(cached: Dict[str, Any], prompt: Optional[str]) -> bool:
    if prompt is None:
        return True
    used = cached.get("prompt_used")
    if not used:
        return False
    # Light check: raw string equality first, fallback to sha1
    return (used == prompt) or (_sha1(used) == _sha1(prompt))

def save_json(path: Path, obj: Dict[str, Any]) -> Path:
    # Ensure directory
    path.parent.mkdir(parents=True, exist_ok=True)
    # Always sanitize to JSON-safe structure
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_json_serialize(obj), f, ensure_ascii=False, indent=2)
    return path


# ---------- decision helpers (phase-level / per-model / final) ----------

def should_use_phase_cache(
    output_dir: str | Path,
    phase_key: str,
    *,
    prompt: str | None = None,
    require_prompt_match: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Path]]:
    p = phase_result_path(output_dir, phase_key)
    cached = load_json_if_exists(p)
    if not cached or not has_minimal_keys(cached):
        return False, None, p
    if require_prompt_match and not _prompt_matches(cached, prompt):
        return False, None, p
    return True, cached, p

def save_phase_result(output_dir: str | Path, phase_key: str, result: Dict[str, Any]) -> Path:
    return save_json(phase_result_path(output_dir, phase_key), result)

def should_use_per_model_cache(
    output_dir: str | Path,
    model_name: str,
    *,
    prompt: str | None = None,
    require_prompt_match: bool = False,
    phase_key: str = "phase2",
):
    # 1) new path in root
    p = per_model_cache_path(output_dir, model_name, phase_key=phase_key)
    cached = load_json_if_exists(p)
    if cached and has_minimal_keys(cached) and (not require_prompt_match or _prompt_matches(cached, prompt)):
        return True, cached, p
    # 2) fallback: legacy location phase2_per_model/
    legacy = Path(output_dir) / f"{phase_key}_per_model" / f"{safe_model_name(model_name)}.json"
    cached_legacy = load_json_if_exists(legacy)
    if cached_legacy and has_minimal_keys(cached_legacy) and (not require_prompt_match or _prompt_matches(cached_legacy, prompt)):
        # Optional: you can migrate by writing to new path here if desired
        # save_json(p, cached_legacy)
        return True, cached_legacy, legacy

    return False, None, p

def save_per_model_result(
    output_dir: str | Path,
    model_name: str,
    result: Dict[str, Any],
    *,
    phase_key: str = "phase2",
) -> Path:
    p = per_model_cache_path(output_dir, model_name, phase_key=phase_key)
    return save_json(p, result)

def try_load_full_results(output_dir: str | Path) -> Optional[Dict[str, Any]]:
    """
    Minimal check for final results. We keep this lenient on purpose.
    Accept either of these shapes:
      { "success": true, "result_data": ... } or
      { "success": true, "phase_results": ... } or
      { "success": true, "results": ... }
    """
    p = comprehensive_path(output_dir)
    obj = load_json_if_exists(p)
    if not obj or obj.get("success") is not True:
        return None
    if any(k in obj for k in ("result_data", "phase_results", "results")):
        return obj
    return None

def per_model_cache_path(output_dir: str | Path, model_name: str, *, phase_key: str = "phase2") -> Path:
    # Store next to other phase JSONs (no subdirectory)
    fname = f"{phase_key}_non_elite_model_{safe_model_name(model_name)}.json"
    return Path(output_dir) / fname
