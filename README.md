# DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot NER

**Official code repository for ACL 2025 submission (anonymized for review)**

This repository contains the implementation of **DiZiNER**, a framework that simulates human pilot annotation processes to achieve state-of-the-art zero-shot Named Entity Recognition through iterative disagreement-guided instruction refinement.

---

## Paper Overview

DiZiNER addresses the persistent gap between zero-shot and supervised NER by mimicking human annotation workflows. Multiple heterogeneous LLMs act as independent annotators labeling shared documents, while a supervisor model analyzes inter-model disagreements to iteratively refine task instructions—mirroring how human annotators establish gold standards through disagreement resolution.

![Figure 1: DiZiNER Framework Overview](figures/figure1_overview.png)

### Three-Stage Iterative Cycle

1. **Independent Cross-Annotation**: Multiple LLM annotators independently perform NER tagging on the same document set
2. **Disagreement Analysis**: Identifies hotspot spans with high annotation disagreement, categorizes error patterns into structured reports
3. **Instruction Refinement**: Supervisor leverages disagreement summaries to revise task guidelines through a 4-phase process

### Main Results

![Table 1: CrossNER Results](figures/table1_crossner.png)

![Table 2: Overall Results](figures/table2_overall.png)

**Performance Highlights**:
- **New SOTA**: Achieved best zero-shot results on 13 out of 18 benchmarks
- **Average Improvement**: +13.6 F1 points over previous best zero-shot systems
- **Gap Reduction**: Narrowed zero-shot to supervised gap from 31.7 to 17.6 F1 points
- **Supervisor Comparison**: Outperformed GPT-4o mini supervisor by +7.5 F1 (CrossNER) and +6.4 F1 (overall)

### Agreement-Performance Correlation

![Figure 2: Agreement Correlation](figures/figure2_correlation.png)

Key findings:
- Higher inter-model agreement consistently predicts better NER performance
- Disagreement-guided refinement is the primary driver of improvements
- Performance gains stem from instruction quality rather than supervisor model scale

### Key Ablation Studies

<p align="center">
  <img src="figures/table3_taskgoal.png" alt="Table 3: Final Task Goal" width="60%">
</p>

**Critical Components**:
1. **Final Task Goal** (-3.7 F1 when removed): Essential for resolving conflicting instructions
2. **Annotator Diversity**: Homogeneous model pools fail to improve beyond iteration 0
3. **Optimal Set Size**: 15-20 samples per iteration achieve best performance
4. **Gold Standard** (-0.4 F1): Disagreement-guided approach outperforms gold supervision

### Datasets Evaluated

**18 NER Benchmarks**:
- **Cross-domain**: CrossNER (AI, Literature, Music, Politics, Science)
- **General**: CoNLL2003, ACE2005, OntoNotes, MultiNERD
- **Biomedical**: AnatEM, BC2GM, BC4CHEMD, BC5CDR, GENIA
- **STEM**: FabNER
- **Social**: BroadTwitter, MIT-Movie, MIT-Restaurant

---

## Installation

### Prerequisites

```bash
# Python 3.11+
pip install torch==2.8.0 transformers==4.55.2 tokenizers==0.21.4 sentence-transformers==5.1.0 datasets==4.0.0
pip install numpy==2.3.2 pandas==2.3.1 scikit-learn==1.7.1 scipy==1.16.1 tqdm==4.67.1 joblib==1.5.1 regex==2025.7.34
pip install matplotlib==3.10.5 seaborn==0.13.2 plotly==6.3.0
pip install openai==1.99.1 anthropic==0.61.0 huggingface-hub==0.34.3 requests==2.32.4 safetensors==0.6.2
pip install jupyterlab==4.4.7 ipywidgets==8.1.7
```

### API Key Setup

DiZiNER requires API keys for OpenRouter (annotator models) and OpenAI (supervisor model).

**Option 1: Environment Variables (Recommended)**

```bash
export OPENROUTER_API_KEY="your_openrouter_key_here"
export OPENAI_API_KEY="your_openai_key_here"
```

**Option 2: Direct Configuration**

Create `config/api_keys.json`:

```json
{
  "openrouter_api_key": "your_openrouter_key_here",
  "openai_api_key": "your_openai_key_here"
}
```

The code checks for keys in this order: environment variables → .env file → config file.

---

## Quick Start

### Reproducing Paper Results

```python
from main_experiments import main_iterative_experiment

# Run DiZiNER on CrossNER-AI
results = main_iterative_experiment(
    benchmark="crossner_ai",
    num_models=8,
    max_iterations=5,
    supervisor_model_name="gpt-5-mini-2025-08-07",
    llm_infer_by_openrouter=True,
    max_common_instructions=5,
    max_patterns=10,
    max_model_specific_instructions=3,
    hotspot_percentile=80,
    coalition_cutoff=0.5
)
```

### Running on Other Benchmarks

```python
# ACE2005
results = main_iterative_experiment(
    benchmark="ACE05",
    num_models=8,
    max_iterations=5,
    supervisor_model_name="gpt-5-mini-2025-08-07",
    llm_infer_by_openrouter=True
)

# CoNLL2003
results = main_iterative_experiment(
    benchmark="conllpp",
    num_models=8,
    max_iterations=5,
    supervisor_model_name="gpt-5-mini-2025-08-07",
    llm_infer_by_openrouter=True
)
```

---

## Implementation Architecture

### Core Pipeline

**Experiment Orchestration** (`main_experiments.py`)
- Coordinates multi-iteration workflow: annotation → analysis → supervision
- Manages experiment configuration and result aggregation

**Annotation** (`annotation_runner.py`, `parallel_annotation.py`, `base_annotator.py`)
- Parallel model processing via ThreadPoolExecutor
- Supervisor instruction integration for iterative refinement
- Result caching and OpenRouter optimization

**Analysis** (`disagreement_analysis_in_pipeline.py`, `agreement_analysis_test.py`, `error_analysis.py`)
- Hotspot identification using disagreement metrics (Dconf, Dtype, Ubnd)
- Inter-annotator agreement calculation (Cohen's/Fleiss' Kappa)
- Error pattern categorization and documentation

**Supervision** (`supervisor_implementation.py`, `base_supervisor.py`)
- 4-phase instruction refinement:
  - Phase 1: Disagreement pattern extraction
  - Phase 2: Model-specific error diagnosis
  - Phase 3: Instruction hierarchization
  - Phase 4: Guideline organization and output

**Utilities** (`utils_experiments.py`, `utils_annotator.py`, `lexical_diversity_grouping.py`)
- Dataset grouping via K-means clustering on sentence embeddings
- BIO-entity conversion and metric calculation
- Model selection and result management

### Workflow

```
1. Dataset Preparation
   └─→ lexical_diversity_grouping.py
       └─→ Creates diverse sample groups

2. Baseline Annotation (Iteration 0)
   └─→ main_experiments.py
       ├─→ parallel_annotation.py (concurrent processing)
       │   └─→ annotation_runner.py
       │       └─→ base_annotator.py
       └─→ Saves model results

3. Analysis Phase
   └─→ run_analysis_pipeline()
       ├─→ agreement_analysis_test.py
       ├─→ disagreement_analysis_in_pipeline.py
       └─→ error_analysis.py

4. Supervision Phase (4-phase refinement)
   └─→ supervisor_implementation.py
       └─→ base_supervisor.py
           └─→ Generates enhanced guidelines

5. Next Iteration (1, 2, ...)
   └─→ Repeats steps 2-4 with updated guidelines
```

### Paper-Implementation Mapping

| Paper Component | Implementation Module |
|-----------------|----------------------|
| Independent Cross-Annotation | `parallel_annotation.py`, `annotation_runner.py` |
| Disagreement Analysis | `disagreement_analysis_in_pipeline.py` |
| Hotspot Identification | Disagreement metrics (Dconf, Dtype, Ubnd) |
| 4-Phase Supervision | `base_supervisor.py` (Phase 1-4) |
| Model Weight Computation | Pairwise strict span F1 calculation |
| Elite Set Selection | Top 50% cumulative weight threshold |
| Instruction Refinement | `supervisor_implementation.py` |

---

## Configuration

### Experimental Settings

**Annotator Models** (8 heterogeneous LLMs via OpenRouter):
- mistral-small3.2:24b
- gpt-oss:20b
- phi4:14b
- qwen3:14b
- gemma3:12b
- deepseek-r1:8b
- llama3.1:8b
- nemotron-nano:8b

**Supervisor Model**: GPT-4o mini (OpenAI API)

**Default Configuration**:
```python
{
    'iteration_document_set': 25,      # Samples per iteration
    'max_iterations': 5,               # Maximum refinement cycles
    'hotspot_threshold': 0.8,          # Top 20% disagreement tokens
    'max_common_instructions': 5,      # Common guideline limit
    'max_patterns': 10,                # Pattern extraction limit
    'max_model_specific_instructions': 3,  # Per-model guideline limit
    'coalition_cutoff': 0.5            # Elite set threshold
}
```

### Tuning Configurations

Three parameter sets used across benchmarks:

| Config | max_common | max_patterns | max_model_spec | limit_changes | max_ratio |
|--------|------------|--------------|----------------|---------------|-----------|
| Stable | 3 | 5 | 2 | True | 0.10 |
| Relaxed | 5 | 8 | 3 | False | 0.20 |
| Aggressive | 10 | 20 | 10 | False | 0.50 |

### Advanced Options

```python
results = main_iterative_experiment(
    benchmark="conllpp",
    num_models=8,
    max_iterations=5,
    
    # Model selection
    drop_worst_annr=True,              # Remove lowest-agreement model
    
    # Supervision control
    supervised_by_gold_standard=False,  # Use disagreement (not gold labels)
    skip_final_goal_update=False,      # Update task goal each iteration
    
    # Instruction limits
    limit_instruction_changes=True,
    max_change_ratio=0.2,
    
    # Analysis thresholds
    hotspot_percentile=80,
    coalition_cutoff=0.5
)
```

---

## Output Structure

```
experiment_results/
├── {benchmark}/
│   ├── models{N}/
│   │   ├── {supervisor_model}/
│   │   │   ├── g{groups}_s{size}_grp{idx}_iter{N}/
│   │   │   │   ├── model_results/
│   │   │   │   │   └── {model_name}.json
│   │   │   │   ├── agreement_analysis/
│   │   │   │   ├── disagreement_analysis/
│   │   │   │   │   └── hotspot_docs/
│   │   │   │   │       └── hotspot_disagreement_analysis.md
│   │   │   │   ├── error_analysis/
│   │   │   │   ├── supervisor_results/
│   │   │   │   │   ├── phase1_disagreement_pattern_analysis.json
│   │   │   │   │   ├── phase2_non_elite_model_analysis.json
│   │   │   │   │   ├── phase3_guideline_integration.json
│   │   │   │   │   └── phase4_enhanced_guidelines.json
│   │   │   │   ├── prompts/
│   │   │   │   │   └── {model}_iter{N}_prompt_template.txt
│   │   │   │   ├── combined_results.json
│   │   │   │   └── experiment_config.json
```

---

## Cost Analysis

Average cost per benchmark (5 iterations):
- **Inference**: $1.90 per iteration
- **Supervision**: $0.77 per iteration
- **Total per iteration**: $2.67
- **Total per benchmark**: ~$13.35

Costs based on OpenRouter and OpenAI API pricing as of August 2025.

---

## Acknowledgments

This work is submitted to ARR (ACL Rolling Review) for consideration at ACL 2025.
