"""
Test Inference Preparation Script
Analyzes test_experiment_prompts_and_results folder to:
1. Count total txt files
2. Identify and remove duplicates based on experiment conditions
3. Create final-test-prompts with deduplicated prompts
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import re
from dataclasses import dataclass

@dataclass
class PromptInfo:
    """Information about a prompt file"""
    benchmark: str
    model_count: str
    supervisor: str
    experiment_condition: str
    annotator_model: str
    file_path: Path
    search_condition: str
    find_by_suffix: str
    
    def get_unique_key(self) -> str:
        """Generate unique key for deduplication (excluding find_by_suffix)"""
        # Check if this is iter0 condition
        if self.is_iter0():
            # For iter0, only benchmark and annotator_model matter
            return f"{self.benchmark}_{self.annotator_model}_iter0"
        else:
            # For non-iter0, use all conditions
            return f"{self.benchmark}_{self.model_count}_{self.supervisor}_{self.experiment_condition}_{self.annotator_model}"
    
    def is_iter0(self) -> bool:
        """Check if this is an iter0 condition"""
        return "_iter0_" in self.experiment_condition or self.experiment_condition.endswith("_iter0")
    
    def get_target_path(self, base_path: Path) -> Path:
        """Generate target path for final prompts"""
        if self.is_iter0():
            # For iter0: benchmark/iter0_condition/prompts/model_prompt_template.txt
            # Clean the experiment condition to only include up to iter0
            clean_condition = self.get_clean_iter0_condition()
            return base_path / self.benchmark / clean_condition / "prompts" / f"{self.annotator_model}_prompt_template.txt"
        else:
            # For non-iter0: benchmark/model_count/supervisor/experiment_condition/prompts/model_prompt_template.txt
            return base_path / self.benchmark / self.model_count / self.supervisor / self.experiment_condition / "prompts" / f"{self.annotator_model}_prompt_template.txt"
    
    def get_clean_iter0_condition(self) -> str:
        """Extract clean iter0 condition (e.g., g20_s25_grp0_iter0)"""
        if not self.is_iter0():
            return self.experiment_condition
        
        # Find the position of iter0
        parts = self.experiment_condition.split('_')
        clean_parts = []
        
        for part in parts:
            clean_parts.append(part)
            if part == 'iter0':
                break
        
        return '_'.join(clean_parts)

class TestInferencePreparator:
    def __init__(self, source_dir: str = "test_experiment_prompts_and_results_dff_numbers", 
                 target_dir: str = "final-test-prompts"):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.prompt_infos: List[PromptInfo] = []
        
    def analyze_source_directory(self) -> Dict:
        """Analyze source directory and extract all prompt information"""
        print(f"Analyzing source directory: {self.source_dir}")
        
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        
        total_files = 0
        analysis_results = {
            'total_files': 0,
            'search_conditions': set(),
            'benchmarks': set(),
            'model_counts': set(),
            'supervisors': set(),
            'annotator_models': set()
        }
        
        # Traverse all search condition folders
        for search_condition_dir in self.source_dir.iterdir():
            if not search_condition_dir.is_dir():
                continue
                
            search_condition = search_condition_dir.name
            analysis_results['search_conditions'].add(search_condition)
            
            # Traverse benchmark folders
            for benchmark_dir in search_condition_dir.iterdir():
                if not benchmark_dir.is_dir():
                    continue
                    
                benchmark = benchmark_dir.name
                analysis_results['benchmarks'].add(benchmark)
                
                # Traverse model count folders
                for model_count_dir in benchmark_dir.iterdir():
                    if not model_count_dir.is_dir() or not model_count_dir.name.startswith('models'):
                        continue
                        
                    model_count = model_count_dir.name
                    analysis_results['model_counts'].add(model_count)
                    
                    # Traverse supervisor folders
                    for supervisor_dir in model_count_dir.iterdir():
                        if not supervisor_dir.is_dir():
                            continue
                            
                        supervisor = supervisor_dir.name
                        analysis_results['supervisors'].add(supervisor)
                        
                        # Traverse experiment condition folders
                        for exp_condition_dir in supervisor_dir.iterdir():
                            if not exp_condition_dir.is_dir():
                                continue
                                
                            exp_condition_full = exp_condition_dir.name
                            
                            # Extract find_by_suffix and clean experiment condition
                            find_by_suffix = ""
                            if exp_condition_full.endswith('_fdbygst'):
                                find_by_suffix = "_fdbygst"
                                exp_condition = exp_condition_full[:-len(find_by_suffix)]
                            elif exp_condition_full.endswith('_fdbypma'):
                                find_by_suffix = "_fdbypma"
                                exp_condition = exp_condition_full[:-len(find_by_suffix)]
                            else:
                                exp_condition = exp_condition_full
                            
                            # Look for prompts folder
                            prompts_dir = exp_condition_dir / "prompts"
                            if prompts_dir.exists():
                                # Find all txt files
                                for txt_file in prompts_dir.glob("*_prompt_template.txt"):
                                    total_files += 1
                                    
                                    # Extract annotator model name (part before _iter)
                                    file_stem = txt_file.stem
                                    iter_match = re.search(r'(.+?)_iter\d+_prompt_template', file_stem)
                                    if iter_match:
                                        annotator_model = iter_match.group(1)
                                        analysis_results['annotator_models'].add(annotator_model)
                                        
                                        prompt_info = PromptInfo(
                                            benchmark=benchmark,
                                            model_count=model_count,
                                            supervisor=supervisor,
                                            experiment_condition=exp_condition,
                                            annotator_model=annotator_model,
                                            file_path=txt_file,
                                            search_condition=search_condition,
                                            find_by_suffix=find_by_suffix
                                        )
                                        self.prompt_infos.append(prompt_info)
        
        analysis_results['total_files'] = total_files
        return analysis_results
    
    def identify_duplicates(self) -> Tuple[Dict[str, List[PromptInfo]], int]:
        """Identify duplicate prompts based on unique key"""
        unique_groups = defaultdict(list)
        
        for prompt_info in self.prompt_infos:
            unique_key = prompt_info.get_unique_key()
            unique_groups[unique_key].append(prompt_info)
        
        # Count duplicates
        duplicates = {k: v for k, v in unique_groups.items() if len(v) > 1}
        unique_count = len(unique_groups)
        
        return unique_groups, unique_count
    
    def create_final_prompts(self, unique_groups: Dict[str, List[PromptInfo]]) -> int:
        """Create final-test-prompts directory with deduplicated prompts"""
        if self.target_dir.exists():
            print(f"Removing existing target directory: {self.target_dir}")
            shutil.rmtree(self.target_dir)
        
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        copied_files = 0
        
        for unique_key, prompt_list in unique_groups.items():
            # Use the first prompt from each group (arbitrary choice for duplicates)
            selected_prompt = prompt_list[0]
            
            # Create target path
            target_path = selected_prompt.get_target_path(self.target_dir)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(selected_prompt.file_path, target_path)
            copied_files += 1
            
            if len(prompt_list) > 1:
                print(f"  Duplicate resolved: {unique_key}")
                for i, prompt in enumerate(prompt_list):
                    status = "SELECTED" if i == 0 else "SKIPPED"
                    print(f"    [{status}] {prompt.search_condition}{prompt.find_by_suffix}: {prompt.file_path}")
            
            if copied_files == 0:
                print(f"    Warning: No prompt found for {selected_prompt.annotator_model}")
            else:
                if selected_prompt.is_iter0():
                    print(f"    Copied iter0 prompt: {selected_prompt.benchmark}/{selected_prompt.get_clean_iter0_condition()}/")
                else:
                    print(f"    Copied prompt: {selected_prompt.benchmark}/{selected_prompt.model_count}/{selected_prompt.supervisor}/...")
        
        return copied_files
    
    def verify_final_files(self, expected_count: int) -> bool:
        """Verify that final-test-prompts contains expected number of txt files"""
        if not self.target_dir.exists():
            print(f"  ✗ Target directory does not exist: {self.target_dir}")
            return False
        
        # Count all txt files in final-test-prompts
        actual_files = list(self.target_dir.glob("**/*_prompt_template.txt"))
        actual_count = len(actual_files)
        
        print(f"\nFinal Verification:")
        print(f"  Expected unique conditions: {expected_count}")
        print(f"  Actual txt files in {self.target_dir}: {actual_count}")
        
        if actual_count == expected_count:
            print(f"  ✓ Verification passed: file count matches unique conditions")
            return True
        else:
            print(f"  ✗ Verification failed: {actual_count} files ≠ {expected_count} expected")
            # Show some example files for debugging
            if actual_files:
                print(f"  Example files found:")
                for i, file_path in enumerate(actual_files[:3]):
                    relative_path = file_path.relative_to(self.target_dir)
                    print(f"    {i+1}. {relative_path}")
                if len(actual_files) > 3:
                    print(f"    ... and {len(actual_files) - 3} more files")
            return False

    def generate_report(self, analysis_results: Dict, unique_groups: Dict, unique_count: int, copied_files: int):
        """Generate comprehensive report"""
        total_files = analysis_results['total_files']
        duplicate_count = total_files - unique_count
        
        # Count iter0 conditions
        iter0_count = sum(1 for prompt_list in unique_groups.values() 
                         if prompt_list[0].is_iter0())
        non_iter0_count = unique_count - iter0_count
        
        print(f"\n{'='*80}")
        print("TEST INFERENCE PREPARATION REPORT")
        print(f"{'='*80}")
        
        print(f"\nSource Directory Analysis:")
        print(f"  Total txt files found: {total_files}")
        print(f"  Unique conditions: {unique_count}")
        print(f"    - iter0 conditions: {iter0_count}")
        print(f"    - non-iter0 conditions: {non_iter0_count}")
        print(f"  Duplicate files: {duplicate_count}")
        
        print(f"\nCondition Breakdown:")
        print(f"  Search conditions: {len(analysis_results['search_conditions'])}")
        for condition in sorted(analysis_results['search_conditions']):
            print(f"    - {condition}")
        
        print(f"  Benchmarks: {len(analysis_results['benchmarks'])}")
        print(f"  Model counts: {sorted(analysis_results['model_counts'])}")
        print(f"  Supervisors: {len(analysis_results['supervisors'])}")
        print(f"  Annotator models: {len(analysis_results['annotator_models'])}")
        
        if duplicate_count > 0:
            print(f"\nDuplicate Details:")
            duplicates = {k: v for k, v in unique_groups.items() if len(v) > 1}
            iter0_duplicates = {k: v for k, v in duplicates.items() 
                              if v[0].is_iter0()}
            non_iter0_duplicates = {k: v for k, v in duplicates.items() 
                                  if not v[0].is_iter0()}
            
            if iter0_duplicates:
                print(f"  iter0 duplicate groups: {len(iter0_duplicates)}")
                for unique_key, prompt_list in list(iter0_duplicates.items())[:3]:  # Show first 3
                    print(f"    Key: {unique_key} ({len(prompt_list)} duplicates)")
                if len(iter0_duplicates) > 3:
                    print(f"    ... and {len(iter0_duplicates) - 3} more iter0 duplicate groups")
            
            if non_iter0_duplicates:
                print(f"  non-iter0 duplicate groups: {len(non_iter0_duplicates)}")
                for unique_key, prompt_list in list(non_iter0_duplicates.items())[:3]:  # Show first 3
                    print(f"    Key: {unique_key} ({len(prompt_list)} duplicates)")
                if len(non_iter0_duplicates) > 3:
                    print(f"    ... and {len(non_iter0_duplicates) - 3} more non-iter0 duplicate groups")
        
        print(f"\nCopy Process Results:")
        print(f"  Files copied to final-test-prompts: {copied_files}")
        print(f"  Target directory: {self.target_dir}")
        
        # Verify counts match
        if copied_files == unique_count:
            print(f"  ✓ Copy verification passed: copied files = unique conditions")
        else:
            print(f"  ✗ Copy verification failed: {copied_files} copied ≠ {unique_count} unique")
    
    def save_detailed_report(self, analysis_results: Dict, unique_groups: Dict):
        """Save detailed report as JSON"""
        report_data = {
            'analysis_summary': {
                'total_files': analysis_results['total_files'],
                'unique_conditions': len(unique_groups),
                'duplicate_count': analysis_results['total_files'] - len(unique_groups)
            },
            'search_conditions': list(analysis_results['search_conditions']),
            'benchmarks': list(analysis_results['benchmarks']),
            'model_counts': list(analysis_results['model_counts']),
            'supervisors': list(analysis_results['supervisors']),
            'annotator_models': list(analysis_results['annotator_models']),
            'unique_conditions': {}
        }
        
        # Add detailed info for each unique condition
        for unique_key, prompt_list in unique_groups.items():
            report_data['unique_conditions'][unique_key] = {
                'count': len(prompt_list),
                'is_iter0': prompt_list[0].is_iter0(),
                'selected_file': str(prompt_list[0].file_path),
                'all_sources': [
                    {
                        'search_condition': p.search_condition,
                        'find_by_suffix': p.find_by_suffix,
                        'file_path': str(p.file_path),
                        'is_iter0': p.is_iter0()
                    }
                    for p in prompt_list
                ]
            }
        
        report_file = self.target_dir / "preparation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"  Detailed report saved: {report_file}")
    
    def run_preparation(self):
        """Run complete preparation process"""
        print(f"Starting test inference preparation...")
        
        # Step 1: Analyze source directory
        analysis_results = self.analyze_source_directory()
        
        # Step 2: Identify duplicates
        unique_groups, unique_count = self.identify_duplicates()
        
        # Step 3: Create final prompts
        copied_files = self.create_final_prompts(unique_groups)
        
        # Step 4: Generate report
        self.generate_report(analysis_results, unique_groups, unique_count, copied_files)
        
        # Step 5: Verify final file count
        verification_passed = self.verify_final_files(unique_count)
        
        # Step 6: Save detailed report
        self.save_detailed_report(analysis_results, unique_groups)
        
        return {
            'total_files': analysis_results['total_files'],
            'unique_count': unique_count,
            'copied_files': copied_files,
            'verification_passed': verification_passed,
            'target_dir': str(self.target_dir)
        }

def main():
    """Main execution function"""
    preparator = TestInferencePreparator()
    results = preparator.run_preparation()
    
    print(f"\nPreparation completed successfully!")
    print(f"Ready for test inference with {results['copied_files']} unique prompt configurations.")

if __name__ == "__main__":
    main()