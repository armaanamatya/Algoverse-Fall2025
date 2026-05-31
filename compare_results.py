#!/usr/bin/env python3
"""
Compare results between original and optimized notebook outputs.

This script helps verify that the optimized notebooks produce
consistent results with the original sequential processing.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import statistics


def load_results(directory: Path, pattern: str) -> List[Dict[str, Any]]:
    """Load all JSON result files matching pattern."""
    results = []
    for file in sorted(directory.glob(pattern)):
        with open(file, 'r', encoding='utf-8') as f:
            results.append(json.load(f))
    return results


def calculate_aggregate_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate statistics across all result files."""
    all_cbt_scores = []
    all_persona_scores = []
    
    for result in results:
        all_cbt_scores.extend([r['score'] for r in result['cbt_adherence_results']])
        all_persona_scores.extend([r['score'] for r in result['persona_consistency_results']])
    
    return {
        'total_files': len(results),
        'total_turns': sum(r['counselor_turns_evaluated'] for r in results),
        'cbt_adherence': {
            'mean': statistics.mean(all_cbt_scores) if all_cbt_scores else 0,
            'median': statistics.median(all_cbt_scores) if all_cbt_scores else 0,
            'stdev': statistics.stdev(all_cbt_scores) if len(all_cbt_scores) > 1 else 0,
            'min': min(all_cbt_scores) if all_cbt_scores else 0,
            'max': max(all_cbt_scores) if all_cbt_scores else 0,
        },
        'persona_consistency': {
            'mean': statistics.mean(all_persona_scores) if all_persona_scores else 0,
            'median': statistics.median(all_persona_scores) if all_persona_scores else 0,
            'stdev': statistics.stdev(all_persona_scores) if len(all_persona_scores) > 1 else 0,
            'min': min(all_persona_scores) if all_persona_scores else 0,
            'max': max(all_persona_scores) if all_persona_scores else 0,
        }
    }


def compare_results(original_stats: Dict, optimized_stats: Dict) -> None:
    """Compare and display differences between original and optimized results."""
    print("=" * 80)
    print("RESULTS COMPARISON: Original vs Optimized")
    print("=" * 80)
    
    print(f"\n📊 Dataset Size:")
    print(f"  Original:  {original_stats['total_files']} files, {original_stats['total_turns']} turns")
    print(f"  Optimized: {optimized_stats['total_files']} files, {optimized_stats['total_turns']} turns")
    
    if original_stats['total_turns'] != optimized_stats['total_turns']:
        print("  ⚠️  WARNING: Different number of turns evaluated!")
    else:
        print("  ✅ Same number of turns evaluated")
    
    print(f"\n📈 CBT Adherence Scores:")
    print(f"  {'Metric':<12} {'Original':<12} {'Optimized':<12} {'Difference':<12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for metric in ['mean', 'median', 'stdev', 'min', 'max']:
        orig_val = original_stats['cbt_adherence'][metric]
        opt_val = optimized_stats['cbt_adherence'][metric]
        diff = opt_val - orig_val
        
        print(f"  {metric.capitalize():<12} {orig_val:<12.3f} {opt_val:<12.3f} {diff:+.3f}")
    
    print(f"\n📈 Persona Consistency Scores:")
    print(f"  {'Metric':<12} {'Original':<12} {'Optimized':<12} {'Difference':<12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for metric in ['mean', 'median', 'stdev', 'min', 'max']:
        orig_val = original_stats['persona_consistency'][metric]
        opt_val = optimized_stats['persona_consistency'][metric]
        diff = opt_val - orig_val
        
        print(f"  {metric.capitalize():<12} {orig_val:<12.3f} {opt_val:<12.3f} {diff:+.3f}")
    
    # Check if results are "close enough" (within 0.5 points on average)
    cbt_diff = abs(original_stats['cbt_adherence']['mean'] - optimized_stats['cbt_adherence']['mean'])
    persona_diff = abs(original_stats['persona_consistency']['mean'] - optimized_stats['persona_consistency']['mean'])
    
    print(f"\n🔍 Validation:")
    if cbt_diff < 0.5 and persona_diff < 0.5:
        print("  ✅ Results are consistent (mean difference < 0.5)")
    elif cbt_diff < 1.0 and persona_diff < 1.0:
        print("  ⚠️  Results are mostly consistent (mean difference < 1.0)")
    else:
        print("  ❌ Results show significant differences (mean difference >= 1.0)")
        print("     This may be due to non-deterministic LLM responses")
    
    print("=" * 80)


def main():
    """Main comparison function."""
    # Define directories
    original_dir = Path("./evaluation_results")
    optimized_dir = Path("./evaluation_results_optimized")
    
    # Check if directories exist
    if not original_dir.exists():
        print(f"❌ Original results directory not found: {original_dir}")
        print("   Run the original notebooks first to generate baseline results.")
        return
    
    if not optimized_dir.exists():
        print(f"❌ Optimized results directory not found: {optimized_dir}")
        print("   Run the optimized notebooks first to generate results.")
        return
    
    # Load results
    print("Loading results...")
    
    # For memory-included version
    original_results = load_results(original_dir, "*_eval.json")
    optimized_results = load_results(optimized_dir, "*_eval_optimized.json")
    
    if not original_results:
        print(f"❌ No original results found in {original_dir}")
        return
    
    if not optimized_results:
        print(f"❌ No optimized results found in {optimized_dir}")
        return
    
    print(f"✅ Loaded {len(original_results)} original result files")
    print(f"✅ Loaded {len(optimized_results)} optimized result files")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    original_stats = calculate_aggregate_stats(original_results)
    optimized_stats = calculate_aggregate_stats(optimized_results)
    
    # Compare results
    compare_results(original_stats, optimized_stats)
    
    # Also check memory-not-included version if available
    optimized_nomem_results = load_results(optimized_dir, "*_eval_optimized_nomem.json")
    if optimized_nomem_results:
        print("\n\n")
        print("=" * 80)
        print("BONUS: Memory-NOT-Included Version Statistics")
        print("=" * 80)
        nomem_stats = calculate_aggregate_stats(optimized_nomem_results)
        
        print(f"\n📊 Dataset: {nomem_stats['total_files']} files, {nomem_stats['total_turns']} turns")
        print(f"\n📈 CBT Adherence: Mean = {nomem_stats['cbt_adherence']['mean']:.3f}")
        print(f"📈 Persona Consistency: Mean = {nomem_stats['persona_consistency']['mean']:.3f}")


if __name__ == "__main__":
    main()
