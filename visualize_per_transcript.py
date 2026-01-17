"""
Per-transcript visualization script for therapy evaluation results
Creates graphs similar to therapy_memincluded.ipynb cells 19
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Configuration
RESULTS_DIR = "output_therapy_memnotincluded/results"
OUTPUT_DIR = "output_therapy_memnotincluded/images"

def load_all_results(results_dir):
    """Load all JSON result files from the directory."""
    results = []
    json_files = glob.glob(os.path.join(results_dir, "*_results.json"))

    for filepath in sorted(json_files):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['source_file'] = os.path.basename(filepath)
            results.append(data)

    return results

def create_per_transcript_visualization(result, output_dir):
    """Create 3-panel visualization for a single transcript."""
    filename = result["filename"].replace(".txt", "")

    # Extract data
    cbt_scores = [r["score"] for r in result["cbt_adherence_results"]]
    persona_scores = [r["score"] for r in result["persona_consistency_results"]]
    turns = [r["turn_number"] for r in result["cbt_adherence_results"]]

    # Memory counts from snapshots if available, otherwise use placeholder
    if "memory_snapshots" in result and result["memory_snapshots"]:
        memory_counts = [s.get("memory_count", 0) for s in result["memory_snapshots"]]
    else:
        # For memnotincluded, we don't have memory snapshots per turn
        # Use a placeholder showing memories were not used
        memory_counts = [0] * len(turns)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Calculate rolling average window
    window = min(10, len(cbt_scores) // 5) if len(cbt_scores) > 10 else 3

    # =========================================================================
    # Panel 1: CBT Adherence
    # =========================================================================
    ax1 = axes[0]
    ax1.plot(turns, cbt_scores, 'b-', linewidth=1, alpha=0.5, label='CBT Score')
    ax1.scatter(turns, cbt_scores, c='blue', s=15, alpha=0.6)

    # Rolling average
    if len(cbt_scores) >= window:
        rolling_avg = np.convolve(cbt_scores, np.ones(window)/window, mode='valid')
        rolling_turns = turns[window-1:]
        ax1.plot(rolling_turns, rolling_avg, 'b-', linewidth=2.5,
                label=f'Rolling Avg ({window}-turn)')

    # Threshold lines
    ax1.axhline(y=7, color='orange', linestyle='--', linewidth=1.5, label='Good Threshold (7)')
    ax1.axhline(y=5, color='red', linestyle='--', linewidth=1.5, label='Decay Warning (5)')

    # Stats annotation
    mean_cbt = np.mean(cbt_scores)
    ax1.axhline(y=mean_cbt, color='blue', linestyle=':', alpha=0.5)
    ax1.text(turns[-1], mean_cbt + 0.3, f'Mean: {mean_cbt:.2f}',
            fontsize=10, color='blue', ha='right')

    ax1.set_xlabel('Turn Number', fontsize=11)
    ax1.set_ylabel('CBT Adherence Score (1-10)', fontsize=11)
    ax1.set_title(f'{filename} - Part B: Instruction Decay (CBT Adherence)',
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 11)
    ax1.set_xlim(min(turns)-5, max(turns)+5)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 2: Persona Consistency
    # =========================================================================
    ax2 = axes[1]
    ax2.plot(turns, persona_scores, 'g-', linewidth=1, alpha=0.5, label='Persona Score')
    ax2.scatter(turns, persona_scores, c='green', s=15, alpha=0.6)

    # Rolling average
    if len(persona_scores) >= window:
        rolling_avg2 = np.convolve(persona_scores, np.ones(window)/window, mode='valid')
        ax2.plot(rolling_turns, rolling_avg2, 'g-', linewidth=2.5,
                label=f'Rolling Avg ({window}-turn)')

    # Threshold lines
    ax2.axhline(y=7, color='orange', linestyle='--', linewidth=1.5, label='Good Threshold (7)')
    ax2.axhline(y=5, color='red', linestyle='--', linewidth=1.5, label='Decay Warning (5)')

    # Stats annotation
    mean_persona = np.mean(persona_scores)
    ax2.axhline(y=mean_persona, color='green', linestyle=':', alpha=0.5)
    ax2.text(turns[-1], mean_persona + 0.3, f'Mean: {mean_persona:.2f}',
            fontsize=10, color='green', ha='right')

    ax2.set_xlabel('Turn Number', fontsize=11)
    ax2.set_ylabel('Persona Consistency Score (1-10)', fontsize=11)
    ax2.set_title(f'{filename} - Part C: Persona Consistency (Boundary Dissolution)',
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 11)
    ax2.set_xlim(min(turns)-5, max(turns)+5)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 3: Memory Stats (or combined score view for memnotincluded)
    # =========================================================================
    ax3 = axes[2]

    # Check if memory counts show actual growth (not just constant)
    memory_varies = len(set(memory_counts)) > 1 if memory_counts else False

    if memory_varies and max(memory_counts) > 0:
        # Has meaningful memory data - show memory growth
        ax3.plot(turns, memory_counts, 'm-', linewidth=2, label='Cumulative Memories')
        ax3.fill_between(turns, 0, memory_counts, alpha=0.3, color='purple')
        ax3.set_ylabel('Number of Stored Memories', fontsize=11)
        ax3.set_title(f'{filename} - Memory Accumulation Over Session',
                     fontsize=13, fontweight='bold')
    else:
        # No memory data - show combined CBT + Persona view
        ax3.plot(turns, cbt_scores, 'b-', linewidth=1.5, alpha=0.7, label='CBT Adherence')
        ax3.plot(turns, persona_scores, 'g-', linewidth=1.5, alpha=0.7, label='Persona Consistency')

        # Add difference/gap visualization
        diff = np.array(persona_scores) - np.array(cbt_scores)
        ax3.fill_between(turns, cbt_scores, persona_scores, alpha=0.2,
                        color='gray', label='Score Gap')

        ax3.axhline(y=7, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        ax3.axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.7)

        ax3.set_ylabel('Score (1-10)', fontsize=11)
        ax3.set_title(f'{filename} - CBT vs Persona Comparison (No Memory Used)',
                     fontsize=13, fontweight='bold')
        ax3.set_ylim(0, 11)

    ax3.set_xlabel('Turn Number', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(min(turns)-5, max(turns)+5)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, f"{filename}_evaluation.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path

def create_aggregate_visualization(all_results, output_dir):
    """Create aggregate visualization across all transcripts."""

    # Aggregate all scores
    all_cbt_scores = []
    all_persona_scores = []

    for result in all_results:
        all_cbt_scores.extend([r["score"] for r in result["cbt_adherence_results"]])
        all_persona_scores.extend([r["score"] for r in result["persona_consistency_results"]])

    turn_numbers = list(range(1, len(all_cbt_scores) + 1))

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    window = min(20, len(all_cbt_scores) // 5) if len(all_cbt_scores) > 20 else 5

    # =========================================================================
    # Panel 1: CBT Adherence Aggregate
    # =========================================================================
    ax1 = axes[0]
    ax1.plot(turn_numbers, all_cbt_scores, 'b-', linewidth=0.8, alpha=0.4, label='CBT Score')

    # Rolling average
    rolling_avg = np.convolve(all_cbt_scores, np.ones(window)/window, mode='valid')
    ax1.plot(range(window//2 + 1, len(rolling_avg) + window//2 + 1), rolling_avg,
            'b-', linewidth=2.5, label=f'Rolling Avg ({window})')

    ax1.axhline(y=7, color='orange', linestyle='--', label='Good Threshold (7)')
    ax1.axhline(y=5, color='red', linestyle='--', label='Decay Warning (5)')

    mean_cbt = np.mean(all_cbt_scores)
    ax1.axhline(y=mean_cbt, color='blue', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Evaluation Number (across all transcripts)', fontsize=11)
    ax1.set_ylabel('CBT Adherence Score (1-10)', fontsize=11)
    ax1.set_title('Part B: Instruction Decay - CBT Adherence Over Time (All Transcripts)',
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.set_ylim(0, 11)
    ax1.grid(True, alpha=0.3)

    # Add stats box
    stats_text = f'Mean: {mean_cbt:.2f}\nStd: {np.std(all_cbt_scores):.2f}\nN: {len(all_cbt_scores)}'
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # =========================================================================
    # Panel 2: Persona Consistency Aggregate
    # =========================================================================
    ax2 = axes[1]
    ax2.plot(turn_numbers, all_persona_scores, 'g-', linewidth=0.8, alpha=0.4, label='Persona Score')

    rolling_avg2 = np.convolve(all_persona_scores, np.ones(window)/window, mode='valid')
    ax2.plot(range(window//2 + 1, len(rolling_avg2) + window//2 + 1), rolling_avg2,
            'g-', linewidth=2.5, label=f'Rolling Avg ({window})')

    ax2.axhline(y=7, color='orange', linestyle='--', label='Good Threshold (7)')
    ax2.axhline(y=5, color='red', linestyle='--', label='Decay Warning (5)')

    mean_persona = np.mean(all_persona_scores)
    ax2.axhline(y=mean_persona, color='green', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Evaluation Number (across all transcripts)', fontsize=11)
    ax2.set_ylabel('Persona Consistency Score (1-10)', fontsize=11)
    ax2.set_title('Part C: Persona Consistency - Professional Tone Over Time (All Transcripts)',
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.set_ylim(0, 11)
    ax2.grid(True, alpha=0.3)

    stats_text2 = f'Mean: {mean_persona:.2f}\nStd: {np.std(all_persona_scores):.2f}\nN: {len(all_persona_scores)}'
    ax2.text(0.98, 0.98, stats_text2, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # =========================================================================
    # Panel 3: Transcript boundaries with both scores
    # =========================================================================
    ax3 = axes[2]

    # Plot with vertical lines at transcript boundaries
    cumulative_count = 0
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for idx, result in enumerate(all_results):
        n_turns = len(result["cbt_adherence_results"])
        start = cumulative_count
        end = cumulative_count + n_turns

        cbt = [r["score"] for r in result["cbt_adherence_results"]]
        persona = [r["score"] for r in result["persona_consistency_results"]]
        x = list(range(start + 1, end + 1))

        label = result["filename"].replace(".txt", "")[-4:]
        ax3.plot(x, cbt, '-', color=colors[idx], linewidth=1.5, alpha=0.8,
                label=f'{label} CBT')
        ax3.plot(x, persona, '--', color=colors[idx], linewidth=1.5, alpha=0.8)

        # Vertical line at transcript boundary
        if idx > 0:
            ax3.axvline(x=start, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        cumulative_count = end

    ax3.axhline(y=7, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax3.axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.7)

    ax3.set_xlabel('Evaluation Number', fontsize=11)
    ax3.set_ylabel('Score (1-10)', fontsize=11)
    ax3.set_title('All Transcripts: CBT (solid) vs Persona (dashed) by Transcript',
                 fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8, ncol=len(all_results))
    ax3.set_ylim(0, 11)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, "alignment_overview.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading results...")
    results = load_all_results(RESULTS_DIR)
    print(f"Loaded {len(results)} result files")

    if not results:
        print("No results found!")
        return

    print("\n" + "=" * 60)
    print("Generating per-transcript visualizations...")
    print("=" * 60)

    # Generate per-transcript visualizations
    for result in results:
        create_per_transcript_visualization(result, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Generating aggregate visualization...")
    print("=" * 60)

    # Generate aggregate visualization
    create_aggregate_visualization(results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")

if __name__ == "__main__":
    main()
