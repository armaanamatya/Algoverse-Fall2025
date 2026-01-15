"""
Visualization script for therapy evaluation results
Generates charts from the evaluation JSON files
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Configuration
RESULTS_DIR = "output_therapy_memnotincluded/results"
OUTPUT_DIR = "output_therapy_memnotincluded/visualizations"

def load_all_results(results_dir):
    """Load all JSON result files from the directory."""
    results = []
    json_files = glob.glob(os.path.join(results_dir, "*_results.json"))

    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['source_file'] = os.path.basename(filepath)
            results.append(data)

    return results

def extract_scores(results):
    """Extract CBT and Persona scores from all results."""
    all_cbt_scores = []
    all_persona_scores = []
    scores_by_transcript = {}

    for result in results:
        transcript_id = result.get('filename', result.get('source_file', 'unknown')).replace('.txt', '').replace('_results.json', '')

        cbt_scores = [r['score'] for r in result.get('cbt_adherence_results', []) if 'score' in r]
        persona_scores = [r['score'] for r in result.get('persona_consistency_results', []) if 'score' in r]

        all_cbt_scores.extend(cbt_scores)
        all_persona_scores.extend(persona_scores)

        scores_by_transcript[transcript_id] = {
            'cbt': cbt_scores,
            'persona': persona_scores,
            'turns': [r['turn_number'] for r in result.get('cbt_adherence_results', []) if 'turn_number' in r]
        }

    return all_cbt_scores, all_persona_scores, scores_by_transcript

def extract_indicators(results):
    """Extract CBT technique indicators from results."""
    indicators = {
        'uses_socratic_questioning': 0,
        'gives_direct_advice': 0,
        'explores_evidence': 0,
        'uses_should_statements': 0
    }
    total_turns = 0

    for result in results:
        for turn in result.get('cbt_adherence_results', []):
            total_turns += 1
            for key in indicators:
                if turn.get(key, False):
                    indicators[key] += 1

    return indicators, total_turns

def create_score_distribution(scores, title, xlabel, output_path, color='steelblue'):
    """Create a histogram of score distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.arange(0.5, 11.5, 1)
    counts, _, patches = ax.hist(scores, bins=bins, color=color, edgecolor='white', alpha=0.7)

    # Add value labels on bars
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.annotate(f'{int(count)}',
                       xy=(patch.get_x() + patch.get_width()/2, count),
                       ha='center', va='bottom', fontsize=9)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.grid(axis='y', alpha=0.3)

    # Add statistics
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    stats_text = f'Mean: {mean_score:.2f}\nMedian: {median_score:.1f}\nN: {len(scores)}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_score_comparison_boxplot(scores_by_transcript, output_path):
    """Create boxplot comparing CBT scores across transcripts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CBT Scores
    transcript_ids = list(scores_by_transcript.keys())
    cbt_data = [scores_by_transcript[t]['cbt'] for t in transcript_ids]

    bp1 = axes[0].boxplot(cbt_data, labels=[t.split('_')[0][-4:] for t in transcript_ids], patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    axes[0].set_xlabel('Transcript ID (last 4 digits)', fontsize=11)
    axes[0].set_ylabel('CBT Adherence Score', fontsize=11)
    axes[0].set_title('CBT Adherence Scores by Transcript', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 11)
    axes[0].grid(axis='y', alpha=0.3)

    # Persona Scores
    persona_data = [scores_by_transcript[t]['persona'] for t in transcript_ids]

    bp2 = axes[1].boxplot(persona_data, labels=[t.split('_')[0][-4:] for t in transcript_ids], patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
    axes[1].set_xlabel('Transcript ID (last 4 digits)', fontsize=11)
    axes[1].set_ylabel('Persona Consistency Score', fontsize=11)
    axes[1].set_title('Persona Consistency Scores by Transcript', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 11)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_indicator_chart(indicators, total_turns, output_path):
    """Create bar chart showing CBT technique indicator frequencies."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = {
        'uses_socratic_questioning': 'Socratic\nQuestioning',
        'gives_direct_advice': 'Direct\nAdvice',
        'explores_evidence': 'Explores\nEvidence',
        'uses_should_statements': 'Should\nStatements'
    }

    x_labels = [labels[k] for k in indicators.keys()]
    counts = list(indicators.values())
    percentages = [c / total_turns * 100 for c in counts]

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    bars = ax.bar(x_labels, percentages, color=colors, edgecolor='white', alpha=0.8)

    # Add value labels
    for bar, count, pct in zip(bars, counts, percentages):
        ax.annotate(f'{pct:.1f}%\n({count}/{total_turns})',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Percentage of Turns (%)', fontsize=12)
    ax.set_title('CBT Technique Indicators Across All Turns', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(percentages) * 1.2 if max(percentages) > 0 else 10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_score_progression(scores_by_transcript, output_path):
    """Create line chart showing score progression over turns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(scores_by_transcript)))

    for idx, (transcript_id, data) in enumerate(scores_by_transcript.items()):
        label = transcript_id[-4:]
        turns = data['turns']

        if turns and data['cbt']:
            # Use rolling average for smoother visualization
            window = 5
            cbt_smooth = np.convolve(data['cbt'], np.ones(window)/window, mode='valid')
            turns_smooth = turns[window-1:]
            axes[0].plot(turns_smooth, cbt_smooth, label=label, color=colors[idx], alpha=0.8, linewidth=1.5)

        if turns and data['persona']:
            persona_smooth = np.convolve(data['persona'], np.ones(window)/window, mode='valid')
            axes[1].plot(turns_smooth, persona_smooth, label=label, color=colors[idx], alpha=0.8, linewidth=1.5)

    axes[0].set_xlabel('Turn Number', fontsize=11)
    axes[0].set_ylabel('CBT Adherence Score (5-turn avg)', fontsize=11)
    axes[0].set_title('CBT Adherence Score Progression', fontsize=12, fontweight='bold')
    axes[0].legend(title='Transcript', fontsize=8)
    axes[0].set_ylim(0, 10)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Turn Number', fontsize=11)
    axes[1].set_ylabel('Persona Consistency Score (5-turn avg)', fontsize=11)
    axes[1].set_title('Persona Consistency Score Progression', fontsize=12, fontweight='bold')
    axes[1].legend(title='Transcript', fontsize=8)
    axes[1].set_ylim(0, 10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_summary_stats(all_cbt_scores, all_persona_scores, scores_by_transcript, output_path):
    """Create a summary statistics visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Overall score distributions (violin plots)
    data_for_violin = [all_cbt_scores, all_persona_scores]
    parts = axes[0, 0].violinplot(data_for_violin, positions=[1, 2], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('steelblue')
    parts['bodies'][1].set_facecolor('forestgreen')
    axes[0, 0].set_xticks([1, 2])
    axes[0, 0].set_xticklabels(['CBT Adherence', 'Persona Consistency'])
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Score Distribution Comparison', fontweight='bold')
    axes[0, 0].set_ylim(0, 11)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Per-transcript mean scores
    transcript_ids = list(scores_by_transcript.keys())
    short_ids = [t[-4:] for t in transcript_ids]
    cbt_means = [np.mean(scores_by_transcript[t]['cbt']) for t in transcript_ids]
    persona_means = [np.mean(scores_by_transcript[t]['persona']) for t in transcript_ids]

    x = np.arange(len(short_ids))
    width = 0.35
    axes[0, 1].bar(x - width/2, cbt_means, width, label='CBT Adherence', color='steelblue', alpha=0.8)
    axes[0, 1].bar(x + width/2, persona_means, width, label='Persona Consistency', color='forestgreen', alpha=0.8)
    axes[0, 1].set_xlabel('Transcript ID')
    axes[0, 1].set_ylabel('Mean Score')
    axes[0, 1].set_title('Mean Scores by Transcript', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(short_ids)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 10)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Score heatmap (CBT vs Persona correlation)
    axes[1, 0].scatter(all_cbt_scores, all_persona_scores, alpha=0.3, c='purple')
    axes[1, 0].set_xlabel('CBT Adherence Score')
    axes[1, 0].set_ylabel('Persona Consistency Score')
    axes[1, 0].set_title('CBT vs Persona Score Correlation', fontweight='bold')
    axes[1, 0].set_xlim(0, 11)
    axes[1, 0].set_ylim(0, 11)
    axes[1, 0].grid(alpha=0.3)

    # Add correlation coefficient
    if len(all_cbt_scores) > 1:
        corr = np.corrcoef(all_cbt_scores, all_persona_scores)[0, 1]
        axes[1, 0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1, 0].transAxes,
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Summary table
    axes[1, 1].axis('off')
    summary_data = [
        ['Metric', 'CBT Adherence', 'Persona Consistency'],
        ['Mean', f'{np.mean(all_cbt_scores):.2f}', f'{np.mean(all_persona_scores):.2f}'],
        ['Median', f'{np.median(all_cbt_scores):.1f}', f'{np.median(all_persona_scores):.1f}'],
        ['Std Dev', f'{np.std(all_cbt_scores):.2f}', f'{np.std(all_persona_scores):.2f}'],
        ['Min', f'{np.min(all_cbt_scores):.0f}', f'{np.min(all_persona_scores):.0f}'],
        ['Max', f'{np.max(all_cbt_scores):.0f}', f'{np.max(all_persona_scores):.0f}'],
        ['N', f'{len(all_cbt_scores)}', f'{len(all_persona_scores)}'],
    ]

    table = axes[1, 1].table(cellText=summary_data, loc='center', cellLoc='center',
                             colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    axes[1, 1].set_title('Summary Statistics', fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading results...")
    results = load_all_results(RESULTS_DIR)
    print(f"Loaded {len(results)} result files")

    if not results:
        print("No results found!")
        return

    # Extract data
    print("\nExtracting scores...")
    all_cbt_scores, all_persona_scores, scores_by_transcript = extract_scores(results)
    print(f"Total CBT scores: {len(all_cbt_scores)}")
    print(f"Total Persona scores: {len(all_persona_scores)}")

    indicators, total_turns = extract_indicators(results)
    print(f"Total turns analyzed: {total_turns}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. CBT Score Distribution
    create_score_distribution(
        all_cbt_scores,
        'CBT Adherence Score Distribution',
        'CBT Adherence Score (1-10)',
        os.path.join(OUTPUT_DIR, '1_cbt_score_distribution.png'),
        color='steelblue'
    )

    # 2. Persona Score Distribution
    create_score_distribution(
        all_persona_scores,
        'Persona Consistency Score Distribution',
        'Persona Consistency Score (1-10)',
        os.path.join(OUTPUT_DIR, '2_persona_score_distribution.png'),
        color='forestgreen'
    )

    # 3. Comparison boxplot
    create_score_comparison_boxplot(
        scores_by_transcript,
        os.path.join(OUTPUT_DIR, '3_score_comparison_boxplot.png')
    )

    # 4. CBT Technique Indicators
    create_indicator_chart(
        indicators,
        total_turns,
        os.path.join(OUTPUT_DIR, '4_cbt_technique_indicators.png')
    )

    # 5. Score Progression
    create_score_progression(
        scores_by_transcript,
        os.path.join(OUTPUT_DIR, '5_score_progression.png')
    )

    # 6. Summary Statistics
    create_summary_stats(
        all_cbt_scores,
        all_persona_scores,
        scores_by_transcript,
        os.path.join(OUTPUT_DIR, '6_summary_statistics.png')
    )

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {f}")

if __name__ == "__main__":
    main()
