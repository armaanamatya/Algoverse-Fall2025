"""
Generate the DC-RS Flow Diagram with Summarization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def create_flow_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 22)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors
    colors = {
        'input': '#E8F4FD',
        'input_border': '#4A90D9',
        'mem0': '#FFE4C4',
        'mem0_dark': '#FFA500',
        'baseline': '#FFB6C1',
        'baseline_dark': '#FF69B4',
        'dcrs': '#E0F0FF',
        'dcrs_dark': '#4169E1',
        'eval': '#DDA0DD',
        'eval_dark': '#9932CC',
        'llm': '#4A4A4A',
        'storage': '#90EE90',
        'summarize': '#98FB98',
        'summarize_dark': '#228B22',
    }

    def draw_box(x, y, w, h, text, color, border_color=None, fontsize=8, bold=False, text_color='black'):
        """Draw a rounded rectangle box with text"""
        if border_color is None:
            border_color = color
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=color, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                weight=weight, color=text_color, wrap=True)

    def draw_circle(x, y, r, color):
        """Draw a circle"""
        circle = Circle((x, y), r, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(circle)

    def draw_arrow(x1, y1, x2, y2, color='black', style='-'):
        """Draw an arrow"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.2,
                                  linestyle=style))

    def draw_line(x1, y1, x2, y2, color='black', style='-'):
        """Draw a line without arrow"""
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.2, linestyle=style)

    # =========================================================================
    # TITLE
    # =========================================================================
    ax.text(8, 21.5, 'DC-RS Pipeline with Periodic Summarization',
            ha='center', va='center', fontsize=14, weight='bold')
    ax.text(8, 21.0, '(Turn Pair = Patient Turn + Generated Counselor Response)',
            ha='center', va='center', fontsize=9, style='italic')

    # =========================================================================
    # INPUT SECTION
    # =========================================================================
    draw_box(8, 20, 3.5, 0.8, 'Input\nPatient Turn (from transcript)',
             colors['input'], colors['input_border'], fontsize=8)

    draw_box(12.5, 20, 2.5, 0.6, 'All Previous Turns',
             colors['input'], colors['input_border'], fontsize=7)

    # =========================================================================
    # SECTION HEADERS (Three columns)
    # =========================================================================
    # Draw section backgrounds
    mem0_bg = FancyBboxPatch((0.3, 4.5), 4.4, 14.5,
                              boxstyle="round,pad=0.02",
                              facecolor='#FFF8DC', edgecolor=colors['mem0_dark'],
                              linewidth=2, linestyle='--', alpha=0.3)
    ax.add_patch(mem0_bg)

    baseline_bg = FancyBboxPatch((5.3, 8), 5.4, 11,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#FFF0F5', edgecolor=colors['baseline_dark'],
                                  linewidth=2, linestyle='--', alpha=0.3)
    ax.add_patch(baseline_bg)

    dcrs_bg = FancyBboxPatch((11, 4.5), 4.7, 14.5,
                              boxstyle="round,pad=0.02",
                              facecolor='#F0F8FF', edgecolor=colors['dcrs_dark'],
                              linewidth=2, linestyle='--', alpha=0.3)
    ax.add_patch(dcrs_bg)

    # Section labels
    draw_circle(1.2, 18.5, 0.15, colors['mem0_dark'])
    ax.text(2.5, 18.5, 'MEM0 (Static Memory)', fontsize=9, weight='bold', color=colors['mem0_dark'])

    draw_circle(6.5, 18.5, 0.15, colors['baseline_dark'])
    ax.text(8, 18.5, 'BASELINE (No Memory)', fontsize=9, weight='bold', color=colors['baseline_dark'])

    draw_circle(12, 18.5, 0.15, colors['dcrs_dark'])
    ax.text(13.5, 18.5, 'DC-RS (Continual Learning)', fontsize=9, weight='bold', color=colors['dcrs_dark'])

    # =========================================================================
    # ARROWS FROM INPUT
    # =========================================================================
    draw_arrow(8, 19.5, 2.5, 18)  # to mem0
    draw_arrow(8, 19.5, 8, 18)    # to baseline context
    draw_arrow(8, 19.5, 13.3, 18) # to dcrs
    draw_arrow(12.5, 19.6, 8, 18.1)  # previous turns to context

    # =========================================================================
    # BASELINE - Context Gathering
    # =========================================================================
    draw_box(8, 17.5, 2.5, 0.6, 'Context Gathering\nget_conversation_context',
             colors['baseline'], fontsize=7)
    draw_arrow(8, 17.1, 8, 16.5)
    draw_box(8, 16, 2.5, 0.6, 'Context: Last 10 turns',
             colors['baseline'], fontsize=7)

    # =========================================================================
    # MEM0 COLUMN
    # =========================================================================
    draw_box(2.5, 17.5, 2.2, 0.5, 'Patient Turn ONLY',
             colors['mem0'], fontsize=7)
    draw_arrow(2.5, 17.2, 2.5, 16.7)

    draw_box(2.5, 16, 2.8, 1.2, 'RETRIEVE MEMORIES:\nget_all_memories(user_id)\n↓\nChromaDB vector search',
             colors['mem0'], fontsize=6)
    draw_arrow(2.5, 15.3, 2.5, 14.7)

    draw_box(2.5, 14, 2.5, 1, 'Retrieved Memories\nPatient facts, preferences,\nhistory',
             colors['mem0'], fontsize=6)
    draw_arrow(2.5, 13.4, 2.5, 12.8)

    draw_box(2.5, 12, 2.8, 1.4, 'PROMPT:\nSystem: CBT_SYSTEM_PROMPT\nMemories: Retrieved facts\nUser: Patient turn\n(NO context window)',
             colors['mem0'], fontsize=6)
    draw_arrow(2.5, 11.2, 2.5, 10.7)

    draw_box(2.5, 10.2, 2, 0.6, '→ LLM Counselor\ngpt-oss:20b',
             colors['llm'], fontsize=6, text_color='white')
    draw_arrow(2.5, 9.8, 2.5, 9.3)

    draw_box(2.5, 8.8, 2, 0.6, 'Generated Response',
             colors['mem0'], fontsize=7)

    # Store to mem0
    draw_arrow(2.5, 8.4, 2.5, 7.8)
    draw_box(2.5, 7, 2.8, 1.2, 'STORE TO MEM0:\nadd_conversation_turn_to_memory()\n↓\nPatient turn → ChromaDB\nCounselor response → ChromaDB',
             colors['mem0_dark'], fontsize=5, text_color='white')

    # =========================================================================
    # BASELINE COLUMN
    # =========================================================================
    draw_box(8, 15, 2.2, 0.5, 'Patient Turn + Context',
             colors['baseline'], fontsize=7)
    draw_arrow(8, 14.7, 8, 14.2)

    draw_box(8, 13.4, 2.8, 1.2, 'PROMPT:\nSystem: CBT_SYSTEM_PROMPT\nContext: Last 10 turns\nUser: Patient turn',
             colors['baseline'], fontsize=6)
    draw_arrow(8, 12.7, 8, 12.2)

    draw_box(8, 11.7, 2, 0.6, '→ LLM Counselor\ngpt-oss:20b',
             colors['llm'], fontsize=6, text_color='white')
    draw_arrow(8, 11.3, 8, 10.8)

    draw_box(8, 10.3, 2, 0.6, 'Generated Response',
             colors['baseline_dark'], fontsize=7, text_color='white')

    # =========================================================================
    # DC-RS COLUMN
    # =========================================================================
    draw_box(13.3, 17.5, 2.2, 0.5, 'Patient Turn ONLY',
             colors['dcrs'], fontsize=7)
    draw_arrow(13.3, 17.2, 13.3, 16.7)

    draw_box(13.3, 15.9, 2.8, 1.3, 'PROMPT:\nSystem: CBT_SYSTEM_PROMPT\nCheatsheet: Strategies\nUser: Patient turn\n(NO context window)',
             colors['dcrs'], fontsize=6)
    draw_arrow(13.3, 15.1, 13.3, 14.6)

    draw_box(13.3, 14.1, 2, 0.6, '→ LLM Counselor\ngpt-oss:20b',
             colors['llm'], fontsize=6, text_color='white')
    draw_arrow(13.3, 13.7, 13.3, 13.2)

    draw_box(13.3, 12.7, 2, 0.6, 'Generated Response',
             colors['dcrs'], fontsize=7)

    # Extraction
    draw_arrow(13.3, 12.3, 13.3, 11.8)
    draw_box(13.3, 11, 2.8, 1.3, 'EXTRACTION PROMPT:\nAnalyze patient-counselor\ninteraction\n↓\nExtract therapeutic strategies\nFilter non-clinical content\n↓\nReturns: CBT techniques,\npatterns, interventions',
             colors['dcrs'], fontsize=5)
    draw_arrow(13.3, 10.2, 13.3, 9.7)

    draw_box(13.3, 9.2, 2, 0.6, '→ LLM Extractor\ngpt-oss:20b\n(TEST-TIME LEARNING)',
             colors['dcrs_dark'], fontsize=5, text_color='white')
    draw_arrow(13.3, 8.8, 13.3, 8.3)

    draw_box(13.3, 7.8, 2.5, 0.7, 'Update Cheatsheet\n(add strategies, dedupe)',
             colors['dcrs'], fontsize=6)

    # =========================================================================
    # NEW: SUMMARIZATION CHECK
    # =========================================================================
    draw_arrow(13.3, 7.4, 13.3, 6.9)

    # Decision diamond (approximated with rotated square)
    diamond_x, diamond_y = 13.3, 6.4
    diamond_size = 0.5
    diamond = plt.Polygon([
        [diamond_x, diamond_y + diamond_size],
        [diamond_x + diamond_size, diamond_y],
        [diamond_x, diamond_y - diamond_size],
        [diamond_x - diamond_size, diamond_y]
    ], facecolor=colors['summarize'], edgecolor=colors['summarize_dark'], linewidth=2)
    ax.add_patch(diamond)
    ax.text(diamond_x, diamond_y, 'N % 100\n== 0?', ha='center', va='center', fontsize=5, weight='bold')

    # Yes path - Summarization
    draw_arrow(13.3 + diamond_size, 6.4, 15, 6.4)
    ax.text(14.2, 6.6, 'Yes', fontsize=6, color=colors['summarize_dark'])

    draw_box(15, 5.5, 1.8, 1.4, 'SUMMARIZE:\nsummarize_cheatsheet()\n↓\nLLM consolidates\nsimilar strategies\n↓\nMax 10 items/category',
             colors['summarize'], colors['summarize_dark'], fontsize=5)

    # Arrow back from summarization
    draw_line(15, 4.7, 15, 4.3)
    draw_line(15, 4.3, 13.3, 4.3)
    draw_arrow(13.3, 4.3, 13.3, 4.8)

    # No path - Continue
    draw_arrow(13.3, 6.4 - diamond_size, 13.3, 5.4)
    ax.text(13.5, 5.7, 'No', fontsize=6)

    # Cheatsheet data structure
    draw_box(13.3, 5, 2.8, 0.9, 'TherapeuticCheatsheet\ncbt_techniques: list\ndistortion_patterns: list\neffective_interventions: list',
             colors['dcrs'], fontsize=5)

    # Loop back arrow
    draw_line(14.8, 5, 15.3, 5)
    draw_line(15.3, 5, 15.3, 17.5)
    draw_line(15.3, 17.5, 14.5, 17.5)
    ax.text(15.4, 11, 'Next turn pair', fontsize=6, rotation=90, va='center')

    # =========================================================================
    # EVALUATION SECTION
    # =========================================================================
    eval_bg = FancyBboxPatch((3.5, 1), 9, 3.2,
                              boxstyle="round,pad=0.02",
                              facecolor='#F5F5F5', edgecolor=colors['eval_dark'],
                              linewidth=2, alpha=0.3)
    ax.add_patch(eval_bg)

    draw_circle(4.2, 3.8, 0.12, colors['eval_dark'])
    ax.text(5.5, 3.8, 'EVALUATION (All Conditions)', fontsize=8, weight='bold', color=colors['eval_dark'])

    # Arrows to evaluation
    draw_arrow(2.5, 8.4, 5.5, 3.5)  # mem0 to eval
    draw_arrow(8, 9.9, 8, 3.5)      # baseline to eval
    draw_arrow(13.3, 12.3, 10.5, 3.5)  # dcrs to eval

    draw_box(8, 3.2, 3, 0.5, 'Generated Response + Context',
             '#FFFFFF', fontsize=7)

    # Two evaluation boxes
    draw_box(5.5, 2, 3, 1.2, 'CBT ADHERENCE PROMPT:\nRate counselor response\non CBT principles\nScore: 0-10\nCriteria: Socratic questioning,\nvalidation, restructuring',
             colors['eval'], fontsize=5)

    draw_box(10.5, 2, 3, 1.2, 'PERSONA CONSISTENCY\nPROMPT:\nCompare to baseline\ntherapeutic stance\nScore: 0-10\nCriteria: Professional tone,\nempathy, boundaries',
             colors['eval'], fontsize=5)

    # LLM Judge boxes
    draw_box(5.5, 0.8, 1.5, 0.4, '→ LLM Judge',
             colors['llm'], fontsize=6, text_color='white')
    draw_box(10.5, 0.8, 1.5, 0.4, '→ LLM Judge',
             colors['llm'], fontsize=6, text_color='white')

    # =========================================================================
    # RESULTS STORAGE
    # =========================================================================
    draw_arrow(8, 0.5, 8, 0.1)
    draw_box(8, -0.3, 3, 0.5, 'Results Storage\nJSON + Visualizations',
             colors['storage'], fontsize=7)

    # =========================================================================
    # LEGEND
    # =========================================================================
    ax.text(0.5, 0.5, 'N = Turn Pair Number', fontsize=7, style='italic')
    ax.text(0.5, 0.1, 'Summarization: Every 100 turn pairs', fontsize=7, style='italic', color=colors['summarize_dark'])

    plt.tight_layout()
    plt.savefig('summarized_flow_diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('summarized_flow_diagram.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: summarized_flow_diagram.png and summarized_flow_diagram.pdf")
    plt.show()

if __name__ == "__main__":
    create_flow_diagram()
