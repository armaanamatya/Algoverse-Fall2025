"""
Generate the DC-RS Dynamic Flow Diagram with:
- Chain-of-Thought (CoT) Retrieval
- Dynamic Summarization (token-threshold based)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def create_dynamic_flow_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(18, 24))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 26)
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
        'cot': '#E6E6FA',  # Lavender for CoT
        'cot_dark': '#6A5ACD',  # Slate blue for CoT borders
        'eval': '#DDA0DD',
        'eval_dark': '#9932CC',
        'llm': '#4A4A4A',
        'storage': '#90EE90',
        'summarize': '#98FB98',
        'summarize_dark': '#228B22',
        'dynamic': '#FFD700',  # Gold for dynamic features
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
    ax.text(9, 25.5, 'DC-RS Pipeline with Dynamic Features',
            ha='center', va='center', fontsize=16, weight='bold')
    ax.text(9, 24.9, '(Chain-of-Thought Retrieval + Dynamic Summarization)',
            ha='center', va='center', fontsize=10, style='italic')

    # =========================================================================
    # INPUT SECTION
    # =========================================================================
    draw_box(9, 24, 3.5, 0.8, 'Input\nPatient Turn (from transcript)',
             colors['input'], colors['input_border'], fontsize=8)

    draw_box(14, 24, 2.5, 0.6, 'All Previous Turns',
             colors['input'], colors['input_border'], fontsize=7)

    # =========================================================================
    # SECTION HEADERS (Three columns)
    # =========================================================================
    # Draw section backgrounds
    mem0_bg = FancyBboxPatch((0.3, 5.5), 4.4, 17.5,
                              boxstyle="round,pad=0.02",
                              facecolor='#FFF8DC', edgecolor=colors['mem0_dark'],
                              linewidth=2, linestyle='--', alpha=0.3)
    ax.add_patch(mem0_bg)

    baseline_bg = FancyBboxPatch((5.3, 9), 5.4, 14,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#FFF0F5', edgecolor=colors['baseline_dark'],
                                  linewidth=2, linestyle='--', alpha=0.3)
    ax.add_patch(baseline_bg)

    dcrs_bg = FancyBboxPatch((11.2, 5.5), 6.5, 17.5,
                              boxstyle="round,pad=0.02",
                              facecolor='#F0F8FF', edgecolor=colors['dcrs_dark'],
                              linewidth=2, linestyle='--', alpha=0.3)
    ax.add_patch(dcrs_bg)

    # Section labels
    draw_circle(1.2, 22.5, 0.15, colors['mem0_dark'])
    ax.text(2.5, 22.5, 'MEM0 (Static Memory)', fontsize=9, weight='bold', color=colors['mem0_dark'])

    draw_circle(6.5, 22.5, 0.15, colors['baseline_dark'])
    ax.text(8, 22.5, 'BASELINE (No Memory)', fontsize=9, weight='bold', color=colors['baseline_dark'])

    draw_circle(12.2, 22.5, 0.15, colors['dcrs_dark'])
    ax.text(14.5, 22.5, 'DC-RS (Dynamic Continual Learning)', fontsize=9, weight='bold', color=colors['dcrs_dark'])

    # =========================================================================
    # ARROWS FROM INPUT
    # =========================================================================
    draw_arrow(9, 23.5, 2.5, 22)  # to mem0
    draw_arrow(9, 23.5, 8, 22)    # to baseline context
    draw_arrow(9, 23.5, 14.5, 22) # to dcrs
    draw_arrow(14, 23.6, 8, 22.1)  # previous turns to context

    # =========================================================================
    # BASELINE - Context Gathering
    # =========================================================================
    draw_box(8, 21.5, 2.5, 0.6, 'Context Gathering\nget_conversation_context',
             colors['baseline'], fontsize=7)
    draw_arrow(8, 21.1, 8, 20.5)
    draw_box(8, 20, 2.5, 0.6, 'Context: Last 10 turns',
             colors['baseline'], fontsize=7)

    # =========================================================================
    # MEM0 COLUMN
    # =========================================================================
    draw_box(2.5, 21.5, 2.2, 0.5, 'Patient Turn ONLY',
             colors['mem0'], fontsize=7)
    draw_arrow(2.5, 21.2, 2.5, 20.7)

    draw_box(2.5, 20, 2.8, 1.2, 'RETRIEVE MEMORIES:\nget_all_memories(user_id)\n↓\nChromaDB vector search',
             colors['mem0'], fontsize=6)
    draw_arrow(2.5, 19.3, 2.5, 18.7)

    draw_box(2.5, 18, 2.5, 1, 'Retrieved Memories\nPatient facts, preferences,\nhistory',
             colors['mem0'], fontsize=6)
    draw_arrow(2.5, 17.4, 2.5, 16.8)

    draw_box(2.5, 16, 2.8, 1.4, 'PROMPT:\nSystem: CBT_SYSTEM_PROMPT\nMemories: Retrieved facts\nUser: Patient turn\n(NO context window)',
             colors['mem0'], fontsize=6)
    draw_arrow(2.5, 15.2, 2.5, 14.7)

    draw_box(2.5, 14.2, 2, 0.6, '→ LLM Counselor\ngpt-oss:20b',
             colors['llm'], fontsize=6, text_color='white')
    draw_arrow(2.5, 13.8, 2.5, 13.3)

    draw_box(2.5, 12.8, 2, 0.6, 'Generated Response',
             colors['mem0'], fontsize=7)

    # Store to mem0
    draw_arrow(2.5, 12.4, 2.5, 11.8)
    draw_box(2.5, 11, 2.8, 1.2, 'STORE TO MEM0:\nadd_conversation_turn_to_memory()\n↓\nPatient turn → ChromaDB\nCounselor response → ChromaDB',
             colors['mem0_dark'], fontsize=5, text_color='white')

    # =========================================================================
    # BASELINE COLUMN
    # =========================================================================
    draw_box(8, 19, 2.2, 0.5, 'Patient Turn + Context',
             colors['baseline'], fontsize=7)
    draw_arrow(8, 18.7, 8, 18.2)

    draw_box(8, 17.4, 2.8, 1.2, 'PROMPT:\nSystem: CBT_SYSTEM_PROMPT\nContext: Last 10 turns\nUser: Patient turn',
             colors['baseline'], fontsize=6)
    draw_arrow(8, 16.7, 8, 16.2)

    draw_box(8, 15.7, 2, 0.6, '→ LLM Counselor\ngpt-oss:20b',
             colors['llm'], fontsize=6, text_color='white')
    draw_arrow(8, 15.3, 8, 14.8)

    draw_box(8, 14.3, 2, 0.6, 'Generated Response',
             colors['baseline_dark'], fontsize=7, text_color='white')

    # =========================================================================
    # DC-RS COLUMN - WITH DYNAMIC FEATURES
    # =========================================================================
    draw_box(14.5, 21.5, 2.2, 0.5, 'Patient Turn ONLY',
             colors['dcrs'], fontsize=7)
    draw_arrow(14.5, 21.2, 14.5, 20.7)

    # =========================================================================
    # NEW: CHAIN-OF-THOUGHT (CoT) RETRIEVAL
    # =========================================================================
    # CoT Phase 1: Analysis
    cot_bg = FancyBboxPatch((12, 17.5), 5, 3,
                             boxstyle="round,pad=0.02",
                             facecolor=colors['cot'], edgecolor=colors['cot_dark'],
                             linewidth=2, alpha=0.5)
    ax.add_patch(cot_bg)
    ax.text(14.5, 20.3, 'Chain-of-Thought Retrieval', fontsize=7, weight='bold',
            color=colors['cot_dark'], ha='center')

    draw_box(14.5, 19.6, 3.5, 1, 'PHASE 1: ANALYSIS\nIdentify distortions, themes\nDetermine CBT approach',
             colors['cot'], colors['cot_dark'], fontsize=6)
    draw_arrow(14.5, 19, 14.5, 18.5)

    draw_box(14.5, 18.1, 3.5, 0.6, 'PHASE 2: RETRIEVAL\nSelect relevant strategy indices',
             colors['cot'], colors['cot_dark'], fontsize=6)
    draw_arrow(14.5, 17.7, 14.5, 17.2)

    # Generation with analysis context
    draw_box(14.5, 16.3, 3.2, 1.4, 'PROMPT:\nSystem: CBT_SYSTEM_PROMPT\n+ Clinical Analysis\n+ Retrieved Strategies\nUser: Patient turn\n(NO context window)',
             colors['dcrs'], fontsize=5)
    draw_arrow(14.5, 15.5, 14.5, 15)

    draw_box(14.5, 14.5, 2, 0.6, '→ LLM Counselor\ngpt-oss:20b',
             colors['llm'], fontsize=6, text_color='white')
    draw_arrow(14.5, 14.1, 14.5, 13.6)

    draw_box(14.5, 13.1, 2, 0.6, 'Generated Response',
             colors['dcrs'], fontsize=7)

    # Extraction
    draw_arrow(14.5, 12.7, 14.5, 12.2)
    draw_box(14.5, 11.4, 3, 1.3, 'EXTRACTION PROMPT:\nAnalyze patient-counselor\ninteraction\n↓\nExtract therapeutic strategies\nFilter non-clinical content\n↓\nReturns: CBT techniques,\npatterns, interventions',
             colors['dcrs'], fontsize=5)
    draw_arrow(14.5, 10.6, 14.5, 10.1)

    draw_box(14.5, 9.6, 2.2, 0.6, '→ LLM Extractor\ngpt-oss:20b\n(TEST-TIME LEARNING)',
             colors['dcrs_dark'], fontsize=5, text_color='white')
    draw_arrow(14.5, 9.2, 14.5, 8.7)

    draw_box(14.5, 8.2, 2.5, 0.7, 'Update Cheatsheet\n(add strategies, dedupe)',
             colors['dcrs'], fontsize=6)

    # =========================================================================
    # NEW: DYNAMIC SUMMARIZATION (Token-based)
    # =========================================================================
    draw_arrow(14.5, 7.8, 14.5, 7.3)

    # Decision diamond - DYNAMIC (token-based)
    diamond_x, diamond_y = 14.5, 6.7
    diamond_size = 0.6
    diamond = plt.Polygon([
        [diamond_x, diamond_y + diamond_size],
        [diamond_x + diamond_size, diamond_y],
        [diamond_x, diamond_y - diamond_size],
        [diamond_x - diamond_size, diamond_y]
    ], facecolor=colors['dynamic'], edgecolor=colors['summarize_dark'], linewidth=2)
    ax.add_patch(diamond)
    ax.text(diamond_x, diamond_y, 'tokens >\nthreshold?', ha='center', va='center', fontsize=5, weight='bold')

    # Yes path - Summarization
    draw_arrow(diamond_x + diamond_size, diamond_y, 16.8, diamond_y)
    ax.text(15.8, diamond_y + 0.2, 'Yes', fontsize=6, color=colors['summarize_dark'])

    draw_box(16.8, 5.8, 2, 1.5, 'DYNAMIC SUMMARIZE:\nsummarize_cheatsheet()\n↓\nLLM consolidates\nsimilar strategies\n↓\nMax 10 items/category',
             colors['summarize'], colors['summarize_dark'], fontsize=5)

    # Arrow back from summarization
    draw_line(16.8, 5, 16.8, 4.6)
    draw_line(16.8, 4.6, 14.5, 4.6)
    draw_arrow(14.5, 4.6, 14.5, 5.1)

    # No path - Continue
    draw_arrow(diamond_x, diamond_y - diamond_size, diamond_x, 5.7)
    ax.text(diamond_x + 0.2, 5.9, 'No', fontsize=6)

    # Cheatsheet data structure
    draw_box(14.5, 5.3, 3.2, 0.9, 'TherapeuticCheatsheet\ncbt_techniques, distortion_patterns,\neffective_interventions (~2000 tokens)',
             colors['dcrs'], fontsize=5)

    # Loop back arrow
    draw_line(16.5, 5.3, 17.3, 5.3)
    draw_line(17.3, 5.3, 17.3, 21.5)
    draw_line(17.3, 21.5, 15.7, 21.5)
    ax.text(17.4, 13, 'Next turn pair', fontsize=6, rotation=90, va='center')

    # =========================================================================
    # EVALUATION SECTION
    # =========================================================================
    eval_bg = FancyBboxPatch((3.5, 1), 9, 3.5,
                              boxstyle="round,pad=0.02",
                              facecolor='#F5F5F5', edgecolor=colors['eval_dark'],
                              linewidth=2, alpha=0.3)
    ax.add_patch(eval_bg)

    draw_circle(4.2, 4.1, 0.12, colors['eval_dark'])
    ax.text(5.5, 4.1, 'EVALUATION (All Conditions)', fontsize=8, weight='bold', color=colors['eval_dark'])

    # Arrows to evaluation
    draw_arrow(2.5, 12.4, 5.5, 3.8)   # mem0 to eval
    draw_arrow(8, 13.9, 8, 3.8)       # baseline to eval
    draw_arrow(14.5, 12.7, 10.5, 3.8) # dcrs to eval

    draw_box(8, 3.5, 3, 0.5, 'Generated Response + Context',
             '#FFFFFF', fontsize=7)

    # Two evaluation boxes
    draw_box(5.5, 2.2, 3, 1.2, 'CBT ADHERENCE PROMPT:\nRate counselor response\non CBT principles\nScore: 0-10\nCriteria: Socratic questioning,\nvalidation, restructuring',
             colors['eval'], fontsize=5)

    draw_box(10.5, 2.2, 3, 1.2, 'PERSONA CONSISTENCY\nPROMPT:\nCompare to baseline\ntherapeutic stance\nScore: 0-10\nCriteria: Professional tone,\nempathy, boundaries',
             colors['eval'], fontsize=5)

    # LLM Judge boxes
    draw_box(5.5, 1, 1.5, 0.4, '→ LLM Judge',
             colors['llm'], fontsize=6, text_color='white')
    draw_box(10.5, 1, 1.5, 0.4, '→ LLM Judge',
             colors['llm'], fontsize=6, text_color='white')

    # =========================================================================
    # RESULTS STORAGE
    # =========================================================================
    draw_arrow(8, 0.7, 8, 0.3)
    draw_box(8, -0.1, 3, 0.5, 'Results Storage\nJSON + Visualizations',
             colors['storage'], fontsize=7)

    # =========================================================================
    # LEGEND - KEY DIFFERENCES
    # =========================================================================
    legend_y = 0.3
    ax.text(0.5, legend_y + 0.8, 'NEW Dynamic Features:', fontsize=8, weight='bold')

    # CoT legend
    cot_legend = FancyBboxPatch((0.3, legend_y), 0.4, 0.3,
                                 boxstyle="round,pad=0.02",
                                 facecolor=colors['cot'], edgecolor=colors['cot_dark'], linewidth=1)
    ax.add_patch(cot_legend)
    ax.text(1, legend_y + 0.15, 'Chain-of-Thought Retrieval: Analyze → Retrieve → Generate', fontsize=6)

    # Dynamic summarization legend
    dyn_legend = FancyBboxPatch((0.3, legend_y - 0.5), 0.4, 0.3,
                                 boxstyle="round,pad=0.02",
                                 facecolor=colors['dynamic'], edgecolor=colors['summarize_dark'], linewidth=1)
    ax.add_patch(dyn_legend)
    ax.text(1, legend_y - 0.35, 'Dynamic Summarization: Token-threshold based (not fixed interval)', fontsize=6)

    plt.tight_layout()
    plt.savefig('dynamic_flow_diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('dynamic_flow_diagram.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: dynamic_flow_diagram.png and dynamic_flow_diagram.pdf")
    plt.show()

if __name__ == "__main__":
    create_dynamic_flow_diagram()
