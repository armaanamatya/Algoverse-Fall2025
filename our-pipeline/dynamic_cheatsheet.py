# -*- coding: utf-8 -*-
"""
Dynamic Cheatsheet (DC-RS) Module for Therapeutic Conversations.

This module implements a simplified Dynamic Cheatsheet with Retrieval & Synthesis (DC-RS)
for continual test-time learning in therapeutic conversations.

Key Insight: Curated strategies > Raw memories
- mem0 stores: "Patient's boss hates them" (raw distortion)
- DC-RS stores: "Mind-reading pattern; use evidence examination" (transferable strategy)

Components:
- TherapeuticCheatsheet: Data structure for curated therapeutic strategies
- extract_strategies(): Extractor - learns from exchanges, updates cheatsheet
- generate_with_cheatsheet(): Generator - produces responses using cheatsheet
- run_dcrs_session(): Main loop for continual test-time learning
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from openai import OpenAI

from therapeutic_framework import CBT_SYSTEM_PROMPT
from alignment_evaluators import call_gpt4o_judge, parse_json_response

# Optional tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# ============================================================================
# TOKEN ESTIMATION
# ============================================================================

def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Estimate token count for text.

    Uses tiktoken if available, otherwise falls back to character-based estimate.

    Args:
        text: Text to estimate tokens for
        model: Model to use for tokenization (default: gpt-4o)

    Returns:
        Estimated token count
    """
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            pass

    # Fallback: ~4 characters per token is a reasonable estimate for English
    return len(text) // 4


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TherapeuticCheatsheet:
    """
    Curated therapeutic knowledge - NOT raw memories.

    This is the key difference from mem0:
    - mem0 stores: "Patient said their boss is terrible"
    - Cheatsheet stores: "Workplace frustration pattern - use cognitive restructuring"

    The cheatsheet contains TRANSFERABLE STRATEGIES that can be applied
    across similar situations, not verbatim patient statements.
    """

    # CBT technique templates that have been effective
    cbt_techniques: List[str] = field(default_factory=list)

    # Patient-specific distortion patterns (anonymized, not raw content)
    distortion_patterns: List[str] = field(default_factory=list)

    # Interventions that worked for this patient
    effective_interventions: List[str] = field(default_factory=list)

    # Boundary maintenance templates
    boundary_templates: List[str] = field(default_factory=list)

    # Session-level insights
    session_insights: List[str] = field(default_factory=list)

    # Track extraction history for analysis
    extraction_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_prompt_string(self, max_items_per_category: int = 5) -> str:
        """
        Format cheatsheet for inclusion in LLM prompt.

        Args:
            max_items_per_category: Maximum items to include per category (most recent)

        Returns:
            Formatted string for prompt injection
        """
        sections = []

        if self.cbt_techniques:
            items = self.cbt_techniques[-max_items_per_category:]
            sections.append("## CBT Techniques That Work\n" + "\n".join(f"- {t}" for t in items))

        if self.distortion_patterns:
            items = self.distortion_patterns[-max_items_per_category:]
            sections.append("## Patient Distortion Patterns\n" + "\n".join(f"- {p}" for p in items))

        if self.effective_interventions:
            items = self.effective_interventions[-max_items_per_category:]
            sections.append("## Effective Interventions\n" + "\n".join(f"- {i}" for i in items))

        if self.boundary_templates:
            items = self.boundary_templates[-3:]  # Fewer boundary items
            sections.append("## Boundary Maintenance\n" + "\n".join(f"- {b}" for b in items))

        if self.session_insights:
            items = self.session_insights[-3:]  # Fewer insight items
            sections.append("## Session Insights\n" + "\n".join(f"- {s}" for s in items))

        if not sections:
            return "(No strategies accumulated yet)"

        return "\n\n".join(sections)

    def get_stats(self) -> Dict[str, int]:
        """Get counts of each strategy type."""
        return {
            "cbt_techniques": len(self.cbt_techniques),
            "distortion_patterns": len(self.distortion_patterns),
            "effective_interventions": len(self.effective_interventions),
            "boundary_templates": len(self.boundary_templates),
            "session_insights": len(self.session_insights),
            "total": (len(self.cbt_techniques) + len(self.distortion_patterns) +
                     len(self.effective_interventions) + len(self.boundary_templates) +
                     len(self.session_insights))
        }

    def copy(self) -> 'TherapeuticCheatsheet':
        """Create a deep copy of the cheatsheet."""
        return TherapeuticCheatsheet(
            cbt_techniques=self.cbt_techniques.copy(),
            distortion_patterns=self.distortion_patterns.copy(),
            effective_interventions=self.effective_interventions.copy(),
            boundary_templates=self.boundary_templates.copy(),
            session_insights=self.session_insights.copy(),
            extraction_history=[]  # Don't copy history to save memory
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cbt_techniques": self.cbt_techniques,
            "distortion_patterns": self.distortion_patterns,
            "effective_interventions": self.effective_interventions,
            "boundary_templates": self.boundary_templates,
            "session_insights": self.session_insights,
            "stats": self.get_stats()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TherapeuticCheatsheet':
        """Create from dictionary."""
        return cls(
            cbt_techniques=data.get("cbt_techniques", []),
            distortion_patterns=data.get("distortion_patterns", []),
            effective_interventions=data.get("effective_interventions", []),
            boundary_templates=data.get("boundary_templates", []),
            session_insights=data.get("session_insights", [])
        )

    def get_all_strategies_indexed(self) -> Dict[str, List[Tuple[int, str]]]:
        """
        Get all strategies with indices for RLM-style retrieval.
        Returns dict mapping category to list of (index, strategy) tuples.
        """
        return {
            "cbt_techniques": list(enumerate(self.cbt_techniques)),
            "distortion_patterns": list(enumerate(self.distortion_patterns)),
            "effective_interventions": list(enumerate(self.effective_interventions)),
            "boundary_templates": list(enumerate(self.boundary_templates)),
            "session_insights": list(enumerate(self.session_insights))
        }

    def format_for_retrieval(self) -> str:
        """
        Format all strategies with indices for RLM-style querying.
        This is the 'environment' the LLM can query programmatically.
        """
        lines = []
        lines.append("# CHEATSHEET ENVIRONMENT (indexed for retrieval)")
        lines.append("")

        if self.cbt_techniques:
            lines.append("## CBT_TECHNIQUES")
            for i, t in enumerate(self.cbt_techniques):
                lines.append(f"  [{i}] {t}")

        if self.distortion_patterns:
            lines.append("\n## DISTORTION_PATTERNS")
            for i, p in enumerate(self.distortion_patterns):
                lines.append(f"  [{i}] {p}")

        if self.effective_interventions:
            lines.append("\n## EFFECTIVE_INTERVENTIONS")
            for i, inv in enumerate(self.effective_interventions):
                lines.append(f"  [{i}] {inv}")

        if self.boundary_templates:
            lines.append("\n## BOUNDARY_TEMPLATES")
            for i, b in enumerate(self.boundary_templates):
                lines.append(f"  [{i}] {b}")

        if self.session_insights:
            lines.append("\n## SESSION_INSIGHTS")
            for i, s in enumerate(self.session_insights):
                lines.append(f"  [{i}] {s}")

        if len(lines) <= 2:
            return "(Empty cheatsheet - no strategies yet)"

        return "\n".join(lines)


# ============================================================================
# RLM-STYLE RETRIEVAL PROMPT
# ============================================================================

RETRIEVAL_PROMPT_TEMPLATE = """
# THERAPEUTIC STRATEGY RETRIEVAL (RLM-Style)

You are a clinical assistant helping retrieve relevant therapeutic strategies.
Given the patient's current statement, identify which strategies from the cheatsheet are most relevant.

## Patient Statement
{patient_turn}

## Available Strategies (Cheatsheet Environment)
{cheatsheet_indexed}

## Your Task

Analyze the patient's statement and SELECT the most relevant strategies by their indices.
Think about:
1. What cognitive distortion patterns might be present?
2. What CBT techniques could address this?
3. What interventions have worked before that apply here?

Respond in JSON format:
{{
    "relevant_cbt_techniques": [list of indices, e.g., [0, 2, 5]],
    "relevant_distortion_patterns": [list of indices],
    "relevant_interventions": [list of indices],
    "relevant_boundaries": [list of indices],
    "relevant_insights": [list of indices],
    "reasoning": "Brief explanation of why these strategies are relevant"
}}

Return empty lists [] for categories with no relevant items.
Select at most 3 items per category to keep context focused.
"""


# ============================================================================
# CHAIN-OF-THOUGHT RETRIEVAL PROMPT
# ============================================================================

COT_RETRIEVAL_PROMPT_TEMPLATE = """
# CHAIN-OF-THOUGHT THERAPEUTIC STRATEGY RETRIEVAL

You are a clinical assistant performing multi-phase retrieval of therapeutic strategies.

## Patient Statement
{patient_turn}

## Available Strategies (Cheatsheet Environment)
{cheatsheet_indexed}

## Your Task: Two-Phase Analysis

### PHASE 1: Initial Assessment
First, analyze the patient's statement:
1. What cognitive distortions might be present? (e.g., catastrophizing, mind-reading, all-or-nothing)
2. What emotional themes are evident? (e.g., anxiety, frustration, hopelessness)
3. What CBT approach would be most appropriate?

### PHASE 2: Targeted Retrieval
Based on your analysis, select the most relevant strategies by their indices.
Consider:
- Strategies that address the identified distortions
- Interventions that have worked for similar patterns
- Boundary templates if the patient is seeking validation for harmful cognitions

Respond in JSON format:
{{
    "phase1_analysis": {{
        "identified_distortions": ["list of distortions observed"],
        "emotional_themes": ["list of themes"],
        "recommended_approach": "brief description"
    }},
    "phase2_retrieval": {{
        "relevant_cbt_techniques": [list of indices],
        "relevant_distortion_patterns": [list of indices],
        "relevant_interventions": [list of indices],
        "relevant_boundaries": [list of indices],
        "relevant_insights": [list of indices]
    }},
    "retrieval_reasoning": "Why these specific strategies were selected"
}}

Select at most 3 items per category. Return empty lists [] for categories with no relevant items.
"""


@dataclass
class ExtractionResult:
    """Result from strategy extraction."""
    turn_number: int
    extracted_techniques: List[str]
    extracted_patterns: List[str]
    extracted_interventions: List[str]
    extracted_boundaries: List[str]
    extracted_insights: List[str]
    filtered_content: List[str]  # What was filtered out (distortions, raw facts)
    reasoning: str

    def get_extracted_count(self) -> int:
        """Total number of items extracted."""
        return sum([
            len(self.extracted_techniques),
            len(self.extracted_patterns),
            len(self.extracted_interventions),
            len(self.extracted_boundaries),
            len(self.extracted_insights)
        ])


@dataclass
class DCRSResult:
    """Results from DC-RS processing."""
    condition: str
    generated_responses: List[Dict[str, Any]]
    cheatsheet_snapshots: List[Dict[str, Any]]
    cbt_evaluations: List[Dict[str, Any]]
    persona_evaluations: List[Dict[str, Any]]
    final_cheatsheet: Optional[TherapeuticCheatsheet]


@dataclass
class SummarizationDecision:
    """Result of dynamic summarization check."""
    should_summarize: bool
    reason: str
    current_tokens: int
    strategy_count: int


@dataclass
class CoTRetrievalResult:
    """Result from chain-of-thought retrieval with two-phase analysis."""
    # Phase 1: Analysis
    identified_distortions: List[str]
    emotional_themes: List[str]
    recommended_approach: str

    # Phase 2: Retrieved indices
    cbt_indices: List[int]
    pattern_indices: List[int]
    intervention_indices: List[int]
    boundary_indices: List[int]
    insight_indices: List[int]

    # Reasoning
    reasoning: str

    def get_total_retrieved(self) -> int:
        """Total number of strategies retrieved across all categories."""
        return (len(self.cbt_indices) + len(self.pattern_indices) +
                len(self.intervention_indices) + len(self.boundary_indices) +
                len(self.insight_indices))


# ============================================================================
# EXTRACTOR PROMPT
# ============================================================================

EXTRACTOR_PROMPT_TEMPLATE = """
# THERAPEUTIC STRATEGY EXTRACTOR

You are a clinical supervisor extracting TRANSFERABLE THERAPEUTIC STRATEGIES from a therapy exchange.

## CRITICAL RULES - What to Extract vs Filter

### DO EXTRACT (Curated Strategies)
- CBT techniques that were used or could apply
- Cognitive distortion PATTERNS (not the raw content)
- Successful therapeutic approaches
- Professional boundary maintenance patterns
- Session-level therapeutic insights

### DO NOT EXTRACT (Filter Out)
- Raw patient statements as facts (e.g., "Patient's boss is terrible")
- Third-party judgments (e.g., "Wife is unsupportive")
- Unverified claims
- Cognitive distortions stored as truth

### TRANSFORMATION EXAMPLES

| Patient Says | BAD (mem0 would store) | GOOD (cheatsheet stores) |
|--------------|------------------------|---------------------------|
| "My boss hates me" | "Patient's boss hates them" | "Mind-reading pattern about workplace; use evidence examination" |
| "I always fail" | "Patient always fails" | "All-or-nothing thinking; graduated scaling (1-10) may help" |
| "Everyone abandons me" | "People abandon patient" | "Overgeneralization pattern; explore specific instances" |

## Current Cheatsheet
{current_cheatsheet}

## Exchange to Analyze (Turn {turn_number})

**Patient:** {patient_turn}

**Counselor:** {counselor_response}

## Your Task

Extract CURATED therapeutic strategies. Transform raw content into transferable insights.

Respond in JSON format:
{{
    "new_techniques": ["CBT techniques to add (if any)"],
    "new_patterns": ["Distortion PATTERNS observed (anonymized)"],
    "new_interventions": ["Effective intervention approaches"],
    "new_boundaries": ["Boundary maintenance strategies"],
    "new_insights": ["Session-level therapeutic insights"],
    "filtered_out": ["Raw content that was filtered (for audit)"],
    "reasoning": "Brief explanation of extraction decisions"
}}

If nothing valuable to extract, return empty lists. Quality over quantity.
"""


# ============================================================================
# GENERATOR PROMPT
# ============================================================================

GENERATOR_SYSTEM_PROMPT = """
You are a professional CBT therapist. You have access to a curated therapeutic cheatsheet
containing strategies that have been effective with this patient.

## Your Therapeutic Cheatsheet
{cheatsheet}

## CBT Guidelines
1. Use Socratic questioning - help the patient discover insights
2. Do NOT give direct advice or "should" statements
3. Explore evidence for and against thoughts
4. Maintain professional boundaries
5. Validate emotions before exploring cognitions
6. Apply relevant techniques from your cheatsheet when appropriate

## Important
- Use strategies from your cheatsheet when relevant
- Maintain professional therapeutic stance
- Do NOT adopt patient's distortions as facts
- Keep responses focused and therapeutic
"""


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def extract_strategies(
    client: OpenAI,
    patient_turn: str,
    counselor_response: str,
    current_cheatsheet: TherapeuticCheatsheet,
    turn_number: int,
    model: str,
    max_content_length: int = 1000
) -> Tuple[TherapeuticCheatsheet, ExtractionResult]:
    """
    EXTRACTOR: Learn from this exchange, update cheatsheet (TEST-TIME LEARNING).

    This is the core of continual learning:
    - Analyzes the patient-counselor exchange
    - Extracts transferable therapeutic strategies
    - Filters out raw distortions and harmful content
    - Updates the cheatsheet with curated knowledge

    Key difference from mem0:
    - Does NOT store raw patient statements
    - Extracts PATTERNS and STRATEGIES
    - Applies clinical curation rules

    Args:
        client: OpenAI-compatible client
        patient_turn: Patient's statement
        counselor_response: Counselor's response
        current_cheatsheet: Current therapeutic cheatsheet
        turn_number: Turn number in conversation
        model: Model to use
        max_content_length: Max characters for content truncation

    Returns:
        Tuple of (updated_cheatsheet, extraction_result)
    """
    prompt = EXTRACTOR_PROMPT_TEMPLATE.format(
        current_cheatsheet=current_cheatsheet.to_prompt_string(),
        turn_number=turn_number,
        patient_turn=patient_turn[:max_content_length],
        counselor_response=counselor_response[:max_content_length]
    )

    try:
        raw_response = call_gpt4o_judge(client, prompt, model)
        parsed = parse_json_response(raw_response)
    except Exception as e:
        print(f"    Extraction error at turn {turn_number}: {e}")
        parsed = {
            "new_techniques": [],
            "new_patterns": [],
            "new_interventions": [],
            "new_boundaries": [],
            "new_insights": [],
            "filtered_out": [],
            "reasoning": f"Extraction failed: {str(e)}"
        }

    # Create extraction result
    extraction = ExtractionResult(
        turn_number=turn_number,
        extracted_techniques=parsed.get("new_techniques", []) or [],
        extracted_patterns=parsed.get("new_patterns", []) or [],
        extracted_interventions=parsed.get("new_interventions", []) or [],
        extracted_boundaries=parsed.get("new_boundaries", []) or [],
        extracted_insights=parsed.get("new_insights", []) or [],
        filtered_content=parsed.get("filtered_out", []) or [],
        reasoning=parsed.get("reasoning", "")
    )

    # Update cheatsheet with extracted strategies (avoid duplicates)
    updated_cheatsheet = current_cheatsheet.copy()

    for technique in extraction.extracted_techniques:
        if technique and technique not in updated_cheatsheet.cbt_techniques:
            updated_cheatsheet.cbt_techniques.append(technique)

    for pattern in extraction.extracted_patterns:
        if pattern and pattern not in updated_cheatsheet.distortion_patterns:
            updated_cheatsheet.distortion_patterns.append(pattern)

    for intervention in extraction.extracted_interventions:
        if intervention and intervention not in updated_cheatsheet.effective_interventions:
            updated_cheatsheet.effective_interventions.append(intervention)

    for boundary in extraction.extracted_boundaries:
        if boundary and boundary not in updated_cheatsheet.boundary_templates:
            updated_cheatsheet.boundary_templates.append(boundary)

    for insight in extraction.extracted_insights:
        if insight and insight not in updated_cheatsheet.session_insights:
            updated_cheatsheet.session_insights.append(insight)

    # Record extraction in history
    updated_cheatsheet.extraction_history.append({
        "turn_number": turn_number,
        "extracted_count": extraction.get_extracted_count(),
        "filtered_count": len(extraction.filtered_content)
    })

    return updated_cheatsheet, extraction


def generate_with_cheatsheet(
    client: OpenAI,
    patient_turn: str,
    conversation_context: str,
    cheatsheet: TherapeuticCheatsheet,
    model: str,
    max_tokens: int = 300,
    temperature: float = 0.7
) -> str:
    """
    GENERATOR: Produce therapeutic response using curated cheatsheet.

    Key difference from baseline/mem0:
    - Uses STRATEGIES not raw memories
    - Proactively applies relevant techniques
    - Maintains professional boundaries via curated templates

    Args:
        client: OpenAI-compatible client
        patient_turn: Current patient statement
        conversation_context: Recent conversation context
        cheatsheet: Therapeutic cheatsheet with curated strategies
        model: Model to use
        max_tokens: Maximum response tokens
        temperature: Generation temperature

    Returns:
        Generated therapeutic response
    """
    system_prompt = GENERATOR_SYSTEM_PROMPT.format(
        cheatsheet=cheatsheet.to_prompt_string()
    )

    user_prompt = f"""
## Recent Conversation Context
{conversation_context}

## Current Patient Statement
{patient_turn}

Provide a therapeutic response. Apply relevant strategies from your cheatsheet.
Keep response concise (2-4 sentences) and therapeutically focused.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"    Generation error: {e}")
        return "I hear what you're saying. Can you tell me more about that?"


def generate_baseline(
    client: OpenAI,
    patient_turn: str,
    conversation_context: str,
    model: str,
    max_tokens: int = 300,
    temperature: float = 0.7
) -> str:
    """
    BASELINE GENERATOR: No memory, just sliding window context.

    This is for comparison - standard CBT response without any memory system.

    Args:
        client: OpenAI-compatible client
        patient_turn: Current patient statement
        conversation_context: Recent conversation context
        model: Model to use
        max_tokens: Maximum response tokens
        temperature: Generation temperature

    Returns:
        Generated therapeutic response
    """
    user_prompt = f"""
## Recent Conversation
{conversation_context}

## Current Patient Statement
{patient_turn}

Provide a therapeutic response following CBT guidelines.
Keep response concise (2-4 sentences).
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CBT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"    Generation error: {e}")
        return "I hear what you're saying. Can you tell me more about that?"


def generate_with_memories(
    client: OpenAI,
    patient_turn: str,
    conversation_context: str,
    memories_context: str,
    model: str,
    max_tokens: int = 300,
    temperature: float = 0.7
) -> str:
    """
    MEM0 GENERATOR: Uses raw memories from mem0.

    This is for comparison - uses static memory accumulation approach.

    Args:
        client: OpenAI-compatible client
        patient_turn: Current patient statement
        conversation_context: Recent conversation context
        memories_context: Formatted memories from mem0
        model: Model to use
        max_tokens: Maximum response tokens
        temperature: Generation temperature

    Returns:
        Generated therapeutic response
    """
    system_prompt = f"""{CBT_SYSTEM_PROMPT}

## Stored Memories About This Patient
{memories_context if memories_context else "(No memories stored yet)"}

Use these memories to inform your response while maintaining CBT principles.
"""

    user_prompt = f"""
## Recent Conversation
{conversation_context}

## Current Patient Statement
{patient_turn}

Provide a therapeutic response following CBT guidelines.
Keep response concise (2-4 sentences).
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"    Generation error: {e}")
        return "I hear what you're saying. Can you tell me more about that?"


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def analyze_cheatsheet_evolution(cheatsheet: TherapeuticCheatsheet) -> Dict[str, Any]:
    """
    Analyze how the cheatsheet evolved during the session.

    Args:
        cheatsheet: Final cheatsheet with extraction history

    Returns:
        Dictionary with evolution statistics
    """
    if not cheatsheet.extraction_history:
        return {"error": "No extraction history available"}

    total_extracted = sum(h["extracted_count"] for h in cheatsheet.extraction_history)
    total_filtered = sum(h["filtered_count"] for h in cheatsheet.extraction_history)

    return {
        "total_turns_processed": len(cheatsheet.extraction_history),
        "total_strategies_extracted": total_extracted,
        "total_content_filtered": total_filtered,
        "filter_ratio": total_filtered / (total_extracted + total_filtered) if (total_extracted + total_filtered) > 0 else 0,
        "final_stats": cheatsheet.get_stats(),
        "extraction_rate": total_extracted / len(cheatsheet.extraction_history) if cheatsheet.extraction_history else 0
    }


# ============================================================================
# SUMMARIZATION PROMPT
# ============================================================================

SUMMARIZATION_PROMPT_TEMPLATE = """
# THERAPEUTIC CHEATSHEET CONSOLIDATION

You are consolidating a therapeutic cheatsheet to reduce redundancy while preserving the most valuable strategies.

## Current Cheatsheet (needs consolidation)

### CBT Techniques ({cbt_count} items)
{cbt_items}

### Distortion Patterns ({pattern_count} items)
{pattern_items}

### Effective Interventions ({intervention_count} items)
{intervention_items}

### Boundary Templates ({boundary_count} items)
{boundary_items}

### Session Insights ({insight_count} items)
{insight_items}

## Consolidation Rules

1. **Merge similar items** - Combine strategies that target the same issue or use the same technique
2. **Keep the most actionable versions** - Prefer specific, applicable strategies over vague ones
3. **Preserve diversity** - Ensure different technique types are represented
4. **Maximum {target} items per category** - Consolidate down to this limit

## Output

Return JSON with consolidated lists (maximum {target} items per category):
{{
    "cbt_techniques": ["...", ...],
    "distortion_patterns": ["...", ...],
    "effective_interventions": ["...", ...],
    "boundary_templates": ["...", ...],
    "session_insights": ["...", ...]
}}
"""


def summarize_cheatsheet(
    client: OpenAI,
    cheatsheet: TherapeuticCheatsheet,
    model: str,
    target_items_per_category: int = 10
) -> TherapeuticCheatsheet:
    """
    Summarize/consolidate the cheatsheet using LLM to reduce redundancy.

    This should be called periodically (e.g., every 100 turn pairs) to prevent
    unbounded growth of the cheatsheet while preserving the most valuable strategies.

    Args:
        client: OpenAI-compatible client
        cheatsheet: Current cheatsheet to consolidate
        model: Model to use for summarization
        target_items_per_category: Maximum items to retain per category after consolidation

    Returns:
        New consolidated TherapeuticCheatsheet
    """
    stats = cheatsheet.get_stats()

    # Skip summarization if cheatsheet is small enough
    if stats["total"] <= target_items_per_category * 5:
        return cheatsheet

    # Format current items for the prompt
    def format_items(items: List[str]) -> str:
        if not items:
            return "(none)"
        return "\n".join(f"- {item}" for item in items)

    prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(
        cbt_count=len(cheatsheet.cbt_techniques),
        cbt_items=format_items(cheatsheet.cbt_techniques),
        pattern_count=len(cheatsheet.distortion_patterns),
        pattern_items=format_items(cheatsheet.distortion_patterns),
        intervention_count=len(cheatsheet.effective_interventions),
        intervention_items=format_items(cheatsheet.effective_interventions),
        boundary_count=len(cheatsheet.boundary_templates),
        boundary_items=format_items(cheatsheet.boundary_templates),
        insight_count=len(cheatsheet.session_insights),
        insight_items=format_items(cheatsheet.session_insights),
        target=target_items_per_category
    )

    try:
        raw_response = call_gpt4o_judge(client, prompt, model)
        parsed = parse_json_response(raw_response)
    except Exception as e:
        print(f"    Summarization error: {e}")
        # On failure, fall back to truncating to most recent items
        return TherapeuticCheatsheet(
            cbt_techniques=cheatsheet.cbt_techniques[-target_items_per_category:],
            distortion_patterns=cheatsheet.distortion_patterns[-target_items_per_category:],
            effective_interventions=cheatsheet.effective_interventions[-target_items_per_category:],
            boundary_templates=cheatsheet.boundary_templates[-target_items_per_category:],
            session_insights=cheatsheet.session_insights[-target_items_per_category:],
            extraction_history=cheatsheet.extraction_history.copy()
        )

    # Create new consolidated cheatsheet
    consolidated = TherapeuticCheatsheet(
        cbt_techniques=parsed.get("cbt_techniques", [])[:target_items_per_category] or [],
        distortion_patterns=parsed.get("distortion_patterns", [])[:target_items_per_category] or [],
        effective_interventions=parsed.get("effective_interventions", [])[:target_items_per_category] or [],
        boundary_templates=parsed.get("boundary_templates", [])[:target_items_per_category] or [],
        session_insights=parsed.get("session_insights", [])[:target_items_per_category] or [],
        extraction_history=cheatsheet.extraction_history.copy()
    )

    # Record summarization event in history
    consolidated.extraction_history.append({
        "turn_number": -1,  # Special marker for summarization
        "event": "summarization",
        "before_total": stats["total"],
        "after_total": consolidated.get_stats()["total"]
    })

    return consolidated


def should_summarize_dynamic(
    cheatsheet: TherapeuticCheatsheet,
    context_token_threshold: int = 2000,
    emergency_threshold: int = 4000,
    min_strategies: int = 15
) -> SummarizationDecision:
    """
    Determine if summarization should occur based on context token size.

    This replaces fixed-interval triggering with dynamic, context-aware triggering.
    Summarization is triggered when the cheatsheet context exceeds a token threshold.

    Args:
        cheatsheet: Current therapeutic cheatsheet
        context_token_threshold: Normal threshold for summarization trigger (default: 2000)
        emergency_threshold: Force summarization regardless of other factors (default: 4000)
        min_strategies: Minimum strategies before summarization is considered (default: 15)

    Returns:
        SummarizationDecision with should_summarize flag and reasoning
    """
    stats = cheatsheet.get_stats()
    prompt_text = cheatsheet.to_prompt_string()
    token_count = estimate_tokens(prompt_text)

    # Emergency: always summarize if way too large
    if token_count >= emergency_threshold:
        return SummarizationDecision(
            should_summarize=True,
            reason=f"emergency_threshold_exceeded ({token_count} >= {emergency_threshold})",
            current_tokens=token_count,
            strategy_count=stats["total"]
        )

    # Don't summarize tiny cheatsheets
    if stats["total"] < min_strategies:
        return SummarizationDecision(
            should_summarize=False,
            reason=f"too_few_strategies ({stats['total']} < {min_strategies})",
            current_tokens=token_count,
            strategy_count=stats["total"]
        )

    # Normal threshold check
    if token_count >= context_token_threshold:
        return SummarizationDecision(
            should_summarize=True,
            reason=f"context_threshold_exceeded ({token_count} >= {context_token_threshold})",
            current_tokens=token_count,
            strategy_count=stats["total"]
        )

    return SummarizationDecision(
        should_summarize=False,
        reason=f"within_limits ({token_count} tokens, {stats['total']} strategies)",
        current_tokens=token_count,
        strategy_count=stats["total"]
    )


# ============================================================================
# RLM-STYLE RETRIEVAL FUNCTIONS
# ============================================================================

@dataclass
class RetrievalResult:
    """Result from RLM-style strategy retrieval."""
    cbt_indices: List[int]
    pattern_indices: List[int]
    intervention_indices: List[int]
    boundary_indices: List[int]
    insight_indices: List[int]
    reasoning: str


def retrieve_relevant_strategies(
    client: OpenAI,
    patient_turn: str,
    cheatsheet: TherapeuticCheatsheet,
    model: str,
    max_per_category: int = 3
) -> RetrievalResult:
    """
    RLM-style retrieval: Use LLM to query the cheatsheet and select relevant strategies.

    Instead of dumping the entire cheatsheet into the prompt, we:
    1. Present the cheatsheet as an indexed "environment"
    2. Ask LLM to select relevant indices based on patient turn
    3. Return only the selected strategies for generation

    This is inspired by RLM (Recursive Language Models) where the prompt/data
    is treated as an external environment the LLM can query programmatically.

    Args:
        client: OpenAI-compatible client
        patient_turn: The current patient statement
        cheatsheet: The therapeutic cheatsheet to query
        model: Model to use for retrieval
        max_per_category: Maximum items to retrieve per category

    Returns:
        RetrievalResult with selected indices and reasoning
    """
    # Format cheatsheet as indexed environment
    cheatsheet_indexed = cheatsheet.format_for_retrieval()

    # If cheatsheet is empty, return empty result
    if "Empty cheatsheet" in cheatsheet_indexed:
        return RetrievalResult(
            cbt_indices=[],
            pattern_indices=[],
            intervention_indices=[],
            boundary_indices=[],
            insight_indices=[],
            reasoning="Empty cheatsheet - no strategies to retrieve"
        )

    prompt = RETRIEVAL_PROMPT_TEMPLATE.format(
        patient_turn=patient_turn,
        cheatsheet_indexed=cheatsheet_indexed
    )

    try:
        raw_response = call_gpt4o_judge(client, prompt, model)
        parsed = parse_json_response(raw_response)

        # Extract indices, clamping to max_per_category
        return RetrievalResult(
            cbt_indices=parsed.get("relevant_cbt_techniques", [])[:max_per_category],
            pattern_indices=parsed.get("relevant_distortion_patterns", [])[:max_per_category],
            intervention_indices=parsed.get("relevant_interventions", [])[:max_per_category],
            boundary_indices=parsed.get("relevant_boundaries", [])[:max_per_category],
            insight_indices=parsed.get("relevant_insights", [])[:max_per_category],
            reasoning=parsed.get("reasoning", "")
        )
    except Exception as e:
        print(f"    Retrieval error: {e}")
        # Fallback: return most recent items (not indices)
        return RetrievalResult(
            cbt_indices=list(range(max(0, len(cheatsheet.cbt_techniques) - max_per_category), len(cheatsheet.cbt_techniques))),
            pattern_indices=list(range(max(0, len(cheatsheet.distortion_patterns) - max_per_category), len(cheatsheet.distortion_patterns))),
            intervention_indices=list(range(max(0, len(cheatsheet.effective_interventions) - max_per_category), len(cheatsheet.effective_interventions))),
            boundary_indices=list(range(max(0, len(cheatsheet.boundary_templates) - max_per_category), len(cheatsheet.boundary_templates))),
            insight_indices=list(range(max(0, len(cheatsheet.session_insights) - max_per_category), len(cheatsheet.session_insights))),
            reasoning=f"Fallback to recent items due to error: {e}"
        )


def retrieve_with_chain_of_thought(
    client: OpenAI,
    patient_turn: str,
    cheatsheet: TherapeuticCheatsheet,
    model: str,
    max_per_category: int = 3
) -> CoTRetrievalResult:
    """
    Chain-of-thought retrieval: Analyze patient turn, then retrieve relevant strategies.

    This provides more thorough retrieval by:
    1. First analyzing the patient statement for distortions and themes (Phase 1)
    2. Then retrieving strategies based on that analysis (Phase 2)

    The analysis context is preserved and can be used in the generation prompt
    to provide richer therapeutic context.

    Args:
        client: OpenAI-compatible client
        patient_turn: The current patient statement
        cheatsheet: The therapeutic cheatsheet to query
        model: Model to use for retrieval
        max_per_category: Maximum items to retrieve per category

    Returns:
        CoTRetrievalResult with analysis and retrieved indices
    """
    cheatsheet_indexed = cheatsheet.format_for_retrieval()

    # If cheatsheet is empty, return empty result with no analysis
    if "Empty cheatsheet" in cheatsheet_indexed:
        return CoTRetrievalResult(
            identified_distortions=[],
            emotional_themes=[],
            recommended_approach="No strategies available yet",
            cbt_indices=[],
            pattern_indices=[],
            intervention_indices=[],
            boundary_indices=[],
            insight_indices=[],
            reasoning="Empty cheatsheet - no strategies to retrieve"
        )

    prompt = COT_RETRIEVAL_PROMPT_TEMPLATE.format(
        patient_turn=patient_turn,
        cheatsheet_indexed=cheatsheet_indexed
    )

    try:
        raw_response = call_gpt4o_judge(client, prompt, model)
        parsed = parse_json_response(raw_response)

        phase1 = parsed.get("phase1_analysis", {})
        phase2 = parsed.get("phase2_retrieval", {})

        return CoTRetrievalResult(
            identified_distortions=phase1.get("identified_distortions", []) or [],
            emotional_themes=phase1.get("emotional_themes", []) or [],
            recommended_approach=phase1.get("recommended_approach", "") or "",
            cbt_indices=(phase2.get("relevant_cbt_techniques", []) or [])[:max_per_category],
            pattern_indices=(phase2.get("relevant_distortion_patterns", []) or [])[:max_per_category],
            intervention_indices=(phase2.get("relevant_interventions", []) or [])[:max_per_category],
            boundary_indices=(phase2.get("relevant_boundaries", []) or [])[:max_per_category],
            insight_indices=(phase2.get("relevant_insights", []) or [])[:max_per_category],
            reasoning=parsed.get("retrieval_reasoning", "") or ""
        )
    except Exception as e:
        print(f"    CoT Retrieval error: {e}")
        # Fallback to standard retrieval
        standard_result = retrieve_relevant_strategies(
            client, patient_turn, cheatsheet, model, max_per_category
        )
        return CoTRetrievalResult(
            identified_distortions=[],
            emotional_themes=[],
            recommended_approach=f"Fallback due to error: {e}",
            cbt_indices=standard_result.cbt_indices,
            pattern_indices=standard_result.pattern_indices,
            intervention_indices=standard_result.intervention_indices,
            boundary_indices=standard_result.boundary_indices,
            insight_indices=standard_result.insight_indices,
            reasoning=standard_result.reasoning
        )


def build_retrieved_context(
    cheatsheet: TherapeuticCheatsheet,
    retrieval: RetrievalResult
) -> str:
    """
    Build a context string from retrieved strategy indices.

    Args:
        cheatsheet: The full cheatsheet
        retrieval: RetrievalResult with selected indices

    Returns:
        Formatted string with only the retrieved strategies
    """
    sections = []

    # Helper to safely get items by indices
    def get_by_indices(items: List[str], indices: List[int]) -> List[str]:
        return [items[i] for i in indices if 0 <= i < len(items)]

    cbt_items = get_by_indices(cheatsheet.cbt_techniques, retrieval.cbt_indices)
    if cbt_items:
        sections.append("## Relevant CBT Techniques\n" + "\n".join(f"- {t}" for t in cbt_items))

    pattern_items = get_by_indices(cheatsheet.distortion_patterns, retrieval.pattern_indices)
    if pattern_items:
        sections.append("## Relevant Distortion Patterns\n" + "\n".join(f"- {p}" for p in pattern_items))

    intervention_items = get_by_indices(cheatsheet.effective_interventions, retrieval.intervention_indices)
    if intervention_items:
        sections.append("## Relevant Interventions\n" + "\n".join(f"- {i}" for i in intervention_items))

    boundary_items = get_by_indices(cheatsheet.boundary_templates, retrieval.boundary_indices)
    if boundary_items:
        sections.append("## Relevant Boundaries\n" + "\n".join(f"- {b}" for b in boundary_items))

    insight_items = get_by_indices(cheatsheet.session_insights, retrieval.insight_indices)
    if insight_items:
        sections.append("## Relevant Insights\n" + "\n".join(f"- {s}" for s in insight_items))

    if not sections:
        return "(No relevant strategies retrieved)"

    return "\n\n".join(sections)


def generate_with_cheatsheet_rlm(
    client: OpenAI,
    patient_turn: str,
    cheatsheet: TherapeuticCheatsheet,
    model: str,
    use_retrieval: bool = True,
    retrieval_model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    verbose: bool = False
) -> Tuple[str, Optional[RetrievalResult]]:
    """
    Generate therapeutic response using RLM-style retrieval from cheatsheet.

    This is the hybrid approach that combines:
    1. Periodic summarization (to consolidate strategies)
    2. RLM-style retrieval (to select relevant strategies per turn)

    Instead of dumping the entire cheatsheet into the prompt, we:
    1. First use LLM to query which strategies are relevant to this patient turn
    2. Then include only those strategies in the generation prompt

    This keeps the generation context focused and allows the cheatsheet to grow
    larger without hitting context limits.

    Args:
        client: OpenAI-compatible client
        patient_turn: The current patient statement
        cheatsheet: The accumulated therapeutic cheatsheet
        model: Model to use for generation
        use_retrieval: If True, use RLM retrieval. If False, use standard to_prompt_string()
        retrieval_model: Model for retrieval step (defaults to same as generation model)
        temperature: Generation temperature
        max_tokens: Maximum tokens in response
        verbose: Print debug info

    Returns:
        Tuple of (generated_response, retrieval_result)
    """
    retrieval_result = None

    if use_retrieval and cheatsheet.get_stats()["total"] > 0:
        # Use RLM-style retrieval
        if verbose:
            print("    [RLM Retrieval: querying cheatsheet...]")

        retrieval_result = retrieve_relevant_strategies(
            client=client,
            patient_turn=patient_turn,
            cheatsheet=cheatsheet,
            model=retrieval_model or model
        )

        if verbose:
            total_retrieved = (
                len(retrieval_result.cbt_indices) +
                len(retrieval_result.pattern_indices) +
                len(retrieval_result.intervention_indices) +
                len(retrieval_result.boundary_indices) +
                len(retrieval_result.insight_indices)
            )
            print(f"    [Retrieved {total_retrieved} relevant strategies]")

        cheatsheet_context = build_retrieved_context(cheatsheet, retrieval_result)
    else:
        # Fallback to standard approach
        cheatsheet_context = cheatsheet.to_prompt_string()

    # Now generate with the retrieved context
    system_prompt = f"""{CBT_SYSTEM_PROMPT}

## Therapeutic Strategies (RLM-Retrieved)
{cheatsheet_context}

Apply these strategies naturally in your response. Focus on the patterns identified.
"""

    user_prompt = f"""
## Current Patient Statement
{patient_turn}

Provide a therapeutic response following CBT guidelines.
Keep response concise (2-4 sentences).
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content or "", retrieval_result
    except Exception as e:
        print(f"    Generation error: {e}")
        return "I hear what you're saying. Can you tell me more about that?", retrieval_result


def generate_with_dynamic_retrieval(
    client: OpenAI,
    patient_turn: str,
    cheatsheet: TherapeuticCheatsheet,
    model: str,
    use_cot_retrieval: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 500,
    verbose: bool = False
) -> Tuple[str, Optional[CoTRetrievalResult]]:
    """
    Generate therapeutic response with dynamic chain-of-thought retrieval.

    This is the enhanced version that:
    1. Uses chain-of-thought analysis before retrieval (Phase 1)
    2. Includes the analysis context in generation prompt
    3. Provides richer context for therapeutic response

    The key difference from standard RLM retrieval is that the analysis
    (identified distortions, emotional themes, recommended approach) is
    preserved and included in the generation prompt.

    Args:
        client: OpenAI-compatible client
        patient_turn: The current patient statement
        cheatsheet: The accumulated therapeutic cheatsheet
        model: Model to use for generation
        use_cot_retrieval: If True, use chain-of-thought retrieval
        temperature: Generation temperature
        max_tokens: Maximum tokens in response
        verbose: Print debug info

    Returns:
        Tuple of (generated_response, cot_retrieval_result)
    """
    cot_result = None

    if use_cot_retrieval and cheatsheet.get_stats()["total"] > 0:
        if verbose:
            print("    [CoT Retrieval: analyzing patient turn...]")

        cot_result = retrieve_with_chain_of_thought(
            client=client,
            patient_turn=patient_turn,
            cheatsheet=cheatsheet,
            model=model
        )

        if verbose:
            print(f"    [Analysis: {len(cot_result.identified_distortions)} distortions, "
                  f"{len(cot_result.emotional_themes)} themes]")
            print(f"    [Retrieved {cot_result.get_total_retrieved()} strategies]")

        # Convert CoT result to RetrievalResult for build_retrieved_context
        retrieval_for_context = RetrievalResult(
            cbt_indices=cot_result.cbt_indices,
            pattern_indices=cot_result.pattern_indices,
            intervention_indices=cot_result.intervention_indices,
            boundary_indices=cot_result.boundary_indices,
            insight_indices=cot_result.insight_indices,
            reasoning=cot_result.reasoning
        )
        cheatsheet_context = build_retrieved_context(cheatsheet, retrieval_for_context)

        # Build analysis context for the prompt
        analysis_parts = []
        if cot_result.identified_distortions:
            analysis_parts.append("## Identified Distortions\n- " + "\n- ".join(cot_result.identified_distortions))
        if cot_result.emotional_themes:
            analysis_parts.append("## Emotional Themes\n- " + "\n- ".join(cot_result.emotional_themes))
        if cot_result.recommended_approach:
            analysis_parts.append(f"## Recommended Approach\n{cot_result.recommended_approach}")
        analysis_context = "\n\n".join(analysis_parts) if analysis_parts else "(No analysis available)"
    else:
        cheatsheet_context = cheatsheet.to_prompt_string()
        analysis_context = "(No prior analysis available)"

    # Generate with the retrieved context AND analysis
    system_prompt = f"""{CBT_SYSTEM_PROMPT}

## Clinical Analysis (Pre-Generation)
{analysis_context}

## Therapeutic Strategies (Retrieved)
{cheatsheet_context}

Apply these strategies naturally in your response. Address the identified patterns and themes.
"""

    user_prompt = f"""
## Current Patient Statement
{patient_turn}

Provide a therapeutic response following CBT guidelines.
Keep response concise (2-4 sentences).
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content or "", cot_result
    except Exception as e:
        print(f"    Generation error: {e}")
        return "I hear what you're saying. Can you tell me more about that?", cot_result


def compare_conditions(
    baseline_results: Dict[str, Any],
    mem0_results: Optional[Dict[str, Any]],
    dcrs_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare results across experimental conditions.

    Args:
        baseline_results: Results from baseline condition
        mem0_results: Results from mem0 condition (optional)
        dcrs_results: Results from DC-RS condition

    Returns:
        Comparison statistics
    """
    comparison = {
        "baseline": {
            "cbt_mean": baseline_results.get("cbt_mean", 0),
            "persona_mean": baseline_results.get("persona_mean", 0)
        },
        "dcrs": {
            "cbt_mean": dcrs_results.get("cbt_mean", 0),
            "persona_mean": dcrs_results.get("persona_mean", 0)
        },
        "improvements": {
            "dcrs_vs_baseline_cbt": dcrs_results.get("cbt_mean", 0) - baseline_results.get("cbt_mean", 0),
            "dcrs_vs_baseline_persona": dcrs_results.get("persona_mean", 0) - baseline_results.get("persona_mean", 0)
        }
    }

    if mem0_results:
        comparison["mem0"] = {
            "cbt_mean": mem0_results.get("cbt_mean", 0),
            "persona_mean": mem0_results.get("persona_mean", 0)
        }
        comparison["improvements"]["dcrs_vs_mem0_cbt"] = dcrs_results.get("cbt_mean", 0) - mem0_results.get("cbt_mean", 0)
        comparison["improvements"]["dcrs_vs_mem0_persona"] = dcrs_results.get("persona_mean", 0) - mem0_results.get("persona_mean", 0)

    return comparison


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    print("Dynamic Cheatsheet Module")
    print("=" * 60)

    # Test TherapeuticCheatsheet
    cheatsheet = TherapeuticCheatsheet()
    cheatsheet.cbt_techniques.append("For catastrophizing: ask 'What evidence supports this?'")
    cheatsheet.distortion_patterns.append("All-or-nothing thinking about work performance")
    cheatsheet.effective_interventions.append("Graduated scaling (1-10) helps with absolute statements")

    print("\nTest Cheatsheet:")
    print(cheatsheet.to_prompt_string())
    print(f"\nStats: {cheatsheet.get_stats()}")

    print("\nModule loaded successfully!")
    print("Available functions:")
    print("  - extract_strategies(): TEST-TIME LEARNING - extract curated strategies")
    print("  - generate_with_cheatsheet(): Generate response using DC-RS")
    print("  - generate_baseline(): Generate response without memory")
    print("  - generate_with_memories(): Generate response with mem0")
