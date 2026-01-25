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
