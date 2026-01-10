# -*- coding: utf-8 -*-
"""LLM Counselor module for generating CBT therapeutic responses."""

from openai import OpenAI
from typing import Optional
from therapeutic_framework import CBT_SYSTEM_PROMPT


def generate_counselor_response(
    client: OpenAI,
    patient_query: str,
    conversation_context: str,
    memories_context: Optional[str] = None,
    turn_number: int = 1,
    model: str = "gpt-4o",
    temperature: float = 0.7
) -> str:
    """Generate a CBT counselor response to a patient query.

    This function creates LLM-generated therapeutic responses that follow
    CBT guidelines. The generated responses are then evaluated by the
    alignment evaluators to measure CBT adherence and persona consistency.

    Args:
        client: OpenAI client instance (supports OpenAI, Ollama, LM Studio)
        patient_query: The current patient statement/question
        conversation_context: Recent conversation history (sliding window)
        memories_context: Optional formatted memories (if using mem0)
        turn_number: Current turn number
        model: LLM model to use for generation
        temperature: Sampling temperature (default 0.7 for natural variation)

    Returns:
        Generated counselor response string

    Example:
        >>> client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        >>> response = generate_counselor_response(
        ...     client=client,
        ...     patient_query="I've been feeling really anxious lately",
        ...     conversation_context="",
        ...     memories_context=None,
        ...     turn_number=1,
        ...     model="gpt-oss:20b",
        ...     temperature=0.7
        ... )
        >>> print(response)
        "I hear that you're experiencing anxiety. Can you help me understand
        what situations tend to trigger these anxious feelings?"
    """
    # Build prompt with CBT guidelines
    prompt_parts = [
        f"## Conversation Context (Turn {turn_number}):",
        conversation_context if conversation_context else "This is the beginning of the conversation.",
        ""
    ]

    # Add memories if provided (for memincluded version)
    if memories_context:
        prompt_parts.extend([
            f"## Stored Memories (Facts Extracted Up to Turn {turn_number}):",
            memories_context,
            ""
        ])

    prompt_parts.extend([
        f"## Current Patient Statement:",
        patient_query,
        "",
        "## Your Task:",
        "As a CBT therapist, provide a therapeutic response to the patient's statement.",
        "Follow the Core CBT Guidelines and maintain a professional, warm demeanor.",
        "Your response should be 2-4 sentences."
    ])

    user_prompt = "\n".join(prompt_parts)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CBT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    print("LLM Counselor Module")
    print("=" * 60)
    print("This module generates CBT-aligned therapeutic responses.")
    print("\nKey function:")
    print("  - generate_counselor_response(): Generate LLM counselor response")
    print("\nUsage:")
    print("  1. Initialize OpenAI-compatible client (OpenAI, Ollama, LM Studio)")
    print("  2. Call generate_counselor_response() with patient query")
    print("  3. Evaluate the generated response with alignment_evaluators")
    print("\nNote: Temperature = 0.7 allows for natural variation while")
    print("maintaining control. Adjust if responses are too repetitive (increase)")
    print("or too chaotic (decrease).")
