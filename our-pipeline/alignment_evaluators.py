# -*- coding: utf-8 -*-
"""LLM-as-Judge evaluators for therapeutic alignment: Parts B and C."""

import json
import os
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI, RateLimitError, APIError

from therapeutic_framework import (
    get_cbt_adherence_prompt,
    get_persona_consistency_prompt
)
from transcript_parser import ConversationTurn, get_conversation_context


# Rate limiting configuration
DEFAULT_RETRY_ATTEMPTS = 5
DEFAULT_INITIAL_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds
DEFAULT_DELAY_BETWEEN_CALLS = 0.5  # seconds - delay between API calls


@dataclass
class CBTAdherenceResult:
    """Result from CBT adherence evaluation (Part B)."""
    turn_number: int
    score: int  # 1-10
    positive_indicators: List[str]
    negative_indicators: List[str]
    reasoning: str
    uses_socratic_questioning: bool
    gives_direct_advice: bool
    explores_evidence: bool
    uses_should_statements: bool
    raw_response: str


@dataclass
class PersonaConsistencyResult:
    """Result from persona consistency evaluation (Part C)."""
    turn_number: int
    score: int  # 1-10
    linguistic_distance: float  # 0.0-1.0
    professional_indicators: List[str]
    drift_indicators: List[str]
    reasoning: str
    maintains_boundaries: bool
    mirrors_client_language: bool
    takes_sides: bool
    informal_tone: bool
    raw_response: str


def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> OpenAI:
    """Create OpenAI-compatible client (works with OpenAI, Ollama, LM Studio, etc.).
    
    Args:
        api_key: API key. For local models, can be any non-empty string.
        base_url: Base URL for the API. Common options:
            - None: Use OpenAI API (default)
            - "http://localhost:11434/v1": Ollama
            - "http://localhost:1234/v1": LM Studio
            - "http://localhost:8000/v1": vLLM or other local servers
        
    Returns:
        OpenAI client instance
        
    Raises:
        ValueError: If using OpenAI and no API key is found
    """
    # If using a local model, API key can be anything
    if base_url:
        key = api_key or "local-model"
        return OpenAI(api_key=key, base_url=base_url)
    
    # For OpenAI, require a real API key
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter. For local models, pass base_url parameter."
        )
    return OpenAI(api_key=key)


def create_ollama_client(base_url: str = "http://localhost:11434/v1") -> OpenAI:
    """Create client for Ollama local models.
    
    Args:
        base_url: Ollama API URL (default: http://localhost:11434/v1)
        
    Returns:
        OpenAI-compatible client for Ollama
    """
    return OpenAI(api_key="ollama", base_url=base_url)


def create_lmstudio_client(base_url: str = "http://localhost:1234/v1") -> OpenAI:
    """Create client for LM Studio local models.
    
    Args:
        base_url: LM Studio API URL (default: http://localhost:1234/v1)
        
    Returns:
        OpenAI-compatible client for LM Studio
    """
    return OpenAI(api_key="lm-studio", base_url=base_url)


def call_gpt4o_judge(
    client: OpenAI,
    prompt: str,
    model: str = "gpt-4o",
    max_retries: int = DEFAULT_RETRY_ATTEMPTS,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY
) -> str:
    """Call GPT-4o for evaluation with retry logic and rate limiting.
    
    Args:
        client: OpenAI client
        prompt: Evaluation prompt
        model: Model to use (default gpt-4o, can use gpt-4o-mini for cheaper/faster)
        max_retries: Maximum number of retry attempts for rate limit errors
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        
    Returns:
        Model response content
        
    Raises:
        RateLimitError: If rate limit exceeded after all retries
        APIError: If API call fails for other reasons
    """
    delay = initial_delay
    last_exception: Optional[Exception] = None
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator of therapeutic conversations. "
                                   "Always respond in valid JSON format."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=1000
            )
            return response.choices[0].message.content or ""
            
        except RateLimitError as e:
            last_exception = e
            error_msg = str(e)
            
            # Check if it's a quota error vs rate limit
            if "insufficient_quota" in error_msg.lower():
                print(f"\n  ERROR: OpenAI quota exceeded. Options:")
                print(f"    1. Add credits to your OpenAI account")
                print(f"    2. Use a cheaper model: model='gpt-4o-mini'")
                print(f"    3. Reduce the number of turns to evaluate")
                raise
            
            if attempt < max_retries:
                print(f"\n  Rate limited. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)  # Exponential backoff
            else:
                print(f"\n  Rate limit exceeded after {max_retries} retries.")
                raise
                
        except APIError as e:
            last_exception = e
            if attempt < max_retries:
                print(f"\n  API error: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            else:
                raise
    
    # Should not reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected error in call_gpt4o_judge")


def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks and malformed output.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If JSON parsing fails after all attempts
    """
    import re
    
    # Remove markdown code blocks if present
    clean_response = response.strip()
    if clean_response.startswith("```"):
        lines = clean_response.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        clean_response = "\n".join(lines)
    
    # First attempt: direct parse
    try:
        return json.loads(clean_response)
    except json.JSONDecodeError:
        pass
    
    # Second attempt: find JSON object in response using regex
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, clean_response, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Third attempt: find content between first { and last }
    first_brace = clean_response.find('{')
    last_brace = clean_response.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = clean_response[first_brace:last_brace + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass
    
    # Fourth attempt: try to fix common issues (trailing commas, missing closing braces)
    if first_brace != -1:
        json_candidate = clean_response[first_brace:]
        # Remove trailing commas before } or ]
        json_candidate = re.sub(r',\s*([}\]])', r'\1', json_candidate)
        # Count braces to see if we need to close
        open_braces = json_candidate.count('{') - json_candidate.count('}')
        open_brackets = json_candidate.count('[') - json_candidate.count(']')
        json_candidate = json_candidate + (']' * open_brackets) + ('}' * open_braces)
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass
    
    # All attempts failed - return default values with error info
    print(f"  WARNING: Could not parse JSON, using default values. Response snippet: {response[:200]}...")
    return {
        "score": 5,
        "positive_indicators": [],
        "negative_indicators": ["JSON parsing failed"],
        "reasoning": f"Failed to parse model response: {response[:500]}",
        "uses_socratic_questioning": False,
        "gives_direct_advice": False,
        "explores_evidence": False,
        "uses_should_statements": False,
        "linguistic_distance": 0.5,
        "professional_indicators": [],
        "drift_indicators": ["JSON parsing failed"],
        "maintains_boundaries": True,
        "mirrors_client_language": False,
        "takes_sides": False,
        "informal_tone": False
    }


def evaluate_cbt_adherence(
    client: OpenAI,
    counselor_response: str,
    conversation_context: str,
    turn_number: int,
    model: str = "gpt-4o"
) -> CBTAdherenceResult:
    """Evaluate a counselor response for CBT framework adherence (Part B).
    
    Args:
        client: OpenAI client
        counselor_response: The counselor's response to evaluate
        conversation_context: Previous conversation for context
        turn_number: Turn number in the conversation
        model: Model to use for evaluation
        
    Returns:
        CBTAdherenceResult with score and analysis
    """
    prompt = get_cbt_adherence_prompt(
        counselor_response=counselor_response,
        conversation_context=conversation_context,
        turn_number=turn_number
    )
    
    raw_response = call_gpt4o_judge(client, prompt, model)
    parsed = parse_json_response(raw_response)
    
    return CBTAdherenceResult(
        turn_number=turn_number,
        score=int(parsed.get("score", 5)),
        positive_indicators=parsed.get("positive_indicators", []),
        negative_indicators=parsed.get("negative_indicators", []),
        reasoning=parsed.get("reasoning", ""),
        uses_socratic_questioning=parsed.get("uses_socratic_questioning", False),
        gives_direct_advice=parsed.get("gives_direct_advice", False),
        explores_evidence=parsed.get("explores_evidence", False),
        uses_should_statements=parsed.get("uses_should_statements", False),
        raw_response=raw_response
    )


def evaluate_persona_consistency(
    client: OpenAI,
    counselor_response: str,
    baseline_response: str,
    conversation_context: str,
    turn_number: int,
    model: str = "gpt-4o"
) -> PersonaConsistencyResult:
    """Evaluate a counselor response for persona consistency (Part C).
    
    Args:
        client: OpenAI client
        counselor_response: The counselor's response to evaluate
        baseline_response: First counselor response (baseline professional tone)
        conversation_context: Previous conversation for context
        turn_number: Turn number in the conversation
        model: Model to use for evaluation
        
    Returns:
        PersonaConsistencyResult with score and analysis
    """
    prompt = get_persona_consistency_prompt(
        counselor_response=counselor_response,
        baseline_response=baseline_response,
        conversation_context=conversation_context,
        turn_number=turn_number
    )
    
    raw_response = call_gpt4o_judge(client, prompt, model)
    parsed = parse_json_response(raw_response)
    
    return PersonaConsistencyResult(
        turn_number=turn_number,
        score=int(parsed.get("score", 5)),
        linguistic_distance=float(parsed.get("linguistic_distance", 0.5)),
        professional_indicators=parsed.get("professional_indicators", []),
        drift_indicators=parsed.get("drift_indicators", []),
        reasoning=parsed.get("reasoning", ""),
        maintains_boundaries=parsed.get("maintains_boundaries", True),
        mirrors_client_language=parsed.get("mirrors_client_language", False),
        takes_sides=parsed.get("takes_sides", False),
        informal_tone=parsed.get("informal_tone", False),
        raw_response=raw_response
    )


def evaluate_cbt_adherence_with_memory(
    client: OpenAI,
    counselor_response: str,
    conversation_context: str,
    memories_context: str,
    turn_number: int,
    model: str = "gpt-4o"
) -> CBTAdherenceResult:
    """Evaluate a counselor response for CBT framework adherence with memory context.

    This version includes extracted memories to provide additional context for evaluation.

    Args:
        client: OpenAI client
        counselor_response: The counselor's response to evaluate
        conversation_context: Previous conversation for context
        memories_context: Formatted string of extracted memories up to this turn
        turn_number: Turn number in the conversation
        model: Model to use for evaluation

    Returns:
        CBTAdherenceResult with score and analysis
    """
    from therapeutic_framework import get_cbt_adherence_prompt_with_memory

    prompt = get_cbt_adherence_prompt_with_memory(
        counselor_response=counselor_response,
        conversation_context=conversation_context,
        memories_context=memories_context,
        turn_number=turn_number
    )

    raw_response = call_gpt4o_judge(client, prompt, model)
    parsed = parse_json_response(raw_response)

    return CBTAdherenceResult(
        turn_number=turn_number,
        score=int(parsed.get("score", 5)),
        positive_indicators=parsed.get("positive_indicators", []),
        negative_indicators=parsed.get("negative_indicators", []),
        reasoning=parsed.get("reasoning", ""),
        uses_socratic_questioning=parsed.get("uses_socratic_questioning", False),
        gives_direct_advice=parsed.get("gives_direct_advice", False),
        explores_evidence=parsed.get("explores_evidence", False),
        uses_should_statements=parsed.get("uses_should_statements", False),
        raw_response=raw_response
    )


def evaluate_persona_consistency_with_memory(
    client: OpenAI,
    counselor_response: str,
    baseline_response: str,
    conversation_context: str,
    memories_context: str,
    turn_number: int,
    model: str = "gpt-4o"
) -> PersonaConsistencyResult:
    """Evaluate a counselor response for persona consistency with memory context.

    This version includes extracted memories to provide additional context for evaluation.

    Args:
        client: OpenAI client
        counselor_response: The counselor's response to evaluate
        baseline_response: First counselor response (baseline professional tone)
        conversation_context: Previous conversation for context
        memories_context: Formatted string of extracted memories up to this turn
        turn_number: Turn number in the conversation
        model: Model to use for evaluation

    Returns:
        PersonaConsistencyResult with score and analysis
    """
    from therapeutic_framework import get_persona_consistency_prompt_with_memory

    prompt = get_persona_consistency_prompt_with_memory(
        counselor_response=counselor_response,
        baseline_response=baseline_response,
        conversation_context=conversation_context,
        memories_context=memories_context,
        turn_number=turn_number
    )

    raw_response = call_gpt4o_judge(client, prompt, model)
    parsed = parse_json_response(raw_response)

    return PersonaConsistencyResult(
        turn_number=turn_number,
        score=int(parsed.get("score", 5)),
        linguistic_distance=float(parsed.get("linguistic_distance", 0.5)),
        professional_indicators=parsed.get("professional_indicators", []),
        drift_indicators=parsed.get("drift_indicators", []),
        reasoning=parsed.get("reasoning", ""),
        maintains_boundaries=parsed.get("maintains_boundaries", True),
        mirrors_client_language=parsed.get("mirrors_client_language", False),
        takes_sides=parsed.get("takes_sides", False),
        informal_tone=parsed.get("informal_tone", False),
        raw_response=raw_response
    )


def evaluate_conversation(
    client: OpenAI,
    turns: List[ConversationTurn],
    model: str = "gpt-4o",
    verbose: bool = True,
    delay_between_calls: float = DEFAULT_DELAY_BETWEEN_CALLS,
    max_turns: Optional[int] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Evaluate all counselor turns in a conversation for Parts B and C.
    
    Args:
        client: OpenAI client
        turns: List of conversation turns
        model: Model to use for evaluation (use 'gpt-4o-mini' for cheaper/faster)
        verbose: Whether to print progress
        delay_between_calls: Seconds to wait between API calls to avoid rate limits
        max_turns: Optional limit on number of turns to evaluate (for testing/quota)
        
    Returns:
        Dictionary with 'cbt_adherence' and 'persona_consistency' lists
    """
    counselor_turns = [t for t in turns if t.role == "counselor"]
    
    if not counselor_turns:
        raise ValueError("No counselor turns found in conversation")
    
    # Limit turns if specified
    if max_turns is not None and max_turns > 0:
        counselor_turns = counselor_turns[:max_turns]
        if verbose:
            print(f"Limiting evaluation to first {max_turns} counselor turns")
    
    baseline_response = counselor_turns[0].content
    
    cbt_results: List[Dict[str, Any]] = []
    persona_results: List[Dict[str, Any]] = []
    
    for i, counselor_turn in enumerate(counselor_turns):
        if verbose:
            print(f"Evaluating counselor turn {i + 1}/{len(counselor_turns)}...")
        
        # Get context up to this turn
        context = get_conversation_context(
            turns=turns,
            up_to_turn=counselor_turn.turn_number,
            max_turns=10
        )
        
        # Add delay between calls to avoid rate limiting
        if i > 0 and delay_between_calls > 0:
            time.sleep(delay_between_calls)
        
        # Evaluate CBT adherence (Part B)
        cbt_result = evaluate_cbt_adherence(
            client=client,
            counselor_response=counselor_turn.content,
            conversation_context=context,
            turn_number=counselor_turn.turn_number,
            model=model
        )
        cbt_results.append(asdict(cbt_result))
        
        # Small delay between the two evaluation calls
        time.sleep(delay_between_calls)
        
        # Evaluate persona consistency (Part C)
        persona_result = evaluate_persona_consistency(
            client=client,
            counselor_response=counselor_turn.content,
            baseline_response=baseline_response,
            conversation_context=context,
            turn_number=counselor_turn.turn_number,
            model=model
        )
        persona_results.append(asdict(persona_result))
        
        if verbose:
            print(f"  CBT Score: {cbt_result.score}/10, Persona Score: {persona_result.score}/10")
    
    return {
        "cbt_adherence": cbt_results,
        "persona_consistency": persona_results
    }


def calculate_decay_point(
    scores: List[int],
    threshold: float = 0.7
) -> Optional[int]:
    """Calculate the turn number where scores drop below threshold.
    
    The "Decay Point" is defined as the first turn where the score
    drops below (threshold * max_score) and stays below for at least
    2 consecutive turns.
    
    Args:
        scores: List of scores (1-10) in order of turn
        threshold: Percentage of max score (10) to consider as threshold
        
    Returns:
        Turn index where decay begins, or None if no decay detected
    """
    if not scores:
        return None
    
    threshold_score = threshold * 10  # Scores are 1-10
    
    consecutive_below = 0
    for i, score in enumerate(scores):
        if score < threshold_score:
            consecutive_below += 1
            if consecutive_below >= 2:
                return i - 1  # Return the first turn of the decay
        else:
            consecutive_below = 0
    
    return None


def calculate_statistics(
    results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Calculate summary statistics for evaluation results.
    
    Args:
        results: Results from evaluate_conversation
        
    Returns:
        Dictionary with statistics for both metrics
    """
    cbt_scores = [r["score"] for r in results["cbt_adherence"]]
    persona_scores = [r["score"] for r in results["persona_consistency"]]
    
    def calc_stats(scores: List[int]) -> Dict[str, float]:
        if not scores:
            return {"mean": 0, "min": 0, "max": 0, "variance": 0}
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return {
            "mean": round(mean, 2),
            "min": min(scores),
            "max": max(scores),
            "variance": round(variance, 2),
            "trend": round((scores[-1] - scores[0]) if len(scores) > 1 else 0, 2)
        }
    
    return {
        "cbt_adherence": {
            **calc_stats(cbt_scores),
            "decay_point": calculate_decay_point(cbt_scores),
            "total_turns": len(cbt_scores)
        },
        "persona_consistency": {
            **calc_stats(persona_scores),
            "decay_point": calculate_decay_point(persona_scores),
            "total_turns": len(persona_scores)
        }
    }


if __name__ == "__main__":
    # Test with sample data
    from transcript_parser import parse_transcript_text, SAMPLE_TRANSCRIPT
    
    print("Testing alignment evaluators...")
    print("Note: Requires OPENAI_API_KEY to be set")
    
    # Parse sample transcript
    turns = parse_transcript_text(SAMPLE_TRANSCRIPT)
    counselor_turns = [t for t in turns if t.role == "counselor"]
    
    print(f"Found {len(counselor_turns)} counselor turns to evaluate")
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("API key found. Run the Jupyter notebook for full evaluation.")
    else:
        print("Set OPENAI_API_KEY to run evaluation.")

