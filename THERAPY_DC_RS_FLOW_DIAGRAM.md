# Therapy DC-RS Data Flow Diagram

## Models Used

- **LLM Counselor**: `gpt-oss:20b` (Lambda Cloud GPU via Ollama)
- **LLM Judge**: `gpt-oss:20b` (Same model for evaluation)
- **Mem0 Embedder**: Uses same model for memory extraction

---

## Three-Condition Data Flow

```mermaid
graph TB
    subgraph Input
        PT[Patient Turn<br/>Raw text from transcript]
        TURNS[All Previous Turns]
    end

    subgraph Context_Gathering
        PT --> CTX[get_conversation_context]
        TURNS --> CTX
        CTX --> CONTEXT[Context: Last 10 turns]
    end

    %% ========================================================================
    %% BASELINE CONDITION
    %% ========================================================================
    subgraph Baseline_Condition["🔴 BASELINE (No Memory)"]
        direction TB

        B_INPUT[Patient Turn + Context]

        B_PROMPT["<b>PROMPT:</b><br/>System: CBT_SYSTEM_PROMPT<br/>Context: Last 10 turns<br/>User: Patient turn"]

        B_LLM["🤖 LLM Counselor<br/>gpt-oss:20b"]

        B_RESPONSE[Generated Response]

        B_INPUT --> B_PROMPT
        B_PROMPT --> B_LLM
        B_LLM --> B_RESPONSE

        style B_INPUT fill:#ffe6e6
        style B_PROMPT fill:#ffcccc
        style B_LLM fill:#ff9999
        style B_RESPONSE fill:#ff6666
    end

    %% ========================================================================
    %% MEM0 CONDITION
    %% ========================================================================
    subgraph Mem0_Condition["🟠 MEM0 (Static Memory)"]
        direction TB

        M_INPUT[Patient Turn ONLY]

        M_RETRIEVE["<b>RETRIEVE MEMORIES:</b><br/>get_all_memories(user_id)<br/>↓<br/>ChromaDB vector search"]

        M_MEMORIES[Retrieved Memories<br/>Patient facts, preferences, history]

        M_PROMPT["<b>PROMPT:</b><br/>System: CBT_SYSTEM_PROMPT<br/><b>Memories:</b> Retrieved facts<br/>User: Patient turn<br/>(NO context window)"]

        M_LLM["🤖 LLM Counselor<br/>gpt-oss:20b"]

        M_RESPONSE[Generated Response]

        M_STORE["<b>STORE TO MEM0:</b><br/>add_conversation_turn_to_memory()<br/>↓<br/>Patient turn → ChromaDB<br/>Counselor response → ChromaDB"]

        M_INPUT --> M_RETRIEVE
        M_RETRIEVE --> M_MEMORIES
        M_MEMORIES --> M_PROMPT
        M_INPUT --> M_PROMPT
        M_PROMPT --> M_LLM
        M_LLM --> M_RESPONSE
        M_RESPONSE --> M_STORE

        style M_INPUT fill:#fff4e6
        style M_RETRIEVE fill:#ffe6cc
        style M_MEMORIES fill:#ffd9b3
        style M_PROMPT fill:#ffcc99
        style M_LLM fill:#ffb366
        style M_RESPONSE fill:#ff9933
        style M_STORE fill:#ff8000
    end

    %% ========================================================================
    %% DC-RS CONDITION
    %% ========================================================================
    subgraph DCRS_Condition["🔵 DC-RS (Continual Learning)"]
        direction TB

        D_INPUT[Patient Turn ONLY]

        D_CHEATSHEET[TherapeuticCheatsheet<br/>cbt_techniques: list<br/>distortion_patterns: list<br/>effective_interventions: list]

        D_PROMPT["<b>PROMPT:</b><br/>System: CBT_SYSTEM_PROMPT<br/><b>Cheatsheet:</b> Strategies<br/>User: Patient turn<br/>(NO context window)"]

        D_LLM1["🤖 LLM Counselor<br/>gpt-oss:20b"]

        D_RESPONSE[Generated Response]

        D_EXTRACT_PROMPT["<b>EXTRACTION PROMPT:</b><br/>Analyze patient-counselor interaction<br/>Extract therapeutic strategies<br/>Filter non-clinical content<br/>↓<br/>Returns: CBT techniques, patterns, interventions"]

        D_LLM2["🤖 LLM Extractor<br/>gpt-oss:20b<br/>(TEST-TIME LEARNING)"]

        D_UPDATE[Updated Cheatsheet<br/>New strategies added<br/>Distortions filtered]

        D_INPUT --> D_PROMPT
        D_CHEATSHEET --> D_PROMPT
        D_PROMPT --> D_LLM1
        D_LLM1 --> D_RESPONSE

        D_RESPONSE --> D_EXTRACT_PROMPT
        D_INPUT --> D_EXTRACT_PROMPT
        D_CHEATSHEET --> D_EXTRACT_PROMPT
        D_EXTRACT_PROMPT --> D_LLM2
        D_LLM2 --> D_UPDATE
        D_UPDATE -.->|"Next turn"| D_CHEATSHEET

        style D_INPUT fill:#e6f2ff
        style D_CHEATSHEET fill:#cce5ff
        style D_PROMPT fill:#b3d9ff
        style D_LLM1 fill:#99ccff
        style D_RESPONSE fill:#66b3ff
        style D_EXTRACT_PROMPT fill:#3399ff
        style D_LLM2 fill:#0080ff
        style D_UPDATE fill:#0066cc
    end

    %% ========================================================================
    %% EVALUATION (SAME FOR ALL)
    %% ========================================================================
    subgraph Evaluation["⚖️ EVALUATION (All Conditions)"]
        direction TB

        E_INPUT[Generated Response + Context]

        E_CBT_PROMPT["<b>CBT ADHERENCE PROMPT:</b><br/>Rate counselor response on CBT principles<br/>Score: 0-10<br/>Criteria: Socratic questioning, validation,<br/>cognitive restructuring, etc."]

        E_CBT_LLM["🤖 LLM Judge<br/>gpt-oss:20b"]

        E_CBT_SCORE[CBT Score: 0-10<br/>+ Reasoning]

        E_PERSONA_PROMPT["<b>PERSONA CONSISTENCY PROMPT:</b><br/>Compare to baseline therapeutic stance<br/>Score: 0-10<br/>Criteria: Professional tone, empathy,<br/>therapeutic boundaries"]

        E_PERSONA_LLM["🤖 LLM Judge<br/>gpt-oss:20b"]

        E_PERSONA_SCORE[Persona Score: 0-10<br/>+ Reasoning]

        E_INPUT --> E_CBT_PROMPT
        E_CBT_PROMPT --> E_CBT_LLM
        E_CBT_LLM --> E_CBT_SCORE

        E_INPUT --> E_PERSONA_PROMPT
        E_PERSONA_PROMPT --> E_PERSONA_LLM
        E_PERSONA_LLM --> E_PERSONA_SCORE

        style E_INPUT fill:#f0f0f0
        style E_CBT_PROMPT fill:#d9d9d9
        style E_CBT_LLM fill:#bfbfbf
        style E_CBT_SCORE fill:#a6a6a6
        style E_PERSONA_PROMPT fill:#d9d9d9
        style E_PERSONA_LLM fill:#bfbfbf
        style E_PERSONA_SCORE fill:#a6a6a6
    end

    %% ========================================================================
    %% CONNECTIONS
    %% ========================================================================
    CONTEXT --> B_INPUT
    PT --> M_INPUT
    PT --> D_INPUT

    B_RESPONSE --> E_INPUT
    M_RESPONSE --> E_INPUT
    D_RESPONSE --> E_INPUT

    E_CBT_SCORE --> RESULTS[Results Storage<br/>JSON + Visualizations]
    E_PERSONA_SCORE --> RESULTS
```

---

## Detailed Prompt Breakdown

### 1. **Generation Prompts**

#### Baseline Prompt
```
System: {CBT_SYSTEM_PROMPT}
  - "You are a Cognitive Behavioral Therapist..."
  - Contains therapeutic guidelines and principles

Context:
  {Last 10 conversation turns}

User: {Patient current turn}
```

#### Mem0 Prompt
```
System: {CBT_SYSTEM_PROMPT}

Memories:
  - Patient has anxiety about work
  - Patient mentioned sister in previous session
  - Patient prefers direct communication style
  [Retrieved from ChromaDB]

User: {Patient current turn}
```

**Note**: No sliding window context - relies purely on retrieved memories

#### DC-RS Prompt
```
System: {CBT_SYSTEM_PROMPT}

Therapeutic Cheatsheet:
  CBT Techniques:
    - "Use Socratic questioning for thought examination"
    - "Apply behavioral activation for low mood"

  Distortion Patterns:
    - "Patient shows catastrophizing tendency"
    - "Watch for black-and-white thinking"

  Effective Interventions:
    - "Validation before challenge works well"
    - "Gradual exposure effective for avoidance"

User: {Patient current turn}
```

**Note**: No sliding window context - relies purely on curated strategies

---

### 2. **Extraction Prompt (DC-RS Only)**

```
Analyze this therapeutic interaction and extract strategies:

Patient Turn: {patient_turn}
Counselor Response: {generated_response}
Current Cheatsheet: {current_cheatsheet}

Extract:
1. CBT techniques used successfully
2. Cognitive distortion patterns identified
3. Effective therapeutic interventions

Filter out:
- Patient's distorted beliefs (don't reinforce)
- Non-clinical observations
- Personal details without therapeutic value

Return JSON:
{
  "cbt_techniques": [...],
  "distortion_patterns": [...],
  "effective_interventions": [...]
}
```

---

### 3. **Evaluation Prompts**

#### CBT Adherence Evaluation
```
System: You are an expert CBT supervisor evaluating therapeutic quality.

Rate the counselor's response on CBT adherence (0-10):

Criteria:
- Uses Socratic questioning (not direct advice)
- Validates emotions while challenging cognitions
- Explores automatic thoughts and beliefs
- Suggests behavioral experiments or homework
- Maintains collaborative stance
- Avoids reassurance without cognitive work

Context: {last_10_turns}
Counselor Response: {generated_response}

Return JSON:
{
  "score": <0-10>,
  "reasoning": "..."
}
```

#### Persona Consistency Evaluation
```
System: Evaluate if counselor maintains professional therapeutic persona.

Compare to baseline therapeutic stance:
Baseline: "I hear that you're experiencing some difficulties. Can you tell me more?"

Rate consistency (0-10):

Criteria:
- Professional empathetic tone
- Appropriate boundaries (not overly casual/friendly)
- Therapeutic focus (not problem-solving/advice-giving)
- Consistent with CBT philosophy

Context: {last_10_turns}
Counselor Response: {generated_response}

Return JSON:
{
  "score": <0-10>,
  "reasoning": "..."
}
```

---

## Key Differences Summary

| Aspect | Baseline | Mem0 | DC-RS |
|--------|----------|------|-------|
| **Context Window** | Last 10 turns | **None** | **None** |
| **Memory System** | None | Raw memories | Curated strategies |
| **Learning** | None | Stores all turns | Extracts patterns |
| **Filtering** | N/A | None | Clinical filtering |
| **LLM Calls/Turn** | 3 (gen + 2 eval) | 3 (gen + 2 eval) | 4 (gen + extract + 2 eval) |
| **Storage Growth** | None | Linear (every turn) | Sublinear (filtered) |
| **Adaptation** | Static | Static accumulation | Continual learning |

---

## Expected Outcomes (Hypothesis)

```
CBT Adherence:     Baseline < Mem0 < DC-RS
Persona Consistency: Baseline ≤ Mem0 < DC-RS

Why DC-RS wins:
1. Strategies > Raw facts for therapeutic consistency
2. Clinical filtering prevents distortion accumulation
3. Test-time learning adapts to patient patterns
```

---

## Example Turn Flow

**Turn 15: Patient says "I always mess everything up"**

### Baseline:
```
→ Sees only last 10 turns
→ Generic CBT response about examining "always" thinking
→ Score: 7/10
```

### Mem0:
```
→ Retrieves: "Patient said 'I'm a failure' in turn 3"
→ Risk: Might reinforce distortion by treating it as fact
→ Score: 6/10 (distortion creep)
```

### DC-RS:
```
→ Cheatsheet has: "Patient shows all-or-nothing thinking pattern"
→ Cheatsheet has: "Effective intervention: Examine evidence for/against"
→ Uses learned strategy specific to this patient
→ Extracts: "All-or-nothing language ('always') triggers distress"
→ Score: 8/10 (pattern-aware, filtered learning)
```

---

## Model Details

**gpt-oss:20b** (GPT-OSS model via Ollama on Lambda Cloud):
- Open-source GPT-style model
- 20 billion parameters
- Runs on Lambda Cloud GPU instance
- Accessed via SSH tunnel: `http://localhost:11434/v1`
- Same model used for:
  - Response generation (all 3 conditions)
  - Strategy extraction (DC-RS)
  - CBT adherence evaluation
  - Persona consistency evaluation
  - Mem0 memory extraction

This ensures fair comparison - differences come from **method**, not model variance.
