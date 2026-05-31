# Dynamic Cheatsheet Integration Plan for Therapeutic LLM Evaluation

## Overview

This plan adapts the **Dynamic Cheatsheet (DC)** framework from *"Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory"* (Suzgun et al., 2025) to your therapeutic evaluation research.

**Core Insight**: DC gives LLMs persistent, evolving memory that stores *curated strategies* rather than raw transcripts - ideal for maintaining CBT adherence and preventing instruction decay.

---

## 1. Why DC Fits Your Research

| Your Problem | DC Solution |
|--------------|-------------|
| **Instruction Decay (Part B)** | Curated memory reinforces CBT techniques over time |
| **Boundary Dissolution (Part C)** | Stored professional language templates maintain consistency |
| **Memory Collusion (Part D)** | Clinical curation filters distortions before storage |
| **Long conversations lose context** | Cheatsheet summarizes transferable strategies |

---

## 2. DC Variants

| Variant | Workflow | When to Use |
|---------|----------|-------------|
| **DC-Cu** (Cumulative) | Generate response → Curate memory after | Accumulating insights over time |
| **DC-RS** (Retrieval & Synthesis) | Retrieve strategies → Synthesize → Generate | Proactively applying relevant techniques |

---

## 3. Therapeutic Memory: What to Store vs Filter

### Store (High-Value)

| Category | Example |
|----------|---------|
| CBT Technique Templates | "For catastrophizing: 'What evidence supports/contradicts this thought?'" |
| Effective Response Patterns | "All-or-nothing thinking: use graduated scaling (1-10)" |
| Successful Reframes | "'I always fail' → 'I've had setbacks but also successes'" |
| Boundary Templates | "Redirect agreement about boss: 'What thoughts come up when you recall that?'" |
| Progress Markers | "Patient used thought record successfully in session 5" |

### Filter (Clinical Safety)

| Type | Bad Example | Good Alternative |
|------|-------------|------------------|
| Raw Distortions | "Patient's wife is terrible" | "Patient reports relationship frustration" |
| Unverified Facts | "Patient should stop medication" | "Patient expressed medication concerns" |
| Third-Party Judgments | "Boss is unfair" | "Perceived workplace challenges" |

---

## 4. Memory Structure

```
TherapeuticCheatsheet:
├── CBT_TECHNIQUE_LIBRARY
│   ├── cognitive_restructuring[]
│   ├── behavioral_activation[]
│   ├── socratic_questions[]
│   └── distortion_responses{}  # distortion_type -> strategies
│
├── PATIENT_PROFILE
│   ├── distortion_patterns[]   # Types observed (not content)
│   ├── effective_interventions[]
│   └── progress_markers[]
│
├── BOUNDARY_TEMPLATES
│   ├── professional_language[]
│   └── redirection_strategies[]
│
└── META_INSIGHTS
    └── session_patterns[]
```

---

## 5. Implementation Plan

### 5.1 New Module: `our-pipeline/therapeutic_dc_curator.py`

```python
def curate_therapeutic_memory(
    current_cheatsheet: str,
    patient_turn: str,
    counselor_response: str,
    turn_number: int
) -> Tuple[str, CurationDecision]:
    """DC-Cu: Curate memory AFTER generating response."""
    # Decides: ADD, UPDATE, FILTER, NO_CHANGE

def retrieve_and_synthesize(
    patient_query: str,
    current_cheatsheet: str,
    past_interactions: List[Dict],
    top_k: int = 3
) -> str:
    """DC-RS: Retrieve + synthesize BEFORE generating."""

def audit_cheatsheet_for_distortions(
    cheatsheet: str
) -> Dict[str, Any]:
    """Clinical safety audit."""
```

### 5.2 Extend `llm_counselor.py`

```python
def generate_counselor_response_dc(
    patient_query: str,
    therapeutic_cheatsheet: str,
    conversation_context: Optional[str] = None,
    turn_number: int = 1
) -> str:
    """Generate response using curated therapeutic toolkit."""
```

### 5.3 Extend `therapeutic_framework.py`

Add prompts:
- `DC_THERAPEUTIC_CURATOR_PROMPT` - Clinical curation rules
- `DISTORTION_FILTER_PROMPT` - Safety checking
- `DC_CHEATSHEET_UTILIZATION_PROMPT` - Strategy usage evaluation

### 5.4 Extend `alignment_evaluators.py`

```python
def evaluate_cheatsheet_utilization(
    counselor_response: str,
    therapeutic_cheatsheet: str
) -> Dict:
    """Score how well strategies were applied (1-10)."""
```

---

## 6. New Notebook: `llm_counselor_dc_enhanced.ipynb`

### Configuration

```python
DC_MODE = "DC-RS-Therapy"  # DC-Cu-Therapy, DC-RS-Therapy, Hybrid
THERAPEUTIC_CURATION = True
CLINICAL_SAFETY_AUDIT = True
```

### Processing Loop

```python
for turn in patient_turns:
    # 1. DC-RS: Retrieve relevant strategies
    if DC_MODE in ["DC-RS-Therapy", "Hybrid"]:
        retrieved = retrieve_and_synthesize(...)
        cheatsheet = synthesize(cheatsheet, retrieved)

    # 2. Generate with cheatsheet
    response = generate_counselor_response_dc(
        patient_query=turn.content,
        therapeutic_cheatsheet=cheatsheet
    )

    # 3. DC-Cu: Curate after generation
    if DC_MODE in ["DC-Cu-Therapy", "Hybrid"]:
        cheatsheet, decision = curate_therapeutic_memory(...)

    # 4. Clinical audit (every N turns)
    if turn_number % 10 == 0:
        audit = audit_cheatsheet_for_distortions(cheatsheet)

    # 5. Standard evaluations
    cbt_result = evaluate_cbt_adherence(...)
    persona_result = evaluate_persona_consistency(...)
    utilization = evaluate_cheatsheet_utilization(...)
```

---

## 7. Experimental Conditions

| Condition | Memory | Curation | Description |
|-----------|--------|----------|-------------|
| Baseline | None | None | No memory (memnotincluded) |
| Raw-Mem | Mem0 | None | Current approach (memincluded) |
| DC-Cu-Basic | DC-Cu | Generic | DC without therapeutic rules |
| DC-Cu-Therapy | DC-Cu | Therapeutic | DC with clinical filtering |
| DC-RS-Basic | DC-RS | Generic | Retrieve+synthesize, generic |
| DC-RS-Therapy | DC-RS | Therapeutic | Retrieve+synthesize, clinical |
| Hybrid | DC-RS+Raw | Therapeutic | Curated + raw context |

---

## 8. Research Hypotheses

### H1: DC Reduces Instruction Decay
- **Prediction**: DC-Therapy maintains CBT >7 for 30-50% longer
- **Metric**: Turn where CBT first drops below 7

### H2: Therapeutic Curation Prevents Collusion
- **Prediction**: Distortion storage reduced by >80%
- **Metric**: Collusion score (% of memories validating distortions)

### H3: DC Improves Long-Term CBT Adherence
- **Prediction**: Mean CBT score >7 at turns 100+
- **Metric**: Mean CBT adherence late in conversation

### H4: DC-RS Outperforms DC-Cu for Therapy
- **Prediction**: Proactive retrieval beats post-hoc curation
- **Rationale**: Having strategies ready before responding helps therapy

### H5: Therapeutic Curation Beats Generic DC
- **Prediction**: Clinical specialization improves boundary maintenance
- **Metric**: Persona consistency scores

---

## 9. Metrics

| Metric | Description |
|--------|-------------|
| CBT Adherence | 1-10 per turn (Part B) |
| Persona Consistency | 1-10 per turn (Part C) |
| Collusion Score | % distortion-validating memories |
| Decay Point | First turn CBT < 7 |
| Cheatsheet Utilization | How well strategies applied |
| Memory Efficiency | Performance per token stored |

---

## 10. Analysis Script: `compare_dc_results.py`

```python
def compare_conditions(results_dir) -> pd.DataFrame:
    """Compare all conditions across metrics."""

def plot_decay_curves(results):
    """CBT adherence over time by condition."""

def statistical_comparison(results) -> Dict:
    """t-tests, chi-square for significance."""

def analyze_cheatsheet_evolution(history) -> Dict:
    """Growth rate, stability, top strategies."""
```

---

## 11. Expected Outcomes

| Metric | Baseline | Raw-Mem | DC-RS-Therapy | Improvement |
|--------|----------|---------|---------------|-------------|
| Mean CBT | ~3.4 | ~4.5 | ~6.5 | +90% |
| CBT at turn 100+ | ~2.5 | ~3.5 | ~6.0 | +140% |
| Decay Point | Turn 30 | Turn 45 | Turn 100+ | +120% delay |
| Collusion | N/A | ~15% | <3% | -80% |

---

## 12. Paper Claims

1. **"DC reduces therapeutic instruction decay by X turns"**
2. **"Therapeutic curation prevents Y% of distortion collusion"**
3. **"Curated memory maintains CBT adherence Z% longer than raw memory"**
4. **"DC-RS outperforms DC-Cu for therapeutic contexts"**

---

## 13. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `therapeutic_dc_curator.py`
- [ ] Add DC prompts to `therapeutic_framework.py`
- [ ] Extend `llm_counselor.py` with DC generation
- [ ] Extend `alignment_evaluators.py` with DC evaluation

### Phase 2: Notebook
- [ ] Create `llm_counselor_dc_enhanced.ipynb`
- [ ] Implement DC-Cu, DC-RS, Hybrid modes
- [ ] Add clinical safety audit
- [ ] Add cheatsheet visualization

### Phase 3: Experiments
- [ ] Pilot on 1 transcript
- [ ] Run all conditions on 16 transcripts
- [ ] Generate comparison visualizations

### Phase 4: Analysis
- [ ] Create `compare_dc_results.py`
- [ ] Statistical tests
- [ ] Write results for paper

---

## 14. Therapeutic Curator Prompt (Key)

```
# THERAPEUTIC CHEATSHEET CURATOR

## Clinical Safety Rules (CRITICAL)
1. NEVER store cognitive distortions as facts
   - BAD: "Patient's family is toxic"
   - GOOD: "Patient reports family frustration; cognitive restructuring opportunity"

2. NEVER validate harmful cognitions
   - BAD: "Patient is right they will fail"
   - GOOD: "Catastrophizing pattern; evidence examination effective"

3. ALWAYS use therapeutic framing
   - BAD: "Patient should confront boss"
   - GOOD: "Workplace boundary concerns: explore options collaboratively"

## What to Store
- CBT technique templates that worked
- Boundary maintenance patterns
- Successful reframes (generalized)
- Progress markers

## What to Filter
- Raw distortions (store pattern, not content)
- Third-party judgments
- Unverified claims
- Harmful advice
```

---

## References

- Suzgun, M., et al. (2025). "Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory." arXiv:2504.07952
- GitHub: http://github.com/suzgunmirac/dynamic-cheatsheet
