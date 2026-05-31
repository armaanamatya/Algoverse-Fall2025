# Directory Structure

Complete directory structure after running all 4 evaluation notebooks.

## Main Project Structure

```
C:\Users\Armaan\Desktop\Algoverse-Fall2025\
│
├── 0518-014_raw/                          # Input transcripts
│   ├── 1000056544.txt
│   ├── 1000056545.txt
│   ├── 1000056546.txt
│   └── ... (16 files total)
│
├── chroma_db/                             # Mem0 storage (shared by all 4 notebooks)
│   ├── chroma.sqlite3                     # All 4 collections stored here
│   └── {uuid}/                            # Vector embeddings folder(s)
│       └── data_level0.bin
│
├── output_therapy_memincluded/
│   ├── images/
│   │   ├── alignment_overview.png         # Aggregate graph
│   │   ├── 1000056544_evaluation.png      # Per-file graphs
│   │   ├── 1000056545_evaluation.png
│   │   └── ...
│   ├── checkpoints/                       # Empty after completion (auto-deleted)
│   │   └── (1000056544_checkpoint.json)   # Only exists mid-run
│   ├── results/
│   │   ├── 1000056544_results.json
│   │   ├── 1000056545_results.json
│   │   └── ...
│   ├── 1000056544_evaluation_log.md       # Markdown logs at root
│   ├── 1000056545_evaluation_log.md
│   └── ...
│
├── output_therapy_memnotincluded/
│   ├── images/
│   │   ├── alignment_overview.png
│   │   ├── 1000056544_evaluation.png
│   │   └── ...
│   ├── checkpoints/
│   ├── results/
│   │   ├── 1000056544_results.json
│   │   └── ...
│   ├── 1000056544_evaluation_log.md
│   └── ...
│
├── output_llm_counselor_memincluded/
│   ├── images/
│   │   ├── llm_counselor_overview.png
│   │   ├── 1000056544_evaluation.png
│   │   └── ...
│   ├── checkpoints/
│   ├── results/
│   │   ├── 1000056544_results.json
│   │   └── ...
│   ├── 1000056544_evaluation_log.md
│   └── ...
│
├── output_llm_counselor_memnotincluded/
│   ├── images/
│   │   ├── llm_counselor_overview.png
│   │   ├── 1000056544_evaluation.png
│   │   └── ...
│   ├── checkpoints/
│   ├── results/
│   │   ├── 1000056544_results.json
│   │   └── ...
│   ├── 1000056544_evaluation_log.md
│   └── ...
│
├── our-pipeline/                          # Shared modules
│   ├── transcript_parser.py
│   ├── alignment_evaluators.py
│   ├── mem0_integration.py
│   ├── therapeutic_framework.py
│   └── llm_counselor.py
│
├── therapy_memincluded.ipynb              # The 4 main notebooks
├── therapy_memnotincluded.ipynb
├── llm_counselor_memincluded.ipynb
├── llm_counselor_memnotincluded.ipynb
│
├── mem0storageexplained.md                # Documentation
├── directory_structure.md                 # This file
└── .env                                   # API keys
```

---

## Mem0 Collections in chroma.sqlite3

All 4 notebooks store memories in the same SQLite database but in separate collections:

```
chroma.sqlite3
├── Collection: therapy_memories_memincluded
│   ├── user_id: session_1000056544 → memories for transcript 1
│   ├── user_id: session_1000056545 → memories for transcript 2
│   └── ...
├── Collection: therapy_memories_memnotincluded
│   ├── user_id: session_1000056544
│   └── ...
├── Collection: llm_counselor_memincluded
│   ├── user_id: llm_session_1000056544
│   └── ...
└── Collection: llm_counselor_memnotincluded
    ├── user_id: llm_session_1000056544
    └── ...
```

---

## User ID Patterns

| Notebook Type | User ID Format | Example (for `1000056544.txt`) |
|---------------|----------------|-------------------------------|
| `therapy_memincluded` | `session_{filename}` | `session_1000056544` |
| `therapy_memnotincluded` | `session_{filename}` | `session_1000056544` |
| `llm_counselor_memincluded` | `llm_session_{filename}` | `llm_session_1000056544` |
| `llm_counselor_memnotincluded` | `llm_session_{filename}` | `llm_session_1000056544` |

---

## Output Directory Summary

| Notebook | Output Directory | Mem0 Collection | Checkpoints |
|----------|-----------------|-----------------|-------------|
| `therapy_memincluded.ipynb` | `./output_therapy_memincluded/` | `therapy_memories_memincluded` | `./output_therapy_memincluded/checkpoints/` |
| `therapy_memnotincluded.ipynb` | `./output_therapy_memnotincluded/` | `therapy_memories_memnotincluded` | `./output_therapy_memnotincluded/checkpoints/` |
| `llm_counselor_memincluded.ipynb` | `./output_llm_counselor_memincluded/` | `llm_counselor_memincluded` | `./output_llm_counselor_memincluded/checkpoints/` |
| `llm_counselor_memnotincluded.ipynb` | `./output_llm_counselor_memnotincluded/` | `llm_counselor_memnotincluded` | `./output_llm_counselor_memnotincluded/checkpoints/` |

---

## Sample Markdown Log Structure

Each transcript generates a markdown log file (e.g., `1000056544_evaluation_log.md`):

```markdown
# Evaluation Log: 1000056544.txt

**Generated:** 2025-01-15 14:30:00
**Model:** gpt-oss:20b
**Memory Enhanced:** Yes/No

## Transcript Info
- Total Turns: 250
- Counselor Turns: 125

---

## Turn-by-Turn Evaluations

### Turn 3
**Counselor Response:**
> Yeah, you do too...

**CBT Adherence Score:** 7/10
**CBT Reasoning:**
> The response shows...

**Persona Consistency Score:** 8/10
**Persona Reasoning:**
> Professional tone maintained...

---

### Turn 5
...

## Memories Stored
1. **[Turn 2, patient]** Patient has never been in therapy before
2. **[Turn 4, patient]** Patient feels nervous about opening up
...

## Summary Statistics
### CBT Adherence
- Mean: 7.2/10
- Min: 5/10
- Max: 9/10

### Persona Consistency
- Mean: 8.1/10
- Min: 6/10
- Max: 10/10

### Memory
- Total Memories Stored: 45
```

---

## Checkpoint System

- **Location**: `{output_dir}/checkpoints/{transcript_stem}_checkpoint.json`
- **Created**: After each turn evaluation
- **Deleted**: Automatically after successful transcript completion
- **Resume**: Set `RESUME_FROM_CHECKPOINT = True` in notebook to resume from last checkpoint

### Checkpoint JSON Structure:
```json
{
  "filename": "1000056544.txt",
  "last_counselor_turn_idx": 45,
  "last_memory_turn": 90,
  "total_counselor_turns": 125,
  "cbt_results": [...],
  "persona_results": [...],
  "memory_snapshots": [...],
  "timestamp": "2025-01-15 14:30:00"
}
```
