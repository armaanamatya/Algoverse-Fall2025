# Mem0 Storage Explained

This document explains how Mem0 stores memories using ChromaDB as the vector store backend.

## Storage Location

Memories are stored in the `chroma_db/` directory:

```
chroma_db/
├── chroma.sqlite3          # Main database (metadata, memory text, collection info)
└── <UUID folders>/         # Vector embeddings for similarity search
    ├── 16da89ae-8e10-4245-aa90-ee94a98bdd62/
    ├── 7d240134-a476-470d-9214-72676ea7f5a7/
    └── ...
```

## Why UUID Folder Names?

ChromaDB uses UUIDs internally to identify segments. Each collection has:
- A **metadata segment** (stored in SQLite)
- A **vector segment** (stored in UUID folders as HNSW index files)

The human-readable collection names are mapped to these UUIDs in the SQLite database.

## Collections in This Project

| Collection Name | Embeddings | Purpose |
|-----------------|------------|---------|
| `therapy_memories_memincluded` | 112 | Memories for memory-included evaluation |
| `therapy_memories_memnotincluded` | 104 | Memories for memory-not-included evaluation |
| `llm_counselor_memincluded` | 33 | LLM counselor session memories |
| `llm_counselor_memnotincluded` | 37 | LLM counselor session memories |
| `therapy_memories_safe_test` | 35 | Safe test run memories |
| `therapy_memories` | 3 | Original test collection |

## Memory Structure

Each memory stored by Mem0 contains:

| Field | Description | Example |
|-------|-------------|---------|
| `data` | The actual memory text | "User became uptight, and others grew angry" |
| `turn_number` | Conversation turn number | 42 |
| `role` | Who said it | "patient" or "counselor" |
| `user_id` | Session identifier | "session_1000056544" |
| `created_at` | Timestamp | "2026-01-07T13:43:06.770943-08:00" |
| `hash` | Unique hash for deduplication | "2854779130464f00c6ab932219e68530" |

## Example Memories

From `therapy_memories_memincluded`:

```
[Turn 42, patient] User became uptight, and others grew angry about the uptightness.
  User ID: session_1000056544

[Turn 34, patient] The friend has a boyfriend
  User ID: session_1000056544

[Turn 56, counselor] User informed counselor about this last week
  User ID: therapy_session

[Turn 30, patient] Wanted to talk a couple of weeks ago when the whole thing started happening
  User ID: session_1000056544
```

## How to Query Memories

### Using Mem0 API (Recommended)

```python
from mem0 import Memory

# Initialize with config
memory = Memory.from_config(config)

# Get all memories for a user/session
memories = memory.get_all(user_id="session_1000056544")

# Search memories by query
results = memory.search("user feeling anxious", user_id="session_1000056544")
```

### Using SQLite Directly

```python
import sqlite3
conn = sqlite3.connect('chroma_db/chroma.sqlite3')
cursor = conn.cursor()

# List all collections
cursor.execute('SELECT id, name FROM collections')
for row in cursor.fetchall():
    print(f"UUID: {row[0]}, Name: {row[1]}")

# Get memories from a collection
cursor.execute("""
    SELECT em.key, em.string_value, em.int_value
    FROM embedding_metadata em
    JOIN embeddings e ON em.id = e.id
    JOIN segments s ON e.segment_id = s.id
    JOIN collections c ON s.collection = c.id
    WHERE c.name = 'therapy_memories_memincluded'
    AND em.key = 'data'
    LIMIT 10
""")
```

## How Mem0 Handles Memory Updates

When you call `memory.add()`, Mem0 automatically:

1. **Compares** new facts with existing memories using semantic similarity
2. **Decides** whether to:
   - `ADD` - New unique memory
   - `UPDATE` - Modify existing memory with new info
   - `DELETE` - Remove contradicted memory
   - `NONE` - No change needed (duplicate)

This is why you don't need to manually manage memory updates - Mem0 handles deduplication and updates automatically.

## Memory Isolation by User ID

Each transcript file uses a unique `user_id` for memory isolation:

```python
USER_ID = f"session_{transcript_file.stem}"
# Example: "session_1000056544"
```

This ensures memories from different therapy sessions don't mix together.

## Checkpoints and Memory Snapshots

The optimized notebooks save memory snapshots at each checkpoint:

```json
{
  "memory_snapshot": {
    "turn_number": 10,
    "memory_count": 25,
    "memories_content": ["memory1...", "memory2...", "..."],
    "cbt_score": 7,
    "persona_score": 8
  }
}
```

This allows tracking how memories grow throughout a conversation.
