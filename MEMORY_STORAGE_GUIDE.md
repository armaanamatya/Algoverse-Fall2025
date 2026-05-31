# Memory Storage Guide

## 📍 Where Memories Are Stored

### **Physical Location**
```
./chroma_db/
```
This is a **ChromaDB vector database** directory in your project root (`c:\Users\Armaan\Desktop\Algoverse-Fall2025\chroma_db\`)

### **Collection Names** (Logical Separation)

The notebooks use different ChromaDB collections to keep memories separate:

#### **`therapy_memincluded.ipynb`**
- **Collection**: `therapy_memories_memincluded`
- **Purpose**: Stores memories when evaluators RECEIVE memory context
- **Location**: `./chroma_db/` (ChromaDB manages internal structure)

#### **`therapy_memnotincluded.ipynb`**
- **Collection**: `therapy_memories_memnotincluded`
- **Purpose**: Stores memories when evaluators DO NOT receive memory context
- **Location**: `./chroma_db/` (ChromaDB manages internal structure)

---

## 🔑 Memory Isolation Per File

### **USER_ID Strategy** (✅ FIXED)

Each transcript file now gets its own isolated memory space:

```python
USER_ID = f"session_{transcript_file.stem}"
```

**Examples:**
- `1000056544.txt` → `USER_ID = "session_1000056544"`
- `1000056545.txt` → `USER_ID = "session_1000056545"`
- `1000056546.txt` → `USER_ID = "session_1000056546"`
- ... etc.

This ensures:
- ✅ Memories from file A don't leak into file B
- ✅ Each file's evaluation is independent
- ✅ Clean memory state for each transcript

---

## 📊 Memory Structure

### **What Gets Stored**

For each conversation turn, Mem0 extracts and stores:
- Patient statements and facts
- Counselor observations
- Emotional states
- Key topics discussed
- Cognitive distortions (if detected)

### **Storage Format**

ChromaDB stores:
1. **Memory text** - The extracted fact/statement
2. **Embeddings** - Vector representation for similarity search
3. **Metadata** - Turn number, role, timestamp, USER_ID

---

## 🗂️ Directory Structure

```
Algoverse-Fall2025/
├── chroma_db/                          # ChromaDB storage
│   ├── therapy_memories_memincluded/   # Collection for memincluded notebook
│   │   ├── session_1000056544/         # Memories for file 1
│   │   ├── session_1000056545/         # Memories for file 2
│   │   └── ...                         # (16 total)
│   └── therapy_memories_memnotincluded/# Collection for memnotincluded notebook
│       ├── session_1000056544/         # Memories for file 1
│       ├── session_1000056545/         # Memories for file 2
│       └── ...                         # (16 total)
│
├── evaluation_results/                 # JSON evaluation outputs
│   ├── 1000056544_eval.json          # Results for file 1
│   ├── 1000056545_eval.json          # Results for file 2
│   └── ...                            # (16 total)
│
└── 0518-014_raw/                      # Source transcripts
    ├── 1000056544.txt
    ├── 1000056545.txt
    └── ...
```

---

## 🔄 Memory Lifecycle

### **During Processing**

1. **Turn Added** → Mem0 extracts facts → Stored in ChromaDB
2. **Evaluation** → Memories retrieved → Passed to judges (memincluded only)
3. **Next Turn** → Process repeats with accumulated memories

### **Between Files**

- New `USER_ID` is set
- Fresh memory space for next file
- Previous file's memories remain in database but isolated

### **Persistence**

- Memories persist in `./chroma_db/` between notebook runs
- Set `RESET_MEMORIES = True` in Section 3 to clear and start fresh
- Set `RESET_MEMORIES = False` to keep existing memories (default)

---

## 🧹 Managing Memory Storage

### **Clear All Memories**
```python
# In notebook Section 3, change:
RESET_MEMORIES = True  # This will clear the collection on next run
```

### **Delete ChromaDB Manually**
```powershell
# Stop Jupyter first, then:
Remove-Item -Recurse -Force ./chroma_db
```

### **View Memory Size**
```powershell
# Check directory size
Get-ChildItem ./chroma_db -Recurse | Measure-Object -Property Length -Sum
```

---

## 📈 Memory Statistics in Output

Each `{filename}_eval.json` includes memory snapshots:

```json
{
  "memory_snapshots": [
    {
      "turn_number": 2,
      "memory_count": 1,
      "memories_in_context": 1,  // Only in memincluded
      "cbt_score": 1,
      "persona_score": 10
    },
    // ... one snapshot per counselor turn
  ]
}
```

- **`memory_count`**: Total memories stored up to this turn
- **`memories_in_context`**: How many memories were passed to evaluators (memincluded only)

---

## ⚠️ Important Notes

1. **ChromaDB is local** - All data stays on your machine
2. **Collections are separate** - memincluded and memnotincluded don't share data
3. **USER_IDs are isolated** - Each file has its own memory space
4. **Persistence** - Memories survive notebook restarts unless you reset
5. **Size** - ChromaDB will grow as you process more files (expect ~10-50MB per collection)

---

## 🔍 Debugging Memory Issues

### **Check if memories are being created:**
```python
# In notebook, after processing:
all_memories = get_all_memories(memory, USER_ID)
print(f"Total memories: {len(all_memories)}")
for mem in all_memories[:5]:
    print(mem)
```

### **Verify USER_ID isolation:**
```python
# Should show different counts for different USER_IDs
user_id_1 = "session_1000056544"
user_id_2 = "session_1000056545"
print(f"File 1 memories: {len(get_all_memories(memory, user_id_1))}")
print(f"File 2 memories: {len(get_all_memories(memory, user_id_2))}")
```
