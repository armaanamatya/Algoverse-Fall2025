# ⚡ Quick Reference: Optimized Notebooks

## 📁 Files Overview

| File | Purpose | Use When |
|------|---------|----------|
| `therapy_memincluded_optimized.ipynb` | Memory-enhanced evaluation | You want memories passed to LLM judges |
| `therapy_memnotincluded_optimized.ipynb` | Standard evaluation | You want evaluation without memory context |
| `OPTIMIZATION_GUIDE.md` | Detailed guide | You need configuration help |
| `compare_results.py` | Validation script | You want to verify results |

---

## ⚙️ Configuration Cheat Sheet

### Hardware-Based Settings

```python
# === LOW-END HARDWARE (4 cores, 8GB RAM) ===
MAX_FILE_WORKERS = 1
MAX_TURN_WORKERS = 4
# Expected time: 2-3 hours

# === MEDIUM HARDWARE (8 cores, 16GB RAM) ===
MAX_FILE_WORKERS = 2
MAX_TURN_WORKERS = 6
# Expected time: 45-90 minutes

# === HIGH-END HARDWARE (16+ cores, 32GB RAM) ===
MAX_FILE_WORKERS = 4
MAX_TURN_WORKERS = 8
# Expected time: 30-45 minutes
```

### Model-Based Settings

```python
# === LOCAL MODELS (Ollama, LM Studio) ===
DELAY_BETWEEN_CALLS = 0.0  # No delay needed

# === OPENAI API ===
DELAY_BETWEEN_CALLS = 0.5  # Respect rate limits
MAX_TURN_WORKERS = 4       # Reduce to avoid rate limits
```

---

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install tqdm
```

### 2. Open Notebook
- Open `therapy_memincluded_optimized.ipynb` in Jupyter

### 3. Run All Cells
- Click "Run All" or execute cells sequentially
- Watch the progress bars!

---

## 📊 What to Expect

### Progress Bars
```
Processing files: 100%|████████████| 16/16 [45:23<00:00, 170.21s/it]
Evaluating 1000056544.txt: 100%|████| 125/125 [02:45<00:00, 1.32s/it]
```

### Final Output
```
============================================================
ALL FILES PROCESSED!
Total time: 45.38 minutes (2722.89 seconds)
Files completed: 16/16
Results saved to: ./evaluation_results_optimized/
============================================================
```

---

## 🔧 Common Adjustments

### Too Slow?
**Increase workers** (if you have resources):
```python
MAX_FILE_WORKERS = 4
MAX_TURN_WORKERS = 8
```

### Out of Memory?
**Decrease workers**:
```python
MAX_FILE_WORKERS = 1
MAX_TURN_WORKERS = 2
```

### Ollama Errors?
**Reduce concurrent load**:
```python
MAX_TURN_WORKERS = 4
DELAY_BETWEEN_CALLS = 0.1
```

### Test First?
**Limit turns for testing**:
```python
MAX_TURNS = 20  # Process only first 20 turns per file
```

---

## 📈 Performance Comparison

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Time** | 4-6 hours | 30-60 min | 5-10x faster |
| **Concurrency** | 1 | 12 | 12x parallel |
| **User Experience** | Print logs | Progress bars | Much better |

---

## ✅ Validation Checklist

After running optimized notebooks:

- [ ] All 16 files processed successfully
- [ ] Results saved to `./evaluation_results_optimized/`
- [ ] No error messages in output
- [ ] Progress bars completed to 100%
- [ ] Total time < 2 hours

Optional validation:
- [ ] Run `python compare_results.py`
- [ ] Verify mean score difference < 0.5

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: tqdm` | `pip install tqdm` |
| Out of memory | Reduce `MAX_FILE_WORKERS` and `MAX_TURN_WORKERS` |
| Ollama connection error | Check `ollama serve` is running |
| ChromaDB lock error | Reduce workers (already thread-safe) |
| Results inconsistent | Normal - LLMs are non-deterministic |

---

## 📝 Output Files

Results are saved as JSON files:
```
./evaluation_results_optimized/
├── 1000056544_eval_optimized.json       # Memory-enhanced
├── 1000056545_eval_optimized.json
├── ...
├── 1000056544_eval_optimized_nomem.json # Without memory
└── ...
```

Each file contains:
- CBT adherence scores
- Persona consistency scores
- Memory snapshots
- Metadata (model, timestamps, etc.)

---

## 🎯 Key Takeaways

1. **5-10x faster** than original notebooks
2. **Thread-safe** memory operations
3. **Configurable** based on your hardware
4. **Progress tracking** with tqdm
5. **Same results** as original (within tolerance)

---

## 📚 Need More Help?

- **Detailed guide**: See `OPTIMIZATION_GUIDE.md`
- **Overview**: See `OPTIMIZATION_SUMMARY.md`
- **Validation**: Run `compare_results.py`
- **Original notebooks**: Still available for reference

---

**Pro Tip**: Start with `MAX_TURNS = 20` to test the setup before running the full dataset!

---

Last updated: 2026-01-12
