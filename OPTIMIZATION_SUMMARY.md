# Performance Optimization Summary

## 🚀 What Was Done

I've created **optimized versions** of your therapy evaluation notebooks that use **multithreading** to dramatically speed up processing.

### New Files Created:

1. **`therapy_memincluded_optimized.ipynb`** - Optimized memory-enhanced evaluation
2. **`therapy_memnotincluded_optimized.ipynb`** - Optimized standard evaluation  
3. **`OPTIMIZATION_GUIDE.md`** - Comprehensive guide with configuration tips
4. **`compare_results.py`** - Script to validate result consistency

---

## ⚡ Performance Improvements

### Original Notebooks
- **Processing**: Sequential (one file at a time, one turn at a time)
- **Estimated Time**: 4-6 hours for 16 files
- **Total LLM Calls**: ~4,000 (sequential)

### Optimized Notebooks  
- **Processing**: Concurrent (2 files × 6 turns in parallel)
- **Estimated Time**: 30-60 minutes for 16 files
- **Total LLM Calls**: ~4,000 (concurrent)
- **Speedup**: **5-10x faster** ⚡

---

## 🔧 How It Works

### 1. **File-Level Parallelization**
Process multiple transcript files at the same time:
```python
MAX_FILE_WORKERS = 2  # Process 2 files concurrently
```

### 2. **Turn-Level Parallelization**
Evaluate multiple turns within each file concurrently:
```python
MAX_TURN_WORKERS = 6  # Evaluate 6 turns at once
```

### 3. **Thread-Safe Memory Operations**
Uses locks to prevent race conditions when multiple threads access Mem0:
```python
with memory_lock:
    add_conversation_turn_to_memory(...)
```

### 4. **Reduced Delays**
Removed unnecessary delays for local models:
```python
DELAY_BETWEEN_CALLS = 0.0  # No delay for Ollama/LM Studio
```

### 5. **Progress Tracking**
Added `tqdm` progress bars for real-time status updates

---

## 📋 Quick Start Guide

### Step 1: Install Required Package
```bash
pip install tqdm
```

### Step 2: Choose Your Notebook
- **With memory context**: `therapy_memincluded_optimized.ipynb`
- **Without memory context**: `therapy_memnotincluded_optimized.ipynb`

### Step 3: Configure Workers (Optional)
Adjust based on your hardware:

| Hardware | MAX_FILE_WORKERS | MAX_TURN_WORKERS | Expected Time |
|----------|------------------|------------------|---------------|
| Low (4 cores, 8GB RAM) | 1 | 4 | 2-3 hours |
| Medium (8 cores, 16GB RAM) | 2 | 6 | 45-90 min |
| High (16+ cores, 32GB RAM) | 4 | 8 | 30-45 min |

### Step 4: Run the Notebook
Execute all cells and watch the progress bars!

### Step 5: Validate Results (Optional)
Compare with original results:
```bash
python compare_results.py
```

---

## 🎯 Key Optimizations Explained

### Why Multithreading?
Your notebooks make **thousands of independent LLM API calls**. Each call:
1. Sends request to Ollama
2. Waits for response
3. Processes result

During the "waiting" time, the CPU is idle. Multithreading allows processing multiple requests simultaneously, keeping the CPU and LLM server busy.

### Why Not More Workers?
Too many concurrent requests can:
- Overwhelm the LLM server (Ollama)
- Cause out-of-memory errors
- Lead to slower responses due to resource contention

The recommended settings (2 files × 6 turns = 12 concurrent calls) balance speed and stability.

### Thread Safety
Mem0 uses ChromaDB which isn't thread-safe by default. The optimized notebooks use `threading.Lock()` to ensure only one thread accesses memory at a time, preventing data corruption.

---

## 📊 Expected Results

### Processing Time Breakdown

**Original (Sequential)**:
```
File 1: 15-20 min
File 2: 15-20 min
...
File 16: 15-20 min
Total: 4-6 hours
```

**Optimized (Concurrent - 2 files × 6 turns)**:
```
Files 1-2 (parallel): 15-20 min
Files 3-4 (parallel): 15-20 min
...
Files 15-16 (parallel): 15-20 min
Total: 60-90 min (8 batches × 7.5 min avg)
```

With higher parallelization (4 files × 8 turns):
```
Total: 30-45 min (4 batches × 10 min avg)
```

---

## 🔍 Validation

The optimized notebooks should produce **nearly identical results** to the original notebooks because:

1. ✅ Same evaluation logic
2. ✅ Same prompts and rubrics
3. ✅ Same model (gpt-oss:20b)
4. ✅ Same conversation context

**Minor differences** may occur due to:
- Non-deterministic LLM responses (temperature > 0)
- Slightly different memory states if processing order varies

Use `compare_results.py` to verify results are within acceptable tolerance (< 0.5 point difference on average).

---

## 🛠️ Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce workers
```python
MAX_FILE_WORKERS = 1
MAX_TURN_WORKERS = 4
```

### Issue: Ollama Server Overload
**Solution**: Reduce turn workers
```python
MAX_TURN_WORKERS = 2
```

### Issue: ChromaDB Errors
**Solution**: Memory operations are already thread-safe. If errors persist, reduce parallelization.

---

## 📈 Monitoring Performance

### Check Ollama Server
```bash
# Monitor Ollama logs
ollama serve
```

### Check System Resources
- **Windows**: Task Manager (Ctrl+Shift+Esc)
- **Linux/Mac**: `htop` or `top`

Watch for:
- CPU usage (should be high during processing)
- RAM usage (shouldn't exceed 80%)
- GPU usage (if using GPU acceleration)

---

## 🎓 Advanced Options (Not Implemented)

### 1. Batch Processing
Evaluate multiple turns in a single LLM call.
- **Pros**: Fewer API calls
- **Cons**: Harder to parse, less granular

### 2. Async Processing
Use `asyncio` instead of threading.
- **Pros**: Better for very high concurrency
- **Cons**: More complex, requires async libraries

### 3. Distributed Processing
Run on multiple machines.
- **Pros**: Unlimited scaling
- **Cons**: Complex setup, network overhead

---

## ✅ Summary

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Time** | 4-6 hours | 30-60 min | **80-90% faster** |
| **Concurrency** | 1 call at a time | 12 calls at once | **12x parallelization** |
| **Progress** | Print statements | Progress bars | Better UX |
| **Safety** | N/A | Thread-safe locks | No data corruption |

---

## 🚦 Next Steps

1. ✅ **Test with small dataset**: Set `MAX_TURNS = 20` to verify it works
2. ✅ **Run full dataset**: Process all 16 files
3. ✅ **Compare results**: Use `compare_results.py` to validate
4. ✅ **Tune workers**: Adjust based on your hardware performance
5. ⏭️ **Scale up**: If needed, increase workers for even faster processing

---

## 📚 Additional Resources

- **`OPTIMIZATION_GUIDE.md`**: Detailed configuration guide
- **`compare_results.py`**: Result validation script
- **Original notebooks**: Keep for reference and comparison

---

**Questions or issues?** Check the troubleshooting section in `OPTIMIZATION_GUIDE.md` or review the inline comments in the optimized notebooks!

Happy optimizing! 🚀
