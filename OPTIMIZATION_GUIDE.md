# Notebook Optimization Guide

## Overview

The original notebooks (`therapy_memincluded.ipynb` and `therapy_memnotincluded.ipynb`) were taking **4-6 hours** to process 16 transcript files due to sequential processing of ~4,000+ LLM API calls.

The optimized versions can reduce this to **30-60 minutes** using concurrent processing.

---

## Performance Bottlenecks Identified

### 1. **Sequential File Processing**
- Processing 16 files one at a time
- Each file has ~125 counselor turns
- Total: ~2,000 counselor turns across all files

### 2. **Sequential Turn Evaluation**
- 2 LLM calls per turn (CBT adherence + Persona consistency)
- Total: ~4,000 LLM API calls
- With 0.1s delay between calls = ~400 seconds just in delays

### 3. **Memory Operations**
- Sequential memory additions for each turn
- Memory retrieval for each evaluation

### 4. **Unnecessary Delays**
- 0.1s delay between calls for local models (Ollama/LM Studio)
- This delay is unnecessary for local inference

---

## Optimization Strategies Implemented

### ✅ **1. Multithreading (Primary Optimization)**

#### File-Level Parallelization
- Process multiple transcript files concurrently
- Recommended: `MAX_FILE_WORKERS = 2-4`
- Limited by memory and CPU resources

#### Turn-Level Parallelization
- Evaluate multiple turns within a file concurrently
- Recommended: `MAX_TURN_WORKERS = 4-8`
- Limited by LLM server capacity

**Expected Speedup**: 4-10x faster depending on hardware

### ✅ **2. Reduced Delays**
- Set `DELAY_BETWEEN_CALLS = 0.0` for local models
- Only use delays for rate-limited APIs (OpenAI)

**Expected Speedup**: 1.5-2x faster

### ✅ **3. Thread-Safe Memory Operations**
- Use `threading.Lock()` for memory operations
- Prevents race conditions when multiple threads access memory

### ✅ **4. Progress Tracking**
- Added `tqdm` progress bars
- Real-time visibility into processing status

---

## New Optimized Notebooks

### 📓 `therapy_memincluded_optimized.ipynb`
- Memory-enhanced evaluation (memories passed to LLM judges)
- Concurrent file and turn processing
- Thread-safe memory operations

### 📓 `therapy_memnotincluded_optimized.ipynb`
- Standard evaluation (no memory context)
- Concurrent file and turn processing
- Thread-safe memory operations

---

## Configuration Parameters

### Parallelization Settings

```python
MAX_FILE_WORKERS = 2  # Number of files to process concurrently
MAX_TURN_WORKERS = 6  # Number of turns to evaluate concurrently per file
```

#### Recommended Values:

| Hardware | MAX_FILE_WORKERS | MAX_TURN_WORKERS | Expected Time |
|----------|------------------|------------------|---------------|
| **Low** (4 cores, 8GB RAM) | 1 | 4 | 2-3 hours |
| **Medium** (8 cores, 16GB RAM) | 2 | 6 | 45-90 min |
| **High** (16+ cores, 32GB RAM) | 4 | 8 | 30-45 min |

### Delay Settings

```python
# For local models (Ollama, LM Studio)
DELAY_BETWEEN_CALLS = 0.0

# For OpenAI API (rate limiting)
DELAY_BETWEEN_CALLS = 0.5
```

---

## Performance Comparison

### Original Notebooks
- **Processing Mode**: Sequential
- **Files**: 16 files, one at a time
- **Turns per file**: ~125 counselor turns
- **LLM calls per turn**: 2 (CBT + Persona)
- **Total LLM calls**: ~4,000
- **Estimated time**: 4-6 hours

### Optimized Notebooks
- **Processing Mode**: Concurrent (2 files × 6 turns)
- **Parallelization**: 12 concurrent LLM calls
- **Estimated time**: 30-60 minutes
- **Speedup**: **5-10x faster**

---

## Usage Instructions

### 1. Install Required Package

```bash
pip install tqdm
```

### 2. Open Optimized Notebook

Choose the appropriate notebook:
- `therapy_memincluded_optimized.ipynb` - with memory context
- `therapy_memnotincluded_optimized.ipynb` - without memory context

### 3. Configure Settings

Adjust parallelization based on your hardware:

```python
MAX_FILE_WORKERS = 2  # Adjust based on CPU/RAM
MAX_TURN_WORKERS = 6  # Adjust based on LLM server capacity
```

### 4. Run the Notebook

Execute all cells. You'll see:
- Progress bars for file and turn processing
- Real-time status updates
- Final timing statistics

### 5. Check Results

Results are saved to:
```
./evaluation_results_optimized/
  ├── 1000056544_eval_optimized.json
  ├── 1000056545_eval_optimized.json
  └── ...
```

---

## Additional Optimization Options

### Option 1: Batch Processing (Not Implemented)
Evaluate multiple turns in a single LLM call by combining prompts.

**Pros**: Fewer API calls
**Cons**: Harder to parse, less granular control

### Option 2: Caching (Not Implemented)
Save intermediate results and resume from checkpoint.

**Pros**: Can resume after interruption
**Cons**: More complex state management

### Option 3: Async Processing (Not Implemented)
Use `asyncio` instead of threading for I/O-bound operations.

**Pros**: Better for high-concurrency scenarios
**Cons**: More complex code, requires async-compatible libraries

---

## Troubleshooting

### Issue: Out of Memory Errors

**Solution**: Reduce parallelization
```python
MAX_FILE_WORKERS = 1
MAX_TURN_WORKERS = 4
```

### Issue: LLM Server Overload

**Solution**: Reduce turn workers
```python
MAX_TURN_WORKERS = 2
```

### Issue: Inconsistent Results

**Solution**: Add small delay between calls
```python
DELAY_BETWEEN_CALLS = 0.1
```

### Issue: ChromaDB Lock Errors

**Solution**: Memory operations are already thread-safe with locks. If you still see errors, reduce parallelization.

---

## Monitoring Performance

### Check Ollama Server Load

```bash
# Monitor Ollama logs
ollama serve

# Check GPU usage (if using GPU)
nvidia-smi -l 1
```

### Check CPU Usage

```bash
# Windows
taskmgr

# Linux/Mac
htop
```

---

## Best Practices

1. **Start Small**: Test with `MAX_TURNS = 20` first
2. **Monitor Resources**: Watch CPU/RAM/GPU usage
3. **Adjust Workers**: Tune based on your hardware
4. **Use Progress Bars**: Keep track of processing status
5. **Save Frequently**: Results are saved per-file automatically

---

## Summary

| Aspect | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Processing** | Sequential | Concurrent | 5-10x faster |
| **Time** | 4-6 hours | 30-60 min | 80-90% reduction |
| **Delays** | 0.1s per call | 0s (local) | 1.5-2x faster |
| **Progress** | Print statements | Progress bars | Better UX |
| **Thread Safety** | N/A | Locks for memory | Safe concurrent access |

---

## Next Steps

1. ✅ Run optimized notebooks with your data
2. ✅ Adjust worker counts based on performance
3. ✅ Compare results with original notebooks
4. ⏭️ Consider implementing caching for very large datasets
5. ⏭️ Explore async processing for even higher concurrency

---

**Questions?** Check the notebook comments or reach out for support!
