# 🔍 How to Monitor RAM and GPU Usage in Your Jupyter Notebooks

## Quick Answer

I've created a complete resource monitoring system for you! Here's what you need to do:

### 1️⃣ Install Required Packages (One-time setup)

```bash
pip install psutil gputil py3nvml
```

✅ **Already installed!** These packages are now ready to use.

### 2️⃣ Add Monitoring to Your Notebooks

Copy and paste these code blocks into your notebooks:

#### At the BEGINNING of your notebook:
```python
# Resource Monitoring Setup
import sys
from pathlib import Path

# Add our-pipeline to path
pipeline_path = Path("./our-pipeline")
if str(pipeline_path) not in sys.path:
    sys.path.insert(0, str(pipeline_path))

from resource_monitor import ResourceMonitor

# Create monitor (change name to match your notebook)
monitor = ResourceMonitor(notebook_name="therapy_memincluded")

# Start monitoring
monitor.start_monitoring()
```

#### At key checkpoints (after Mem0 init, after loading data, etc.):
```python
monitor.monitor_checkpoint("After Mem0 Initialization")
```

#### At the END of your notebook:
```python
# Stop monitoring and show summary
monitor.stop_monitoring()

# Save report
report_path = monitor.save_report()
print(f"📄 Report saved to: {report_path}")
```

---

## 📊 What You'll See

### Real-time Output Example:

```
============================================================
📊 Initial State Resource Usage
============================================================
⏰ Time: 2026-01-12T21:45:00

💾 RAM:
   Used: 8.45 GB (52.3%)
   Available: 7.71 GB

🖥️  CPU: 12.5%

🎮 GPU (1 device(s)):
   GPU 0 (NVIDIA GeForce RTX 3060):
      Memory: 1024 MB / 12288 MB (8.3%)
      Utilization: 5.2%
      Temperature: 45.0°C
============================================================
```

### Final Summary Example:

```
============================================================
📈 Summary Statistics
============================================================

💾 RAM Usage:
   Peak: 14.23 GB (88.2%)
   Average: 11.56 GB (71.6%)
   Minimum: 8.45 GB (52.3%)
   Delta: +5.78 GB

🖥️  CPU Usage:
   Peak: 89.3%
   Average: 52.1%

🎮 GPU 0 (NVIDIA GeForce RTX 3060):
   Peak Memory: 8192 MB (66.7%)
   Average Memory: 5234 MB
   Peak Utilization: 95.2%
   Average Utilization: 68.4%
============================================================
```

---

## 📁 Files Created for You

| File | Purpose |
|------|---------|
| `our-pipeline/resource_monitor.py` | Main monitoring module |
| `RESOURCE_MONITORING_GUIDE.md` | Complete documentation |
| `QUICK_START_MONITORING.py` | Ready-to-copy code snippets |
| `resource_monitoring_snippets.py` | Additional code examples |

---

## 🎯 Monitoring Strategy for Your Notebooks

### For `therapy_memincluded.ipynb`:

1. **Start monitoring** after imports
2. **Checkpoint** after Mem0 initialization
3. **Checkpoint** after loading transcripts
4. **Checkpoint** every 5 files in processing loop
5. **Stop monitoring** at the end

### For `therapy_memnotincluded.ipynb`:

Same strategy! This lets you compare resource usage between the two approaches.

---

## 🔄 Comparing Both Notebooks

After running both notebooks with monitoring, you'll get JSON reports in `./resource_reports/`:

```
resource_reports/
├── therapy_memincluded_resources_20260112_214500.json
└── therapy_memnotincluded_resources_20260112_220130.json
```

Use the comparison code in `QUICK_START_MONITORING.py` to see the differences!

---

## 💡 Expected Results

### Typical Resource Usage:

**therapy_memincluded.ipynb** (with memory context):
- **RAM**: 12-20 GB peak
- **GPU**: 8-16 GB VRAM (if using local Ollama)
- **CPU**: 40-80% during processing

**therapy_memnotincluded.ipynb** (without memory context):
- **RAM**: Slightly lower (10-18 GB peak)
- **GPU**: Similar (8-16 GB VRAM)
- **CPU**: Similar (40-80%)

The memory-included version may use slightly more RAM because it:
1. Stores memories in ChromaDB
2. Retrieves and formats memories for each evaluation
3. Passes additional context to the LLM judges

---

## 🚨 Troubleshooting

### "No GPU detected"
- Normal if you don't have an NVIDIA GPU
- Monitoring will still work for RAM and CPU

### "GPUtil not installed"
```bash
pip install gputil
```

### High RAM usage warning
- Check if you're processing too many files at once
- Consider processing files sequentially
- Clear unused variables: `del large_variable`

---

## 📚 Documentation Files

For more details, see:

1. **`RESOURCE_MONITORING_GUIDE.md`** - Complete guide with examples
2. **`QUICK_START_MONITORING.py`** - Copy-paste code blocks
3. **`resource_monitoring_snippets.py`** - Additional snippets

---

## ✅ Quick Checklist

Before running your notebooks:

- [x] ✅ Packages installed (`psutil`, `gputil`, `py3nvml`)
- [ ] Add monitoring code to beginning of notebook
- [ ] Add checkpoints at key locations
- [ ] Add stop monitoring at end
- [ ] Run notebook and check output
- [ ] Review JSON report in `resource_reports/`
- [ ] Compare both notebooks

---

## 🎓 Example: Adding to therapy_memincluded.ipynb

Here's exactly where to add the code:

```python
# ============================================================================
# Cell 1: Existing imports
# ============================================================================
import sys
import os
from pathlib import Path
# ... existing imports ...

# ============================================================================
# Cell 2: NEW - Add Resource Monitoring
# ============================================================================
pipeline_path = Path("./our-pipeline")
if str(pipeline_path) not in sys.path:
    sys.path.insert(0, str(pipeline_path))

from resource_monitor import ResourceMonitor
monitor = ResourceMonitor(notebook_name="therapy_memincluded")
monitor.start_monitoring()

# ============================================================================
# Cell 3: Model Configuration (existing)
# ============================================================================
# ... your existing model config ...

# ============================================================================
# Cell 4: Initialize Mem0 (existing)
# ============================================================================
# ... your existing Mem0 init ...

# ADD THIS at the end of the cell:
monitor.monitor_checkpoint("After Mem0 Initialization")

# ============================================================================
# Cell 5: Load Transcripts (existing)
# ============================================================================
# ... your existing transcript loading ...

# ADD THIS at the end of the cell:
monitor.monitor_checkpoint("After Loading Transcripts")

# ============================================================================
# Cell 6: Process Turns (existing)
# ============================================================================
for file_idx, transcript_file in enumerate(transcript_files, 1):
    # ... existing processing code ...
    
    # ADD THIS inside the loop:
    if file_idx % 5 == 0:
        monitor.monitor_checkpoint(f"After Processing {file_idx} Files")

# ============================================================================
# Cell 7: NEW - Stop Monitoring
# ============================================================================
monitor.stop_monitoring()
report_path = monitor.save_report()
print(f"✅ Monitoring complete! Report: {report_path}")
```

---

## 🎉 You're All Set!

The monitoring system is ready to use. Just copy the code snippets into your notebooks and run them!

**Questions?** Check `RESOURCE_MONITORING_GUIDE.md` for detailed documentation.

---

*Created: 2026-01-12*  
*Packages installed: psutil, gputil, py3nvml*  
*Ready to monitor: ✅*
