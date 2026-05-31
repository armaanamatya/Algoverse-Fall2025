# Resource Monitoring Guide for Jupyter Notebooks

This guide shows you how to monitor RAM, GPU, and CPU usage in your therapy evaluation notebooks.

---

## 📦 Installation

First, install the required monitoring packages:

```bash
pip install psutil gputil py3nvml
```

**Package purposes:**
- `psutil` - RAM and CPU monitoring (required)
- `gputil` - GPU monitoring via NVIDIA GPUs (optional, only if you have NVIDIA GPU)
- `py3nvml` - Advanced NVIDIA GPU monitoring (optional)

---

## 🚀 Quick Start

### Step 1: Add Monitoring to Your Notebook

Add this cell at the **beginning** of your notebook (after imports):

```python
# Cell 1: Setup Resource Monitoring
import sys
from pathlib import Path

# Import the resource monitor
sys.path.insert(0, str(Path("./our-pipeline")))
from resource_monitor import ResourceMonitor

# Create monitor instance
monitor = ResourceMonitor(notebook_name="therapy_memincluded")  # or "therapy_memnotincluded"

# Start monitoring
monitor.start_monitoring()
```

### Step 2: Add Checkpoints During Execution

Add checkpoint cells at key points in your notebook:

```python
# After loading transcripts
monitor.monitor_checkpoint("After Loading Transcripts")
```

```python
# After processing first file
monitor.monitor_checkpoint("After Processing File 1")
```

```python
# After processing all files
monitor.monitor_checkpoint("After All Files Processed")
```

### Step 3: Stop Monitoring at the End

Add this cell at the **end** of your notebook:

```python
# Stop monitoring and show summary
monitor.stop_monitoring()

# Save detailed report to JSON file
monitor.save_report()
```

---

## 📊 Example Output

### Initial State
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

### Checkpoint
```
============================================================
📊 After Processing File 1 Resource Usage
============================================================
⏰ Time: 2026-01-12T21:50:15

💾 RAM:
   Used: 12.34 GB (76.5%)
   Available: 3.82 GB

🖥️  CPU: 45.8%

🎮 GPU (1 device(s)):
   GPU 0 (NVIDIA GeForce RTX 3060):
      Memory: 4567 MB / 12288 MB (37.2%)
      Utilization: 78.5%
      Temperature: 68.0°C
============================================================
```

### Final Summary
```
============================================================
🏁 Monitoring Complete: therapy_memincluded
============================================================
⏰ End time: 2026-01-12T22:15:30
⏱️  Duration: 1830.5 seconds

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

## 📝 Complete Example for therapy_memincluded.ipynb

Here's how to add monitoring to your existing notebook:

### Cell 1 (Add after existing imports):
```python
# ============================================================================
# RESOURCE MONITORING SETUP
# ============================================================================
from resource_monitor import ResourceMonitor

monitor = ResourceMonitor(notebook_name="therapy_memincluded")
monitor.start_monitoring()
```

### Cell 2 (After Mem0 initialization):
```python
monitor.monitor_checkpoint("After Mem0 Initialization")
```

### Cell 3 (After loading transcripts):
```python
monitor.monitor_checkpoint("After Loading Transcripts")
```

### Cell 4 (Inside your processing loop - add this every 5 files):
```python
# Inside the file processing loop
for file_idx, transcript_file in enumerate(transcript_files, 1):
    # ... existing processing code ...
    
    # Add checkpoint every 5 files
    if file_idx % 5 == 0:
        monitor.monitor_checkpoint(f"After Processing {file_idx} Files")
```

### Cell 5 (At the very end):
```python
# ============================================================================
# STOP MONITORING AND SAVE REPORT
# ============================================================================
monitor.stop_monitoring()
report_path = monitor.save_report()

print(f"\n✅ Resource monitoring complete!")
print(f"📄 Report saved to: {report_path}")
```

---

## 🔍 Monitoring Without GPU

If you don't have a GPU or GPUtil isn't installed, the monitor will still work for RAM and CPU:

```
⚠️ GPUtil not installed. GPU monitoring disabled.
Install with: pip install gputil

============================================================
📊 Initial State Resource Usage
============================================================
⏰ Time: 2026-01-12T21:45:00

💾 RAM:
   Used: 8.45 GB (52.3%)
   Available: 7.71 GB

🖥️  CPU: 12.5%

🎮 GPU: No GPU detected or monitoring unavailable
============================================================
```

---

## 📁 Output Files

Resource reports are saved to `./resource_reports/` directory:

```
resource_reports/
├── therapy_memincluded_resources_20260112_214500.json
└── therapy_memnotincluded_resources_20260112_220130.json
```

### JSON Report Structure:
```json
{
  "notebook_name": "therapy_memincluded",
  "start_time": "2026-01-12T21:45:00",
  "end_time": "2026-01-12T22:15:30",
  "duration_seconds": 1830.5,
  "snapshots": [
    {
      "timestamp": "2026-01-12T21:45:00",
      "ram_used_gb": 8.45,
      "ram_percent": 52.3,
      "ram_available_gb": 7.71,
      "cpu_percent": 12.5,
      "gpu_count": 1,
      "gpu_usage": [
        {
          "id": 0,
          "name": "NVIDIA GeForce RTX 3060",
          "memory_used_mb": 1024,
          "memory_total_mb": 12288,
          "memory_percent": 8.3,
          "gpu_util_percent": 5.2,
          "temperature": 45.0
        }
      ]
    }
  ],
  "summary": {
    "ram": {
      "peak_gb": 14.23,
      "average_gb": 11.56,
      "min_gb": 8.45,
      "peak_percent": 88.2,
      "average_percent": 71.6,
      "delta_gb": 5.78
    },
    "cpu": {
      "peak_percent": 89.3,
      "average_percent": 52.1
    },
    "gpu": [
      {
        "id": 0,
        "name": "NVIDIA GeForce RTX 3060",
        "peak_memory_mb": 8192,
        "average_memory_mb": 5234,
        "peak_utilization_percent": 95.2,
        "average_utilization_percent": 68.4
      }
    ]
  }
}
```

---

## 🔄 Comparing Both Notebooks

To compare resource usage between the two notebooks:

### 1. Run both notebooks with monitoring
```python
# In therapy_memincluded.ipynb
monitor = ResourceMonitor(notebook_name="therapy_memincluded")

# In therapy_memnotincluded.ipynb
monitor = ResourceMonitor(notebook_name="therapy_memnotincluded")
```

### 2. Load and compare the JSON reports
```python
import json
from pathlib import Path

# Load reports
reports_dir = Path("./resource_reports")
mem_included = json.load(open(reports_dir / "therapy_memincluded_resources_TIMESTAMP.json"))
mem_not_included = json.load(open(reports_dir / "therapy_memnotincluded_resources_TIMESTAMP.json"))

# Compare RAM usage
print("RAM Comparison:")
print(f"Memory-Included Peak: {mem_included['summary']['ram']['peak_gb']:.2f} GB")
print(f"Memory-Not-Included Peak: {mem_not_included['summary']['ram']['peak_gb']:.2f} GB")
print(f"Difference: {mem_included['summary']['ram']['peak_gb'] - mem_not_included['summary']['ram']['peak_gb']:+.2f} GB")

# Compare GPU usage (if available)
if 'gpu' in mem_included['summary'] and 'gpu' in mem_not_included['summary']:
    print("\nGPU Memory Comparison:")
    print(f"Memory-Included Peak: {mem_included['summary']['gpu'][0]['peak_memory_mb']:.0f} MB")
    print(f"Memory-Not-Included Peak: {mem_not_included['summary']['gpu'][0]['peak_memory_mb']:.0f} MB")
```

---

## 💡 Best Practices

### 1. **Take Strategic Checkpoints**
Don't checkpoint too frequently (slows down execution). Good checkpoint locations:
- After initialization
- After loading data
- Every N files processed (e.g., every 5 files)
- Before/after memory-intensive operations
- At the end

### 2. **Monitor Long-Running Notebooks**
For notebooks that run for hours:
```python
# Add periodic checkpoints in your processing loop
for i, turn in enumerate(turns):
    # ... process turn ...
    
    # Checkpoint every 50 turns
    if i % 50 == 0:
        monitor.monitor_checkpoint(f"Turn {i}")
```

### 3. **Check for Memory Leaks**
If RAM usage keeps increasing without plateauing, you might have a memory leak:
```python
# Look at the delta in the summary
# If delta is very large and positive, investigate memory leaks
```

### 4. **GPU Monitoring for Ollama**
If using Ollama with GPU, monitor to ensure:
- GPU memory doesn't exceed capacity
- GPU utilization is reasonable (not 0% or 100% constantly)

---

## 🐛 Troubleshooting

### "GPUtil not installed"
```bash
pip install gputil
```

### "No GPU detected" (but you have one)
- Make sure NVIDIA drivers are installed
- Try installing: `pip install py3nvml`
- Check if GPU is visible: `nvidia-smi` in terminal

### "Permission denied" when saving report
```python
# Specify a different output directory
monitor.save_report(output_dir="./my_reports")
```

### High RAM usage
- Check if you're loading too many files at once
- Consider processing files one at a time
- Clear variables you don't need: `del large_variable`

---

## 📊 Visualization (Optional)

Create plots from your monitoring data:

```python
import matplotlib.pyplot as plt
import json

# Load report
with open("resource_reports/therapy_memincluded_resources_TIMESTAMP.json") as f:
    report = json.load(f)

# Extract data
timestamps = [s['timestamp'] for s in report['snapshots']]
ram_usage = [s['ram_used_gb'] for s in report['snapshots']]
cpu_usage = [s['cpu_percent'] for s in report['snapshots']]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(range(len(ram_usage)), ram_usage, 'b-o', label='RAM Usage (GB)')
ax1.set_ylabel('RAM (GB)')
ax1.set_title('Resource Usage Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(len(cpu_usage)), cpu_usage, 'r-o', label='CPU Usage (%)')
ax2.set_ylabel('CPU (%)')
ax2.set_xlabel('Checkpoint Number')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resource_usage_plot.png', dpi=150)
plt.show()
```

---

## 🎯 Expected Resource Usage

### Typical RAM Usage:
- **Base Jupyter + Python**: 1-2 GB
- **Loading Mem0 + ChromaDB**: +2-3 GB
- **Loading Ollama model (20B)**: +8-12 GB (if running locally)
- **Processing transcripts**: +1-3 GB per file
- **Total Expected**: 12-20 GB for memory-included version

### Typical GPU Usage (if using local Ollama):
- **Model loaded**: 8-16 GB VRAM (for 20B model)
- **During inference**: 60-95% GPU utilization
- **Idle**: 5-10% GPU utilization

### CPU Usage:
- **During LLM calls**: 40-80%
- **During memory operations**: 20-40%
- **Idle**: 5-15%

---

## ✅ Quick Checklist

Before running your notebooks:

- [ ] Install monitoring packages: `pip install psutil gputil py3nvml`
- [ ] Add `ResourceMonitor` import to notebook
- [ ] Add `monitor.start_monitoring()` at beginning
- [ ] Add checkpoints at key locations
- [ ] Add `monitor.stop_monitoring()` at end
- [ ] Add `monitor.save_report()` to save results
- [ ] Check `resource_reports/` directory for output files

---

*Last Updated: 2026-01-12*
