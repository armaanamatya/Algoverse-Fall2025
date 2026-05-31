# Resource Monitoring Code Snippets
# Copy and paste these into your Jupyter notebooks

# ==============================================================================
# SNIPPET 1: Add to the BEGINNING of your notebook (after existing imports)
# ==============================================================================

"""
# Resource Monitoring Setup
import sys
from pathlib import Path

# Import the resource monitor
pipeline_path = Path("./our-pipeline")
if str(pipeline_path) not in sys.path:
    sys.path.insert(0, str(pipeline_path))

from resource_monitor import ResourceMonitor

# Create monitor instance
# Change notebook_name to match your notebook
monitor = ResourceMonitor(notebook_name="therapy_memincluded")  # or "therapy_memnotincluded"

# Start monitoring
monitor.start_monitoring()
"""

# ==============================================================================
# SNIPPET 2: Add AFTER Mem0 initialization
# ==============================================================================

"""
# Checkpoint: After Mem0 Initialization
monitor.monitor_checkpoint("After Mem0 Initialization")
"""

# ==============================================================================
# SNIPPET 3: Add AFTER loading transcripts
# ==============================================================================

"""
# Checkpoint: After Loading Transcripts
monitor.monitor_checkpoint("After Loading Transcripts")
"""

# ==============================================================================
# SNIPPET 4: Add INSIDE your file processing loop (every 5 files)
# ==============================================================================

"""
# Inside the file processing loop, add this:
if file_idx % 5 == 0:
    monitor.monitor_checkpoint(f"After Processing {file_idx}/{len(transcript_files)} Files")
"""

# ==============================================================================
# SNIPPET 5: Add at the END of your notebook
# ==============================================================================

"""
# Stop monitoring and generate summary
monitor.stop_monitoring()

# Save detailed report to JSON file
report_path = monitor.save_report()

print(f"\\n✅ Resource monitoring complete!")
print(f"📄 Report saved to: {report_path}")
"""

# ==============================================================================
# SNIPPET 6: OPTIONAL - Compare two notebook reports
# ==============================================================================

"""
# Compare Resource Usage Between Notebooks
import json
from pathlib import Path

# Load the two most recent reports
reports_dir = Path("./resource_reports")
report_files = sorted(reports_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

if len(report_files) >= 2:
    # Load reports
    report1 = json.load(open(report_files[0]))
    report2 = json.load(open(report_files[1]))
    
    print("="*60)
    print("Resource Usage Comparison")
    print("="*60)
    
    print(f"\\nNotebook 1: {report1['notebook_name']}")
    print(f"  Duration: {report1['duration_seconds']:.1f} seconds")
    print(f"  Peak RAM: {report1['summary']['ram']['peak_gb']:.2f} GB ({report1['summary']['ram']['peak_percent']:.1f}%)")
    print(f"  Average RAM: {report1['summary']['ram']['average_gb']:.2f} GB")
    print(f"  Peak CPU: {report1['summary']['cpu']['peak_percent']:.1f}%")
    
    print(f"\\nNotebook 2: {report2['notebook_name']}")
    print(f"  Duration: {report2['duration_seconds']:.1f} seconds")
    print(f"  Peak RAM: {report2['summary']['ram']['peak_gb']:.2f} GB ({report2['summary']['ram']['peak_percent']:.1f}%)")
    print(f"  Average RAM: {report2['summary']['ram']['average_gb']:.2f} GB")
    print(f"  Peak CPU: {report2['summary']['cpu']['peak_percent']:.1f}%")
    
    # Calculate differences
    ram_diff = report1['summary']['ram']['peak_gb'] - report2['summary']['ram']['peak_gb']
    time_diff = report1['duration_seconds'] - report2['duration_seconds']
    
    print(f"\\nDifferences:")
    print(f"  RAM: {ram_diff:+.2f} GB ({report1['notebook_name']} vs {report2['notebook_name']})")
    print(f"  Time: {time_diff:+.1f} seconds")
    
    # GPU comparison if available
    if 'gpu' in report1['summary'] and 'gpu' in report2['summary']:
        gpu1_mem = report1['summary']['gpu'][0]['peak_memory_mb']
        gpu2_mem = report2['summary']['gpu'][0]['peak_memory_mb']
        print(f"  GPU Memory: {(gpu1_mem - gpu2_mem):+.0f} MB")
        
    print("="*60)
else:
    print("⚠️ Need at least 2 reports to compare. Run both notebooks first.")
"""

# ==============================================================================
# SNIPPET 7: OPTIONAL - Plot resource usage over time
# ==============================================================================

"""
# Visualize Resource Usage
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load most recent report
reports_dir = Path("./resource_reports")
report_files = sorted(reports_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

if report_files:
    with open(report_files[0]) as f:
        report = json.load(f)
    
    # Extract data
    snapshots = report['snapshots']
    ram_usage = [s['ram_used_gb'] for s in snapshots]
    cpu_usage = [s['cpu_percent'] for s in snapshots]
    checkpoint_nums = list(range(len(snapshots)))
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # RAM plot
    ax1.plot(checkpoint_nums, ram_usage, 'b-o', linewidth=2, markersize=6, label='RAM Usage')
    ax1.axhline(y=report['summary']['ram']['peak_gb'], color='r', linestyle='--', 
                label=f"Peak: {report['summary']['ram']['peak_gb']:.2f} GB")
    ax1.set_ylabel('RAM Usage (GB)', fontsize=12)
    ax1.set_title(f"Resource Usage: {report['notebook_name']}", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CPU plot
    ax2.plot(checkpoint_nums, cpu_usage, 'g-o', linewidth=2, markersize=6, label='CPU Usage')
    ax2.axhline(y=report['summary']['cpu']['peak_percent'], color='r', linestyle='--',
                label=f"Peak: {report['summary']['cpu']['peak_percent']:.1f}%")
    ax2.set_ylabel('CPU Usage (%)', fontsize=12)
    ax2.set_xlabel('Checkpoint Number', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{report['notebook_name']}_resource_usage.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved to: {plot_filename}")
    
    plt.show()
else:
    print("⚠️ No reports found. Run monitoring first.")
"""
