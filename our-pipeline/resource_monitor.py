"""
Resource Monitoring Utilities for Jupyter Notebooks
Tracks RAM, GPU, and CPU usage during notebook execution
"""

import psutil
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Try to import GPU monitoring libraries
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPUtil not installed. GPU monitoring disabled.")
    print("Install with: pip install gputil")

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class ResourceSnapshot:
    """Single snapshot of resource usage"""
    timestamp: str
    ram_used_gb: float
    ram_percent: float
    ram_available_gb: float
    cpu_percent: float
    gpu_count: int
    gpu_usage: List[Dict[str, float]]  # List of {id, name, memory_used_mb, memory_total_mb, gpu_util_percent}
    
    def to_dict(self):
        return asdict(self)


class ResourceMonitor:
    """Monitor system resources during notebook execution"""
    
    def __init__(self, notebook_name: str = "notebook"):
        self.notebook_name = notebook_name
        self.snapshots: List[ResourceSnapshot] = []
        self.start_time = None
        self.end_time = None
        
        # Initialize GPU monitoring if available
        self.gpu_available = GPU_AVAILABLE
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except:
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False
    
    def get_ram_info(self) -> Dict[str, float]:
        """Get current RAM usage"""
        mem = psutil.virtual_memory()
        return {
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent,
            "available_gb": mem.available / (1024**3),
            "total_gb": mem.total / (1024**3)
        }
    
    def get_cpu_info(self) -> float:
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def get_gpu_info(self) -> List[Dict[str, float]]:
        """Get GPU usage information"""
        gpu_info = []
        
        if not self.gpu_available:
            return gpu_info
        
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_percent": (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                    "gpu_util_percent": gpu.load * 100,
                    "temperature": gpu.temperature
                })
        except Exception as e:
            print(f"⚠️ Error reading GPU info: {e}")
        
        return gpu_info
    
    def take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage"""
        ram_info = self.get_ram_info()
        cpu_percent = self.get_cpu_info()
        gpu_info = self.get_gpu_info()
        
        snapshot = ResourceSnapshot(
            timestamp=datetime.now().isoformat(),
            ram_used_gb=ram_info["used_gb"],
            ram_percent=ram_info["percent"],
            ram_available_gb=ram_info["available_gb"],
            cpu_percent=cpu_percent,
            gpu_count=len(gpu_info),
            gpu_usage=gpu_info
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def start_monitoring(self):
        """Start monitoring session"""
        self.start_time = datetime.now()
        self.snapshots = []
        print(f"🔍 Resource monitoring started for: {self.notebook_name}")
        print(f"⏰ Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Take initial snapshot
        initial = self.take_snapshot()
        self.print_snapshot(initial, "Initial State")
    
    def print_snapshot(self, snapshot: ResourceSnapshot, label: str = "Current"):
        """Print a formatted snapshot"""
        print(f"\n{'='*60}")
        print(f"📊 {label} Resource Usage")
        print(f"{'='*60}")
        print(f"⏰ Time: {snapshot.timestamp}")
        print(f"\n💾 RAM:")
        print(f"   Used: {snapshot.ram_used_gb:.2f} GB ({snapshot.ram_percent:.1f}%)")
        print(f"   Available: {snapshot.ram_available_gb:.2f} GB")
        print(f"\n🖥️  CPU: {snapshot.cpu_percent:.1f}%")
        
        if snapshot.gpu_count > 0:
            print(f"\n🎮 GPU ({snapshot.gpu_count} device(s)):")
            for gpu in snapshot.gpu_usage:
                print(f"   GPU {gpu['id']} ({gpu['name']}):")
                print(f"      Memory: {gpu['memory_used_mb']:.0f} MB / {gpu['memory_total_mb']:.0f} MB ({gpu['memory_percent']:.1f}%)")
                print(f"      Utilization: {gpu['gpu_util_percent']:.1f}%")
                print(f"      Temperature: {gpu['temperature']:.1f}°C")
        else:
            print(f"\n🎮 GPU: No GPU detected or monitoring unavailable")
        print(f"{'='*60}\n")
    
    def monitor_checkpoint(self, label: str = "Checkpoint"):
        """Take and print a checkpoint snapshot"""
        snapshot = self.take_snapshot()
        self.print_snapshot(snapshot, label)
        return snapshot
    
    def stop_monitoring(self):
        """Stop monitoring and generate summary"""
        self.end_time = datetime.now()
        final = self.take_snapshot()
        
        print(f"\n{'='*60}")
        print(f"🏁 Monitoring Complete: {self.notebook_name}")
        print(f"{'='*60}")
        print(f"⏰ End time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Duration: {(self.end_time - self.start_time).total_seconds():.1f} seconds")
        
        self.print_snapshot(final, "Final State")
        
        # Calculate statistics
        self.print_summary()
    
    def print_summary(self):
        """Print summary statistics"""
        if len(self.snapshots) < 2:
            print("⚠️ Not enough snapshots for summary")
            return
        
        ram_used = [s.ram_used_gb for s in self.snapshots]
        ram_percent = [s.ram_percent for s in self.snapshots]
        cpu_percent = [s.cpu_percent for s in self.snapshots]
        
        print(f"\n{'='*60}")
        print(f"📈 Summary Statistics")
        print(f"{'='*60}")
        print(f"\n💾 RAM Usage:")
        print(f"   Peak: {max(ram_used):.2f} GB ({max(ram_percent):.1f}%)")
        print(f"   Average: {sum(ram_used)/len(ram_used):.2f} GB ({sum(ram_percent)/len(ram_percent):.1f}%)")
        print(f"   Minimum: {min(ram_used):.2f} GB ({min(ram_percent):.1f}%)")
        print(f"   Delta: {ram_used[-1] - ram_used[0]:+.2f} GB")
        
        print(f"\n🖥️  CPU Usage:")
        print(f"   Peak: {max(cpu_percent):.1f}%")
        print(f"   Average: {sum(cpu_percent)/len(cpu_percent):.1f}%")
        
        # GPU summary
        if self.snapshots[0].gpu_count > 0:
            for gpu_id in range(self.snapshots[0].gpu_count):
                gpu_mem = [s.gpu_usage[gpu_id]['memory_used_mb'] for s in self.snapshots if len(s.gpu_usage) > gpu_id]
                gpu_util = [s.gpu_usage[gpu_id]['gpu_util_percent'] for s in self.snapshots if len(s.gpu_usage) > gpu_id]
                
                if gpu_mem:
                    print(f"\n🎮 GPU {gpu_id} ({self.snapshots[0].gpu_usage[gpu_id]['name']}):")
                    print(f"   Peak Memory: {max(gpu_mem):.0f} MB ({max(gpu_mem)/self.snapshots[0].gpu_usage[gpu_id]['memory_total_mb']*100:.1f}%)")
                    print(f"   Average Memory: {sum(gpu_mem)/len(gpu_mem):.0f} MB")
                    print(f"   Peak Utilization: {max(gpu_util):.1f}%")
                    print(f"   Average Utilization: {sum(gpu_util)/len(gpu_util):.1f}%")
        
        print(f"{'='*60}\n")
    
    def save_report(self, output_dir: str = "./resource_reports"):
        """Save monitoring report to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.notebook_name}_resources_{timestamp}.json"
        filepath = output_path / filename
        
        report = {
            "notebook_name": self.notebook_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "summary": self._calculate_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Resource report saved to: {filepath}")
        return filepath
    
    def _calculate_summary(self) -> Dict:
        """Calculate summary statistics"""
        if len(self.snapshots) < 2:
            return {}
        
        ram_used = [s.ram_used_gb for s in self.snapshots]
        ram_percent = [s.ram_percent for s in self.snapshots]
        cpu_percent = [s.cpu_percent for s in self.snapshots]
        
        summary = {
            "ram": {
                "peak_gb": max(ram_used),
                "average_gb": sum(ram_used) / len(ram_used),
                "min_gb": min(ram_used),
                "peak_percent": max(ram_percent),
                "average_percent": sum(ram_percent) / len(ram_percent),
                "delta_gb": ram_used[-1] - ram_used[0]
            },
            "cpu": {
                "peak_percent": max(cpu_percent),
                "average_percent": sum(cpu_percent) / len(cpu_percent)
            }
        }
        
        # Add GPU summary if available
        if self.snapshots[0].gpu_count > 0:
            summary["gpu"] = []
            for gpu_id in range(self.snapshots[0].gpu_count):
                gpu_mem = [s.gpu_usage[gpu_id]['memory_used_mb'] for s in self.snapshots if len(s.gpu_usage) > gpu_id]
                gpu_util = [s.gpu_usage[gpu_id]['gpu_util_percent'] for s in self.snapshots if len(s.gpu_usage) > gpu_id]
                
                if gpu_mem:
                    summary["gpu"].append({
                        "id": gpu_id,
                        "name": self.snapshots[0].gpu_usage[gpu_id]['name'],
                        "peak_memory_mb": max(gpu_mem),
                        "average_memory_mb": sum(gpu_mem) / len(gpu_mem),
                        "peak_utilization_percent": max(gpu_util),
                        "average_utilization_percent": sum(gpu_util) / len(gpu_util)
                    })
        
        return summary


def create_monitoring_cell_code(notebook_name: str) -> str:
    """Generate code to paste into notebook cells"""
    return f'''# Resource Monitoring Setup
import sys
from pathlib import Path

# Add our-pipeline to path if not already there
pipeline_path = Path("./our-pipeline")
if str(pipeline_path) not in sys.path:
    sys.path.insert(0, str(pipeline_path))

from resource_monitor import ResourceMonitor

# Create monitor instance
monitor = ResourceMonitor(notebook_name="{notebook_name}")

# Start monitoring
monitor.start_monitoring()
'''


def create_checkpoint_code() -> str:
    """Generate code for checkpoint monitoring"""
    return '''# Take a checkpoint snapshot
monitor.monitor_checkpoint("After Processing Turns")
'''


def create_stop_code() -> str:
    """Generate code to stop monitoring"""
    return '''# Stop monitoring and show summary
monitor.stop_monitoring()

# Save report to file
monitor.save_report()
'''


if __name__ == "__main__":
    # Test the monitor
    print("Testing Resource Monitor...")
    monitor = ResourceMonitor("test")
    monitor.start_monitoring()
    
    # Simulate some work
    time.sleep(2)
    monitor.monitor_checkpoint("Mid-test")
    
    time.sleep(2)
    monitor.stop_monitoring()
    monitor.save_report()
