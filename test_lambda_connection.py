import sys
import os
import time
from pathlib import Path

# Add our-pipeline to path
pipeline_path = Path("./our-pipeline")
if str(pipeline_path) not in sys.path:
    sys.path.insert(0, str(pipeline_path))

from alignment_evaluators import create_lambda_cloud_client
import requests

print("Testing connection to Ollama API...")
try:
    test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
    models_data = test_response.json()
    print(f"✓ Connection successful!")
    print(f"  Available models: {[m.get('name', 'unknown') for m in models_data.get('models', [])]}")
    
    # Check if gpt-oss:20b is available
    model_names = [m.get('name', '') for m in models_data.get('models', [])]
    if 'gpt-oss:20b' not in model_names:
        print(f"\n⚠ WARNING: gpt-oss:20b not found in available models!")
        print(f"  You may need to pull it on the server: ollama pull gpt-oss:20b")
except Exception as e:
    print(f"✗ Connection test failed: {e}")
    print("Make sure:")
    print("  1. SSH tunnel is active: ssh -L 11434:localhost:11434 ubuntu@209.20.159.159")
    print("  2. Ollama is running on the server: ollama serve")
    sys.exit(1)

print("\nCreating client with extended timeout (5 minutes)...")
# Create client with longer timeout for model loading
from openai import OpenAI
client = OpenAI(
    api_key="lambda-cloud",
    base_url="http://localhost:11434/v1",
    timeout=300.0  # 5 minutes - 20B model can take time to load
)

print("\nSending request to gpt-oss:20b...")
print("(Note: First request may take 1-2 minutes if model needs to load into GPU memory)")
print("Waiting for response (this may take a while on first load)...")
start_time = time.time()

# Start a progress indicator in a separate thread
import threading
progress_stop = threading.Event()

def show_progress():
    """Show progress dots every 10 seconds"""
    while not progress_stop.is_set():
        time.sleep(10)
        if not progress_stop.is_set():
            elapsed = time.time() - start_time
            print(f"  Still waiting... ({elapsed:.0f}s elapsed)")

progress_thread = threading.Thread(target=show_progress, daemon=True)
progress_thread.start()

try:
    # Use with_options for per-request timeout override if needed
    response = client.with_options(timeout=300.0).chat.completions.create(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    progress_stop.set()  # Stop progress indicator
    elapsed = time.time() - start_time
    print(f"\n✓ Response received in {elapsed:.2f} seconds:")
    print(response.choices[0].message.content)
except Exception as e:
    progress_stop.set()  # Stop progress indicator
    elapsed = time.time() - start_time
    print(f"\n✗ Error after {elapsed:.2f} seconds:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")
    print("\nTroubleshooting:")
    print("  1. Check if Ollama is running on server: ssh ubuntu@209.20.159.159 'ps aux | grep ollama'")
    print("  2. Check if model is loaded: ssh ubuntu@209.20.159.159 'ollama list'")
    print("  3. Check GPU memory: ssh ubuntu@209.20.159.159 'nvidia-smi'")
    raise
