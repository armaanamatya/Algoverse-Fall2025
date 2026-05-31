# Lambda Cloud GPU Setup Guide

This guide explains how to use Lambda Cloud GPU instances instead of running models locally.

## Overview

Lambda Cloud provides on-demand GPU instances that can run Ollama or vLLM to serve models remotely. This allows you to:
- Use powerful GPUs (H100, A100, etc.) without owning hardware
- Scale up/down as needed
- Pay only for what you use (~$2.99/hour for H100)

## GPU Options & Pricing

### Which GPU for Which Model?

| GPU | VRAM | $/hr | Models That Fit |
|-----|------|------|-----------------|
| **1× A10** | 24GB | ~$0.60 | llama3.1:8b, mistral:7b, qwen2.5:7b |
| **1× A100 40GB** | 40GB | ~$1.10 | llama3.1:8b, mistral:7b, phi3:14b |
| **1× A100 80GB** | 80GB | ~$1.29 | **gpt-oss:20b**, llama3.1:70b-q4, mixtral:8x7b |
| **1× H100 80GB** | 80GB | ~$2.49 | Same as A100 80GB, but 2-3× faster |
| **8× A100 80GB** | 640GB | ~$10.32 | llama3.1:70b full, large models |

### Model VRAM Requirements

| Model | Approx VRAM | Minimum GPU |
|-------|-------------|-------------|
| llama3.1:8b | ~16GB | A10 (24GB) ✅ |
| mistral:7b | ~14GB | A10 (24GB) ✅ |
| qwen2.5:7b | ~14GB | A10 (24GB) ✅ |
| phi3:14b | ~28GB | A100 40GB ✅ |
| **gpt-oss:20b** | ~40GB | **A100 80GB** ✅ |
| mixtral:8x7b | ~48GB | A100 80GB ✅ |
| llama3.1:70b-q4 | ~40GB | A100 80GB ✅ |
| llama3.1:70b | ~140GB | 8× A100 or 2× H100 |

### Recommendations for Your 4 Notebooks

| Budget | GPU | Model | Est. Cost (16 transcripts) |
|--------|-----|-------|---------------------------|
| **Cheapest** | 1× A10 | llama3.1:8b | ~$1.20-2.40 |
| **Best Value** | 1× A100 80GB | gpt-oss:20b | ~$2.58-3.87 |
| **Fastest** | 1× H100 | gpt-oss:20b | ~$2.49-4.98 |

**Note**: If you want to use `gpt-oss:20b` (your current model), you need at least **A100 80GB**. For cheaper options, switch to `llama3.1:8b` in your notebooks.

---

## Step 1: Launch a Lambda Cloud Instance

1. **Sign up/Login** at [cloud.lambdalabs.com](https://cloud.lambdalabs.com)
2. **Add SSH Key** (required):
   - Generate if needed: `ssh-keygen -t ed25519 -C "your_email@example.com"`
   - Copy your public key: `~/.ssh/id_ed25519.pub` (or `C:\Users\<you>\.ssh\id_ed25519.pub` on Windows)
   - Go to Dashboard → SSH Keys → Add SSH Key
3. **Choose an instance type**:
   - **Cheapest**: 1× A10 (24GB) ~$0.60/hr - use with llama3.1:8b
   - **For gpt-oss:20b**: 1× A100 80GB ~$1.29/hr (recommended)
   - **Fastest**: 1× H100 80GB ~$2.49/hr
4. **Launch the instance**:
   - Select your preferred region (closest = lower latency)
   - Choose Ubuntu 22.04 with Lambda Stack (pre-installed CUDA, PyTorch, etc.)
   - Click Launch and note the **IP address**

## Step 2: Set Up Ollama on Lambda Cloud Instance

SSH into your instance:

```bash
ssh ubuntu@<instance-ip>
```

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull the GPT-OSS-20B model:

```bash
ollama pull gpt-oss:20b
```

Start Ollama server:

```bash
ollama serve
```

**Note**: For production, you may want to run Ollama as a service or use `nohup`:

```bash
nohup ollama serve > ollama.log 2>&1 &
```

## Step 3: Connect from Your Local Machine

You have two options:

### Option A: SSH Tunnel (Recommended - More Secure)

Create an SSH tunnel to forward the Ollama port:

```bash
ssh -L 11434:localhost:11434 ubuntu@<instance-ip>
```

Keep this terminal open. Now your local machine can access Ollama at `http://localhost:11434`.

### Option B: Direct Connection (Less Secure)

If you expose the port directly (not recommended for production), you can connect directly:

```bash
# On Lambda Cloud instance, allow port 11434 (if firewall enabled)
# Then use: http://<instance-ip>:11434/v1
```

## Step 4: Configure Your Code

Update your configuration files to use Lambda Cloud:

### In `therapy_memnotincluded.py` or notebooks:

```python
# OPTION D: Use Lambda Cloud GPU instance
USE_LAMBDA_CLOUD = True
LAMBDA_CLOUD_BASE_URL = "http://localhost:11434/v1"  # For SSH tunnel
# OR
# LAMBDA_CLOUD_BASE_URL = "http://<instance-ip>:11434/v1"  # For direct connection
LAMBDA_CLOUD_MODEL = "gpt-oss:20b"
```

### For Mem0 Integration:

The Mem0 configuration will automatically use the Lambda Cloud instance when `USE_LAMBDA_CLOUD = True`.

## Step 5: Test the Connection

Run a simple test:

```python
from our_pipeline.alignment_evaluators import create_lambda_cloud_client

client = create_lambda_cloud_client(base_url="http://localhost:11434/v1")
response = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Step 6: Shut Down When Done

**Important**: Lambda Cloud charges by the hour. Always shut down instances when not in use:

1. Go to Lambda Cloud dashboard
2. Select your instance
3. Click "Terminate" or "Stop"

## Alternative: Using vLLM on Lambda Cloud

If you prefer vLLM over Ollama:

1. Install vLLM on the instance:
```bash
pip install vllm
```

2. Start vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model gpt-oss-20b \
    --port 8000
```

3. Update your config:
```python
LAMBDA_CLOUD_BASE_URL = "http://localhost:8000/v1"  # vLLM uses port 8000
```

## Troubleshooting

### Connection Refused
- Make sure Ollama is running on the instance: `ollama serve`
- Check SSH tunnel is active (if using tunnel)
- Verify firewall settings if using direct connection

### Model Not Found
- Pull the model on the instance: `ollama pull gpt-oss:20b`
- Verify model name matches exactly

### Slow Performance
- Check GPU utilization: `nvidia-smi`
- Consider using a larger instance or multiple GPUs
- Verify model is loaded on GPU (not CPU)

## Cost Optimization

- **Shut down instances** when not in use
- Use **spot instances** if available (cheaper but can be interrupted)
- **Monitor usage** through Lambda Cloud dashboard
- Consider **reserved instances** for long-term projects

## Security Notes

- **SSH tunnels are recommended** over direct connections
- Use **SSH keys** instead of passwords
- Don't expose ports publicly without proper authentication
- Consider using **VPN** for additional security

---

## Quick Reference (Copy-Paste Commands)

### On Lambda Instance (after SSH in)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model (choose one based on your GPU)
ollama pull llama3.1:8b      # For A10 (cheapest)
ollama pull gpt-oss:20b      # For A100 80GB+

# Start Ollama in background
nohup ollama serve > ollama.log 2>&1 &

# Verify it's running
curl http://localhost:11434/api/tags

# Check GPU usage
nvidia-smi
```

### On Your Local Machine (Windows)
```bash
# Start SSH tunnel (keep this terminal open)
ssh -L 11434:localhost:11434 ubuntu@<INSTANCE_IP>

# Then run your notebooks - they connect via localhost:11434
jupyter notebook therapy_memnotincluded.ipynb
```

### Changing Model in Notebooks

In any of your 4 notebooks, update the configuration cell:
```python
# For cheaper A10 GPU
OLLAMA_MODEL = "llama3.1:8b"

# For A100 80GB (your current setup)
OLLAMA_MODEL = "gpt-oss:20b"
```

---

## Your 4 Notebooks

These notebooks are configured to work with Lambda Cloud via SSH tunnel:

1. `therapy_memincluded.ipynb` - Therapy eval WITH memory context
2. `therapy_memnotincluded.ipynb` - Therapy eval WITHOUT memory context
3. `llm_counselor_memincluded.ipynb` - LLM counselor WITH memory
4. `llm_counselor_memnotincluded.ipynb` - LLM counselor WITHOUT memory

All use the same configuration pattern - just change `OLLAMA_MODEL` to match your GPU.
