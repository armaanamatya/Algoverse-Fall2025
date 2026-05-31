# Lambda Cloud + Ollama Setup Guide

This guide explains how to set up two Lambda Cloud GPU instances running Ollama, and configure the notebooks for parallel execution by two different users.

---

## Overview

| Notebook | User | Lambda Instance | SSH Tunnel Port |
|----------|------|-----------------|-----------------|
| `llm_counselor_memincluded.ipynb` | Person A | Instance 1 (IP: `<LAMBDA_IP_1>`) | 11434 |
| `llm_counselor_memnotincluded.ipynb` | Person B | Instance 2 (IP: `<LAMBDA_IP_2>`) | 11435 |

---

## Part 1: Lambda Cloud Instance Setup

### 1.1 Create Lambda Cloud Account & Instance

1. Go to [Lambda Cloud](https://lambdalabs.com/cloud)
2. Create an account and add payment method
3. Launch a GPU instance:
   - Recommended: gpu_1x_h100_pcie at $2.49/GPU/hr
   - Select Ubuntu 22.04
   - Add your SSH public key (see below if you don't have one)

### 1.2 Generate SSH Key (if needed)

```bash
# On your local machine (Windows PowerShell or Git Bash)
ssh-keygen -t ed25519 -C "your_email@example.com"

# View your public key to add to Lambda
cat ~/.ssh/id_ed25519.pub
```

### 1.3 Note Your Instance IPs

After launching, note the public IP addresses:
- **Instance 1 IP**: `___.___.___.__` (for Person A - memincluded)
- **Instance 2 IP**: `___.___.___.__` (for Person B - memnotincluded)

---

## Part 2: Install Ollama on Lambda Instance

SSH into each Lambda instance and run the following commands:

### 2.1 Connect to Instance

```bash
# Person A (Instance 1)
ssh ubuntu@<LAMBDA_IP_1>

# Person B (Instance 2)
ssh ubuntu@<LAMBDA_IP_2>
```

### 2.2 Install Ollama

```bash
# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

### 2.3 Pull Required Models

```bash
# Pull the main LLM model (gpt-oss:20b)
ollama pull gpt-oss:20b

# Pull the embedding model for mem0
ollama pull nomic-embed-text

# Verify models are downloaded
ollama list
```

Expected output:
```
NAME                ID              SIZE      MODIFIED
gpt-oss:20b         xxxxx           ~40GB     Just now
nomic-embed-text    xxxxx           ~274MB    Just now
```

### 2.4 Start Ollama Server

```bash
# Start Ollama server (runs on port 11434 by default)
ollama serve
```

### 2.5 Test Ollama is Running (on Lambda instance)

```bash
# Test the API
curl http://localhost:11434/api/tags

# Test a simple generation
curl http://localhost:11434/api/generate -d '{
  "model": "gpt-oss:20b",
  "prompt": "Hello",
  "stream": false
}'
```

---

## Part 3: SSH Tunnel Setup (Local Machine)

Create SSH tunnels to forward Lambda's Ollama port to your local machine.

### 3.1 Person A - memincluded notebook (Port 11434)

```bash
# Open terminal and run:
ssh -L 11434:localhost:11434 ubuntu@<LAMBDA_IP_1>

# Keep this terminal open while running the notebook
```

### 3.2 Person B - memnotincluded notebook (Port 11435)

```bash
# Open terminal and run:
ssh -L 11435:localhost:11434 ubuntu@<LAMBDA_IP_2>

# Keep this terminal open while running the notebook
```

### 3.3 Verify Tunnel is Working (Local Machine)

```bash
# Person A - test port 11434
curl http://localhost:11434/api/tags

# Person B - test port 11435
curl http://localhost:11435/api/tags
```

You should see a JSON response listing the available models.

---

## Part 4: Notebook Configuration

### 4.1 Configure `llm_counselor_memincluded.ipynb` (Person A)

Edit **Cell 2** (Configuration):

```python
# Cell 2: Configuration
USE_OLLAMA = False  # Set to False - we're using Lambda
USE_LAMBDA_CLOUD = True  # Enable Lambda Cloud

# Lambda Cloud Configuration
LAMBDA_CLOUD_BASE_URL = "http://localhost:11434/v1"  # Port 11434 for Person A
LAMBDA_CLOUD_MODEL = "gpt-oss:20b"

# Model configuration
COUNSELOR_MODEL = "gpt-oss:20b"
JUDGE_MODEL = "gpt-oss:20b"

print(f"Configuration:")
print(f"  Backend: Lambda Cloud GPU (Instance 1)")
print(f"  Base URL: {LAMBDA_CLOUD_BASE_URL}")
print(f"  Counselor Model: {COUNSELOR_MODEL}")
print(f"  Judge Model: {JUDGE_MODEL}")
print(f"  Memory Access: YES")
```

Edit **Cell 3** (Initialize Client):

```python
# Cell 3: Initialize OpenAI Client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="lambda"  # Ollama doesn't need a real key
)
MODEL = "gpt-oss:20b"
print(f"Using Lambda Cloud GPU instance")
print(f"  Base URL: http://localhost:11434/v1")
print(f"  Model: {MODEL}")
```

Edit **Cell 5** (Initialize Mem0):

```python
# Cell 5: Initialize Mem0
import shutil

RESET_MEMORIES = True  # Set to True for fresh run

# Separate ChromaDB path for this notebook
CHROMA_DB_PATH = "./chroma_db_memincluded"

if RESET_MEMORIES and Path(CHROMA_DB_PATH).exists():
    shutil.rmtree(CHROMA_DB_PATH)
    print(f"Deleted existing {CHROMA_DB_PATH} folder for fresh start")

USER_ID = "patient_0518_014"

# Lambda Cloud Mem0 configuration
mem0_base_url = "http://localhost:11434"  # Without /v1

mem_config = create_mem0_config_with_llm(
    llm_provider="ollama",
    model="gpt-oss:20b",
    base_url=mem0_base_url
)
mem_config["vector_store"]["config"]["collection_name"] = "llm_counselor_memincluded"
mem_config["vector_store"]["config"]["path"] = CHROMA_DB_PATH

# Embedder configuration - nomic-embed-text
mem_config["embedder"] = {
    "provider": "ollama",
    "config": {
        "ollama_base_url": mem0_base_url,
        "model": "nomic-embed-text",
        "embedding_dims": 512
    }
}

print(f"Embedder config: {mem_config['embedder']}")

memory = initialize_mem0(
    config=mem_config,
    reset_collection=RESET_MEMORIES
)
print(f"Initialized Mem0 with Lambda Cloud LLM: gpt-oss:20b")
print(f"Embedding model: nomic-embed-text")
print(f"ChromaDB Path: {CHROMA_DB_PATH}")
print(f"USER_ID: {USER_ID}")
```

---

### 4.2 Configure `llm_counselor_memnotincluded.ipynb` (Person B)

Edit **Cell 2** (Configuration):

```python
# Cell 2: Configuration
USE_OLLAMA = False
USE_LAMBDA_CLOUD = True
USE_OPENAI = False

LAMBDA_CLOUD_BASE_URL = "http://localhost:11435/v1"  # Port 11435 for Person B
LAMBDA_CLOUD_MODEL = "gpt-oss:20b"

COUNSELOR_MODEL = "gpt-oss:20b"
JUDGE_MODEL = "gpt-oss:20b"

print(f"Configuration:")
print(f"  Backend: Lambda Cloud GPU (Instance 2)")
print(f"  Base URL: {LAMBDA_CLOUD_BASE_URL}")
print(f"  Counselor Model: {COUNSELOR_MODEL}")
print(f"  Judge Model: {JUDGE_MODEL}")
print(f"  Memory Access: NO")
print("  Make sure SSH tunnel is active: ssh -L 11435:localhost:11434 ubuntu@<LAMBDA_IP_2>")
```

Edit **Cell 3** (Initialize Client):

```python
# Cell 3: Initialize OpenAI Client
client = OpenAI(
    base_url="http://localhost:11435/v1",  # Port 11435
    api_key="lambda"
)
MODEL = "gpt-oss:20b"
print(f"Using Lambda Cloud GPU instance")
print(f"  Base URL: http://localhost:11435/v1")
print(f"  Model: {MODEL}")
print("Client created successfully!")
```

Edit **Cell 5** (Initialize Mem0):

```python
# Cell 5: Initialize Mem0
import shutil

RESET_MEMORIES = True

# Separate ChromaDB path for this notebook
CHROMA_DB_PATH = "./chroma_db_memnotincluded"

if RESET_MEMORIES and Path(CHROMA_DB_PATH).exists():
    try:
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"Deleted existing {CHROMA_DB_PATH} folder for fresh start")
    except PermissionError as e:
        print(f"Warning: Could not delete {CHROMA_DB_PATH}")
        RESET_MEMORIES = False

USER_ID = "patient_0518_014"

# Lambda Cloud Mem0 configuration
mem0_base_url = "http://localhost:11435"  # Without /v1, Port 11435

mem_config = create_mem0_config_with_llm(
    llm_provider="ollama",
    model="gpt-oss:20b",
    base_url=mem0_base_url
)
mem_config["vector_store"]["config"]["collection_name"] = "llm_counselor_memnotincluded"
mem_config["vector_store"]["config"]["path"] = CHROMA_DB_PATH

# Embedder configuration - nomic-embed-text
mem_config["embedder"] = {
    "provider": "ollama",
    "config": {
        "ollama_base_url": mem0_base_url,
        "model": "nomic-embed-text",
        "embedding_dims": 512
    }
}

print(f"Embedder config: {mem_config['embedder']}")

memory = initialize_mem0(
    config=mem_config,
    reset_collection=RESET_MEMORIES
)
print(f"Initialized Mem0 with Lambda Cloud LLM: gpt-oss:20b")
print(f"Embedding model: nomic-embed-text")
print(f"Collection: llm_counselor_memnotincluded")
print(f"ChromaDB Path: {CHROMA_DB_PATH}")
print(f"USER_ID: {USER_ID}")
```

---

## Part 5: Quick Reference

### SSH Tunnel Commands

```bash
# Person A (memincluded) - Port 11434
ssh -L 11434:localhost:11434 ubuntu@<LAMBDA_IP_1>

# Person B (memnotincluded) - Port 11435
ssh -L 11435:localhost:11434 ubuntu@<LAMBDA_IP_2>
```

### Test Connectivity

```bash
# Person A
curl http://localhost:11434/api/tags

# Person B
curl http://localhost:11435/api/tags
```

### Configuration Summary

| Setting | Person A (memincluded) | Person B (memnotincluded) |
|---------|------------------------|---------------------------|
| SSH Tunnel Port | 11434 | 11435 |
| Base URL | `http://localhost:11434/v1` | `http://localhost:11435/v1` |
| Mem0 URL | `http://localhost:11434` | `http://localhost:11435` |
| LLM Model | `gpt-oss:20b` | `gpt-oss:20b` |
| Embedding Model | `nomic-embed-text` | `nomic-embed-text` |
| ChromaDB Path | `./chroma_db_memincluded` | `./chroma_db_memnotincluded` |
| Collection | `llm_counselor_memincluded` | `llm_counselor_memnotincluded` |

---

## Part 6: Troubleshooting

### Issue: Connection refused

```bash
# Check if Ollama is running on Lambda instance
ssh ubuntu@<LAMBDA_IP> "curl localhost:11434/api/tags"

# If not running, start it:
ssh ubuntu@<LAMBDA_IP> "nohup ollama serve > ollama.log 2>&1 &"
```

### Issue: Model not found

```bash
# SSH into instance and pull the model
ssh ubuntu@<LAMBDA_IP>
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

### Issue: SSH tunnel disconnects

```bash
# Use autossh for persistent tunnels (install first: sudo apt install autossh)
autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -L 11434:localhost:11434 ubuntu@<LAMBDA_IP_1>
```

### Issue: Mem0 embedding errors

Make sure the embedding model is pulled:
```bash
ollama pull nomic-embed-text
```

Verify embedder config in notebook:
```python
mem_config["embedder"] = {
    "provider": "ollama",
    "config": {
        "ollama_base_url": "http://localhost:11434",  # or 11435
        "model": "nomic-embed-text",
        "embedding_dims": 512
    }
}
```

---

## Part 7: Running the Notebooks

### Pre-flight Checklist

- [ ] Lambda instances are running
- [ ] Ollama is running on both instances (`ollama serve`)
- [ ] Both models are pulled (`gpt-oss:20b`, `nomic-embed-text`)
- [ ] SSH tunnels are active (keep terminals open)
- [ ] Connectivity test passes (`curl http://localhost:1143X/api/tags`)
- [ ] Notebook configuration matches your assigned port

### Execution Order

1. **Start SSH tunnel** (keep terminal open)
2. **Open Jupyter notebook**
3. **Run Cell 1-5** (imports, config, client, load files, mem0)
4. **Run Cell 6** (main processing loop) - this takes the longest
5. **Run Cell 7-9** (memory audit, save results, visualization)

### Monitoring Progress

Both notebooks have checkpoint support. If interrupted:
- Set `RESUME_FROM_CHECKPOINT = True` in Cell 6
- Re-run from Cell 6

Check progress in:
- `./output_llm_counselor_memincluded/checkpoints/`
- `./output_llm_counselor_memnotincluded/checkpoints/`

---

## Appendix: Direct Connection (No SSH Tunnel)

If SSH tunnels are problematic, you can connect directly (less secure):

### On Lambda Instance - Allow External Connections

```bash
# Edit Ollama config to listen on all interfaces
sudo systemctl stop ollama
OLLAMA_HOST=0.0.0.0 ollama serve
```

### In Notebook - Use Direct IP

```python
# Person A
LAMBDA_CLOUD_BASE_URL = "http://<LAMBDA_IP_1>:11434/v1"
mem0_base_url = "http://<LAMBDA_IP_1>:11434"

# Person B
LAMBDA_CLOUD_BASE_URL = "http://<LAMBDA_IP_2>:11434/v1"
mem0_base_url = "http://<LAMBDA_IP_2>:11434"
```

**Note**: This exposes Ollama to the internet. Only use for testing and ensure Lambda security groups are properly configured.
