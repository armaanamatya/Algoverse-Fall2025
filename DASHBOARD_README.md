# Counsel Chat Dashboard

Streamlit dashboard for viewing, loading, and querying the multi-turn counsel chat dataset with mem0 memory integration.

## Features

- **Dataset Viewer**: Browse conversations from the multi-turn-counsel-chat dataset
- **Memory Loading**: Load conversations into mem0 with configurable chunking
- **LLM Querying**: Query Lambda Labs LLM with retrieved memory chunks for context-aware responses

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

You have two options for API keys:

#### Option A: Streamlit Secrets (Recommended for local development)

1. Copy the example secrets file:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

2. Edit `.streamlit/secrets.toml` and add your API keys:
   ```toml
   LAMBDA_API_KEY = "your-lambda-api-key-here"
   OPENAI_API_KEY = "your-openai-api-key-here"
   ```

#### Option B: Enter in Dashboard

You can also enter API keys directly in the dashboard sidebar when running the app.

### 3. Run the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Usage

### Dataset Viewer Tab

- Browse conversations from the dataset
- View original questions, answers, and multi-turn message exchanges
- Navigate through pages of conversations

### Load Memories Tab

1. Enter your OpenAI API key (for embeddings) in the sidebar
2. Configure chunking parameters (chunk size, overlap)
3. Click "Load Dataset into mem0"
4. Wait for the loading process to complete

### Query LLM Tab

1. Ensure memories are loaded (from Load Memories tab)
2. Enter your Lambda API key in the sidebar
3. Select a Lambda model
4. Enter your query in the text area
5. Click "Query LLM"
6. View retrieved memory chunks and LLM response

## Configuration

### Chunking Parameters

- **Chunk Size**: Maximum characters per chunk (default: 1000)
- **Chunk Overlap**: Characters to overlap between chunks (default: 200)

### LLM Settings

- **Lambda Model**: Select from available models
- **Memory Search Limit**: Number of memory chunks to retrieve (default: 5)

## Notes

- The dataset file should be located at `./data/multi_turn_counsel_chat.json`
- Vector store (ChromaDB) will be created at `./chroma_db/`
- API keys are never hardcoded - use environment variables or Streamlit secrets

## Troubleshooting

- **Import errors**: Ensure all dependencies are installed and the `our-pipeline` directory is accessible
- **API errors**: Verify your API keys are correct and have sufficient credits
- **Memory loading fails**: Check that your OpenAI API key is valid and has access to embedding models

