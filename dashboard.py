# -*- coding: utf-8 -*-
"""Streamlit dashboard for multi-turn counsel chat dataset with mem0 integration."""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add our-pipeline to path
pipeline_path = Path(__file__).parent / "our-pipeline"
sys.path.insert(0, str(pipeline_path))

# Import modules from our-pipeline
import importlib.util

# Load process_datasets module
process_datasets_path = pipeline_path / "process_datasets.py"
spec = importlib.util.spec_from_file_location("process_datasets", process_datasets_path)
process_datasets_module = importlib.util.module_from_spec(spec)
sys.modules["process_datasets"] = process_datasets_module
spec.loader.exec_module(process_datasets_module)

# Load basic module
basic_path = pipeline_path / "basic.py"
spec2 = importlib.util.spec_from_file_location("basic", basic_path)
basic_module = importlib.util.module_from_spec(spec2)
sys.modules["basic"] = basic_module
spec2.loader.exec_module(basic_module)

# Import functions
parse_huggingface_dataset = process_datasets_module.parse_huggingface_dataset
transform_multi_turn_row = process_datasets_module.transform_multi_turn_row
process_multi_turn_dataset = process_datasets_module.process_multi_turn_dataset
create_mem0_config = basic_module.create_mem0_config
Mem0DatasetLoader = basic_module.Mem0DatasetLoader


def load_huggingface_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse HuggingFace dataset format.
    
    Args:
        file_path: Path to dataset JSON file
        
    Returns:
        List of conversation rows
    """
    return parse_huggingface_dataset(file_path)


def display_conversation(conversation: Dict[str, Any], index: int) -> None:
    """Display a single conversation in chat format.
    
    Args:
        conversation: Conversation data dictionary
        index: Conversation index number
    """
    st.subheader(f"Conversation {index + 1}")
    
    question_text = conversation.get("questionText", "")
    answer_text = conversation.get("answerText", "")
    messages = conversation.get("messages", [])
    
    if question_text:
        with st.expander("📝 Original Question", expanded=False):
            st.write(question_text)
    
    if answer_text:
        with st.expander("💡 Original Answer", expanded=False):
            st.write(answer_text)
    
    if messages and len(messages) > 0:
        st.markdown(f"### 💬 Conversation Messages ({len(messages)} messages)")
        
        # Create a container for the chat messages
        chat_container = st.container()
        
        with chat_container:
            # Use Streamlit's chat message component if available, otherwise use styled containers
            use_chat_message = hasattr(st, "chat_message")
            
            for msg_idx, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    continue
                    
                role = str(msg.get("role", "unknown")).lower()
                content = str(msg.get("content", "")).strip()
                
                if not content:
                    continue
                
                # Use Streamlit's native chat message component (available in Streamlit >= 1.28.0)
                if use_chat_message:
                    try:
                        if role == "counselor":
                            with st.chat_message("assistant"):
                                st.write(content)
                        elif role == "client":
                            with st.chat_message("user"):
                                st.write(content)
                        else:
                            with st.chat_message("user"):
                                st.write(f"**{role.capitalize()}:** {content}")
                    except Exception:
                        # Fallback if chat_message fails
                        if role == "counselor":
                            st.info(f"**Counselor:** {content}")
                        elif role == "client":
                            st.write(f"**Client:** {content}")
                        else:
                            st.write(f"**{role.capitalize()}:** {content}")
                else:
                    # Fallback for older Streamlit versions - use styled containers
                    if role == "counselor":
                        st.markdown(
                            f'<div style="background-color: #e3f2fd; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #2196F3;">'
                            f'<strong style="color: #1976D2;">Counselor:</strong><br>{content}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    elif role == "client":
                        st.markdown(
                            f'<div style="background-color: #f5f5f5; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #757575;">'
                            f'<strong style="color: #424242;">Client:</strong><br>{content}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.write(f"**{role.capitalize()}:** {content}")
    elif messages is not None and len(messages) == 0:
        st.info("ℹ️ This conversation has no messages (empty messages array).")
    else:
        st.warning("⚠️ No conversation messages found in this entry.")
        # Debug: show what keys are available
        with st.expander("Debug: Conversation structure"):
            st.json(conversation)


def call_lambda_api(
    prompt: str,
    api_key: str,
    model: str = "meta-llama/Llama-3.1-70B-Instruct-Turbo",
    api_endpoint: Optional[str] = None
) -> str:
    """Call Lambda Labs API for inference.
    
    Args:
        prompt: Input prompt text
        api_key: Lambda Labs API key
        model: Model name to use
        
    Returns:
        Generated response text
        
    Raises:
        requests.RequestException: If API call fails
        ValueError: If response format is unexpected
    """
    # Lambda Labs API endpoints - try multiple possible endpoints
    # If custom endpoint provided, use it first
    if api_endpoint:
        possible_urls = [api_endpoint]
    else:
        # Default Lambda Labs endpoints to try
        possible_urls = [
            "https://api.lambdalabs.com/v1/completions",
            "https://cloud.lambdalabs.com/api/v1/completions",
            "https://api.cloud.lambdalabs.com/v1/completions"
        ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    last_error = None
    for url in possible_urls:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            # Handle different response formats
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                # Try "text" first, then "message" with "content"
                if "text" in choice:
                    return choice["text"]
                elif "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
            
            # If format is different, try to extract any text response
            if "output" in result:
                return str(result["output"])
            if "response" in result:
                return str(result["response"])
                
            raise ValueError(f"Unexpected API response format. Response: {result}")
            
        except (requests.RequestException, ValueError) as e:
            last_error = e
            # Try next URL if this one fails
            continue
    
    # If all URLs failed, raise the last error with helpful message
    if last_error:
        error_msg = f"Lambda API request failed for all endpoints. Last error: {str(last_error)}"
        if isinstance(last_error, requests.RequestException) and hasattr(last_error, 'response') and last_error.response is not None:
            try:
                error_detail = last_error.response.json()
                error_msg += f"\nAPI Error Details: {error_detail}"
            except:
                error_msg += f"\nResponse Status: {last_error.response.status_code}"
                error_msg += f"\nResponse Text: {last_error.response.text[:200] if hasattr(last_error.response, 'text') else 'N/A'}"
        error_msg += f"\n\nTried endpoints: {', '.join(possible_urls)}"
        error_msg += "\n\nNote: If DNS resolution fails, the API endpoint might be incorrect or your network might be blocking the connection."
        raise requests.RequestException(error_msg) from last_error
    
    raise requests.RequestException("No endpoints available to try")


def query_llm_with_memory(
    query: str,
    memory_loader: Mem0DatasetLoader,
    lambda_api_key: str,
    user_id: str = "default",
    limit: int = 5,
    model: str = "meta-llama/Llama-3.1-70B-Instruct-Turbo",
    api_endpoint: Optional[str] = None
) -> Dict[str, Any]:
    """Query LLM with retrieved memory chunks.
    
    Args:
        query: User query string
        memory_loader: Mem0DatasetLoader instance
        lambda_api_key: Lambda Labs API key
        user_id: User ID for memory search
        limit: Number of memory chunks to retrieve
        model: Lambda model to use
        
    Returns:
        Dictionary with retrieved chunks and LLM response
    """
    # Search for relevant memories
    search_results = memory_loader.search(query=query, user_id=user_id, limit=limit)
    
    # Extract context from retrieved memories
    context_parts: List[str] = []
    retrieved_chunks: List[Dict[str, Any]] = []
    
    if "results" in search_results:
        for result in search_results["results"]:
            memory_text = result.get("memory", "")
            if memory_text:
                context_parts.append(memory_text)
                retrieved_chunks.append({
                    "text": memory_text,
                    "metadata": result.get("metadata", {})
                })
    
    # Build prompt with context
    context = "\n\n".join(context_parts)
    
    if context:
        full_prompt = f"""You are a helpful assistant with access to counseling conversation data. Use the following context from similar counseling conversations to inform your response.

Context from counseling conversations:
{context}

User question: {query}

Please provide a helpful response based on the context above:"""
    else:
        full_prompt = f"User question: {query}\n\nPlease provide a helpful response:"
    
    # Call Lambda API
    llm_response = call_lambda_api(full_prompt, lambda_api_key, model, api_endpoint=api_endpoint)
    
    return {
        "retrieved_chunks": retrieved_chunks,
        "llm_response": llm_response,
        "num_chunks": len(retrieved_chunks)
    }


def load_memories_to_mem0(
    dataset_path: str,
    openai_api_key: Optional[str],
    lambda_api_key: Optional[str],
    vector_store_path: str = "./chroma_db",
    collection_name: str = "counsel_chat_memories",
    user_id: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_huggingface: bool = False
) -> Optional[Mem0DatasetLoader]:
    """Load dataset into mem0 memory store.
    
    Args:
        dataset_path: Path to dataset JSON file
        openai_api_key: OpenAI API key for embeddings (optional if use_huggingface=True)
        lambda_api_key: Lambda API key (for future use, not used here)
        vector_store_path: Path for vector store
        collection_name: Name for the collection
        user_id: User ID for memories
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        use_huggingface: If True, use HuggingFace embeddings instead of OpenAI
        
    Returns:
        Mem0DatasetLoader instance if successful, None otherwise
    """
    if not use_huggingface and not openai_api_key:
        return None
    
    embedder_provider: Optional[str] = None
    embedder_config: Optional[Dict[str, Any]] = None
    
    if use_huggingface:
        # Use HuggingFace embeddings (free, local, no API quota)
        embedder_provider = "huggingface"
        embedder_config = {
            "model": "sentence-transformers/all-MiniLM-L6-v2"  # Fast, lightweight model
        }
    elif openai_api_key:
        # Set environment variable following pattern from process_datasets.py
        os.environ["OPENAI_API_KEY"] = openai_api_key
        embedder_provider = "openai"
        embedder_config = {
            "model": "text-embedding-3-small"
        }
    
    # Use different collection name based on embedder to avoid conflicts
    embedder_suffix = "hf" if use_huggingface else "openai"
    unique_collection_name = f"{collection_name}_{embedder_suffix}"
    
    config = create_mem0_config(
        vector_store_provider="chroma",
        collection_name=unique_collection_name,
        vector_store_path=vector_store_path,
        embedder_provider=embedder_provider,
        embedder_config=embedder_config
    )
    
    loader = Mem0DatasetLoader(config=config, user_id=user_id)
    
    # Process and load dataset
    try:
        count = process_multi_turn_dataset(
            file_path=dataset_path,
            memory_loader=loader,
            user_id=user_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            verbose=False
        )
        return loader
    except Exception as e:
        st.error(f"Error loading memories: {str(e)}")
        return None


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Counsel Chat Dashboard",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 Multi-Turn Counsel Chat Dashboard")
    st.markdown("Load, view, and query counseling conversations with mem0 memory integration")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Keys - load from environment variables (.env file)
        st.subheader("API Keys")
        
        # Load from environment variables (set in .env file)
        lambda_api_key = os.getenv("LAMBDA_API_KEY", "")
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        # Display status of API keys
        if lambda_api_key:
            st.success("✓ Lambda API Key loaded from .env")
        else:
            st.warning("⚠ Lambda API Key not found in .env")
        
        if openai_api_key:
            st.success("✓ OpenAI API Key loaded from .env")
        else:
            st.warning("⚠ OpenAI API Key not found in .env")
        
        st.caption("API keys are loaded from .env file. Edit .env to change keys.")
        
        # Embedder selection
        st.subheader("Embedding Settings")
        use_huggingface = st.checkbox(
            "Use HuggingFace embeddings (free, no API quota)",
            value=False,
            help="Use local HuggingFace embeddings instead of OpenAI. Avoids API quota limits but requires downloading models (~80MB first time)."
        )
        
        if use_huggingface:
            st.info("ℹ️ Using HuggingFace embeddings - no API quota needed!")
        else:
            st.info("ℹ️ Using OpenAI embeddings - requires API quota")
        
        # Chunking parameters
        st.subheader("Chunking Parameters")
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Maximum characters per chunk"
        )
        
        chunk_overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=1000,
            value=200,
            step=50,
            help="Characters to overlap between chunks"
        )
        
        # Model selection
        st.subheader("LLM Settings")
        lambda_model = st.selectbox(
            "Lambda Model",
            options=[
                "meta-llama/Llama-3.1-70B-Instruct-Turbo",
                "meta-llama/Llama-3.1-8B-Instruct-Turbo",
                "mistralai/Mistral-7B-Instruct-v0.2"
            ],
            index=0
        )
        
        lambda_api_endpoint = st.text_input(
            "Lambda API Endpoint (optional)",
            value="",
            help="Custom API endpoint URL. Leave empty to use default Lambda Labs endpoints. Example: https://api.example.com/v1/completions"
        )
        
        search_limit = st.number_input(
            "Memory Search Limit",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of memory chunks to retrieve for queries"
        )
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📚 Dataset Viewer", "💾 Load Memories", "🔍 Query LLM"])
    
    # Tab 1: Dataset Viewer
    with tab1:
        st.header("Dataset Viewer")
        
        dataset_path = "./data/multi_turn_counsel_chat.json"
        
        if not Path(dataset_path).exists():
            st.error(f"Dataset file not found: {dataset_path}")
            return
        
        # Load dataset
        if "conversations" not in st.session_state:
            with st.spinner("Loading dataset..."):
                try:
                    conversations = load_huggingface_dataset(dataset_path)
                    
                    # Validate conversations have messages
                    valid_conversations = []
                    for conv in conversations:
                        if isinstance(conv, dict):
                            # Ensure messages is a list
                            if "messages" in conv and isinstance(conv["messages"], list):
                                valid_conversations.append(conv)
                            elif "messages" not in conv:
                                # Add empty messages if missing
                                conv["messages"] = []
                                valid_conversations.append(conv)
                            else:
                                valid_conversations.append(conv)
                        else:
                            valid_conversations.append(conv)
                    
                    st.session_state.conversations = valid_conversations
                    st.session_state.dataset_loaded = True
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
        
        conversations = st.session_state.conversations
        
        # Count conversations with messages
        conversations_with_messages = sum(1 for conv in conversations if conv.get("messages") and len(conv.get("messages", [])) > 0)
        
        st.success(f"Loaded {len(conversations)} conversations ({conversations_with_messages} with messages)")
        
        # Pagination
        conversations_per_page = 5
        total_pages = (len(conversations) + conversations_per_page - 1) // conversations_per_page
        
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("◀ Previous") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
        with col2:
            st.write(f"Page {st.session_state.current_page} of {total_pages}")
        with col3:
            if st.button("Next ▶") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
        
        # Display conversations for current page
        start_idx = (st.session_state.current_page - 1) * conversations_per_page
        end_idx = start_idx + conversations_per_page
        
        for idx in range(start_idx, min(end_idx, len(conversations))):
            display_conversation(conversations[idx], idx)
            st.divider()
    
    # Tab 2: Load Memories
    with tab2:
        st.header("Load Memories into mem0")
        
        if not use_huggingface and not openai_api_key:
            st.warning("⚠️ Please set OPENAI_API_KEY in your .env file OR enable HuggingFace embeddings in the sidebar to load memories.")
            return
        
        dataset_path = "./data/multi_turn_counsel_chat.json"
        
        if not Path(dataset_path).exists():
            st.error(f"Dataset file not found: {dataset_path}")
            return
        
        if st.button("Load Dataset into mem0", type="primary"):
            with st.spinner("Loading memories into mem0 (this may take a while)..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Initializing mem0...")
                    progress_bar.progress(10)
                    
                    loader = load_memories_to_mem0(
                        dataset_path=dataset_path,
                        openai_api_key=openai_api_key,
                        lambda_api_key=lambda_api_key,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        use_huggingface=use_huggingface
                    )
                    
                    if loader:
                        progress_bar.progress(100)
                        status_text.text("Memories loaded successfully!")
                        st.session_state.memory_loader = loader
                        st.session_state.embedder_type = "huggingface" if use_huggingface else "openai"
                        st.success("✅ Dataset loaded into mem0 successfully!")
                    else:
                        st.error("Failed to load memories.")
                        
                except Exception as e:
                    st.error(f"Error loading memories: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        if "memory_loader" in st.session_state:
            # Check if embedder type matches current selection
            if "embedder_type" in st.session_state:
                current_embedder = "huggingface" if use_huggingface else "openai"
                if st.session_state.embedder_type != current_embedder:
                    st.error(f"⚠️ **Embedder Mismatch!** Memories were loaded with **{st.session_state.embedder_type}** embeddings, but you have **{current_embedder}** selected.")
                    st.info("💡 **Solution:** Click the button below to clear the old memories, then reload with your selected embedder.")
                    if st.button("🗑️ Clear Old Memories and Reload", type="primary"):
                        del st.session_state.memory_loader
                        del st.session_state.embedder_type
                        st.success("✅ Old memories cleared! Now click 'Load Dataset into mem0' again.")
                        st.rerun()
                else:
                    st.success("✅ Memories are loaded and ready for querying!")
            else:
                st.success("✅ Memories are loaded and ready for querying!")
    
    # Tab 3: Query LLM
    with tab3:
        st.header("Query LLM with Memory")
        
        if "memory_loader" not in st.session_state:
            st.warning("Please load memories first in the 'Load Memories' tab.")
            return
        
        if not lambda_api_key:
            st.warning("⚠️ Please set LAMBDA_API_KEY in your .env file to query the LLM.")
            return
        
        memory_loader = st.session_state.memory_loader
        
        # Query input
        user_query = st.text_area(
            "Enter your query:",
            height=100,
            placeholder="e.g., How do counselors help clients with self-esteem issues?"
        )
        
        if st.button("Query LLM", type="primary") and user_query:
            with st.spinner("Searching memories and generating response..."):
                try:
                    result = query_llm_with_memory(
                        query=user_query,
                        memory_loader=memory_loader,
                        lambda_api_key=lambda_api_key,
                        limit=search_limit,
                        model=lambda_model,
                        api_endpoint=lambda_api_endpoint.strip() if lambda_api_endpoint else None
                    )
                    
                    # Display retrieved chunks
                    st.subheader(f"Retrieved Memory Chunks ({result['num_chunks']})")
                    if result["retrieved_chunks"]:
                        for idx, chunk in enumerate(result["retrieved_chunks"], 1):
                            with st.expander(f"Chunk {idx}", expanded=False):
                                st.write(chunk["text"])
                                if chunk.get("metadata"):
                                    st.caption(f"Metadata: {chunk['metadata']}")
                    else:
                        st.info("No relevant chunks found.")
                    
                    st.divider()
                    
                    # Display LLM response
                    st.subheader("LLM Response")
                    st.write(result["llm_response"])
                    
                except requests.RequestException as e:
                    st.error(f"API request failed: {str(e)}")
                except Exception as e:
                    st.error(f"Error querying LLM: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

