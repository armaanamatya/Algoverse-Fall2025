# -*- coding: utf-8 -*-
"""Process downloaded Hugging Face datasets into mem0 chunks."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from basic import (
    create_mem0_config,
    chunk_dataset,
    add_chunks_to_mem0,
    Mem0DatasetLoader
)


def parse_huggingface_dataset(
    file_path: str
) -> List[Dict[str, Any]]:
    """Parse Hugging Face dataset JSON format.
    
    Args:
        file_path: Path to Hugging Face dataset JSON file
        
    Returns:
        List of row dictionaries extracted from the dataset
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON structure is invalid
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected dictionary structure in {file_path}")
    
    if "rows" not in data:
        raise ValueError(f"No 'rows' key found in {file_path}")
    
    rows: List[Dict[str, Any]] = []
    for row_item in data["rows"]:
        if "row" in row_item:
            rows.append(row_item["row"])
    
    return rows


def transform_counsel_chat_row(
    row: Dict[str, Any]
) -> Dict[str, Any]:
    """Transform counsel-chat row into format for mem0.
    
    Args:
        row: Raw row from counsel-chat dataset
        
    Returns:
        Transformed row with 'text' field and metadata
    """
    question_text = row.get("questionText", "")
    answer_text = row.get("answerText", "")
    
    combined_text = f"Question: {question_text}\n\nAnswer: {answer_text}"
    
    transformed: Dict[str, Any] = {
        "text": combined_text,
        "questionID": row.get("questionID"),
        "questionTitle": row.get("questionTitle", ""),
        "questionLink": row.get("questionLink", ""),
        "topic": row.get("topic", ""),
        "therapistInfo": row.get("therapistInfo", ""),
        "therapistURL": row.get("therapistURL", ""),
        "upvotes": row.get("upvotes", 0),
        "views": row.get("views", 0)
    }
    
    return transformed


def transform_multi_turn_row(
    row: Dict[str, Any]
) -> Dict[str, Any]:
    """Transform multi-turn-counsel-chat row into format for mem0.
    
    Args:
        row: Raw row from multi-turn-counsel-chat dataset
        
    Returns:
        Transformed row with 'text' field and metadata
    """
    question_text = row.get("questionText", "")
    answer_text = row.get("answerText", "")
    messages = row.get("messages", [])
    
    messages_text = ""
    if messages:
        messages_text = "\n\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages
        ])
    
    combined_text = f"Question: {question_text}\n\nAnswer: {answer_text}"
    if messages_text:
        combined_text = f"{combined_text}\n\nConversation:\n{messages_text}"
    
    transformed: Dict[str, Any] = {
        "text": combined_text,
        "questionText": question_text,
        "answerText": answer_text,
        "num_messages": len(messages)
    }
    
    return transformed


def process_counsel_chat_dataset(
    file_path: str,
    memory_loader: Mem0DatasetLoader,
    user_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    verbose: bool = True
) -> int:
    """Process counsel-chat dataset into mem0.
    
    Args:
        file_path: Path to counsel-chat JSON file
        memory_loader: Mem0DatasetLoader instance
        user_id: User ID for memories
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        verbose: Whether to show progress
        
    Returns:
        Number of chunks added to mem0
    """
    if verbose:
        print(f"Loading counsel-chat dataset from {file_path}...")
    
    rows = parse_huggingface_dataset(file_path)
    
    if verbose:
        print(f"Found {len(rows)} rows")
    
    transformed_rows: List[Dict[str, Any]] = []
    for row in rows:
        transformed_rows.append(transform_counsel_chat_row(row))
    
    if verbose:
        print("Chunking dataset...")
    
    chunked_items = chunk_dataset(
        dataset=transformed_rows,
        text_key="text",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_metadata=True
    )
    
    if verbose:
        print(f"Created {len(chunked_items)} chunks")
        print("Adding chunks to mem0...")
    
    count = add_chunks_to_mem0(
        memory=memory_loader.memory,
        chunks=chunked_items,
        user_id=user_id,
        text_key="text",
        metadata_keys=None,
        verbose=verbose
    )
    
    return count


def process_multi_turn_dataset(
    file_path: str,
    memory_loader: Mem0DatasetLoader,
    user_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    verbose: bool = True
) -> int:
    """Process multi-turn-counsel-chat dataset into mem0.
    
    Args:
        file_path: Path to multi-turn-counsel-chat JSON file
        memory_loader: Mem0DatasetLoader instance
        user_id: User ID for memories
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        verbose: Whether to show progress
        
    Returns:
        Number of chunks added to mem0
    """
    if verbose:
        print(f"Loading multi-turn-counsel-chat dataset from {file_path}...")
    
    rows = parse_huggingface_dataset(file_path)
    
    if verbose:
        print(f"Found {len(rows)} rows")
    
    transformed_rows: List[Dict[str, Any]] = []
    for row in rows:
        transformed_rows.append(transform_multi_turn_row(row))
    
    if verbose:
        print("Chunking dataset...")
    
    chunked_items = chunk_dataset(
        dataset=transformed_rows,
        text_key="text",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_metadata=True
    )
    
    if verbose:
        print(f"Created {len(chunked_items)} chunks")
        print("Adding chunks to mem0...")
    
    count = add_chunks_to_mem0(
        memory=memory_loader.memory,
        chunks=chunked_items,
        user_id=user_id,
        text_key="text",
        metadata_keys=None,
        verbose=verbose
    )
    
    return count


def process_all_datasets(
    counsel_chat_path: str,
    multi_turn_path: str,
    vector_store_path: str = "./chroma_db",
    collection_name: str = "counsel_chat_memories",
    user_id: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    verbose: bool = True,
    openai_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o-mini"
) -> Dict[str, int]:
    """Process both datasets into mem0.
    
    Args:
        counsel_chat_path: Path to counsel-chat JSON file
        multi_turn_path: Path to multi-turn-counsel-chat JSON file
        vector_store_path: Path for vector store
        collection_name: Name for the collection
        user_id: User ID for memories
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        verbose: Whether to show progress
        openai_api_key: OpenAI API key for embeddings and LLM
        llm_model: LLM model to use (default: gpt-4o)
        
    Returns:
        Dictionary mapping dataset names to number of chunks added
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    embedder_config: Optional[Dict[str, Any]] = None
    if openai_api_key:
        embedder_config = {
            "model": "text-embedding-3-small"
        }
    
    config = create_mem0_config(
        vector_store_provider="chroma",
        collection_name=collection_name,
        vector_store_path=vector_store_path,
        embedder_provider="openai" if openai_api_key else None,
        embedder_config=embedder_config
    )
    
    if openai_api_key:
        config["llm"] = {
            "provider": "openai",
            "config": {
                "model": llm_model
            }
        }
    
    loader = Mem0DatasetLoader(config=config, user_id=user_id)
    
    results: Dict[str, int] = {}
    
    if verbose:
        print("=" * 60)
        print("Processing Counsel Chat Dataset")
        print("=" * 60)
    
    results["counsel_chat"] = process_counsel_chat_dataset(
        file_path=counsel_chat_path,
        memory_loader=loader,
        user_id=user_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        verbose=verbose
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("Processing Multi-Turn Counsel Chat Dataset")
        print("=" * 60)
    
    results["multi_turn_counsel_chat"] = process_multi_turn_dataset(
        file_path=multi_turn_path,
        memory_loader=loader,
        user_id=user_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        verbose=verbose
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("Processing Complete")
        print("=" * 60)
        print(f"Counsel Chat: {results['counsel_chat']} chunks added")
        print(f"Multi-Turn: {results['multi_turn_counsel_chat']} chunks added")
        print(f"Total: {sum(results.values())} chunks added")
    
    return results


if __name__ == "__main__":
    # Get OpenAI API key from environment variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it using: export OPENAI_API_KEY='your-key-here' "
            "or in PowerShell: $env:OPENAI_API_KEY = 'your-key-here'"
        )
    
    results = process_all_datasets(
        counsel_chat_path="./data/counsel_chat.json",
        multi_turn_path="./data/multi_turn_counsel_chat.json",
        vector_store_path="./chroma_db",
        collection_name="counsel_chat_memories",
        user_id="default",
        chunk_size=1000,
        chunk_overlap=200,
        verbose=True,
        openai_api_key=OPENAI_API_KEY,
        llm_model="gpt-4o-mini"
    )
