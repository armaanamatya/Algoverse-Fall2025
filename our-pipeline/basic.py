# -*- coding: utf-8 -*-
"""Scalable template for loading multiple datasets into mem0 with chunking support."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
from tqdm import tqdm
import pandas as pd
from mem0 import Memory


def create_mem0_config(
    vector_store_provider: str,
    collection_name: str,
    vector_store_path: str,
    embedder_provider: Optional[str] = None,
    embedder_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create mem0 configuration dictionary.
    
    Args:
        vector_store_provider: Vector store provider (e.g., 'chroma', 'pinecone')
        collection_name: Name for the collection
        vector_store_path: Path to store vector database
        embedder_provider: Optional embedder provider (e.g., 'openai', 'huggingface')
        embedder_config: Optional embedder configuration
    
    Returns:
        Configuration dictionary for mem0
    """
    config: Dict[str, Any] = {
        "vector_store": {
            "provider": vector_store_provider,
            "config": {
                "collection_name": collection_name,
                "path": vector_store_path
            }
        }
    }
    
    if embedder_provider:
        config["embedder"] = {
            "provider": embedder_provider
        }
        if embedder_config:
            config["embedder"]["config"] = embedder_config
    
    return config


def load_json_dataset(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load dataset from JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        List of data items
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected JSON structure in {file_path}")


def load_csv_dataset(
    file_path: Union[str, Path],
    text_column: str,
    id_column: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load dataset from CSV file.
    
    Args:
        file_path: Path to CSV file
        text_column: Name of column containing text data
        id_column: Optional column name for IDs
    
    Returns:
        List of data items with 'text' and optionally 'id' keys
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required column is missing
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV")
    
    items: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        item: Dict[str, Any] = {"text": str(row[text_column])}
        if id_column and id_column in df.columns:
            item["id"] = str(row[id_column])
        else:
            item["id"] = str(idx)
        items.append(item)
    
    return items


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int = 0
) -> List[str]:
    """Split text into chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks: List[str] = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
        
        start = end - chunk_overlap
    
    return chunks


def chunk_dataset(
    dataset: List[Dict[str, Any]],
    text_key: str = "text",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    preserve_metadata: bool = True
) -> List[Dict[str, Any]]:
    """Chunk a dataset into smaller pieces.
    
    Args:
        dataset: List of data items
        text_key: Key in each item containing text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks
        preserve_metadata: Whether to preserve original item metadata in chunks
    
    Returns:
        List of chunked items
    """
    chunked_items: List[Dict[str, Any]] = []
    
    for item in dataset:
        if text_key not in item:
            continue
        
        text = str(item[text_key])
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_item: Dict[str, Any] = {
                "text": chunk,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks)
            }
            
            if preserve_metadata:
                for key, value in item.items():
                    if key != text_key:
                        chunk_item[f"original_{key}"] = value
            
            chunked_items.append(chunk_item)
    
    return chunked_items


def add_chunks_to_mem0(
    memory: Memory,
    chunks: List[Dict[str, Any]],
    user_id: str,
    text_key: str = "text",
    metadata_keys: Optional[List[str]] = None,
    verbose: bool = True
) -> int:
    """Add chunks to mem0 memory store.
    
    Args:
        memory: mem0 Memory instance
        chunks: List of chunk dictionaries
        user_id: User ID for the memories
        text_key: Key containing the text content
        metadata_keys: Optional list of keys to include as metadata
        verbose: Whether to show progress bar
    
    Returns:
        Number of chunks added
    """
    iterator = chunks
    if verbose:
        iterator = tqdm(chunks, desc="Adding chunks to mem0")
    
    added_count = 0
    for chunk in iterator:
        if text_key not in chunk:
            continue
        
        metadata: Dict[str, Any] = {}
        if metadata_keys:
            for key in metadata_keys:
                if key in chunk:
                    metadata[key] = chunk[key]
        else:
            for key, value in chunk.items():
                if key != text_key:
                    metadata[key] = value
        
        try:
            memory.add(
                messages=chunk[text_key],
                user_id=user_id,
                metadata=metadata
            )
            added_count += 1
        except Exception as e:
            if verbose:
                print(f"Error adding chunk: {e}")
            raise
    
    return added_count


def load_and_process_dataset(
    dataset_path: Union[str, Path],
    dataset_type: str,
    memory: Memory,
    user_id: str,
    text_column: Optional[str] = None,
    id_column: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 200,
    metadata_keys: Optional[List[str]] = None,
    verbose: bool = True
) -> int:
    """Load, optionally chunk, and add dataset to mem0.
    
    Args:
        dataset_path: Path to dataset file
        dataset_type: Type of dataset ('json' or 'csv')
        memory: mem0 Memory instance
        user_id: User ID for the memories
        text_column: Column name for CSV datasets (required for CSV)
        id_column: Optional ID column for CSV datasets
        chunk_size: Optional chunk size (None = no chunking)
        chunk_overlap: Overlap between chunks if chunking
        metadata_keys: Optional list of metadata keys to preserve
        verbose: Whether to show progress
    
    Returns:
        Number of items added to mem0
    
    Raises:
        ValueError: If dataset_type is invalid or required params missing
    """
    if dataset_type == "json":
        dataset = load_json_dataset(dataset_path)
        text_key = "memory" if "memory" in (dataset[0] if dataset else {}) else "text"
    elif dataset_type == "csv":
        if not text_column:
            raise ValueError("text_column is required for CSV datasets")
        dataset = load_csv_dataset(dataset_path, text_column, id_column)
        text_key = "text"
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")
    
    if not dataset:
        if verbose:
            print(f"No data found in {dataset_path}")
        return 0
    
    if chunk_size:
        dataset = chunk_dataset(
            dataset,
            text_key=text_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_metadata=True
        )
    
    return add_chunks_to_mem0(
        memory=memory,
        chunks=dataset,
        user_id=user_id,
        text_key=text_key,
        metadata_keys=metadata_keys,
        verbose=verbose
    )


class Mem0DatasetLoader:
    """Loader for managing multiple datasets in mem0."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        user_id: str = "default"
    ):
        """Initialize mem0 dataset loader.
        
        Args:
            config: mem0 configuration dictionary
            user_id: Default user ID for memories
        """
        self.memory = Memory.from_config(config)
        self.user_id = user_id
        self.loaded_datasets: List[str] = []
    
    def load_dataset(
        self,
        dataset_path: Union[str, Path],
        dataset_type: str,
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 200,
        metadata_keys: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        verbose: bool = True
    ) -> int:
        """Load a single dataset into mem0.
        
        Args:
            dataset_path: Path to dataset file
            dataset_type: Type of dataset ('json' or 'csv')
            text_column: Column name for CSV datasets
            id_column: Optional ID column for CSV datasets
            chunk_size: Optional chunk size
            chunk_overlap: Overlap between chunks
            metadata_keys: Optional metadata keys to preserve
            user_id: Optional user ID (uses default if not provided)
            verbose: Whether to show progress
        
        Returns:
            Number of items added
        """
        effective_user_id = user_id if user_id else self.user_id
        
        count = load_and_process_dataset(
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            memory=self.memory,
            user_id=effective_user_id,
            text_column=text_column,
            id_column=id_column,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata_keys=metadata_keys,
            verbose=verbose
        )
        
        self.loaded_datasets.append(str(dataset_path))
        return count
    
    def load_multiple_datasets(
        self,
        datasets: List[Dict[str, Any]],
        verbose: bool = True
    ) -> Dict[str, int]:
        """Load multiple datasets into mem0.
        
        Args:
            datasets: List of dataset configuration dictionaries, each with:
                - 'path': Path to dataset
                - 'type': Dataset type ('json' or 'csv')
                - 'text_column': Optional, for CSV
                - 'chunk_size': Optional chunk size
                - 'user_id': Optional user ID
                - Other optional parameters
            verbose: Whether to show progress
        
        Returns:
            Dictionary mapping dataset paths to counts of items added
        """
        results: Dict[str, int] = {}
        
        for dataset_config in datasets:
            path = dataset_config["path"]
            if verbose:
                print(f"\nLoading dataset: {path}")
            
            try:
                count = self.load_dataset(
                    dataset_path=path,
                    dataset_type=dataset_config["type"],
                    text_column=dataset_config.get("text_column"),
                    id_column=dataset_config.get("id_column"),
                    chunk_size=dataset_config.get("chunk_size"),
                    chunk_overlap=dataset_config.get("chunk_overlap", 200),
                    metadata_keys=dataset_config.get("metadata_keys"),
                    user_id=dataset_config.get("user_id"),
                    verbose=verbose
                )
                results[path] = count
            except Exception as e:
                if verbose:
                    print(f"Error loading {path}: {e}")
                results[path] = 0
                raise
        
        return results
    
    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Search memories.
        
        Args:
            query: Search query
            user_id: Optional user ID (uses default if not provided)
            limit: Optional limit on results
        
        Returns:
            Search results
        """
        effective_user_id = user_id if user_id else self.user_id
        return self.memory.search(query, user_id=effective_user_id, limit=limit)


# Example usage
if __name__ == "__main__":
    # Example 1: Basic configuration
    config = create_mem0_config(
        vector_store_provider="chroma",
        collection_name="multi_dataset_memories",
        vector_store_path="./chroma_db"
    )
    
    # Example 2: With custom embedder
    # config = create_mem0_config(
    #     vector_store_provider="chroma",
    #     collection_name="multi_dataset_memories",
    #     vector_store_path="./chroma_db",
    #     embedder_provider="openai",
    #     embedder_config={"model": "text-embedding-3-small"}
    # )
    
    # Initialize loader
    loader = Mem0DatasetLoader(config, user_id="user_1")
    
    # Example: Load multiple datasets
    datasets_config = [
        {
            "path": "./data/dataset1.json",
            "type": "json",
            "chunk_size": 1000,
            "chunk_overlap": 200
        },
        {
            "path": "./data/dataset2.csv",
            "type": "csv",
            "text_column": "content",
            "id_column": "id",
            "chunk_size": 500,
            "user_id": "user_2"
        }
    ]
    
    # Load all datasets
    # results = loader.load_multiple_datasets(datasets_config)
    # print(f"Loaded datasets: {results}")
    
    # Search example
    # results = loader.search("example query", limit=5)
    # print(f"Search results: {results}")

