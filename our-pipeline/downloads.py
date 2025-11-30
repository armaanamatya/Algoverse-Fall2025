# -*- coding: utf-8 -*-
"""Download rows from Hugging Face datasets."""

import json
from pathlib import Path
from typing import Dict, Any
import requests


def download_multi_turn_counsel_chat(
    output_path: str
) -> Dict[str, Any]:
    """Download first rows from multi-turn-counsel-chat dataset.
    
    Args:
        output_path: Path to save the downloaded data as JSON
        
    Returns:
        Dictionary containing the downloaded data
        
    Raises:
        requests.RequestException: If the HTTP request fails
        ValueError: If the response is invalid
    """
    url = "https://datasets-server.huggingface.co/first-rows?dataset=Jingy2000%2Fmulti-turn-counsel-chat&config=default&split=train"
    
    response = requests.get(url)
    response.raise_for_status()
    
    data: Dict[str, Any] = response.json()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return data


def download_counsel_chat(
    output_path: str,
    offset: int = 0,
    length: int = 100
) -> Dict[str, Any]:
    """Download rows from counsel-chat dataset.
    
    Args:
        output_path: Path to save the downloaded data as JSON
        offset: Starting offset for rows
        length: Number of rows to download
        
    Returns:
        Dictionary containing the downloaded data
        
    Raises:
        requests.RequestException: If the HTTP request fails
        ValueError: If the response is invalid
    """
    url = f"https://datasets-server.huggingface.co/rows?dataset=nbertagnolli%2Fcounsel-chat&config=default&split=train&offset={offset}&length={length}"
    
    response = requests.get(url)
    response.raise_for_status()
    
    data: Dict[str, Any] = response.json()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return data


def download_all_datasets(
    multi_turn_output_path: str,
    counsel_chat_output_path: str
) -> Dict[str, Dict[str, Any]]:
    """Download both datasets.
    
    Args:
        multi_turn_output_path: Path to save multi-turn-counsel-chat data
        counsel_chat_output_path: Path to save counsel-chat data
        
    Returns:
        Dictionary mapping dataset names to their downloaded data
    """
    results: Dict[str, Dict[str, Any]] = {}
    
    print("Downloading multi-turn-counsel-chat dataset...")
    results["multi_turn_counsel_chat"] = download_multi_turn_counsel_chat(
        output_path=multi_turn_output_path
    )
    
    print("Downloading counsel-chat dataset...")
    results["counsel_chat"] = download_counsel_chat(
        output_path=counsel_chat_output_path
    )
    
    return results


if __name__ == "__main__":
    # Download both datasets
    results = download_all_datasets(
        multi_turn_output_path="./data/multi_turn_counsel_chat.json",
        counsel_chat_output_path="./data/counsel_chat.json"
    )
    
    print(f"Downloaded {len(results)} datasets")
    print(f"Multi-turn dataset saved to: ./data/multi_turn_counsel_chat.json")
    print(f"Counsel-chat dataset saved to: ./data/counsel_chat.json")
