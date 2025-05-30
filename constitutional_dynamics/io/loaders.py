"""
Data Loaders - IO components for loading embeddings and aligned examples

This module provides functions for loading embeddings and aligned examples
from various file formats.
"""

import json
import sys
import logging
from typing import Dict, List, Any, Optional, Union

try:
    from rich.console import Console
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False
    console = None

logger = logging.getLogger("constitutional_dynamics.io.loaders")


def load_embeddings(file_path: str) -> Dict[str, Union[List[float], Dict[str, List[float]]]]:
    """
    Load embeddings from a file in various formats.

    Args:
        file_path: Path to the embeddings file

    Returns:
        Dictionary mapping identifiers to embeddings
    """
    if USE_RICH:
        console.print(f"📊 Loading embeddings from [cyan]{file_path}[/]...")
    else:
        logger.info(f"Loading embeddings from {file_path}...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        error_msg = f"Error loading embeddings file: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Handle different formats
    embeddings = {}

    # Format 1: Direct mapping of ID to embedding vector
    if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
        embeddings = data

    # Format 2: Report format with summaries containing embeddings
    elif isinstance(data, dict) and "summaries" in data:
        for summary in data["summaries"]:
            if "path" in summary and "embedding" in summary:
                embeddings[summary["path"]] = summary["embedding"]

    # Format 3: List of objects with path and embedding
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        for item in data:
            if "path" in item and "embedding" in item:
                embeddings[item["path"]] = item["embedding"]
            elif "id" in item and "embedding" in item:
                embeddings[item["id"]] = item["embedding"]

    # Format 4: File similarities format
    elif isinstance(data, dict) and "file_similarities" in data:
        file_similarities = data["file_similarities"]
        # Extract unique files
        unique_files = set()
        for file, neighbors in file_similarities.items():
            unique_files.add(file)
            for neighbor in neighbors:
                if "path" in neighbor:
                    unique_files.add(neighbor["path"])

        # Create dummy embeddings (can't recover actual embeddings)
        for file in unique_files:
            embeddings[file] = [0.0] * 1024  # Placeholder

    if not embeddings:
        error_msg = "No valid embeddings found in the file"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if USE_RICH:
        console.print(f"✅ Loaded [green]{len(embeddings)}[/] embeddings")
    else:
        logger.info(f"Loaded {len(embeddings)} embeddings")

    return embeddings


def aligned_examples(file_path: str, dimension: int = 1024) -> List[List[float]]:
    """
    Load examples of aligned behavior from a file.

    Args:
        file_path: Path to the aligned examples file
        dimension: Dimensionality of the embedding space

    Returns:
        List of aligned example vectors
    """
    if USE_RICH:
        console.print(f"🎯 Loading aligned examples from [cyan]{file_path}[/]...")
    else:
        logger.info(f"Loading aligned examples from {file_path}...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        error_msg = f"Error loading aligned examples file: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Handle different formats
    examples = []

    # Format 1: List of embeddings
    if isinstance(data, list):
        examples = data

    # Format 2: Dictionary with "aligned_examples" key
    elif isinstance(data, dict) and "aligned_examples" in data:
        examples = data["aligned_examples"]

    # Format 3: Dictionary mapping IDs to embeddings
    elif isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
        examples = list(data.values())

    # Format 4: List of objects with embedding
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        for item in data:
            if "embedding" in item:
                examples.append(item["embedding"])

    # Validate and normalize examples
    valid_examples = []
    for example in examples:
        if isinstance(example, list) and len(example) > 0:
            # Pad or truncate to match dimension
            if len(example) < dimension:
                example = example + [0.0] * (dimension - len(example))
            elif len(example) > dimension:
                example = example[:dimension]
            valid_examples.append(example)

    if not valid_examples:
        error_msg = "No valid aligned examples found in the file"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if USE_RICH:
        console.print(f"✅ Loaded [green]{len(valid_examples)}[/] aligned examples")
    else:
        logger.info(f"Loaded {len(valid_examples)} aligned examples")

    return valid_examples
