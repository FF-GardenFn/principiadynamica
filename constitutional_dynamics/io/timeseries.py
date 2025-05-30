"""
Time Series - IO components for detecting and ordering time-series data

This module provides functions for detecting and ordering time-series data
from embeddings.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple

try:
    from rich.console import Console
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False
    console = None

logger = logging.getLogger("constitutional_dynamics.io.timeseries")


def detect_and_order_time_series(embeddings: Dict[str, List[float]],
                                 prefix_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Detect and order embeddings as a time series if they follow a pattern.

    Args:
        embeddings: Dictionary of embeddings
        prefix_pattern: Optional prefix pattern to identify time series

    Returns:
        List of ordered states
    """
    # If no prefix pattern, try to detect one
    if not prefix_pattern:
        # Look for common patterns like "state_1", "v1_", etc.
        prefixes = {}
        for key in embeddings.keys():
            parts = key.split("_")
            if len(parts) > 1 and parts[-1].isdigit():
                prefix = "_".join(parts[:-1]) + "_"
                prefixes[prefix] = prefixes.get(prefix, 0) + 1

        # Use the most common prefix if it covers enough keys
        if prefixes:
            most_common = max(prefixes.items(), key=lambda x: x[1])
            if most_common[1] >= 2:  # At least 2 matches
                prefix_pattern = most_common[0]
                if USE_RICH:
                    console.print(f"🔍 Detected time series pattern: [cyan]{prefix_pattern}[/]")
                else:
                    logger.info(f"Detected time series pattern: {prefix_pattern}")

    # If we have a prefix pattern, extract the time series
    if prefix_pattern:
        time_series = []

        # Extract all matching keys and sort by number
        matching_keys = []
        for key in embeddings.keys():
            if key.startswith(prefix_pattern):
                try:
                    # Extract the numeric part
                    suffix = key[len(prefix_pattern):]
                    if "_" in suffix:
                        suffix = suffix.split("_")[0]
                    num = int(suffix)
                    matching_keys.append((num, key))
                except ValueError:
                    # Not a numeric suffix
                    pass

        # Sort by numeric order
        matching_keys.sort()

        # Create ordered time series
        for idx, (num, key) in enumerate(matching_keys):
            time_series.append({
                "id": key,
                "embedding": embeddings[key],
                "timestamp": float(num),  # Use the numeric part as timestamp
                "index": idx
            })

        if USE_RICH:
            console.print(f"⏱️ Ordered [green]{len(time_series)}[/] states in time series")
        else:
            logger.info(f"Ordered {len(time_series)} states in time series")

        return time_series
    else:
        # No pattern detected, try to use timestamps in keys
        timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}[T_]\d{2}:\d{2}:\d{2})')
        timestamp_keys = []

        for key in embeddings.keys():
            match = timestamp_pattern.search(key)
            if match:
                timestamp_str = match.group(1)
                try:
                    # Convert to a sortable format
                    timestamp_str = timestamp_str.replace('T', ' ').replace('_', ' ')
                    import datetime
                    timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    timestamp_keys.append((timestamp, key))
                except ValueError:
                    pass

        if timestamp_keys:
            # Sort by timestamp
            timestamp_keys.sort()

            # Create ordered time series
            time_series = []
            for idx, (timestamp, key) in enumerate(timestamp_keys):
                time_series.append({
                    "id": key,
                    "embedding": embeddings[key],
                    "timestamp": timestamp.timestamp(),
                    "index": idx
                })

            if USE_RICH:
                console.print(f"⏱️ Ordered [green]{len(time_series)}[/] states by timestamp")
            else:
                logger.info(f"Ordered {len(time_series)} states by timestamp")

            return time_series

    # If no pattern or timestamps detected, return all embeddings in arbitrary order
    logger.warning("No time series pattern detected. Returning embeddings in arbitrary order.")
    time_series = []
    for idx, (key, embedding) in enumerate(embeddings.items()):
        time_series.append({
            "id": key,
            "embedding": embedding,
            "timestamp": float(idx),  # Use index as timestamp
            "index": idx
        })

    return time_series


def extract_subsequence(time_series: List[Dict[str, Any]],
                        start_idx: Optional[int] = None,
                        end_idx: Optional[int] = None,
                        window_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Extract a subsequence from a time series.

    Args:
        time_series: List of time series states
        start_idx: Optional start index
        end_idx: Optional end index
        window_size: Optional window size (used if start_idx is provided but end_idx is not)

    Returns:
        Subsequence of the time series
    """
    if not time_series:
        return []

    # Default to full sequence
    if start_idx is None:
        start_idx = 0
    if end_idx is None and window_size is not None:
        end_idx = min(start_idx + window_size, len(time_series))
    if end_idx is None:
        end_idx = len(time_series)

    # Validate indices
    start_idx = max(0, min(start_idx, len(time_series) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(time_series)))

    # Extract subsequence
    subsequence = time_series[start_idx:end_idx]

    # Update indices
    for i, state in enumerate(subsequence):
        state["index"] = i

    return subsequence
