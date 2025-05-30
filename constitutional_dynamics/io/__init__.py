"""
Constitutional Dynamics IO Module

This module provides input/output functionality for the Constitutional Dynamics package,
including loading embeddings, detecting time series, and live metrics collection.
"""

from .loaders import load_embeddings, aligned_examples
from .timeseries import detect_and_order_time_series
from .live import create_collector, LiveMetricsCollector

__all__ = [
    "load_embeddings",
    "aligned_examples",
    "detect_and_order_time_series",
    "create_collector",
    "LiveMetricsCollector"
]
