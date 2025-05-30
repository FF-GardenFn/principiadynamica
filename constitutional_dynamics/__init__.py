"""
Constitutional Dynamics 0.1 - Vector-space alignment metrics & state-transition calculus

A framework for analyzing and monitoring alignment as a trajectory through
embedding space rather than a static property.
"""

__version__ = "0.1.0"

# Import from core module
from .core.space import AlignmentVectorSpace
from .core.transition import analyze_transition, predict_trajectory
from .core.metrics import calculate_stability_metrics, evaluate_alignment_robustness
from .core.optimise import AlignmentOptimizer, GraphEnhancedAlignmentOptimizer

# Import from io module
from .io.loaders import load_embeddings, aligned_examples
from .io.timeseries import detect_and_order_time_series
from .io.live import create_collector, LiveMetricsCollector

# Import from vis module
from .vis.visualizer import create_visualizer, TrajectoryVisualizer

# Import from integrations module
from .integrations.graph import create_graph_manager, GraphManager
from .integrations.quantum import create_annealer, QuantumAnnealer
from .integrations.strategist import create_strategist, MetaStrategist

# Import from cli module
# CLI main function is not imported here as it's only intended to be run via
# python -m constitutional_dynamics.cli.main or the constitutional-dynamics console script

# Import from cfg module
from .cfg import (
    load_config,
    get_default_config,
    load_logging_config,
    get_default_logging_config,
    configure_logging,
    DEFAULT_CONFIG_PATH,
    DEFAULT_LOGGING_PATH
)

# Define what's available when using "from constitutional_dynamics import *"
__all__ = [
    # Version
    "__version__",

    # Core
    "AlignmentVectorSpace",
    "analyze_transition",
    "predict_trajectory",
    "calculate_stability_metrics",
    "evaluate_alignment_robustness",
    "AlignmentOptimizer",
    "GraphEnhancedAlignmentOptimizer",

    # IO
    "load_embeddings",
    "aligned_examples",
    "detect_and_order_time_series",
    "create_collector",
    "LiveMetricsCollector",

    # Visualization
    "create_visualizer",
    "TrajectoryVisualizer",

    # Integrations
    "create_graph_manager",
    "GraphManager",
    "create_annealer",
    "QuantumAnnealer",
    "create_strategist",
    "MetaStrategist",

    # CLI entry point is not exposed in the public API

    # Configuration
    "load_config",
    "get_default_config",
    "load_logging_config",
    "get_default_logging_config",
    "configure_logging",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_LOGGING_PATH"
]
