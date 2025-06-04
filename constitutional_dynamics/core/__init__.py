"""
Constitutional Dynamics Core Module

This module provides the core functionality for the Constitutional Dynamics package,
including the AlignmentVectorSpace class, transition analysis, metrics
calculation, and optimization.
"""

from .space import AlignmentVectorSpace
from .transition import (
    analyze_transition, 
    predict_trajectory, 
    compute_activation_probability, 
    compute_residual_potentiality,
    compute_activation
)
from .metrics import (
    calculate_stability_metrics,
    evaluate_alignment_robustness
)
from .optimise import AlignmentOptimizer, GraphEnhancedAlignmentOptimizer

# Note: compute_alignment_score is a method of AlignmentVectorSpace, not a standalone function

__all__ = [
    "AlignmentVectorSpace",
    "analyze_transition",
    "predict_trajectory",
    "compute_activation_probability",
    "compute_residual_potentiality",
    "compute_activation",
    "calculate_stability_metrics",
    "evaluate_alignment_robustness",
    "AlignmentOptimizer",
    "GraphEnhancedAlignmentOptimizer"
]
