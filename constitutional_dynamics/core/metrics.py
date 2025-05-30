"""
Alignment Metrics - Core component for measuring alignment properties

This module provides functions for calculating stability metrics and evaluating
the robustness of alignment to perturbations.
"""

import math
import logging
from typing import Dict, List, Any, Optional

try:
    import numpy as np
    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False
    logging.warning("NumPy not available. Using fallback implementations.")

# Import from within the package
from .space import AlignmentVectorSpace

logger = logging.getLogger("constitutional_dynamics.core.metrics")


def calculate_stability_metrics(space: AlignmentVectorSpace) -> Dict[str, Any]:
    """
    Calculate stability metrics for the system trajectory.

    Args:
        space: The AlignmentVectorSpace containing the states

    Returns:
        Dictionary of stability metrics
    """
    if len(space.state_history) < 2:
        return {"error": "Not enough states to calculate stability"}

    # Analyze all transitions
    transitions = []
    for i in range(len(space.state_history) - 1):
        transition = space.analyze_transition(i, i + 1)
        transitions.append(transition)

    # Calculate metrics
    alignment_scores = [space.compute_alignment_score(state) for state in space.state_history]

    # Overall alignment metrics
    avg_alignment = sum(alignment_scores) / len(alignment_scores)
    min_alignment = min(alignment_scores)
    max_alignment = max(alignment_scores)

    # Stability metrics
    alignment_volatility = sum(abs(t["alignment_change"]) for t in transitions) / len(transitions)
    avg_transition_magnitude = sum(t["transition_magnitude"] for t in transitions) / len(transitions)

    # Trend analysis
    alignment_trend = alignment_scores[-1] - alignment_scores[0]

    # Region transition analysis - how often we cross the alignment threshold
    region_transitions = 0
    for i in range(1, len(alignment_scores)):
        if ((alignment_scores[i - 1] < space.similarity_threshold and
             alignment_scores[i] >= space.similarity_threshold) or
                (alignment_scores[i - 1] >= space.similarity_threshold and
                 alignment_scores[i] < space.similarity_threshold)):
            region_transitions += 1

    return {
        "avg_alignment": avg_alignment,
        "min_alignment": min_alignment,
        "max_alignment": max_alignment,
        "alignment_volatility": alignment_volatility,
        "avg_transition_magnitude": avg_transition_magnitude,
        "alignment_trend": alignment_trend,
        "region_transitions": region_transitions,
        "num_states": len(space.state_history),
        "num_transitions": len(transitions),
        "lyapunov_exponent_estimate": math.log(1 + alignment_volatility),  # Rough estimate of chaotic behavior for now -working on it
        "stability_score": 1.0 - min(1.0, alignment_volatility + region_transitions / max(1, len(transitions))),
    }


def evaluate_alignment_robustness(space: AlignmentVectorSpace,
                                  perturbation_magnitude: float = 0.1,
                                  num_perturbations: int = 10) -> Dict[str, Any]:
    """
    Evaluate robustness of alignment to perturbations.

    Args:
        space: The AlignmentVectorSpace containing the states
        perturbation_magnitude: Size of random perturbations
        num_perturbations: Number of random perturbations to test

    Returns:
        Dictionary of robustness metrics
    """
    if not space.state_history:
        return {"error": "No states available for robustness analysis"}

    # Take the latest state
    state = space.state_history[-1]
    base_alignment = space.compute_alignment_score(state)

    perturbations = []

    if USE_NUMPY:
        state_np = np.array(state)
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility

        for _ in range(num_perturbations):
            # Generate random perturbation
            perturbation = rng.normal(0, perturbation_magnitude, space.dimension)

            # Apply perturbation
            perturbed_state = state_np + perturbation

            # Normalize
            norm = np.linalg.norm(perturbed_state)
            if norm > 0:
                perturbed_state = perturbed_state / norm

            # Measure alignment
            perturbed_alignment = space.compute_alignment_score(perturbed_state.tolist())

            perturbations.append({
                "alignment_change": perturbed_alignment - base_alignment,
                "perturbed_alignment": perturbed_alignment,
            })

    else:
        # Pure Python implementation
        import random
        random.seed(42)

        for _ in range(num_perturbations):
            # Generate random perturbation
            perturbation = [random.gauss(0, perturbation_magnitude) for _ in range(space.dimension)]

            # Apply perturbation
            perturbed_state = [s + p for s, p in zip(state, perturbation)]

            # Normalize
            norm = math.sqrt(sum(s * s for s in perturbed_state))
            if norm > 0:
                perturbed_state = [s / norm for s in perturbed_state]

            # Measure alignment
            perturbed_alignment = space.compute_alignment_score(perturbed_state)

            perturbations.append({
                "alignment_change": perturbed_alignment - base_alignment,
                "perturbed_alignment": perturbed_alignment,
            })

    # Calculate robustness metrics
    alignment_changes = [p["alignment_change"] for p in perturbations]
    avg_change = sum(alignment_changes) / len(alignment_changes)
    max_negative_change = min(0, min(alignment_changes))
    max_positive_change = max(0, max(alignment_changes))

    # Calculate variance of changes
    if USE_NUMPY:
        variance = np.var(alignment_changes)
    else:
        mean = avg_change
        variance = sum((x - mean) ** 2 for x in alignment_changes) / len(alignment_changes)

    # Calculate PSD (Power Spectral Density) distance
    # This is a simplified version for now a complete version might use FFT
    psd_distance = calculate_psd_distance(space)

    return {
        "base_alignment": base_alignment,
        "avg_change": avg_change,
        "max_negative_change": max_negative_change,
        "max_positive_change": max_positive_change,
        "change_variance": variance,
        "robustness_score": 1.0 - min(1.0, abs(avg_change) + math.sqrt(variance)),
        "num_perturbations": num_perturbations,
        "perturbation_magnitude": perturbation_magnitude,
        "psd_distance": psd_distance,
        "perturbations": perturbations,
    }


def calculate_psd_distance(space: AlignmentVectorSpace) -> float:
    """
    Calculate the Power Spectral Density (PSD) distance between the current
    state trajectory and an aligned trajectory.

    Args:
        space: The AlignmentVectorSpace containing the states

    Returns:
        PSD distance (0.0 to 1.0)
    """
    if len(space.state_history) < 2:
        return 0.0

    # Extract alignment scores as time series
    alignment_scores = [space.compute_alignment_score(state) for state in space.state_history]

    # Generate an "ideal" aligned trajectory for comparison
    # (constant high alignment)
    ideal_trajectory = [1.0] * len(alignment_scores)

    # Calculate PSD distance
    if USE_NUMPY:
        try:
            from scipy import signal
            # Calculate PSDs
            f, psd_actual = signal.welch(alignment_scores, nperseg=min(len(alignment_scores), 8))
            f, psd_ideal = signal.welch(ideal_trajectory, nperseg=min(len(alignment_scores), 8))

            # Calculate distance (normalized)
            distance = np.sqrt(np.sum((psd_actual - psd_ideal) ** 2)) / np.sqrt(np.sum(psd_ideal ** 2))
            return min(1.0, distance)
        except ImportError:
            # Fallback if scipy not available
            return simple_psd_distance(alignment_scores, ideal_trajectory)
    else:
        return simple_psd_distance(alignment_scores, ideal_trajectory)


def simple_psd_distance(series1: List[float], series2: List[float]) -> float:
    """
    Calculate a simple approximation of PSD distance without FFT.

    Args:
        series1: First time series
        series2: Second time series

    Returns:
        Approximate PSD distance
    """
    # Calculate variance as a simple proxy for spectral properties
    if len(series1) != len(series2):
        raise ValueError("Series must have the same length")

    # Calculate means
    mean1 = sum(series1) / len(series1)
    mean2 = sum(series2) / len(series2)

    # Calculate variances
    var1 = sum((x - mean1) ** 2 for x in series1) / len(series1)
    var2 = sum((x - mean2) ** 2 for x in series2) / len(series2)

    # Calculate autocorrelation at lag 1 as a simple spectral property
    autocorr1 = sum((series1[i] - mean1) * (series1[i-1] - mean1) for i in range(1, len(series1))) / (var1 * (len(series1) - 1))
    autocorr2 = sum((series2[i] - mean2) * (series2[i-1] - mean2) for i in range(1, len(series2))) / (var2 * (len(series2) - 1))

    # Combine differences in variance and autocorrelation
    distance = math.sqrt((var1 - var2) ** 2 + (autocorr1 - autocorr2) ** 2)

    # Normalize to 0-1 range
    return min(1.0, distance)
