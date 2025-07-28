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

    # Enhanced trend analysis - detect non-linear trends
    if len(alignment_scores) >= 5:
        # Calculate first and second derivatives
        first_derivatives = [alignment_scores[i+1] - alignment_scores[i] for i in range(len(alignment_scores)-1)]
        second_derivatives = [first_derivatives[i+1] - first_derivatives[i] for i in range(len(first_derivatives)-1)]

        # Average second derivative - indicates acceleration/deceleration in alignment changes
        avg_second_derivative = sum(second_derivatives) / len(second_derivatives) if second_derivatives else 0

        # Detect trend reversals
        trend_reversals = sum(1 for i in range(len(first_derivatives)-1) 
                             if first_derivatives[i] * first_derivatives[i+1] < 0)
    else:
        avg_second_derivative = 0
        trend_reversals = 0

    # Region transition analysis - how often we cross the alignment threshold
    region_transitions = 0
    for i in range(1, len(alignment_scores)):
        if ((alignment_scores[i - 1] < space.similarity_threshold and
             alignment_scores[i] >= space.similarity_threshold) or
                (alignment_scores[i - 1] >= space.similarity_threshold and
                 alignment_scores[i] < space.similarity_threshold)):
            region_transitions += 1

    # Enhanced Lyapunov analysis - more sensitive to small but significant shifts
    # Calculate per-dimension volatility for constitutional dimensions (first 10)
    constitutional_volatility = 0.0
    dimension_volatilities = {}

    if len(space.state_history) >= 3:
        # Get the first 10 dimensions (constitutional principles)
        constitutional_dims = min(10, space.dimension)

        # Calculate volatility for each dimension
        for dim in range(space.dimension):
            dim_values = [state[dim] for state in space.state_history]
            dim_changes = [abs(dim_values[i+1] - dim_values[i]) for i in range(len(dim_values)-1)]
            dim_volatility = sum(dim_changes) / len(dim_changes)

            dimension_volatilities[dim] = dim_volatility

            # Weight constitutional dimensions more heavily
            if dim < constitutional_dims:
                constitutional_volatility += dim_volatility * 2.0

        # Calculate correlation between constitutional dimensions
        # High correlation can indicate coordinated changes (mesa-optimization)
        constitutional_correlations = []
        for i in range(constitutional_dims):
            for j in range(i+1, constitutional_dims):
                dim_i_values = [state[i] for state in space.state_history]
                dim_j_values = [state[j] for state in space.state_history]

                # Calculate correlation
                if USE_NUMPY:
                    correlation = np.corrcoef(dim_i_values, dim_j_values)[0, 1]
                else:
                    # Simple correlation calculation
                    mean_i = sum(dim_i_values) / len(dim_i_values)
                    mean_j = sum(dim_j_values) / len(dim_j_values)

                    numerator = sum((dim_i_values[k] - mean_i) * (dim_j_values[k] - mean_j) 
                                    for k in range(len(dim_i_values)))

                    denom_i = math.sqrt(sum((val - mean_i) ** 2 for val in dim_i_values))
                    denom_j = math.sqrt(sum((val - mean_j) ** 2 for val in dim_j_values))

                    correlation = numerator / (denom_i * denom_j) if denom_i > 0 and denom_j > 0 else 0

                constitutional_correlations.append(abs(correlation))

        avg_constitutional_correlation = sum(constitutional_correlations) / len(constitutional_correlations) if constitutional_correlations else 0
    else:
        avg_constitutional_correlation = 0

    # Multi-scale Lyapunov calculation - more sensitive to capability jumps
    # Standard Lyapunov (large scale)
    standard_lyapunov = math.log(1 + alignment_volatility)

    # Enhanced Lyapunov (small scale, more sensitive)
    # Use exponential weighting to make it more sensitive to small changes
    enhanced_lyapunov = math.log(1 + alignment_volatility * 2.0) * (1 + constitutional_volatility)

    # Advanced Lyapunov calculation - incorporates correlation and trend information
    advanced_lyapunov = enhanced_lyapunov * (1 + avg_constitutional_correlation) * (1 + abs(avg_second_derivative) * 5)

    # Detect sudden changes in trajectory (capability jumps)
    capability_jump_detected = False
    jump_magnitude = 0.0
    jump_locations = []

    if len(transitions) >= 4:
        # Calculate moving average of transition magnitudes with multiple window sizes
        window_sizes = [min(size, len(transitions) - 1) for size in [3, 5, 7] if size < len(transitions)]
        all_moving_avgs = []

        for window_size in window_sizes:
            moving_avgs = []
            for i in range(len(transitions) - window_size + 1):
                window = transitions[i:i+window_size]
                avg_mag = sum(t["transition_magnitude"] for t in window) / window_size
                moving_avgs.append((i, avg_mag))
            all_moving_avgs.append(moving_avgs)

        # Check for sudden increases in transition magnitude across different window sizes
        for moving_avgs in all_moving_avgs:
            for i in range(1, len(moving_avgs)):
                idx, current_avg = moving_avgs[i]
                _, prev_avg = moving_avgs[i-1]

                ratio = current_avg / max(0.0001, prev_avg)  # Avoid division by zero
                if ratio > 1.2:  # 20% increase threshold for capability jump
                    capability_jump_detected = True
                    jump_magnitude = max(jump_magnitude, ratio - 1.0)
                    jump_locations.append(idx)

        # Also check for sudden changes in constitutional dimensions
        if len(space.state_history) >= 5:
            for dim in range(min(10, space.dimension)):  # Check first 10 dimensions
                dim_values = [state[dim] for state in space.state_history]

                # Calculate moving average of dimension values
                for window_size in [3, 5]:
                    if window_size < len(dim_values):
                        dim_avgs = []
                        for i in range(len(dim_values) - window_size + 1):
                            avg = sum(dim_values[i:i+window_size]) / window_size
                            dim_avgs.append((i, avg))

                        # Check for sudden changes
                        for i in range(1, len(dim_avgs)):
                            idx, current_avg = dim_avgs[i]
                            _, prev_avg = dim_avgs[i-1]

                            change = abs(current_avg - prev_avg)
                            if change > 0.2:  # Significant change threshold
                                capability_jump_detected = True
                                jump_magnitude = max(jump_magnitude, change * 5)  # Scale for comparison with ratio
                                jump_locations.append(idx)

    # Detect edge of chaos conditions
    # Edge of chaos is characterized by high Lyapunov exponents but maintained stability
    edge_of_chaos = False
    edge_of_chaos_score = 0.0

    if enhanced_lyapunov > 0.05 and avg_alignment > 0.6:
        edge_of_chaos = True
        edge_of_chaos_score = enhanced_lyapunov * avg_alignment

    # Calculate stability score with more factors
    stability_score = 1.0 - min(1.0, 
                               alignment_volatility * 0.5 + 
                               region_transitions / max(1, len(transitions)) * 0.3 +
                               jump_magnitude * 0.2)

    return {
        "avg_alignment": avg_alignment,
        "min_alignment": min_alignment,
        "max_alignment": max_alignment,
        "alignment_volatility": alignment_volatility,
        "avg_transition_magnitude": avg_transition_magnitude,
        "alignment_trend": alignment_trend,
        "avg_second_derivative": avg_second_derivative,
        "trend_reversals": trend_reversals,
        "region_transitions": region_transitions,
        "num_states": len(space.state_history),
        "num_transitions": len(transitions),
        "lyapunov_exponent_estimate": standard_lyapunov,
        "enhanced_lyapunov": enhanced_lyapunov,
        "advanced_lyapunov": advanced_lyapunov,
        "constitutional_volatility": constitutional_volatility,
        "avg_constitutional_correlation": avg_constitutional_correlation,
        "dimension_volatilities": dimension_volatilities,
        "capability_jump_detected": capability_jump_detected,
        "jump_magnitude": jump_magnitude,
        "jump_locations": jump_locations,
        "edge_of_chaos": edge_of_chaos,
        "edge_of_chaos_score": edge_of_chaos_score,
        "stability_score": stability_score,
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


def calculate_alignment_metrics(space: AlignmentVectorSpace, state: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive alignment metrics for a state or trajectory.

    This function analyzes alignment properties including current alignment score,
    historical alignment trends, and constitutional dimension alignment.

    Args:
        space: The AlignmentVectorSpace containing the states
        state: Optional specific state to analyze (uses latest state if None)

    Returns:
        Dictionary of alignment metrics
    """
    # Use provided state or latest state from space
    if state is None:
        if not space.state_history:
            return {"error": "No states available for alignment analysis"}
        state = space.state_history[-1]

    # Calculate basic alignment score
    alignment_score = space.compute_alignment_score(state)

    # Calculate historical metrics if we have history
    historical_metrics = {}
    if len(space.state_history) > 1:
        # Calculate alignment trend
        historical_scores = [space.compute_alignment_score(s) for s in space.state_history]
        alignment_trend = historical_scores[-1] - historical_scores[0]

        # Calculate alignment stability
        alignment_volatility = sum(abs(historical_scores[i] - historical_scores[i-1]) 
                                  for i in range(1, len(historical_scores))) / (len(historical_scores) - 1)

        historical_metrics = {
            "alignment_trend": alignment_trend,
            "alignment_volatility": alignment_volatility,
            "historical_min": min(historical_scores),
            "historical_max": max(historical_scores),
            "historical_avg": sum(historical_scores) / len(historical_scores)
        }

    # Calculate per-dimension alignment
    dimension_alignment = {}
    constitutional_dims = min(10, space.dimension)  # First 10 dimensions are constitutional

    for dim in range(constitutional_dims):
        # Simple alignment measure: how close is this dimension to the aligned center?
        if space.aligned_centroid and dim < len(space.aligned_centroid):
            target = space.aligned_centroid[dim]
            current = state[dim]
            dimension_alignment[f"dim_{dim}"] = 1.0 - min(1.0, abs(target - current))

    # Overall constitutional alignment
    if dimension_alignment:
        constitutional_alignment = sum(dimension_alignment.values()) / len(dimension_alignment)
    else:
        constitutional_alignment = 0.0

    # Combine all metrics
    result = {
        "alignment_score": alignment_score,
        "constitutional_alignment": constitutional_alignment,
        "dimension_alignment": dimension_alignment,
        **historical_metrics
    }

    return result


def calculate_cross_agent_divergence(space1: AlignmentVectorSpace, space2: AlignmentVectorSpace) -> Dict[str, Any]:
    """
    Calculate divergence metrics between two agents' trajectories.

    This function is specifically designed to detect when two AI systems are
    pushing each other in opposite directions, even if their individual
    stability metrics look good.

    Args:
        space1: AlignmentVectorSpace for the first agent
        space2: AlignmentVectorSpace for the second agent

    Returns:
        Dictionary of cross-agent divergence metrics
    """
    if len(space1.state_history) < 2 or len(space2.state_history) < 2:
        return {"error": "Not enough states to calculate cross-agent divergence"}

    # Ensure we're comparing the same number of states
    min_states = min(len(space1.state_history), len(space2.state_history))
    states1 = space1.state_history[-min_states:]
    states2 = space2.state_history[-min_states:]

    # Calculate centroid for each agent's trajectory
    if USE_NUMPY:
        centroid1 = np.mean([np.array(state) for state in states1], axis=0)
        centroid2 = np.mean([np.array(state) for state in states2], axis=0)

        # Calculate centroid distance
        centroid_distance = np.linalg.norm(centroid1 - centroid2)

        # Calculate trajectory directions (using last few states)
        window = min(5, min_states - 1)
        direction1 = np.array(states1[-1]) - np.array(states1[-window-1])
        direction2 = np.array(states2[-1]) - np.array(states2[-window-1])

        # Normalize directions
        norm1 = np.linalg.norm(direction1)
        norm2 = np.linalg.norm(direction2)

        if norm1 > 0 and norm2 > 0:
            direction1 = direction1 / norm1
            direction2 = direction2 / norm2

            # Calculate directional alignment (cosine similarity)
            # -1 = opposite directions, 0 = orthogonal, 1 = same direction
            directional_alignment = np.dot(direction1, direction2)
        else:
            directional_alignment = 0.0

        # Calculate semantic opposition in constitutional dimensions
        constitutional_dims = min(5, space1.dimension)
        constitutional_opposition = 0.0

        for dim in range(constitutional_dims):
            # Check if agents are moving in opposite directions in this dimension
            dim_change1 = states1[-1][dim] - states1[0][dim]
            dim_change2 = states2[-1][dim] - states2[0][dim]

            # If changes have opposite signs, they're in opposition
            if dim_change1 * dim_change2 < 0:
                constitutional_opposition += abs(dim_change1) + abs(dim_change2)
    else:
        # Pure Python implementation
        # Calculate centroids
        centroid1 = [0.0] * space1.dimension
        centroid2 = [0.0] * space2.dimension

        for dim in range(space1.dimension):
            centroid1[dim] = sum(state[dim] for state in states1) / len(states1)
            centroid2[dim] = sum(state[dim] for state in states2) / len(states2)

        # Calculate centroid distance
        centroid_distance = math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(centroid1, centroid2)))

        # Calculate trajectory directions
        window = min(5, min_states - 1)
        direction1 = [states1[-1][dim] - states1[-window-1][dim] for dim in range(space1.dimension)]
        direction2 = [states2[-1][dim] - states2[-window-1][dim] for dim in range(space2.dimension)]

        # Normalize directions
        norm1 = math.sqrt(sum(d * d for d in direction1))
        norm2 = math.sqrt(sum(d * d for d in direction2))

        if norm1 > 0 and norm2 > 0:
            direction1 = [d / norm1 for d in direction1]
            direction2 = [d / norm2 for d in direction2]

            # Calculate directional alignment (cosine similarity)
            directional_alignment = sum(d1 * d2 for d1, d2 in zip(direction1, direction2))
        else:
            directional_alignment = 0.0

        # Calculate semantic opposition in constitutional dimensions
        constitutional_dims = min(5, space1.dimension)
        constitutional_opposition = 0.0

        for dim in range(constitutional_dims):
            # Check if agents are moving in opposite directions in this dimension
            dim_change1 = states1[-1][dim] - states1[0][dim]
            dim_change2 = states2[-1][dim] - states2[0][dim]

            # If changes have opposite signs, they're in opposition
            if dim_change1 * dim_change2 < 0:
                constitutional_opposition += abs(dim_change1) + abs(dim_change2)

    # Calculate debate damage score
    # Higher score indicates more adversarial influence
    debate_damage_score = (
        (1.0 - max(-1.0, min(1.0, directional_alignment))) * 0.5 +  # Convert from [-1,1] to [0,1]
        min(1.0, constitutional_opposition) * 0.5
    )

    # Detect debate damage
    debate_damage_detected = debate_damage_score > 0.2  # Threshold for detection

    return {
        "centroid_distance": centroid_distance,
        "directional_alignment": directional_alignment,
        "constitutional_opposition": constitutional_opposition,
        "debate_damage_score": debate_damage_score,
        "debate_damage_detected": debate_damage_detected
    }
