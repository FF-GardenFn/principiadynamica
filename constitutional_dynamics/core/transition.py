"""
Transition Analysis - Core component for analyzing state transitions

This module provides functions for analyzing transitions between states in the
alignment vector space, including trajectory prediction and STC wrappers.
"""

import logging
import math
from typing import Dict, List, Any, Optional

try:
    import numpy as np
    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False
    logging.warning("NumPy not available. Using fallback implementations.")

# Import from within the package
from .space import AlignmentVectorSpace

logger = logging.getLogger("constitutional_dynamics.core.transition")


def analyze_transition(space: AlignmentVectorSpace, state1_idx: int, state2_idx: int) -> Dict[str, Any]:
    """
    Analyze the transition between two states.

    Args:
        space: The AlignmentVectorSpace containing the states
        state1_idx: Index of the first state
        state2_idx: Index of the second state

    Returns:
        Dictionary with transition analysis
    """
    return space.analyze_transition(state1_idx, state2_idx)


def predict_trajectory(space: AlignmentVectorSpace, start_state_idx: int, steps: int = 5) -> List[Dict[str, Any]]:
    """
    Predict future trajectory based on recent transitions.

    Args:
        space: The AlignmentVectorSpace containing the states
        start_state_idx: Index of starting state
        steps: Number of prediction steps

    Returns:
        List of predicted future states and their metrics
    """
    if start_state_idx < 0 or start_state_idx >= len(space.state_history):
        raise ValueError("Invalid starting state index")

    # Need at least 2 states to predict trajectory
    if len(space.state_history) < 2:
        return [{"error": "Not enough states to predict trajectory"}]

    # Get the most recent transition to use as basis
    recent_transitions = []
    for i in range(max(0, len(space.state_history) - 5), len(space.state_history) - 1):
        transition = space.analyze_transition(i, i + 1)
        recent_transitions.append(transition)

    if not recent_transitions:
        return [{"error": "No transitions available to base prediction on"}]

    # Use average recent transition for prediction
    if USE_NUMPY:
        # Get average transition vector
        transition_vectors = []
        for t in recent_transitions:
            state1 = space.state_history[t["state1_idx"]]
            state2 = space.state_history[t["state2_idx"]]
            vector = np.array(state2) - np.array(state1)
            # Scale by time difference to normalize
            if t["time_delta"] > 0:
                vector = vector / t["time_delta"]
            transition_vectors.append(vector)

        avg_transition = np.mean(np.array(transition_vectors), axis=0)

        # Start with the current state
        current_state = np.array(space.state_history[start_state_idx])

        # Predict future states
        predictions = []
        for step in range(steps):
            # Apply transition
            next_state = current_state + avg_transition

            # Normalize the state (optional)
            norm = np.linalg.norm(next_state)
            if norm > 0:
                next_state = next_state / norm

            # Calculate alignment
            alignment = space.compute_alignment_score(next_state.tolist())

            predictions.append({
                "step": step + 1,
                "predicted_state": next_state.tolist(),
                "predicted_alignment": alignment,
            })

            # Update for next step
            current_state = next_state

    else:
        # Pure Python implementation
        # Get average transition vector
        avg_transition = [0.0] * space.dimension
        for t in recent_transitions:
            state1 = space.state_history[t["state1_idx"]]
            state2 = space.state_history[t["state2_idx"]]
            vector = [s2 - s1 for s1, s2 in zip(state1, state2)]

            # Scale by time difference to normalize
            if t["time_delta"] > 0:
                vector = [v / t["time_delta"] for v in vector]

            # Add to running sum
            avg_transition = [a + v for a, v in zip(avg_transition, vector)]

        # Divide by number of transitions
        avg_transition = [v / len(recent_transitions) for v in avg_transition]

        # Start with the current state
        current_state = space.state_history[start_state_idx].copy()

        # Predict future states
        predictions = []
        for step in range(steps):
            # Apply transition
            next_state = [c + t for c, t in zip(current_state, avg_transition)]

            # Normalize the state (optional)
            norm = math.sqrt(sum(s * s for s in next_state))
            if norm > 0:
                next_state = [s / norm for s in next_state]

            # Calculate alignment
            alignment = space.compute_alignment_score(next_state)

            predictions.append({
                "step": step + 1,
                "predicted_state": next_state,
                "predicted_alignment": alignment,
            })

            # Update for next step
            current_state = next_state

    return predictions


# STC (State-Transition Calculus) wrappers
def compute_activation(state_value: float, time_delta: float, memory_decay: float = 0.2) -> float:
    """
    Compute the activation function φ(a_i, t) from State-Transition Calculus.
    This is a simplified version representing memory decay.

    Args:
        state_value: The value of the state (e.g., alignment score) at its observation time.
        time_delta: Time elapsed since the state was observed (current_time - observation_time).
        memory_decay: Memory decay rate (τ), characteristic time for decay.

    Returns:
        Activation value, representing the current influence of the past state.
    """
    # φ(a_i, t) = a_i * e^(-t/τ) simplified version for now also working on it
    # Since a full activation will most likely also depend on the lyapunov_exponent_estimate in ./space.py
    # Represents the decayed value/influence of state_value after time_delta.
    if memory_decay <= 0:
        # Avoid division by zero or math domain error with exp if time_delta is also non-positive
        # If no decay, activation is just the state_value (if time_delta is zero) or could be an error.
        # For simplicity, if decay rate is invalid, assume full decay or no decay based on time_delta.
        return state_value if time_delta == 0 else 0.0

    return state_value * math.exp(-time_delta / memory_decay)


def compute_residual_potentiality(state: List[float], perturbation_magnitude: float = 0.1) -> List[float]:
    """
    Compute the residual potentiality b(a_res) from State-Transition Calculus.

    Args:
        state: The state vector
        perturbation_magnitude: Magnitude of perturbation to apply

    Returns:
        Perturbed state representing residual potentiality
    """
    if USE_NUMPY:
        # Add random perturbation
        perturbation = np.random.normal(0, perturbation_magnitude, len(state))
        perturbed_state = np.array(state) + perturbation
        # Normalize
        norm = np.linalg.norm(perturbed_state)
        if norm > 0:
            perturbed_state = perturbed_state / norm
        return perturbed_state.tolist()
    else:
        # Pure Python implementation
        import random
        perturbed_state = []
        for s in state:
            # Add random perturbation
            perturbation = random.normalvariate(0, perturbation_magnitude)
            perturbed_state.append(s + perturbation)
        # Normalize
        norm = math.sqrt(sum(s * s for s in perturbed_state))
        if norm > 0:
            perturbed_state = [s / norm for s in perturbed_state]
        return perturbed_state
