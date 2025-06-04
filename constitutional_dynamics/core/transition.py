"""
Transition Analysis - Core component for analyzing state transitions

This module provides functions for analyzing transitions between states in the
alignment vector space, including trajectory prediction and STC wrappers.
"""

import logging
import math
from typing import Dict, List, Any, Optional, Union
import random

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


def compute_activation_probability(
    subset_weight: float = 1.0,
    state: Optional[List[float]] = None,
    memory: Optional[Dict[str, Any]] = None,
    environment: Optional[Dict[str, Any]] = None
) -> float:
    """
    Compute the activation probability φ for a state subset based on STC principles.

    This function calculates the probability of a state subset being activated,
    taking into account the subset's inherent weight, memory effects, and
    environmental factors.

    Args:
        subset_weight: Weight of the state subset (W')
        state: Current state vector (optional)
        memory: Memory factors (optional)
        environment: Environmental factors (optional)

    Returns:
        Activation probability between 0 and 1
    """
    # Start with the base weight
    activation_prob = subset_weight

    # Apply memory effects if provided
    if memory is not None:
        memory_factor = memory.get("decay_factor", 0.9)
        time_delta = memory.get("time_delta", 0.0)

        # Apply exponential decay based on time
        if time_delta > 0:
            memory_effect = math.exp(-time_delta * (1.0 - memory_factor))
            activation_prob *= memory_effect

    # Apply environmental factors if provided
    if environment is not None:
        env_factor = environment.get("factor", 1.0)
        activation_prob *= env_factor

    # Ensure the probability stays in [0, 1]
    activation_prob = max(0.0, min(1.0, activation_prob))

    return activation_prob


# Helper for cosine similarity if not using a full space object
def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not USE_NUMPY: # Already defined at module level
        # Pure Python
        dot_product = sum(x * y for x, y in zip(vec1, vec2))
        norm_vec1 = math.sqrt(sum(x * x for x in vec1))
        norm_vec2 = math.sqrt(sum(x * x for x in vec2))
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)
    else:
        # NumPy
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        dot_product = np.dot(vec1_np, vec2_np)
        norm_vec1 = np.linalg.norm(vec1_np)
        norm_vec2 = np.linalg.norm(vec2_np)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

def compute_residual_potentiality(
    state: List[float], 
    perturbation_magnitude: float = 0.1
) -> Dict[str, Any]: # ✨ CHANGED return type
    """
    Compute the residual potentiality b(a_res) from State-Transition Calculus.
    This involves applying a perturbation to the state and assessing its impact.

    Args:
        state: The state vector
        perturbation_magnitude: Magnitude of perturbation to apply

    Returns:
        A dictionary containing:
            - original_state: The input state vector.
            - perturbed_state: The state vector after perturbation and normalization.
            - perturbation_vector: The random noise vector applied.
            - potentiality_score: A measure of how much the perturbation shifted
                                 the state (1 - cosine_similarity). Higher means more shift.
    """
    perturbation_vector: List[float]
    perturbed_state_intermediate: List[float]

    if USE_NUMPY:
        state_np = np.array(state)
        perturbation_vector_np = np.random.normal(0, perturbation_magnitude, len(state))
        perturbed_state_intermediate_np = state_np + perturbation_vector_np

        perturbation_vector = perturbation_vector_np.tolist()
        perturbed_state_intermediate = perturbed_state_intermediate_np.tolist()
    else:
        # Pure Python implementation
         # random.seed(42) # Considering if seeding is needed here for reproducibility...
        perturbation_vector = [random.gauss(0, perturbation_magnitude) for _ in range(len(state))]
        perturbed_state_intermediate = [s + p for s, p in zip(state, perturbation_vector)]

    # Normalize the perturbed state
    norm = math.sqrt(sum(s * s for s in perturbed_state_intermediate))
    final_perturbed_state: List[float]
    if norm > 0:
        final_perturbed_state = [s / norm for s in perturbed_state_intermediate]
    else:
        final_perturbed_state = perturbed_state_intermediate # Or handle as all zeros if that's preferred for a zero vector

    # Calculate potentiality_score (1 - cosine_similarity between original and perturbed)
    # the returned score is higher if the perturbation causes a larger directional change.
    similarity = _cosine_similarity(state, final_perturbed_state)
    potentiality_score = 1.0 - similarity # Ranges 0 (no change) to 2 (opposite direction), typically 0-1 for small changes.
                                          # To keep it 0-1, could use (1 - similarity) / 2 if similarity is -1 to 1.
                                          # Cosine similarity for unit vectors is between -1 and 1.
                                          # If state and final_perturbed_state are unit vectors, then 1-similarity gives good measure.
                                          # Let's assume state is also somewhat normalized or its norm doesn't affect cosine too much.
                                          # For better interpretability, ensure score is 0-1, so perhaps:
    potentiality_score = (1.0 - similarity) / 2.0 # Now 0 (identical) to 1 (orthogonal or opposite)
                                                 # if original similarity was -1 (opposite), score is 1.
                                                 # if original similarity was 0 (orthogonal), score is 0.5.
                                                 # if original similarity was 1 (identical), score is 0.
                                                 # This scaling seems more intuitive for a 0-1 "instability" or "shift" score.


    return {
        "original_state": state,
        "perturbed_state": final_perturbed_state,
        "perturbation_vector": perturbation_vector,
        "potentiality_score": potentiality_score 
        # "specific_indicators": {} # Placeholder for future work and implementation
    }
