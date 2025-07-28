"""
AlignmentVectorSpace - Core component for alignment vector dynamics

This module implements the AlignmentVectorSpace class, which models alignment
as a vector space where states, regions, and trajectories represent different
aspects of alignment behavior.
"""

import json
import math
import time
from typing import Dict, List, Tuple, Any, Set, Optional, Union
import copy
import logging

try:
    import numpy as np
    from numpy.typing import NDArray
    from scipy.spatial import ConvexHull

    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False
    logging.warning("NumPy/SciPy not available. Using fallback implementations.")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rich_print

    USE_RICH = True
    console = Console()
except ImportError:
    USE_RICH = False
    console = None


class AlignmentVectorSpace:
    """
    Models alignment as a vector space where:
    - Vectors represent behavior/response states
    - Regions in the space represent aligned vs. misaligned behaviors
    - Trajectories through the space represent behavioral evolution
    """

    def __init__(self,
                 dimension: int = 1024,
                 memory_decay: float = 0.2,
                 similarity_threshold: float = 0.7):
        """
        Initialize the alignment vector space.

        Args:
            dimension: Dimensionality of the embedding space
            memory_decay: Rate at which memory of past states decays
            similarity_threshold: Threshold for considering states similar
        """
        self.dimension = dimension
        self.memory_decay = memory_decay
        self.similarity_threshold = similarity_threshold

        # Alignment regions (populated by load_aligned_examples)
        self.aligned_regions = []  # List of vectors defining "aligned" behavior
        self.aligned_centroid = None  # Center of the aligned region
        self.aligned_boundary = None  # Boundary vectors of aligned region

        # Memory of past states
        self.state_history = []  # List of past states
        self.state_timestamps = []  # Timestamps of state observations

        # Cache for transition data
        self.transition_cache = {}  # Maps (state1, state2) -> transition info

        self.logger = logging.getLogger("constitutional_dynamics.core.space")

    def load_aligned_examples(self, examples_path: str) -> bool:
        """
        Load examples of aligned behavior to define the "aligned region".

        Args:
            examples_path: Path to JSON file with aligned examples

        Returns:
            Success status
        """
        try:
            with open(examples_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different formats
            if isinstance(data, dict) and "aligned_examples" in data:
                examples = data["aligned_examples"]
            elif isinstance(data, list):
                examples = data
            else:
                examples = list(data.values())  # Assume dict of embeddings

            # Filter and validate examples
            valid_examples = []
            for example in examples:
                # Handle case where example is a dict with metadata
                if isinstance(example, dict) and "embedding" in example:
                    embedding = example["embedding"]
                else:
                    embedding = example

                # Validate embedding
                if isinstance(embedding, list) and len(embedding) > 0:
                    # Pad or truncate to match dimension
                    if len(embedding) < self.dimension:
                        embedding = embedding + [0.0] * (self.dimension - len(embedding))
                    elif len(embedding) > self.dimension:
                        embedding = embedding[:self.dimension]

                    valid_examples.append(embedding)

            if len(valid_examples) == 0:
                self.logger.warning("No valid aligned examples found")
                return False

            self.aligned_regions = valid_examples

            # Compute aligned region centroid
            if USE_NUMPY:
                self.aligned_centroid = np.mean(np.array(valid_examples), axis=0)

                # If we have enough examples and scipy, compute convex hull
                if len(valid_examples) >= self.dimension + 1:
                    try:
                        hull = ConvexHull(np.array(valid_examples))
                        self.aligned_boundary = [valid_examples[i] for i in hull.vertices]
                    except Exception as e:
                        self.logger.warning(f"Could not compute convex hull: {e}")
                        # Fallback: use examples as boundary
                        self.aligned_boundary = valid_examples
                else:
                    self.aligned_boundary = valid_examples
            else:
                # Simple centroid calculation without numpy
                centroid = [0.0] * self.dimension
                for example in valid_examples:
                    for i in range(self.dimension):
                        centroid[i] += example[i] / len(valid_examples)
                self.aligned_centroid = centroid
                self.aligned_boundary = valid_examples

            self.logger.info(f"Loaded {len(valid_examples)} aligned examples")
            return True

        except Exception as e:
            self.logger.error(f"Error loading aligned examples: {e}")
            return False

    def define_alignment_region(self, center: List[float], radius: float = 0.3) -> bool:
        """
        Manually define an alignment region as a hypersphere.

        Args:
            center: Center vector of the aligned region
            radius: Radius of the aligned region hypersphere

        Returns:
            Success status
        """
        if len(center) != self.dimension:
            self.logger.error(f"Center vector dimension {len(center)} does not match space dimension {self.dimension}")
            return False

        self.aligned_centroid = center

        # Generate boundary points around the center
        if USE_NUMPY:
            # Generate random unit vectors
            boundary_points = []
            for _ in range(min(10, self.dimension)):
                # Random unit vector
                point = np.random.randn(self.dimension)
                point = point / np.linalg.norm(point)
                # Scale by radius and add to center
                point = center + radius * point
                boundary_points.append(point.tolist())
            self.aligned_boundary = boundary_points
        else:
            # Simple boundary approximation without numpy
            boundary_points = []
            for i in range(min(10, self.dimension)):
                point = [0.0] * self.dimension
                point[i % self.dimension] = radius
                for j in range(self.dimension):
                    point[j] += center[j]
                boundary_points.append(point)
            self.aligned_boundary = boundary_points

        self.aligned_regions = [center] + boundary_points
        self.logger.info(f"Defined alignment region with center and {len(boundary_points)} boundary points")
        return True

    def add_state(self, state: List[float], timestamp: Optional[float] = None) -> int:
        """
        Add a state vector to the history.

        Args:
            state: Vector representing the state
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            Index of the added state
        """
        if len(state) != self.dimension:
            # Pad or truncate to match dimension
            if len(state) < self.dimension:
                state = state + [0.0] * (self.dimension - len(state))
            else:
                state = state[:self.dimension]

        if timestamp is None:
            timestamp = time.time()

        self.state_history.append(state)
        self.state_timestamps.append(timestamp)

        # Clear cache entries involving this state
        self.transition_cache = {}

        return len(self.state_history) - 1

    def compute_alignment_score(self, state: List[float], 
                               use_enhanced_scoring: bool = True,
                               constitutional_weights: Optional[Dict[int, float]] = None) -> float:
        """
        Compute how aligned a state is with the defined alignment region.

        Enhanced scoring considers multiple factors beyond simple cosine similarity:
        1. Weighted constitutional dimensions
        2. Distance from aligned region boundary
        3. Consistency across dimensions

        Args:
            state: Vector representing the state
            use_enhanced_scoring: Whether to use enhanced scoring algorithm
            constitutional_weights: Optional weights for constitutional dimensions

        Returns:
            Alignment score (0.0 to 1.0)
        """
        if not self.aligned_centroid:
            self.logger.warning("No alignment region defined")
            return 0.5  # Neutral score if no region defined

        # Ensure state has correct dimension
        if len(state) != self.dimension:
            if len(state) < self.dimension:
                state = state + [0.0] * (self.dimension - len(state))
            else:
                state = state[:self.dimension]

        # Basic alignment: Compute similarity to aligned centroid
        similarity = self.compute_similarity(state, self.aligned_centroid)

        # Scale to 0-1 range (similarity is -1 to 1 for cosine)
        basic_alignment_score = (similarity + 1) / 2

        if not use_enhanced_scoring:
            return basic_alignment_score

        # Enhanced alignment scoring

        # 1. Weighted constitutional dimensions (first 10 dimensions)
        constitutional_dims = min(10, self.dimension)

        # Default weights prioritize key constitutional principles
        if constitutional_weights is None:
            constitutional_weights = {
                0: 1.5,  # Helpfulness
                1: 1.5,  # Harmlessness
                2: 1.5,  # Honesty
                3: 1.2,  # Non-deception
                4: 1.2,  # Transparency
                5: 1.0,  # Fairness
                6: 1.0,  # Privacy
                7: 1.0,  # Autonomy
                8: 1.0,  # Reliability
                9: 1.0   # Robustness
            }

        # Calculate weighted constitutional alignment
        constitutional_scores = []
        for dim in range(constitutional_dims):
            if dim < len(state) and dim < len(self.aligned_centroid):
                # How close is this dimension to the aligned value?
                target = self.aligned_centroid[dim]
                current = state[dim]
                dim_score = 1.0 - min(1.0, abs(target - current))

                # Apply weight
                weight = constitutional_weights.get(dim, 1.0)
                constitutional_scores.append(dim_score * weight)

        # Average weighted constitutional score
        if constitutional_scores:
            weighted_constitutional_score = sum(constitutional_scores) / sum(constitutional_weights.get(i, 1.0) 
                                                                           for i in range(len(constitutional_scores)))
        else:
            weighted_constitutional_score = 0.5

        # 2. Distance from aligned region boundary
        # If we have boundary points, calculate minimum distance to boundary
        boundary_distance_score = 0.5  # Default neutral score
        if self.aligned_boundary:
            # Calculate distances to all boundary points
            distances = []
            for boundary_point in self.aligned_boundary:
                if USE_NUMPY:
                    distance = np.linalg.norm(np.array(state) - np.array(boundary_point))
                else:
                    distance = math.sqrt(sum((s - b) ** 2 for s, b in zip(state, boundary_point)))
                distances.append(distance)

            # Minimum distance to boundary
            min_distance = min(distances) if distances else 0

            # Distance to centroid
            if USE_NUMPY:
                centroid_distance = np.linalg.norm(np.array(state) - np.array(self.aligned_centroid))
            else:
                centroid_distance = math.sqrt(sum((s - c) ** 2 for s, c in zip(state, self.aligned_centroid)))

            # If distance to centroid < min distance to boundary, we're inside the region
            if centroid_distance < min_distance:
                # Inside aligned region - score based on how deep inside
                boundary_distance_score = 0.5 + 0.5 * (1 - centroid_distance / min_distance)
            else:
                # Outside aligned region - score based on how far outside
                boundary_distance_score = 0.5 - 0.5 * min(1.0, (centroid_distance - min_distance) / min_distance)

        # 3. Consistency across dimensions
        # Check if constitutional dimensions are consistently aligned
        consistency_score = 0.5  # Default neutral score
        if constitutional_dims > 1:
            constitutional_values = [state[i] for i in range(constitutional_dims) if i < len(state)]
            if constitutional_values:
                # Calculate variance of constitutional values
                mean_val = sum(constitutional_values) / len(constitutional_values)
                variance = sum((v - mean_val) ** 2 for v in constitutional_values) / len(constitutional_values)

                # Lower variance = higher consistency
                consistency_score = 1.0 - min(1.0, variance * 5)  # Scale variance to 0-1 range

        # Combine scores with appropriate weights
        # Basic alignment: 30%, Constitutional: 40%, Boundary: 20%, Consistency: 10%
        enhanced_score = (
            basic_alignment_score * 0.3 +
            weighted_constitutional_score * 0.4 +
            boundary_distance_score * 0.2 +
            consistency_score * 0.1
        )

        # Ensure score is in 0-1 range
        return max(0.0, min(1.0, enhanced_score))

    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute similarity between two vectors (cosine similarity).

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score (-1.0 to 1.0)
        """
        # Ensure vectors have correct dimension
        if len(vec1) != self.dimension:
            if len(vec1) < self.dimension:
                vec1 = vec1 + [0.0] * (self.dimension - len(vec1))
            else:
                vec1 = vec1[:self.dimension]

        if len(vec2) != self.dimension:
            if len(vec2) < self.dimension:
                vec2 = vec2 + [0.0] * (self.dimension - len(vec2))
            else:
                vec2 = vec2[:self.dimension]

        # Compute cosine similarity
        if USE_NUMPY:
            # NumPy implementation
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(v1, v2) / (norm1 * norm2)
        else:
            # Pure Python implementation
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)

    def analyze_transition(self, state1_idx: int, state2_idx: int) -> Dict[str, Any]:
        """
        Analyze the transition between two states.

        Args:
            state1_idx: Index of the first state
            state2_idx: Index of the second state

        Returns:
            Dictionary with transition analysis
        """
        # Check cache first
        cache_key = (state1_idx, state2_idx)
        if cache_key in self.transition_cache:
            return self.transition_cache[cache_key]

        # Validate indices
        if state1_idx < 0 or state1_idx >= len(self.state_history):
            raise ValueError(f"Invalid state1_idx: {state1_idx}")
        if state2_idx < 0 or state2_idx >= len(self.state_history):
            raise ValueError(f"Invalid state2_idx: {state2_idx}")

        state1 = self.state_history[state1_idx]
        state2 = self.state_history[state2_idx]
        t1 = self.state_timestamps[state1_idx]
        t2 = self.state_timestamps[state2_idx]

        # Compute basic transition metrics
        time_delta = t2 - t1
        similarity = self.compute_similarity(state1, state2)
        alignment1 = self.compute_alignment_score(state1)
        alignment2 = self.compute_alignment_score(state2)
        alignment_change = alignment2 - alignment1

        # Compute vector from state1 to state2
        if USE_NUMPY:
            s1 = np.array(state1)
            s2 = np.array(state2)
            transition_vector = s2 - s1
            transition_magnitude = np.linalg.norm(transition_vector)

            # Compute vector from state1 to aligned centroid
            if self.aligned_centroid is not None:
                ac = np.array(self.aligned_centroid)
                to_aligned_vector = ac - s1
                to_aligned_magnitude = np.linalg.norm(to_aligned_vector)

                # Compute angle between transition vector and to-aligned vector
                if transition_magnitude > 0 and to_aligned_magnitude > 0:
                    cos_angle = np.dot(transition_vector, to_aligned_vector) / (transition_magnitude * to_aligned_magnitude)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    toward_aligned_region = cos_angle > 0
                else:
                    angle = 0.0
                    toward_aligned_region = False
            else:
                to_aligned_vector = None
                to_aligned_magnitude = 0.0
                angle = 0.0
                toward_aligned_region = False
        else:
            # Pure Python implementation
            transition_vector = [s2 - s1 for s1, s2 in zip(state1, state2)]
            transition_magnitude = math.sqrt(sum(v * v for v in transition_vector))

            # Compute vector from state1 to aligned centroid
            if self.aligned_centroid is not None:
                to_aligned_vector = [ac - s1 for s1, ac in zip(state1, self.aligned_centroid)]
                to_aligned_magnitude = math.sqrt(sum(v * v for v in to_aligned_vector))

                # Compute angle between transition vector and to-aligned vector
                if transition_magnitude > 0 and to_aligned_magnitude > 0:
                    dot_product = sum(a * b for a, b in zip(transition_vector, to_aligned_vector))
                    cos_angle = dot_product / (transition_magnitude * to_aligned_magnitude)
                    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clip to valid range
                    angle = math.acos(cos_angle)
                    toward_aligned_region = cos_angle > 0
                else:
                    angle = 0.0
                    toward_aligned_region = False
            else:
                to_aligned_vector = None
                to_aligned_magnitude = 0.0
                angle = 0.0
                toward_aligned_region = False

        # Prepare result
        result = {
            "state1_idx": state1_idx,
            "state2_idx": state2_idx,
            "time_delta": time_delta,
            "similarity": similarity,
            "alignment1": alignment1,
            "alignment2": alignment2,
            "alignment_change": alignment_change,
            "transition_magnitude": transition_magnitude,
            "toward_aligned_region": toward_aligned_region,
            "angle_to_aligned": angle,
        }

        # Cache the result
        self.transition_cache[cache_key] = result
        return result
