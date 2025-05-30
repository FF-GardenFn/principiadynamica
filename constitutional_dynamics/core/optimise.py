"""
Optimization - Core component for multi-objective optimization

This module provides classes for optimizing alignment trajectories using
QUBO formulation and graph-enhanced techniques.
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union

try:
    import numpy as np
    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False
    logging.warning("NumPy not available. Using fallback implementations.")

logger = logging.getLogger("constitutional_dynamics.core.optimise")


class AlignmentOptimizer:
    """
    Optimize alignment state transitions using QUBO formulation.

    This is the base optimizer class that implements the Prize-Collecting
    Traveling Salesman Problem (PCTSP) approach to finding optimal
    alignment trajectories.
    """

    def __init__(self, states: Optional[List[Dict[str, Any]]] = None, config: Optional[Dict[str, Any]] = None, debug: bool = False):
        """
        Initialize the alignment optimizer.

        Args:
            states: List of state dictionaries with embeddings and metadata
            config: Configuration dictionary
            debug: Whether to enable debug logging
        """
        self.states = states or []
        self.config = config or {}
        self.debug = debug

        # Default parameters
        self.flow_constraint_strength = self.config.get("flow_constraint_strength", 5.0)
        self.lambda_weight = self.config.get("lambda_weight", 0.35)

        # Setup logging
        if debug:
            logger.setLevel(logging.DEBUG)

    def generate_costs(self, phi_scores, psd_scores, context_info=None) -> List[Dict[str, Any]]:
        """
        Generate cost values for each state and transition.

        Args:
            phi_scores: Dictionary mapping state IDs to alignment scores
            psd_scores: Dictionary mapping state IDs to PSD distance scores
            context_info: Optional context information

        Returns:
            List of cost dictionaries
        """
        costs = []

        for i, state in enumerate(self.states):
            state_id = state.get("id", i)

            # Get alignment score (phi)
            phi_score = phi_scores.get(state_id, 0.5)

            # Get PSD distance score
            psd_score = psd_scores.get(state_id, 0.0)

            # Calculate time domain penalty (1 - phi_score)
            time_penalty = 1.0 - phi_score

            # Calculate spectral domain penalty (lambda * psd_distance)
            spectral_penalty = self.lambda_weight * psd_score

            # Total cost is weighted sum
            total_cost = time_penalty + spectral_penalty

            # Check if state is critical for alignment
            is_critical = state.get("critical_for_alignment", False)

            costs.append({
                "state_id": state_id,
                "phi_score": phi_score,
                "psd_score": psd_score,
                "time_penalty": time_penalty,
                "spectral_penalty": spectral_penalty,
                "total_cost": total_cost,
                "is_critical": is_critical
            })

        return costs

    def build_qubo(self, phi_scores, psd_scores, context_info=None) -> Dict[Tuple[Any, Any], float]:
        """
        Build QUBO for alignment optimization.

        Args:
            phi_scores: Dictionary mapping state IDs to alignment scores
            psd_scores: Dictionary mapping state IDs to PSD distance scores
            context_info: Optional context information

        Returns:
            QUBO dictionary mapping variable tuples to weights
        """
        # Generate costs
        costs = self.generate_costs(phi_scores, psd_scores, context_info)

        # Initialize QUBO dictionary
        Q = defaultdict(float)

        # Number of states
        n = len(self.states)

        # Add node selection costs
        for i, cost in enumerate(costs):
            var = (i, i)  # Diagonal element for node selection

            # Node cost is the total cost (time + spectral penalties)
            Q[var] += cost["total_cost"]

            # If critical, add strong incentive to include
            if cost["is_critical"]:
                Q[var] -= 10.0  # Strong negative cost (incentive)

        # Add flow constraints
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # Flow constraint: each node has at most one incoming and one outgoing edge
                for k in range(n):
                    if j != k and i != k:
                        # Penalize selecting both (i,j) and (i,k)
                        Q[((i, j), (i, k))] += self.flow_constraint_strength

                        # Penalize selecting both (j,i) and (k,i)
                        Q[((j, i), (k, i))] += self.flow_constraint_strength

        return Q

    def solve_qubo(self, Q, num_reads=1000):
        """
        Solve the QUBO problem using available solver.

        This is a placeholder that should be implemented by subclasses
        or overridden by integration with quantum solvers.

        Args:
            Q: QUBO dictionary
            num_reads: Number of reads for the solver

        Returns:
            Dictionary with solution information
        """
        logger.warning("Base solve_qubo called - should be implemented by subclass or integration")

        # Simple greedy solution as fallback
        solution = {}
        energy = 0.0

        # Sort variables by their coefficients (ascending)
        sorted_vars = sorted(Q.items(), key=lambda x: x[1])

        # Select variables with negative coefficients (beneficial to include)
        for var, coeff in sorted_vars:
            if coeff < 0:
                solution[var] = 1
                energy += coeff
            else:
                solution[var] = 0

        return {
            "solution": solution,
            "energy": energy,
            "num_reads": 1,
            "solver": "greedy_fallback"
        }

    def decode_solution(self, solution_dict):
        """
        Decode the QUBO solution into a path through states.

        Args:
            solution_dict: Dictionary with solution information

        Returns:
            Dictionary with decoded path and metrics
        """
        solution = solution_dict["solution"]
        energy = solution_dict["energy"]

        # Extract selected nodes and edges
        selected_nodes = []
        selected_edges = []

        for var, val in solution.items():
            if val == 1:  # Selected variable
                if var[0] == var[1]:  # Diagonal element = node
                    selected_nodes.append(var[0])
                else:  # Off-diagonal element = edge
                    selected_edges.append(var)

        # Construct path from edges
        path = []
        if selected_edges:
            # Find starting node (one with no incoming edges)
            incoming = {e[1] for e in selected_edges}
            outgoing = {e[0] for e in selected_edges}
            start_candidates = outgoing - incoming

            if start_candidates:
                start = min(start_candidates)
            else:
                # Fallback: pick any node
                start = min(outgoing)

            # Build path
            current = start
            path = [current]

            while True:
                # Find outgoing edge from current node
                next_edges = [(i, j) for (i, j) in selected_edges if i == current]
                if not next_edges:
                    break

                # Move to next node
                current = next_edges[0][1]
                path.append(current)

                # Remove edge to avoid cycles
                selected_edges.remove(next_edges[0])

                # Check if we've completed a cycle
                if current == start or not selected_edges:
                    break

        # If no path was constructed, use selected nodes
        if not path and selected_nodes:
            path = sorted(selected_nodes)

        # Map path indices to state information
        path_info = []
        for idx in path:
            if 0 <= idx < len(self.states):
                state = self.states[idx]
                path_info.append({
                    "state_id": state.get("id", idx),
                    "index": idx,
                    "metadata": state.get("metadata", {})
                })

        return {
            "path": path,
            "path_info": path_info,
            "energy": energy,
            "num_states": len(path),
            "solver": solution_dict.get("solver", "unknown")
        }

    def optimize(self, phi_scores, psd_scores, context_info=None, num_reads=1000):
        """
        Run the full optimization process.

        Args:
            phi_scores: Dictionary mapping state IDs to alignment scores
            psd_scores: Dictionary mapping state IDs to PSD distance scores
            context_info: Optional context information
            num_reads: Number of reads for the solver

        Returns:
            Dictionary with optimization results
        """
        # Build QUBO
        Q = self.build_qubo(phi_scores, psd_scores, context_info)

        # Solve QUBO
        solution = self.solve_qubo(Q, num_reads)

        # Decode solution
        result = self.decode_solution(solution)

        return result


class GraphEnhancedAlignmentOptimizer(AlignmentOptimizer):
    """
    Graph-enhanced alignment optimizer that uses a consequence graph
    to bias the QUBO toward high-alignment paths.
    """

    def __init__(self, graph_manager, *args, **kwargs):
        """
        Initialize the graph-enhanced optimizer.

        Args:
            graph_manager: Graph manager for accessing the consequence graph
            *args, **kwargs: Arguments to pass to parent class
        """
        super().__init__(*args, **kwargs)
        self.gm = graph_manager

        # Graph enhancement factors
        self.cascade_node_factor = self.config.get("cascade_node_factor", 0.5)
        self.chain_edge_factor = self.config.get("chain_edge_factor", 0.3)
        self.alignment_penalty = self.config.get("alignment_penalty", 10.0)

    def build_qubo(self, phi_scores, psd_scores, context_info=None):
        """
        Build QUBO with graph enhancements.

        Args:
            phi_scores: Dictionary mapping state IDs to alignment scores
            psd_scores: Dictionary mapping state IDs to PSD distance scores
            context_info: Optional context information

        Returns:
            Enhanced QUBO dictionary
        """
        # Get base QUBO from parent class
        Q = super().build_qubo(phi_scores, psd_scores, context_info)

        # Cache - we hit these several times
        aligned_states = set()
        if context_info and "aligned_states" in context_info:
            aligned_states = set(context_info["aligned_states"])

        # Node tweaks
        for idx, state in enumerate(self.states):
            state_id = state.get("id", idx)
            var = (idx, idx)  # Node variable tuple

            # Get alignment impact from graph
            impact = self.gm.get_alignment_impact(state_id) if hasattr(self.gm, "get_alignment_impact") else 0.0

            # Subtract impact => higher impact => more negative => more attractive
            Q[var] -= self.cascade_node_factor * impact

            # Penalize already aligned states
            if state_id in aligned_states:
                Q[var] += self.alignment_penalty

        # Edge tweaks
        for i, src_state in enumerate(self.states):
            src_id = src_state.get("id", i)
            for j, tgt_state in enumerate(self.states):
                tgt_id = tgt_state.get("id", j)
                if i == j:
                    continue

                var = (i, j)
                if var not in Q:  # Parent may prune some transitions
                    continue

                # Get transition strength from graph
                strength = self.gm.get_transition_strength(src_id, tgt_id) if hasattr(self.gm, "get_transition_strength") else 0.0

                # Subtract strength => higher strength => more negative => more attractive
                Q[var] -= self.chain_edge_factor * strength

        return Q
