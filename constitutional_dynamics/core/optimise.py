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

        This implementation provides a more sophisticated QUBO-vert solver
        that handles vertical transitions better, particularly for recovery paths.

        Args:
            Q: QUBO dictionary
            num_reads: Number of reads for the solver

        Returns:
            Dictionary with solution information
        """
        # Check if we should use the improved QUBO-vert solver
        use_qubo_vert = True

        if use_qubo_vert:
            logger.info("Using QUBO-vert solver for optimal recovery path planning")
            return self._solve_qubo_vert(Q, num_reads)
        else:
            logger.warning("Base solve_qubo called - should be implemented by subclass or integration")
            return self._solve_qubo_greedy(Q)

    def _solve_qubo_greedy(self, Q):
        """
        Simple greedy solution as fallback.

        Args:
            Q: QUBO dictionary

        Returns:
            Dictionary with solution information
        """
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

    def _solve_qubo_vert(self, Q, num_reads=1000):
        """
        QUBO-vert solver that handles vertical transitions better.

        This solver is specifically designed for recovery path planning,
        using gradient information to find optimal paths from misaligned
        to aligned states.

        Args:
            Q: QUBO dictionary
            num_reads: Number of reads for the solver

        Returns:
            Dictionary with solution information
        """
        # Initialize solution
        solution = {}
        energy = 0.0

        # Extract node and edge variables
        node_vars = {}
        edge_vars = {}

        for var, coeff in Q.items():
            if var[0] == var[1]:  # Diagonal element = node
                node_vars[var] = coeff
            else:  # Off-diagonal element = edge
                edge_vars[var] = coeff

        # Sort nodes by their coefficients (ascending)
        sorted_nodes = sorted(node_vars.items(), key=lambda x: x[1])

        # First pass: select nodes with strong negative coefficients (beneficial to include)
        selected_nodes = set()
        for var, coeff in sorted_nodes:
            if coeff < -0.3:  # More aggressive threshold for node selection
                solution[var] = 1
                energy += coeff
                selected_nodes.add(var[0])
            else:
                solution[var] = 0

        # If no nodes were selected, select at least the most beneficial ones
        if not selected_nodes and sorted_nodes:
            # Select top 3 most beneficial nodes (or all if fewer than 3)
            for var, _ in sorted_nodes[:min(3, len(sorted_nodes))]:
                solution[var] = 1
                energy += node_vars[var]
                selected_nodes.add(var[0])

        # Identify constitutional dimensions for each state
        constitutional_dims = {}
        for i, state in enumerate(self.states):
            if i < len(self.states):
                # Extract constitutional dimensions (first 10)
                if 'embedding' in state:
                    embedding = state['embedding']
                    constitutional_dims[i] = embedding[:min(10, len(embedding))]
                elif 'state' in state:
                    state_vec = state['state']
                    constitutional_dims[i] = state_vec[:min(10, len(state_vec))]

        # Second pass: build a path through selected nodes
        # Sort edges by their coefficients (ascending)
        sorted_edges = sorted(edge_vars.items(), key=lambda x: x[1])

        # Categorize edges by their recovery characteristics
        vertical_edges = []  # Improving alignment
        constitutional_edges = []  # Improving constitutional dimensions
        horizontal_edges = []  # Other transitions

        for var, coeff in sorted_edges:
            i, j = var[0], var[1]

            # Skip if either node is out of range
            if i >= len(self.states) or j >= len(self.states):
                continue

            # Check if this is a vertical transition (improving alignment)
            # We assume later indices represent more aligned states
            if j > i:
                # Check if this transition improves constitutional dimensions
                if i in constitutional_dims and j in constitutional_dims:
                    const_i = constitutional_dims[i]
                    const_j = constitutional_dims[j]

                    # Count improved dimensions
                    improved_dims = sum(1 for k in range(min(len(const_i), len(const_j))) 
                                      if abs(const_j[k]) > abs(const_i[k]))

                    if improved_dims >= 3:  # At least 3 dimensions improved
                        constitutional_edges.append((var, coeff, improved_dims))
                    else:
                        vertical_edges.append((var, coeff))
                else:
                    vertical_edges.append((var, coeff))
            else:
                horizontal_edges.append((var, coeff))

        # Sort constitutional edges by number of improved dimensions (descending) and coefficient
        constitutional_edges.sort(key=lambda x: (-x[2], x[1]))

        # Sort vertical edges by alignment improvement (j-i) and coefficient
        vertical_edges.sort(key=lambda x: (x[0][1] - x[0][0], x[1]))

        # Select constitutional edges first (prioritize constitutional recovery)
        for var, coeff, _ in constitutional_edges:
            i, j = var[0], var[1]

            # Only consider edges from selected nodes or to critical nodes
            if i in selected_nodes:
                solution[var] = 1
                energy += coeff
                selected_nodes.add(j)  # Add destination node

                # Also ensure the destination node is selected
                node_var = (j, j)
                if node_var in node_vars and solution.get(node_var, 0) == 0:
                    solution[node_var] = 1
                    energy += node_vars[node_var]

        # Then select vertical edges (general alignment improvement)
        for var, coeff in vertical_edges:
            i, j = var[0], var[1]

            # Only consider edges from selected nodes
            if i in selected_nodes and j not in selected_nodes:
                solution[var] = 1
                energy += coeff
                selected_nodes.add(j)  # Add destination node

                # Also ensure the destination node is selected
                node_var = (j, j)
                if node_var in node_vars and solution.get(node_var, 0) == 0:
                    solution[node_var] = 1
                    energy += node_vars[node_var]

        # Build a connected path
        # Identify disconnected components in the selected nodes
        components = []
        unassigned = set(selected_nodes)

        while unassigned:
            # Start a new component
            current = next(iter(unassigned))
            component = {current}
            to_explore = {current}
            unassigned.remove(current)

            # Expand component
            while to_explore:
                node = to_explore.pop()

                # Find connected nodes
                for var in solution:
                    if isinstance(var, tuple) and len(var) == 2 and var[0] != var[1] and solution[var] == 1:
                        i, j = var

                        if i == node and j in unassigned:
                            component.add(j)
                            to_explore.add(j)
                            unassigned.remove(j)
                        elif j == node and i in unassigned:
                            component.add(i)
                            to_explore.add(i)
                            unassigned.remove(i)

            components.append(component)

        # If multiple components, connect them
        if len(components) > 1:
            # Sort components by minimum node index (ascending)
            components.sort(key=lambda c: min(c))

            # Connect adjacent components
            for i in range(len(components) - 1):
                comp1 = components[i]
                comp2 = components[i + 1]

                # Find best edge to connect these components
                best_edge = None
                best_coeff = float('inf')

                for var, coeff in sorted_edges:
                    i, j = var

                    if (i in comp1 and j in comp2) or (i in comp2 and j in comp1):
                        if coeff < best_coeff:
                            best_edge = var
                            best_coeff = coeff

                # Add connecting edge if found
                if best_edge:
                    solution[best_edge] = 1
                    energy += best_coeff

        # Fill in any gaps in the path with horizontal edges if needed
        for var, coeff in horizontal_edges:
            i, j = var[0], var[1]

            # Only consider edges between selected nodes that aren't already connected
            if i in selected_nodes and j in selected_nodes:
                # Check if there's already a path from i to j
                path_exists = False
                for v in solution:
                    if isinstance(v, tuple) and len(v) == 2 and v[0] != v[1] and solution[v] == 1:
                        if v[0] == i and v[1] == j:
                            path_exists = True
                            break

                if not path_exists:
                    solution[var] = 1
                    energy += coeff

        return {
            "solution": solution,
            "energy": energy,
            "num_reads": num_reads,
            "solver": "qubo_vert_enhanced"
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
