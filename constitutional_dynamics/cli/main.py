"""
Command-Line Interface for Constitutional Dynamics

This module provides the main entry point for the Constitutional Dynamics package,
the flagship application of the PrincipiaDynamica project. It allows users to
analyze AI model alignment trajectories using State Transition Calculus (STC)
principles, either from static embedding files or live system telemetry.
"""

import os
import sys
import time
import json
import argparse
import logging
import logging.config
import numpy as np
import yaml  # For loading configuration files
from typing import Dict, List, Any, Optional, Tuple  # Added Tuple for type annotations


try:
    import importlib.resources

    PKG_CFG_PATH = "constitutional_dynamics.cfg"
except ImportError:
    # Fallback for Python < 3.7 or if __init__.py is missing in cfg
    # This part is less robust for finding package data.
    # It's better to ensure importlib.resources can be used.
    PKG_CFG_PATH = None
    logging.getLogger("constitutional_dynamics.cli").warning(
        "importlib.resources not available or cfg not a package. "
        "Loading bundled configs might fail."
    )

# Core application imports using relative paths
from ..core.space import AlignmentVectorSpace
from ..core.transition import predict_trajectory
from ..core.metrics import calculate_stability_metrics, evaluate_alignment_robustness, calculate_psd_distance
from ..io.loaders import load_embeddings, aligned_examples
from ..io.timeseries import detect_and_order_time_series
from ..io.live import create_collector
from ..vis.visualizer import create_visualizer
from ..integrations.graph import create_graph_manager

# Quantum and Optimise are imported conditionally/locally to avoid hard dependencies at module load time
from ..integrations.quantum import create_annealer, DWAVE_AVAILABLE, QAOA_AVAILABLE
# from ..core.optimise import AlignmentOptimizer, GraphEnhancedAlignmentOptimizer

# Logger for this module
logger = logging.getLogger("constitutional_dynamics.cli")


def _get_bundled_config_path(filename: str) -> Optional[str]:
    """Helper to get the path to a bundled config file."""
    if PKG_CFG_PATH:
        try:
            # Python 3.9+ style
            if hasattr(importlib.resources, 'files'):
                return str(importlib.resources.files(PKG_CFG_PATH).joinpath(filename))
            # Python 3.7, 3.8 style
            else:
                with importlib.resources.path(PKG_CFG_PATH, filename) as p:
                    return str(p)
        except (FileNotFoundError, TypeError,
                ModuleNotFoundError):  # ModuleNotFoundError for edge cases with namespace packages
            logger.error(f"Bundled config file '{filename}' not found in '{PKG_CFG_PATH}'.")
    return None


def setup_logging(log_config_path: Optional[str] = None) -> None:
    """
    Set up logging configuration for the application.

    Prioritizes user-provided config file, then bundled 'logging.yaml',
    then basicConfig.

    Args:
        log_config_path: Path to a custom logging configuration YAML file.
    """
    config_loaded = False
    if log_config_path and os.path.exists(log_config_path):
        try:
            with open(log_config_path, 'r') as f:
                log_conf = yaml.safe_load(f)
            logging.config.dictConfig(log_conf)
            logger.info(f"Loaded logging configuration from: {log_config_path}")
            config_loaded = True
        except Exception as e:
            logging.basicConfig(level=logging.WARNING)  # Minimal logger for this error
            logger.error(f"Error loading custom logging config '{log_config_path}': {e}. Using fallback.",
                         exc_info=True)

    if not config_loaded:
        bundled_log_config_path = _get_bundled_config_path('logging.yaml')
        if bundled_log_config_path and os.path.exists(bundled_log_config_path):
            try:
                with open(bundled_log_config_path, 'r') as f:
                    log_conf = yaml.safe_load(f)
                logging.config.dictConfig(log_conf)
                logger.info(f"Loaded bundled logging configuration from: {bundled_log_config_path}")
                config_loaded = True
            except Exception as e:
                logging.basicConfig(level=logging.WARNING)
                logger.error(f"Error loading bundled logging config: {e}. Using basicConfig.", exc_info=True)

    if not config_loaded:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger.info("Using basic logging configuration.")


def _update_dict_recursively(target_dict: Dict, source_dict: Dict) -> Dict:
    """Recursively update a dictionary with values from another."""
    for key, value in source_dict.items():
        if isinstance(value, dict) and isinstance(target_dict.get(key), dict):
            _update_dict_recursively(target_dict[key], value)
        else:
            target_dict[key] = value
    return target_dict


def load_app_config(user_config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load application configuration.

    Loads in order:
    1. Bundled `defaults.yaml`.
    2. User-provided configuration file (overrides defaults).

    Args:
        user_config_path: Path to a custom user configuration YAML file.

    Returns:
        A dictionary containing the application configuration.
    """
    # Start with empty or truly minimal hardcoded defaults if bundled fails
    config = {}

    bundled_defaults_path = _get_bundled_config_path('defaults.yaml')
    if bundled_defaults_path and os.path.exists(bundled_defaults_path):
        try:
            with open(bundled_defaults_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.debug(f"Loaded bundled default configuration from: {bundled_defaults_path}")
        except Exception as e:
            logger.error(f"Error loading bundled default config: {e}. Proceeding with minimal/no defaults.",
                         exc_info=True)
            # Define essential minimal defaults here if loading fails critically
            config = {"memory": {}, "alignment": {}, "optimizer": {}, "visualization": {}, "live": {}}

    if user_config_path and os.path.exists(user_config_path):
        try:
            with open(user_config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            _update_dict_recursively(config, user_config)  # Update defaults with user config
            logger.info(f"Loaded user configuration from: {user_config_path}")
        except Exception as e:
            logger.error(f"Error loading user config '{user_config_path}': {e}. Using previously loaded defaults.",
                         exc_info=True)

    # Ensure all top-level keys exist to prevent KeyErrors later
    # (This assumes `config` was populated by bundled defaults or the critical error fallback)
    expected_keys = ["memory", "alignment", "optimizer", "visualization", "live"]
    for key in expected_keys:
        if key not in config:
            config[key] = {}  # Ensure section exists
            logger.warning(f"Configuration section '{key}' was missing; initialized as empty.")

    return config


def _parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Constitutional Dynamics - AI alignment trajectory analysis using State Transition Calculus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows defaults in help
    )

    # Input/output
    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument("-e", "--embeddings", help="Path to embeddings JSON file (required for batch mode).")
    io_group.add_argument("-a", "--aligned", help="Path to JSON file with aligned example embeddings.")
    io_group.add_argument("-o", "--output", help="Output JSON file for analysis results.")

    # Time series
    ts_group = parser.add_argument_group('Time Series Options')
    ts_group.add_argument("-p", "--prefix", help="Prefix pattern for time series detection (e.g., 'state_').")

    # Alignment & Analysis
    align_group = parser.add_argument_group('Alignment & Analysis Options')
    align_group.add_argument("-t", "--threshold", type=float, help="Alignment similarity threshold (overrides config).")
    align_group.add_argument("--dimension", type=int,
                             help="Dimension of embedding space (default auto-detected or 1024).")
    align_group.add_argument("--steps", type=int, default=5, help="Number of prediction steps for trajectory analysis.")
    align_group.add_argument("--spectral", action="store_true", help="Enable spectral (PSD) distance analysis.")

    # Optimization
    opt_group = parser.add_argument_group('Optimization Options')
    opt_group.add_argument("--optimize", action="store_true",
                           help="Run optimization to find an optimal alignment path.")
    opt_group.add_argument("--lambda", type=float, dest="lambda_weight",
                           help="Weight (λ) for spectral vs. time domain in cost function (overrides config).")
    opt_group.add_argument("--quantum", action="store_true",
                           help="Use quantum annealing for optimization (if SDKs are available).")

    # Live mode
    live_group = parser.add_argument_group('Live Mode Options')
    live_group.add_argument("--live", action="store_true",
                            help="Enable live metrics collection mode (e.g., from system stats).")
    live_group.add_argument("--interval", type=float,
                            help="Sampling interval for live mode in seconds (overrides config).")

    # Integrations
    int_group = parser.add_argument_group('Integration Options')
    int_group.add_argument("--graph",
                           help="Neo4j connection URI (e.g., bolt://user:pass@host:port) for graph integration.")

    # Configuration
    cfg_group = parser.add_argument_group('Configuration Options')
    cfg_group.add_argument("--config", help="Path to custom application configuration YAML file.")
    cfg_group.add_argument("--log-config", help="Path to custom logging configuration YAML file.")

    return parser.parse_args()


def _apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply command-line arguments as overrides to the loaded configuration."""
    if args.threshold is not None:
        config.setdefault("alignment", {})["similarity_threshold"] = args.threshold
    if args.lambda_weight is not None:
        config.setdefault("optimizer", {})["lambda_weight"] = args.lambda_weight
    if args.interval is not None:
        config.setdefault("live", {})["interval"] = args.interval
    # `args.dimension` is handled directly in main logic


# --- Main Application Logic Functions ---

def run_batch_mode(args: argparse.Namespace, config: Dict[str, Any], visualizer: Any) -> None:
    """Run the application in batch mode using provided embedding files."""
    logger.info("Starting batch mode analysis.")

    if not args.embeddings:
        logger.error("Embeddings file (--embeddings) is required for batch mode.")
        sys.exit(1)

    embeddings_data = load_embeddings(args.embeddings)
    if not embeddings_data:  # load_embeddings should raise error, but defensive check
        logger.error(f"No embeddings loaded from {args.embeddings}. Exiting.")
        sys.exit(1)

    # Determine dimension
    # Use args.dimension if provided, else infer from first embedding, else default
    inferred_dimension = len(next(iter(embeddings_data.values())))
    dimension = args.dimension if args.dimension is not None else inferred_dimension
    if args.dimension is not None and args.dimension != inferred_dimension:
        logger.warning(
            f"Provided dimension {args.dimension} differs from inferred dimension {inferred_dimension}. "
            f"Using provided dimension: {args.dimension}."
        )
    elif args.dimension is None:
        logger.info(f"Inferred embedding dimension: {dimension}")

    # Initialize AlignmentVectorSpace
    # This is a central component for tracking state evolution as per STC.
    space = AlignmentVectorSpace(
        dimension=dimension,
        memory_decay=config["memory"].get("decay_rate", 0.2),
        similarity_threshold=config["alignment"].get("similarity_threshold", 0.7)
    )

    # Define aligned region
    if args.aligned:
        # load_aligned_examples is a method of AlignmentVectorSpace now
        space.load_aligned_examples(args.aligned)
        if not space.aligned_regions:  # Check if loading actually populated it
            logger.warning(
                f"No valid aligned examples loaded from {args.aligned}. Alignment scores may be less meaningful.")
    else:
        first_embedding_key = next(iter(embeddings_data.keys()))
        first_embedding_vec = embeddings_data[first_embedding_key]
        space.define_alignment_region(first_embedding_vec, radius=0.3)  # Example radius
        logger.warning(
            f"No aligned examples provided (--aligned). Using embedding for '{first_embedding_key}' "
            "to define a default alignment region (center + 0.3 radius). "
            "Provide aligned examples for more meaningful analysis."
        )

    time_series_states = detect_and_order_time_series(embeddings_data, args.prefix)
    if not time_series_states or len(time_series_states) < 2:
        logger.error("Not enough states (minimum 2) found or ordered for analysis. Exiting.")
        sys.exit(1)

    for state_data in time_series_states:
        space.add_state(state_data["embedding"], state_data["timestamp"])
    logger.info(f"Processed {len(space.state_history)} states into alignment vector space.")

    # Perform analyses
    alignment_scores_history = [space.compute_alignment_score(state_vec) for state_vec in space.state_history]
    stability = calculate_stability_metrics(space)
    # Note: Lyapunov estimate in stability is a rough heuristic, as per STC theory development
    logger.info(f"Calculated stability metrics. Avg Alignment: {stability.get('avg_alignment', 'N/A'):.3f}")

    trajectory_predictions = predict_trajectory(space, len(space.state_history) - 1, args.steps)
    logger.info(f"Predicted {len(trajectory_predictions)} future trajectory steps.")

    robustness_analysis = evaluate_alignment_robustness(space)
    logger.info(f"Evaluated alignment robustness. Score: {robustness_analysis.get('robustness_score', 'N/A'):.3f}")

    # Optional PSD distance calculation
    psd_dist = None
    if args.spectral:
        # calculate_psd_distance needs the space object
        psd_dist = calculate_psd_distance(space)
        logger.info(f"Calculated PSD distance: {psd_dist:.3f}")
        robustness_analysis["psd_distance_trajectory"] = psd_dist  # Add to results

    # Visualization
    visualizer.visualize_alignment_history(alignment_scores_history, threshold=space.similarity_threshold)
    visualizer.print_trajectory_analysis(stability, trajectory_predictions)  # Pass full stability dict
    visualizer.print_robustness_analysis(robustness_analysis)
    if args.spectral and psd_dist is not None:
        visualizer.print_spectral_analysis({"psd_distance_trajectory": psd_dist})

    if args.graph:
        # This connection supports the STC concept of observing state transitions in a persistent, queryable manner.
        connect_to_graph(args.graph, space)  # Pass full space object

    if args.optimize:
        # Optimization seeks paths that are "good" according to STC-derived costs (alignment, stability).
        run_optimization(args, config, space, visualizer)

    # Compile and save results
    results_output = {
        "metadata": {
            "source_file": args.embeddings,
            "aligned_examples_file": args.aligned,
            "dimension": dimension,
            "alignment_threshold": space.similarity_threshold,
            "num_states_analyzed": len(space.state_history),
            "analysis_timestamp": time.time()
        },
        "alignment_trajectory": alignment_scores_history,
        "stability_metrics": stability,
        "robustness_analysis": robustness_analysis,
        "trajectory_predictions": trajectory_predictions,
    }
    if psd_dist is not None:  # Only include if calculated
        results_output["spectral_analysis"] = {"psd_distance_trajectory": psd_dist}

    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results_output, f, indent=2, cls=_NumpyAwareEncoder)  # Handle NumPy types if present
            logger.info(f"Full analysis results saved to {args.output}")
        except Exception as e:
            logger.error(f"Failed to save results to {args.output}: {e}")


def run_live_mode(args: argparse.Namespace, config: Dict[str, Any], visualizer: Any) -> None:
    """Run the application in live metrics collection mode."""
    logger.info("Starting live metrics collection mode. Press Ctrl+C to stop and analyze.")

    # For live mode, STC principles are applied to real-time system telemetry.
    # The "constitution" here is a desired low-resource usage state.
    live_config = config.get("live", {})
    collector = create_collector(live_config)

    # Fixed dimension for system metrics (e.g., CPU, Mem, Disk I/O, Net I/O etc.)
    # This dimension should match what `collector.get_metrics_stream()` yields.
    SYSTEM_METRICS_DIMENSION = getattr(collector, 'dimension', 7)  # Try to get from collector, else default
    logger.info(f"Live mode expecting metrics vector of dimension: {SYSTEM_METRICS_DIMENSION}")

    space = AlignmentVectorSpace(
        dimension=SYSTEM_METRICS_DIMENSION,
        memory_decay=config["memory"].get("decay_rate", 0.2),
        similarity_threshold=config["alignment"].get("similarity_threshold", 0.7)
    )

    # Define "aligned" as low resource usage for system telemetry.
    # This definition is part of the "constitution" for this live mode.
    # Example: [CPU_usr, CPU_sys, MEM_perc, D_read, D_write, N_sent, N_recv]
    # Values represent target "good" state (e.g. <30% CPU, <70% Mem, low I/O)
    # These values would ideally be normalized (0-1) by the collector.
    aligned_center = live_config.get("aligned_center_metrics", [0.3, 0.3, 0.7, 0.1, 0.1, 0.1, 0.1])
    if len(aligned_center) != SYSTEM_METRICS_DIMENSION:
        logger.warning(
            f"Live mode 'aligned_center_metrics' dimension mismatch. Expected {SYSTEM_METRICS_DIMENSION}, got {len(aligned_center)}. Adjusting or using fallback.")
        # Fallback or adjust - for now, let's assume it should match or use a default
        aligned_center = [0.5] * SYSTEM_METRICS_DIMENSION  # Generic fallback

    space.define_alignment_region(aligned_center, radius=live_config.get("aligned_radius_metrics", 0.3))
    logger.info(f"Live mode alignment region defined around center: {aligned_center} with radius.")

    graph_manager = None
    if args.graph:
        graph_manager = connect_to_graph(args.graph, space)  # Initial empty space for graph connection

    alignment_scores_history = []
    try:
        collector.start()
        logger.info("Collector started. Streaming metrics...")
        for state_idx, vector_data in enumerate(collector.get_metrics_stream()):
            # Assuming vector_data is {'vector': list, 'timestamp': float} or just the list
            if isinstance(vector_data, dict):
                state_vec = vector_data.get('vector')
                timestamp = vector_data.get('timestamp', time.time())
            else:
                state_vec = vector_data
                timestamp = time.time()

            if not state_vec or len(state_vec) != SYSTEM_METRICS_DIMENSION:
                logger.warning(f"Skipping invalid live data point: {state_vec}")
                continue

            current_state_idx_in_space = space.add_state(state_vec,
                                                         timestamp)  # Note: this returns actual index in space.state_history
            alignment_score = space.compute_alignment_score(state_vec)
            alignment_scores_history.append(alignment_score)

            if graph_manager:
                _add_live_data_to_graph(graph_manager, space, current_state_idx_in_space, state_vec, alignment_score)

            # Limit history for live visualization
            if len(alignment_scores_history) > 100:
                alignment_scores_history = alignment_scores_history[-100:]

            if (state_idx + 1) % live_config.get("refresh_interval_states", 10) == 0:
                if sys.stdout.isatty():  # Only clear if it's a real terminal
                    os.system('cls' if os.name == 'nt' else 'clear')
                visualizer.visualize_alignment_history(alignment_scores_history, threshold=space.similarity_threshold)
                print(
                    f"Collected {state_idx + 1} live states. Last alignment: {alignment_score:.3f}. Press Ctrl+C to stop.")

    except KeyboardInterrupt:
        logger.info("Live metrics collection interrupted by user.")
    finally:
        collector.stop()
        logger.info("Collector stopped.")

    if len(space.state_history) >= 2:
        logger.info("Analyzing collected live data...")
        stability = calculate_stability_metrics(space)
        # robustness might be less meaningful for live system stats unless perturbations are simulated
        # robustness = evaluate_alignment_robustness(space) # Potentially skip or adapt

        visualizer.visualize_alignment_history(alignment_scores_history, threshold=space.similarity_threshold)
        visualizer.print_trajectory_analysis(stability)  # Predictions might not be useful here
        # visualizer.print_robustness_analysis(robustness)

        if args.optimize:
            # Optimization can suggest "ideal" telemetry patterns or identify risky sequences.
            run_optimization(args, config, space, visualizer)
        logger.info("Live mode analysis complete.")
    else:
        logger.warning("Not enough data collected in live mode for full analysis.")


def _add_live_data_to_graph(graph_manager: Any, space: AlignmentVectorSpace, current_idx_in_space: int,
                            state_vec: List[float], alignment_score: float):
    """Helper to add live state and transition to graph."""
    # `current_idx_in_space` is 0-based index in space.state_history
    state_id = f"live_state_{current_idx_in_space}"
    timestamp = space.state_timestamps[current_idx_in_space]

    graph_manager.add_state(state_id, state_vec, {
        "timestamp": timestamp,
        "alignment_score": alignment_score,
        "source": "live_metrics"
    })

    if current_idx_in_space > 0:
        prev_state_idx_in_space = current_idx_in_space - 1
        prev_state_id = f"live_state_{prev_state_idx_in_space}"

        prev_state_vec = space.state_history[prev_state_idx_in_space]
        prev_alignment_score = space.compute_alignment_score(prev_state_vec)

        graph_manager.add_transition(prev_state_id, state_id, {
            "timestamp": timestamp,  # Timestamp of the TO state
            "alignment_change": alignment_score - prev_alignment_score
        })


def run_optimization(args: argparse.Namespace, config: Dict[str, Any], space: AlignmentVectorSpace,
                     visualizer: Any) -> None:
    """
    Run the alignment path optimization process.
    This leverages STC concepts by encoding alignment scores (phi) and spectral
    properties (PSD distance) into a QUBO problem to find optimal trajectories.
    """
    logger.info("Preparing and running alignment path optimization...")

    try:
        # These imports are deferred as optimization is an optional path
        from ..core.optimise import AlignmentOptimizer, GraphEnhancedAlignmentOptimizer
        from ..integrations.quantum import create_annealer
    except ImportError as e:
        logger.error(
            f"Optimization requires additional dependencies not found: {e}. "
            "Please install with 'pip install constitutional-dynamics[quantum]' or relevant extras."
        )
        return

    # Prepare data for optimizer
    states_for_optimizer = []
    phi_scores_map = {}
    psd_scores_map = {}  # For PSD of individual states or windows if applicable

    # The STC framework views each recorded state as a potential node in an alignment trajectory.
    for i, state_vec in enumerate(space.state_history):
        # Using a simple index-based ID for optimizer states
        state_id = str(i)  # Optimizer might expect string IDs
        phi_score = space.compute_alignment_score(state_vec)

        # Mock PSD score for individual states - for actual use, this would need a meaningful calculation
        # e.g., PSD of a small window around the state, or if states represent signals themselves.
        # The current `calculate_psd_distance` is for the whole trajectory.
        # For STC, spectral properties of state transitions or local trajectory segments might be relevant.
        mock_psd_individual_state = 0.1 * (1.0 - phi_score)
        logger.debug(f"Using mock PSD score {mock_psd_individual_state} for state {state_id}")

        states_for_optimizer.append({
            "id": state_id,  # Ensure optimizer handles string IDs
            "embedding": state_vec,  # Original embedding
            "metadata": {"timestamp": space.state_timestamps[i], "original_index": i}
        })
        phi_scores_map[state_id] = phi_score
        psd_scores_map[state_id] = mock_psd_individual_state

    optimizer_config = config.get("optimizer", {})
    optimizer_instance: AlignmentOptimizer  # Type hint

    if args.graph and hasattr(args,
                              'graph_uri_for_optimizer'):  # Assuming graph URI might be passed differently for optimizer
        # Graph-enhanced optimization uses learned consequences or preferred STC transitions from a knowledge graph.
        graph_manager = create_graph_manager(args.graph_uri_for_optimizer)  # Or reuse existing if appropriate
        if graph_manager:
            logger.info("Using GraphEnhancedAlignmentOptimizer.")
            optimizer_instance = GraphEnhancedAlignmentOptimizer(
                graph_manager=graph_manager,
                states=states_for_optimizer,
                config=optimizer_config,
                debug=logger.isEnabledFor(logging.DEBUG)
            )
        else:
            logger.warning("Graph URI provided but failed to create manager. Using standard optimizer.")
            optimizer_instance = AlignmentOptimizer(
                states=states_for_optimizer, config=optimizer_config, debug=logger.isEnabledFor(logging.DEBUG)
            )
    else:
        logger.info("Using AlignmentOptimizer.")
        optimizer_instance = AlignmentOptimizer(
            states=states_for_optimizer, config=optimizer_config, debug=logger.isEnabledFor(logging.DEBUG)
        )

    # The QUBO formulation translates STC alignment goals into an optimization problem.
    qubo = optimizer_instance.build_qubo(phi_scores_map, psd_scores_map)
    logger.info(f"Built QUBO with {len(qubo)} terms for optimization.")

    solution_details: Dict[str, Any]
    if args.quantum:
        # Quantum annealing explores the solution space based on STC principles encoded in the QUBO.
        logger.info("Attempting to solve QUBO with quantum annealer.")
        annealer = create_annealer(
            simulation_mode=not DWAVE_AVAILABLE and not QAOA_AVAILABLE,  # Force sim if no SDKs
            annealer_type=optimizer_config.get("quantum_solver_type", "dwave"),  # e.g. 'dwave' or 'qaoa'
            token=os.environ.get("DWAVE_API_TOKEN"),  # Example of sourcing token
            verbose=logger.isEnabledFor(logging.DEBUG)
        )
        solution_details = annealer.solve_qubo(
            qubo,
            num_reads=optimizer_config.get("quantum_num_reads", 1000)
        )
    else:
        logger.info("Solving QUBO with classical optimizer (fallback/default).")
        # Base optimizer's solve_qubo is a placeholder; real solution comes from optimize()
        # which calls the (potentially overridden) solve_qubo.
        # For now, assume AlignmentOptimizer.optimize() calls its own solve_qubo (which is a fallback).
        # A more effcient way would be to have a dedicated classical solver for the base class.
        # This could be integrated later for now its only me, the books, and the ai so capacities are limited.
        solution_details = optimizer_instance.solve_qubo(qubo,
                                                         num_reads=optimizer_config.get("quantum_num_reads", 1000))

    optimization_result = optimizer_instance.decode_solution(solution_details)
    visualizer.print_optimization_results(optimization_result)

def connect_to_graph(graph_uri: str, space: AlignmentVectorSpace) -> Optional[
    Any]:  # Return type should be GraphManager
    """
    Connect to Neo4j graph database and populate with trajectory data.
    This allows persisting and querying STC state transitions.
    """
    logger.info(f"Attempting to connect to graph database at {graph_uri}...")
    graph_manager = None  # Default to None
    try:
        # create_graph_manager is already imported
        auth_tuple: Optional[Tuple[str, str]] = None
        if "@" in graph_uri and "://" in graph_uri:
            uri_parts = graph_uri.split("://")[1].split("@")
            if len(uri_parts) > 1 and ":" in uri_parts[0]:
                username, password = uri_parts[0].split(":", 1)
                auth_tuple = (username, password)

        graph_manager = create_graph_manager(uri=graph_uri, auth=auth_tuple)

        if graph_manager:
            logger.info("Successfully connected to graph database.")
            # Populate graph with states and transitions from the current space
            # This should reflect the observed STC trajectory.
            num_added_states = 0
            num_added_transitions = 0
            for i, state_vec_g in enumerate(space.state_history):
                state_id_g = f"batch_state_{i}"
                alignment_score_g = space.compute_alignment_score(state_vec_g)
                metadata_g = {
                    "timestamp": space.state_timestamps[i],
                    "alignment_score": alignment_score_g,
                    "source": "batch_analysis"
                }
                graph_manager.add_state(state_id_g, state_vec_g, metadata_g)
                num_added_states += 1

                if i > 0:
                    prev_state_id_g = f"batch_state_{i - 1}"
                    prev_alignment_score_g = space.compute_alignment_score(space.state_history[i - 1])
                    transition_metadata_g = {
                        "timestamp": space.state_timestamps[i],
                        "alignment_change": alignment_score_g - prev_alignment_score_g
                    }
                    graph_manager.add_transition(prev_state_id_g, state_id_g, transition_metadata_g)
                    num_added_transitions += 1
            logger.info(f"Populated graph with {num_added_states} states and {num_added_transitions} transitions.")
        else:
            logger.error("Failed to create graph manager instance.")
    except ImportError:
        logger.error(
            "Neo4j integration requires 'neo4j' Python driver. "
            "Install with 'pip install constitutional-dynamics[graph]'."
        )
    except Exception as e:
        logger.error(f"Error connecting to or populating graph database: {e}", exc_info=True)

    return graph_manager


# Helper for JSON serialization if NumPy types are involved
class _NumpyAwareEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):  # Requires NumPy import if used
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Entry point for the CLI
def main():
    """Main function to orchestrate CLI operations."""
    args = _parse_arguments()

    # Setup logging first, using user-provided config if available
    setup_logging(args.log_config)
    logger.info("Constitutional Dynamics CLI started.")
    logger.debug(f"CLI arguments: {args}")

    # Load application configuration (defaults overridden by file, then by CLI)
    config = load_app_config(args.config)
    _apply_cli_overrides(config, args)
    logger.debug(f"Effective configuration: {config}")

    visualizer = create_visualizer(
        use_rich=config.get("visualization", {}).get("use_rich", True),
        width=config.get("visualization", {}).get("sparkline_width", 60)
    )

    visualizer.print_message(
        "Welcome to Constitutional Dynamics by PrincipiaDynamica!That's one small step for man, one giant leap for mankind.",
        style="bold green"
    )

    try:
        if args.live:
            run_live_mode(args, config, visualizer)
        elif args.embeddings:  # Embeddings file is primary trigger for batch mode
            run_batch_mode(args, config, visualizer)
        else:
            # No action specified if not live and no embeddings
            logger.warning("No embeddings file provided and not in live mode. Nothing to do.")
            _parse_arguments().print_help()  # Show help if no actionable arguments
            sys.exit(0)

        logger.info("Constitutional Dynamics CLI finished.")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
