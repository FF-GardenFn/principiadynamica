# Usage Guide

Constitutional Dynamics, an application from the PrincipiaDynamica project, offers tools to analyze and monitor AI alignment as a dynamic process rather than a static property. This guide provides an overview of how to use Constitutional Dynamics for alignment analysis and monitoring.

## Command-Line Interface

Constitutional Dynamics provides a command-line interface (CLI) for analyzing alignment vector dynamics. The basic syntax is:

```bash
python -m constitutional_dynamics [OPTIONS] [COMMAND]
```

### Basic Commands

#### Analyzing Embeddings

To analyze embeddings from a file:

```bash
python -m constitutional_dynamics --embeddings path/to/embeddings.json --output report.json
```

This will:
1. Load embeddings from the specified file
2. Analyze alignment dynamics
3. Generate a report with alignment scores, stability metrics, and predictions

#### Defining Aligned Behavior

To define what "aligned behavior" looks like:

```bash
python -m constitutional_dynamics --embeddings path/to/embeddings.json --aligned path/to/aligned_examples.json
```

The `aligned_examples.json` file should contain examples of embeddings that represent aligned behavior.

#### Live Monitoring

To monitor alignment in real-time using system metrics:

```bash
python -m constitutional_dynamics --live --interval 1.0
```

This will:
1. Collect system metrics (CPU, memory, network, disk) every 1.0 seconds
2. Convert metrics to vectors in alignment space
3. Analyze alignment dynamics in real-time
4. Display a live visualization of alignment trajectory

Press Ctrl+C to stop monitoring and see the final analysis.

### Configuration

Constitutional Dynamics uses configuration files to control its behavior. The default configuration is stored in `cfg/defaults.yaml`. You can override these settings with your own configuration file:

```bash
python -m constitutional_dynamics --config path/to/custom_config.yaml --log-config path/to/logging_config.yaml
```

This allows you to customize parameters like thresholds, decay rates, and other settings without modifying the code.

### Advanced Options

#### Spectral Analysis

To enable spectral (PSD) analysis:

```bash
python -m constitutional_dynamics --embeddings path/to/embeddings.json --spectral
```

This adds frequency-domain analysis to detect periodic patterns in alignment.

#### Optimization

To find optimal alignment paths:

```bash
python -m constitutional_dynamics --embeddings path/to/embeddings.json --optimize
```

This uses optimization techniques to find paths through embedding space that maximize alignment.

#### Quantum Optimization

If you have access to a D-Wave quantum annealer:

```bash
python -m constitutional_dynamics --embeddings path/to/embeddings.json --optimize --quantum
```

This uses quantum annealing for optimization, which can be more effective for complex alignment landscapes.

#### Graph Integration

To store alignment states and transitions in a Neo4j graph database:

```bash
python -m constitutional_dynamics --embeddings path/to/embeddings.json --graph bolt://username:password@localhost:7687
```

This allows you to:
1. Visualize alignment trajectories in Neo4j Bloom
2. Query alignment patterns using Cypher
3. Analyze alignment transitions over time

## Python API

Constitutional Dynamics can also be used as a Python library for more advanced use cases.

### Basic Usage

```python
import constitutional_dynamics

# Create an alignment vector space
space = constitutional_dynamics.AlignmentVectorSpace(dimension=1024)

# Define aligned region
space.load_aligned_examples("aligned_examples.json")

# Add states
space.add_state(embedding1, timestamp1)
space.add_state(embedding2, timestamp2)

# Analyze transitions
transition = constitutional_dynamics.analyse_transition(space, 0, 1)

# Calculate stability metrics
metrics = constitutional_dynamics.calculate_stability_metrics(space)

# Predict future trajectory
predictions = constitutional_dynamics.predict_trajectory(space, start_state_idx=1, steps=5)

# Evaluate robustness
robustness = constitutional_dynamics.evaluate_alignment_robustness(space)

# Visualize results
visualizer = constitutional_dynamics.create_visualizer()
visualizer.visualize_alignment_history([space.compute_alignment_score(state) for state in space.state_history])
visualizer.print_trajectory_analysis(metrics, predictions)
```

### Live Monitoring

```python
import constitutional_dynamics

# Create alignment vector space
space = constitutional_dynamics.AlignmentVectorSpace(dimension=7)  # 7 dimensions for metrics

# Define aligned region (low resource usage is "aligned")
space.define_alignment_region([0.3, 0.3, 0.7, 0.1, 0.1, 0.1, 0.1], radius=0.3)

# Create metrics collector
collector = constitutional_dynamics.create_collector()

# Start collector
collector.start()

# Process metrics stream
for vector in collector.get_metrics_stream():
    # Add state to space
    state_idx = space.add_state(vector)

    # Compute alignment score
    alignment = space.compute_alignment_score(vector)

    # Do something with the alignment score
    print(f"Alignment: {alignment:.2f}")

    # Stop after 100 samples
    if state_idx >= 100:
        break

# Stop collector
collector.stop()

# Calculate stability metrics
metrics = constitutional_dynamics.calculate_stability_metrics(space)
print(f"Stability score: {metrics['stability_score']:.2f}")
```

### Graph Integration

```python
import constitutional_dynamics

# Create graph manager
graph_manager = constitutional_dynamics.create_graph_manager("bolt://username:password@localhost:7687")

# Add states and transitions
graph_manager.add_state("state_1", embedding1, {"alignment_score": 0.8})
graph_manager.add_state("state_2", embedding2, {"alignment_score": 0.7})
graph_manager.add_transition("state_1", "state_2", {"alignment_change": -0.1})

# Query aligned states
aligned_states = graph_manager.get_aligned_states(threshold=0.7)
```

### Advanced: Circuit Tracer Bridge for Mechanistic Interventions (Experimental)

The Circuit Tracer Bridge allows integrating behavioral monitoring with mechanistic interpretability insights. This is an experimental feature.

```python
from constitutional_dynamics import AlignmentVectorSpace, AlignmentThermostat
# Assuming you have your model, and mock or real circuit tracer components:
# from constitutional_dynamics.integrations.circuit_tracer_bridge.examples.common_mocks import MockCircuitTracer, MockModelInterface

# 1. Initialize Constitutional Dynamics monitor
cd_monitor = AlignmentVectorSpace(dimension=768) 
# ... define aligned region for cd_monitor ...

# 2. Initialize (mock or real) Circuit Tracer and Model Interface
# circuit_tracer_instance = RealCircuitTracer() # From Anthropic's library
# model_interface_instance = YourModelWrapper()

# Using placeholder/mock names for this example:
# For actual use, these would be real or more sophisticated mock objects.
# See circuit_tracer_bridge/examples/ for runnable demos with detailed mocks.
class Placeholder: pass
circuit_tracer_instance = Placeholder() # Replace with actual or better mock
model_interface_instance = Placeholder() # Replace with actual or better mock

# 3. Create the AlignmentThermostat
thermostat = AlignmentThermostat(
    cd_monitor_instance=cd_monitor,
    circuit_tracer_instance=circuit_tracer_instance,
    model_interface=model_interface_instance,
    threshold=0.7,  # Alignment score threshold
    auto_stabilize=True
)

# 4. Simulate a model output and run the feedback loop
# current_embedding = get_model_output_embedding_as_list()
# original_prompt = "The prompt that generated this output"
#
# result = thermostat.run_feedback_loop(
#     current_model_output_embedding=current_embedding,
#     original_prompt_for_trace=original_prompt
# )
#
# if result.get("intervention_applied"):
#     print(f"Intervention Result: Improved={result.get('improved')}, New Score={result.get('new_score')}")
#     if result.get("stability_improved"):
#         print(f"System stability also improved. New Lyapunov est: {result.get('lyapunov_after')}")
# else:
#     print(f"No intervention applied. Reason: {result.get('reason')}")
```

Note: This example is kept high-level. Point users to your circuit_tracer_bridge/examples/ directory and the bridge's own README.md for more detailed, runnable examples with the necessary mock setups.

### Programmatic Configuration Access

```python
import constitutional_dynamics

# Load the default application configuration
default_config = constitutional_dynamics.get_default_config()
print("Default memory decay rate:", default_config.get("memory", {}).get("decay_rate"))

# Load a custom config file (merging with defaults)
# my_config = constitutional_dynamics.load_config("path/to/my_custom_config.yaml")

# Configure logging programmatically (uses bundled logging.yaml by default)
# constitutional_dynamics.configure_logging() 
# Or with a custom logging config:
# constitutional_dynamics.load_logging_config("path/to/my_logging_config.yaml")
```

## Potential Workflows

### Analyzing Model Behavior Over Time

1. Generate embeddings for model outputs at different points in time
2. Define what "aligned" behavior looks like using examples
3. Analyze the trajectory of alignment scores
4. Identify when and how alignment changes

### Monitoring Deployment

1. Set up live monitoring of system metrics
2. Define alignment thresholds based on normal operation
3. Configure alerts for alignment drift
4. Use graph integration to store historical alignment data

### Optimizing for Alignment

1. Define alignment objectives (time domain and spectral)
2. Use optimization to find paths that maximize alignment
3. Apply quantum optimization for complex alignment landscapes
4. Analyze the resulting paths to understand alignment dynamics

### Dynamic Intervention and Oversight

1. Set up live or batch monitoring with Constitutional Dynamics
2. Define alignment thresholds and stability parameters for the AlignmentThermostat
3. When issues are detected, leverage the Circuit Tracer Bridge to perform (simulated or real) mechanistic analysis
4. Apply mechanistically-informed interventions (e.g., feature suppression) via the AlignmentThermostat
5. Continuously verify intervention effectiveness and adapt strategies using the MetaStrategist feedback loop

## Next Steps

- Explore the [API Reference](api/core.md) for detailed documentation of all functions and classes
- Learn about the [Circuit Tracer Bridge](api/integrations.md#circuit-tracer-bridge) for mechanistic interpretability integration
- Check out the [Mathematical Backbone](index.md#mathematical-backbone) to understand the theory behind Constitutional Dynamics
