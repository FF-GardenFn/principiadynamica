# Installation Guide

This guide will help you install Constitutional Dynamics and its dependencies.

## Requirements

Constitutional Dynamics requires Python 3.8 or later. The following dependencies are required for the full functionality:

### Core Dependencies
- **NumPy** and **SciPy**: For mathematical operations and vector space calculations
- **PyYAML**: For configuration file parsing
- **Rich**: For enhanced console output and visualizations

### Optional Dependencies
- **Neo4j Python Driver**: For graph database integration
- **D-Wave Ocean SDK**: For quantum annealing optimization
- **Anthropic/OpenAI SDK**: For LLM strategist integration

## Installation Methods

### From PyPI (Recommended)

```bash
pip install constitutional-dynamics
```

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/FF-GardenFn/principiadynamica.git
   cd principiadynamica
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

### With Optional Dependencies

To install Constitutional Dynamics with optional dependencies:

```bash
# For graph database integration
pip install constitutional-dynamics[graph]

# For quantum optimization
pip install constitutional-dynamics[quantum]

# For development (includes testing tools)
pip install constitutional-dynamics[dev]

# For all optional dependencies
pip install constitutional-dynamics[all]
```

## Configuration

Constitutional Dynamics uses YAML configuration files for settings and logging configuration. The default configuration files are located in the `cfg/` directory:

- `defaults.yaml`: Default configuration for memory decay, alignment thresholds, etc.
- `logging.yaml`: Logging configuration

You can provide your own configuration files using the `--config` and `--log-config` command-line options:

```bash
python -m constitutional_dynamics --config my_config.yaml --log-config my_logging.yaml
```

### Default Configuration

The default configuration includes:

```yaml
# Memory and decay parameters
memory:
  decay_rate: 0.2  # τ (tau) - memory decay rate
  window_size: 10  # Number of states to consider for history

# Alignment thresholds
alignment:
  similarity_threshold: 0.7  # Threshold for considering a state aligned

# Optimizer settings
optimizer:
  lambda_weight: 0.35  # λ (lambda) - weight for spectral vs time domain
  flow_constraint_strength: 5.0  # Strength of flow constraints in QUBO
  quantum_num_reads: 1000  # Number of reads for quantum annealer

# Visualization
visualization:
  sparkline_width: 60  # Width of sparkline visualization
  use_rich: true  # Whether to use Rich for visualization

# Live collection
live:
  interval: 1.0  # Seconds between samples
  gauss_sigma: 2.0  # Smoothing factor (Gaussian)
  max_queue: 512  # Maximum queue size for samples
```

## Verifying Installation

To verify that Constitutional Dynamics is installed correctly, run:

```bash
python -m constitutional_dynamics --help
```

This should display the help message for the Constitutional Dynamics command-line interface.

## Troubleshooting

### Missing Dependencies

If you encounter errors about missing dependencies, try installing the specific dependency:

```bash
pip install <dependency-name>
```

### Neo4j Connection Issues

If you're having trouble connecting to Neo4j, ensure:
1. Neo4j is running and accessible
2. You have the correct connection URI, username, and password
3. The Neo4j Python driver is installed: `pip install neo4j`

### D-Wave Quantum Annealer Access

To use the D-Wave quantum annealer:
1. Sign up for a D-Wave account at https://cloud.dwavesys.com/
2. Install the D-Wave Ocean SDK: `pip install dwave-ocean-sdk`
3. Configure your API token: `dwave config create`

If you don't have access to a quantum annealer, Constitutional Dynamics will automatically fall back to classical simulation.
