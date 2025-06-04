# Core API Reference

The `northstar.core` module provides the core functionality for alignment vector dynamics, including the `AlignmentVectorSpace` class, transition analysis, metrics calculation, and optimization.

## AlignmentVectorSpace

```python
class AlignmentVectorSpace(dimension=1024, memory_decay=0.2, similarity_threshold=0.7)
```

Models alignment as a vector space where:
- Vectors represent behavior/response states
- Regions in the space represent aligned vs. misaligned behaviors
- Trajectories through the space represent behavioral evolution

### Parameters

- **dimension** (`int`, default=1024): Dimensionality of the embedding space
- **memory_decay** (`float`, default=0.2): Rate at which memory of past states decays (τ)
- **similarity_threshold** (`float`, default=0.7): Threshold for considering states similar

### Methods

#### `load_aligned_examples`

```python
load_aligned_examples(examples_path: str) -> bool
```

Load examples of aligned behavior to define the "aligned region".

**Parameters**:
- **examples_path** (`str`): Path to JSON file with aligned examples

**Returns**:
- `bool`: Success status

#### `define_alignment_region`

```python
define_alignment_region(center: List[float], radius: float = 0.3) -> bool
```

Manually define an alignment region as a hypersphere.

**Parameters**:
- **center** (`List[float]`): Center vector of the aligned region
- **radius** (`float`, default=0.3): Radius of the aligned region hypersphere

**Returns**:
- `bool`: Success status

#### `add_state`

```python
add_state(state: List[float], timestamp: Optional[float] = None) -> int
```

Add a state vector to the history.

**Parameters**:
- **state** (`List[float]`): Vector representing the state
- **timestamp** (`float`, optional): Timestamp (defaults to current time)

**Returns**:
- `int`: Index of the added state

#### `compute_alignment_score`

```python
compute_alignment_score(state: List[float]) -> float
```

Compute how aligned a state is with the defined alignment region.

**Parameters**:
- **state** (`List[float]`): Vector representing the state

**Returns**:
- `float`: Alignment score (0.0 to 1.0)

#### `compute_similarity`

```python
compute_similarity(vec1: List[float], vec2: List[float]) -> float
```

Compute similarity between two vectors (cosine similarity).

**Parameters**:
- **vec1** (`List[float]`): First vector
- **vec2** (`List[float]`): Second vector

**Returns**:
- `float`: Similarity score (-1.0 to 1.0)

#### `analyze_transition`

```python
analyze_transition(state1_idx: int, state2_idx: int) -> Dict[str, Any]
```

Analyze the transition between two states.

**Parameters**:
- **state1_idx** (`int`): Index of the first state
- **state2_idx** (`int`): Index of the second state

**Returns**:
- `Dict[str, Any]`: Dictionary with transition analysis

## Transition Analysis

### `analyse_transition`

```python
analyse_transition(space: AlignmentVectorSpace, state1_idx: int, state2_idx: int) -> Dict[str, Any]
```

Analyze the transition between two states.

**Parameters**:
- **space** (`AlignmentVectorSpace`): The AlignmentVectorSpace containing the states
- **state1_idx** (`int`): Index of the first state
- **state2_idx** (`int`): Index of the second state

**Returns**:
- `Dict[str, Any]`: Dictionary with transition analysis including:
  - `state1_idx`: Index of the first state
  - `state2_idx`: Index of the second state
  - `time_delta`: Time difference between states
  - `similarity`: Similarity between states
  - `alignment1`: Alignment score of first state
  - `alignment2`: Alignment score of second state
  - `alignment_change`: Change in alignment
  - `transition_magnitude`: Magnitude of the transition vector
  - `toward_aligned_region`: Whether the transition is toward the aligned region
  - `angle_to_aligned`: Angle between transition vector and vector to aligned region

### `predict_trajectory`

```python
predict_trajectory(space: AlignmentVectorSpace, start_state_idx: int, steps: int = 5) -> List[Dict[str, Any]]
```

Predict future trajectory based on recent transitions.

**Parameters**:
- **space** (`AlignmentVectorSpace`): The AlignmentVectorSpace containing the states
- **start_state_idx** (`int`): Index of starting state
- **steps** (`int`, default=5): Number of prediction steps

**Returns**:
- `List[Dict[str, Any]]`: List of predicted future states and their metrics

### State-Transition Calculus (STC) Functions

#### `compute_activation`

```python
compute_activation(state_value: float, time_delta: float, memory_decay: float = 0.2) -> float
```

Compute the activation function φ(a_i, t) from State-Transition Calculus.

**Parameters**:
- **state_value** (`float`): The value of the state (alignment score)
- **time_delta** (`float`): Time since the state was observed
- **memory_decay** (`float`, default=0.2): Memory decay rate (τ)

**Returns**:
- `float`: Activation value

#### `compute_residual_potentiality`

```python
compute_residual_potentiality(state: List[float], perturbation_magnitude: float = 0.1) -> Dict[str, Any]
```

Compute the residual potentiality b(a_res) from State-Transition Calculus.
This involves applying a perturbation to the state and assessing its impact.

**Parameters**:
- **state** (`List[float]`): The state vector
- **perturbation_magnitude** (`float`, default=0.1): Magnitude of perturbation to apply

**Returns**:
- `Dict[str, Any]`: A dictionary containing:
  - `original_state`: The input state vector
  - `perturbed_state`: The state vector after perturbation and normalization
  - `perturbation_vector`: The random noise vector applied
  - `potentiality_score`: A measure of how much the perturbation shifted the state (higher means more shift)

## Metrics

### `calculate_stability_metrics`

```python
calculate_stability_metrics(space: AlignmentVectorSpace) -> Dict[str, Any]
```

Calculate stability metrics for the system trajectory.

**Parameters**:
- **space** (`AlignmentVectorSpace`): The AlignmentVectorSpace containing the states

**Returns**:
- `Dict[str, Any]`: Dictionary of stability metrics including:
  - `avg_alignment`: Average alignment score
  - `min_alignment`: Minimum alignment score
  - `max_alignment`: Maximum alignment score
  - `alignment_volatility`: Volatility of alignment scores
  - `avg_transition_magnitude`: Average magnitude of transitions
  - `alignment_trend`: Trend in alignment scores
  - `region_transitions`: Number of transitions across the alignment threshold
  - `stability_score`: Overall stability score (0.0 to 1.0)

### `evaluate_alignment_robustness`

```python
evaluate_alignment_robustness(space: AlignmentVectorSpace, perturbation_magnitude: float = 0.1, num_perturbations: int = 10) -> Dict[str, Any]
```

Evaluate robustness of alignment to perturbations.

**Parameters**:
- **space** (`AlignmentVectorSpace`): The AlignmentVectorSpace containing the states
- **perturbation_magnitude** (`float`, default=0.1): Size of random perturbations
- **num_perturbations** (`int`, default=10): Number of random perturbations to test

**Returns**:
- `Dict[str, Any]`: Dictionary of robustness metrics including:
  - `base_alignment`: Alignment score of the original state
  - `avg_change`: Average change in alignment due to perturbations
  - `max_negative_change`: Maximum decrease in alignment
  - `max_positive_change`: Maximum increase in alignment
  - `change_variance`: Variance of alignment changes
  - `robustness_score`: Overall robustness score (0.0 to 1.0)
  - `psd_distance`: Power Spectral Density distance

### `calculate_psd_distance`

```python
calculate_psd_distance(space: AlignmentVectorSpace) -> float
```

Calculate the Power Spectral Density (PSD) distance between the current state trajectory and an aligned trajectory.

**Parameters**:
- **space** (`AlignmentVectorSpace`): The AlignmentVectorSpace containing the states

**Returns**:
- `float`: PSD distance (0.0 to 1.0)

## Optimization

### `AlignmentOptimizer`

```python
class AlignmentOptimizer(states=None, config=None, debug=False)
```

Optimize alignment state transitions using QUBO formulation.

**Parameters**:
- **states** (`List[Dict[str, Any]]`, optional): List of state dictionaries with embeddings and metadata
- **config** (`Dict[str, Any]`, optional): Configuration dictionary
- **debug** (`bool`, default=False): Whether to enable debug logging

#### Methods

##### `generate_costs`

```python
generate_costs(phi_scores, psd_scores, context_info=None) -> List[Dict[str, Any]]
```

Generate cost values for each state and transition.

**Parameters**:
- **phi_scores** (`Dict[str, float]`): Dictionary mapping state IDs to alignment scores
- **psd_scores** (`Dict[str, float]`): Dictionary mapping state IDs to PSD distance scores
- **context_info** (`Dict[str, Any]`, optional): Optional context information

**Returns**:
- `List[Dict[str, Any]]`: List of cost dictionaries

##### `build_qubo`

```python
build_qubo(phi_scores, psd_scores, context_info=None) -> Dict[Tuple[Any, Any], float]
```

Build QUBO for alignment optimization.

**Parameters**:
- **phi_scores** (`Dict[str, float]`): Dictionary mapping state IDs to alignment scores
- **psd_scores** (`Dict[str, float]`): Dictionary mapping state IDs to PSD distance scores
- **context_info** (`Dict[str, Any]`, optional): Optional context information

**Returns**:
- `Dict[Tuple[Any, Any], float]`: QUBO dictionary mapping variable tuples to weights

##### `solve_qubo`

```python
solve_qubo(Q, num_reads=1000)
```

Solve the QUBO problem using available solver.

**Parameters**:
- **Q** (`Dict[Tuple[Any, Any], float]`): QUBO dictionary
- **num_reads** (`int`, default=1000): Number of reads for the solver

**Returns**:
- `Dict[str, Any]`: Dictionary with solution information

##### `decode_solution`

```python
decode_solution(solution_dict)
```

Decode the QUBO solution into a path through states.

**Parameters**:
- **solution_dict** (`Dict[str, Any]`): Dictionary with solution information

**Returns**:
- `Dict[str, Any]`: Dictionary with decoded path and metrics

##### `optimize`

```python
optimize(phi_scores, psd_scores, context_info=None, num_reads=1000)
```

Run the full optimization process.

**Parameters**:
- **phi_scores** (`Dict[str, float]`): Dictionary mapping state IDs to alignment scores
- **psd_scores** (`Dict[str, float]`): Dictionary mapping state IDs to PSD distance scores
- **context_info** (`Dict[str, Any]`, optional): Optional context information
- **num_reads** (`int`, default=1000): Number of reads for the solver

**Returns**:
- `Dict[str, Any]`: Dictionary with optimization results

### `GraphEnhancedAlignmentOptimizer`

```python
class GraphEnhancedAlignmentOptimizer(graph_manager, *args, **kwargs)
```

Graph-enhanced alignment optimizer that uses a consequence graph to bias the QUBO toward high-alignment paths.

**Parameters**:
- **graph_manager** (`GraphManager`): Graph manager for accessing the consequence graph
- **\*args, \*\*kwargs**: Arguments to pass to parent class (`AlignmentOptimizer`)

#### Methods

##### `build_qubo`

```python
build_qubo(phi_scores, psd_scores, context_info=None)
```

Build QUBO with graph enhancements.

**Parameters**:
- **phi_scores** (`Dict[str, float]`): Dictionary mapping state IDs to alignment scores
- **psd_scores** (`Dict[str, float]`): Dictionary mapping state IDs to PSD distance scores
- **context_info** (`Dict[str, Any]`, optional): Optional context information

**Returns**:
- `Dict[Tuple[Any, Any], float]`: Enhanced QUBO dictionary
