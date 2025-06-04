# Integrations API Reference

The `constitutional_dynamics.integrations` module provides integrations with external systems for the Constitutional Dynamics package, including Neo4j graph database, D-Wave quantum annealer, and LLM strategist.

## Graph Integration

The `constitutional_dynamics.integrations.graph` module provides a graph manager for storing and querying alignment states and transitions in a Neo4j graph database.

### `GraphManager`

```python
class GraphManager(uri: str, auth: Optional[Tuple[str, str]] = None, database: str = "neo4j")
```

Manager for storing and querying alignment states and transitions in a Neo4j graph database.

**Parameters**:
- **uri** (`str`): Neo4j connection URI (e.g., bolt://localhost:7687)
- **auth** (`Tuple[str, str]`, optional): Optional tuple of (username, password)
- **database** (`str`, default="neo4j"): Neo4j database name

#### Methods

##### `add_state`

```python
add_state(state_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> bool
```

Add a state node to the graph.

**Parameters**:
- **state_id** (`str`): Unique identifier for the state
- **embedding** (`List[float]`): Vector embedding of the state
- **metadata** (`Dict[str, Any]`, optional): Optional metadata for the state

**Returns**:
- `bool`: Success status

##### `add_transition`

```python
add_transition(from_state_id: str, to_state_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool
```

Add a transition edge between two states.

**Parameters**:
- **from_state_id** (`str`): ID of the source state
- **to_state_id** (`str`): ID of the target state
- **metadata** (`Dict[str, Any]`, optional): Optional metadata for the transition

**Returns**:
- `bool`: Success status

##### `get_state`

```python
get_state(state_id: str) -> Optional[Dict[str, Any]]
```

Get a state from the graph.

**Parameters**:
- **state_id** (`str`): ID of the state to retrieve

**Returns**:
- `Dict[str, Any]` or `None`: State data or None if not found

##### `get_transitions`

```python
get_transitions(state_id: str, direction: str = "outgoing") -> List[Dict[str, Any]]
```

Get transitions for a state.

**Parameters**:
- **state_id** (`str`): ID of the state
- **direction** (`str`, default="outgoing"): 'outgoing', 'incoming', or 'both'

**Returns**:
- `List[Dict[str, Any]]`: List of transitions

##### `get_alignment_impact`

```python
get_alignment_impact(state_id: str) -> float
```

Calculate the alignment impact of a state based on its transitions.

**Parameters**:
- **state_id** (`str`): ID of the state

**Returns**:
- `float`: Alignment impact score

##### `get_transition_strength`

```python
get_transition_strength(from_state_id: str, to_state_id: str) -> float
```

Get the strength of a transition between two states.

**Parameters**:
- **from_state_id** (`str`): ID of the source state
- **to_state_id** (`str`): ID of the target state

**Returns**:
- `float`: Transition strength

##### `get_aligned_states`

```python
get_aligned_states(threshold: float = 0.7, limit: int = 100) -> List[str]
```

Get states with alignment score above threshold.

**Parameters**:
- **threshold** (`float`, default=0.7): Alignment score threshold
- **limit** (`int`, default=100): Maximum number of states to return

**Returns**:
- `List[str]`: List of state IDs

##### `close`

```python
close()
```

Close the Neo4j connection.

### `create_graph_manager`

```python
create_graph_manager(uri: str, auth: Optional[Tuple[str, str]] = None, database: str = "neo4j") -> Optional[GraphManager]
```

Create a new GraphManager instance.

**Parameters**:
- **uri** (`str`): Neo4j connection URI
- **auth** (`Tuple[str, str]`, optional): Optional tuple of (username, password)
- **database** (`str`, default="neo4j"): Neo4j database name

**Returns**:
- `GraphManager` or `None`: GraphManager instance or None if Neo4j is not available

## Quantum Integration

The `constitutional_dynamics.integrations.quantum` module provides a wrapper for the D-Wave quantum annealer, with fallbacks to classical simulation when quantum hardware is not available.

### `QuantumAnnealer`

```python
class QuantumAnnealer(simulation_mode: Optional[bool] = None, annealer_type: str = "dwave", token: Optional[str] = None, endpoint: Optional[str] = None, solver_name: Optional[str] = None, verbose: bool = False)
```

Wrapper for quantum annealing solvers with classical simulation fallbacks.

**Parameters**:
- **simulation_mode** (`bool`, optional): Force simulation mode (None = auto-detect)
- **annealer_type** (`str`, default="dwave"): Type of annealer ("dwave" or "qaoa")
- **token** (`str`, optional): D-Wave API token
- **endpoint** (`str`, optional): D-Wave API endpoint
- **solver_name** (`str`, optional): D-Wave solver name
- **verbose** (`bool`, default=False): Enable verbose logging

#### Methods

##### `solve_qubo`

```python
solve_qubo(Q: Dict[Tuple[Any, Any], float], num_reads: int = 1000, annealing_time: Optional[float] = None, chain_strength: Optional[float] = None, sa_schedule: str = "geometric", initial_temperature: Optional[float] = None, **kwargs) -> Dict[str, Any]
```

Solve a QUBO problem using quantum annealing or simulation.

**Parameters**:
- **Q** (`Dict[Tuple[Any, Any], float]`): QUBO dictionary mapping variable tuples to weights
- **num_reads** (`int`, default=1000): Number of samples to collect
- **annealing_time** (`float`, optional): Annealing time in microseconds (D-Wave only)
- **chain_strength** (`float`, optional): Chain strength for embedding (D-Wave only)
- **sa_schedule** (`str`, default="geometric"): Schedule type for simulated annealing
- **initial_temperature** (`float`, optional): Initial temperature for simulated annealing
- **\*\*kwargs**: Additional solver-specific parameters

**Returns**:
- `Dict[str, Any]`: Dictionary with solution information including:
  - `samples`: List of sample dictionaries
  - `num_reads`: Number of reads performed
  - `best_solution`: Best solution found
  - `best_energy`: Energy of the best solution
  - `solver`: Solver used ("dwave", "qaoa", or "simulated_annealing")
  - `solve_time`: Time taken to solve the problem

### `create_annealer`

```python
create_annealer(simulation_mode=None, annealer_type="dwave", token=None, endpoint=None, solver_name=None, verbose=False) -> QuantumAnnealer
```

Create a new QuantumAnnealer instance.

**Parameters**:
- **simulation_mode** (`bool`, optional): Force simulation mode (None = auto-detect)
- **annealer_type** (`str`, default="dwave"): Type of annealer ("dwave" or "qaoa")
- **token** (`str`, optional): D-Wave API token
- **endpoint** (`str`, optional): D-Wave API endpoint
- **solver_name** (`str`, optional): D-Wave solver name
- **verbose** (`bool`, default=False): Enable verbose logging

**Returns**:
- `QuantumAnnealer`: QuantumAnnealer instance

## Strategist Integration

The `constitutional_dynamics.integrations.strategist` module provides a MetaStrategist class that generates and evaluates alignment strategies using LLMs.

### `StrategyResult`

```python
@dataclass
class StrategyResult:
    strategy_id: str
    title: str
    description: str
    steps: List[str]
    confidence: float
    tags: List[str]
    metadata: Dict[str, Any]
```

Container for strategy generation results.

### `LLMInterface`

```python
class LLMInterface(provider: str = "anthropic", api_key: Optional[str] = None, model: Optional[str] = None)
```

Interface for interacting with Language Models.

**Parameters**:
- **provider** (`str`, default="anthropic"): LLM provider ("anthropic", "openai", etc.)
- **api_key** (`str`, optional): API key for the provider
- **model** (`str`, optional): Model name to use

#### Methods

##### `generate`

```python
generate(prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str
```

Generate text using the LLM.

**Parameters**:
- **prompt** (`str`): Prompt to send to the LLM
- **max_tokens** (`int`, default=1000): Maximum number of tokens to generate
- **temperature** (`float`, default=0.7): Temperature for generation

**Returns**:
- `str`: Generated text

### `MetaStrategist`

```python
class MetaStrategist(llm: Optional[LLMInterface] = None, strategy_prompt: Optional[str] = None, refinement_prompt: Optional[str] = None)
```

AI-powered strategy generator and evaluator for alignment.

**Parameters**:
- **llm** (`LLMInterface`, optional): LLM interface for strategy generation
- **strategy_prompt** (`str`, optional): Template for strategy generation
- **refinement_prompt** (`str`, optional): Template for strategy refinement

#### Methods

##### `generate_strategy`

```python
generate_strategy(context: Dict[str, Any], metrics: Optional[Dict[str, float]] = None, constraints: Optional[Dict[str, Any]] = None, num_candidates: int = 1) -> StrategyResult
```

Generate an alignment strategy based on context and metrics.

**Parameters**:
- **context** (`Dict[str, Any]`): Operational context for strategy generation
- **metrics** (`Dict[str, float]`, optional): Alignment metrics to address
- **constraints** (`Dict[str, Any]`, optional): Optional constraints to apply
- **num_candidates** (`int`, default=1): Number of candidate strategies to generate

**Returns**:
- `StrategyResult`: StrategyResult containing the generated strategy

##### `refine_strategy`

```python
refine_strategy(strategy: StrategyResult, feedback: str, context: Optional[Dict[str, Any]] = None) -> StrategyResult
```

Refine an existing strategy based on feedback.

**Parameters**:
- **strategy** (`StrategyResult`): Existing strategy to refine
- **feedback** (`str`): Feedback to incorporate
- **context** (`Dict[str, Any]`, optional): Optional updated context

**Returns**:
- `StrategyResult`: Refined StrategyResult

### `create_strategist`

```python
create_strategist(provider: str = "anthropic", api_key: Optional[str] = None, model: Optional[str] = None) -> MetaStrategist
```

Create a new MetaStrategist instance.

**Parameters**:
- **provider** (`str`, default="anthropic"): LLM provider ("anthropic", "openai", etc.)
- **api_key** (`str`, optional): API key for the provider
- **model** (`str`, optional): Model name to use

**Returns**:
- `MetaStrategist`: MetaStrategist instance

## Circuit Tracer Bridge

The `constitutional_dynamics.integrations.circuit_tracer_bridge` module provides a bridge between Constitutional Dynamics' alignment monitoring capabilities and Anthropic's Circuit Tracer mechanistic interpretability tools. It introduces key components like the AlignmentThermostat to orchestrate this interaction, specialized monitors, and an intervention framework.

### `AlignmentThermostat`

```python
class AlignmentThermostat(cd_monitor_instance: Any, circuit_tracer_instance: Any, model_interface: Any, threshold: float = 0.7, stability_weight: float = 0.3, auto_stabilize: bool = True)
```

A feedback loop system that combines Constitutional Dynamics' alignment monitoring with Circuit Tracer's mechanistic interpretability to detect and correct alignment issues.

**Parameters**:
- **cd_monitor_instance** (`Any`): An instance of Constitutional Dynamics' monitoring system (typically an AlignmentVectorSpace or similar)
- **circuit_tracer_instance** (`Any`): An instance of Circuit Tracer for mechanistic analysis
- **model_interface** (`Any`): Interface to the model for applying interventions
- **threshold** (`float`, default=0.7): Alignment score threshold below which interventions are triggered
- **stability_weight** (`float`, default=0.3): Weight given to stability metrics in activation probability
- **auto_stabilize** (`bool`, default=True): Whether to automatically stabilize the system based on drift detection

#### Methods

##### `run_feedback_loop`

```python
run_feedback_loop(current_model_output_embedding: List[float], original_prompt_for_trace: str, content_type: str = "default") -> Dict[str, Any]
```

Execute the complete feedback loop from monitoring to intervention verification.

**Parameters**:
- **current_model_output_embedding** (`List[float]`): Embedding of the current model output
- **original_prompt_for_trace** (`str`): The original prompt that generated the output
- **content_type** (`str`, default="default"): Type of content being evaluated (for threshold selection)

**Returns**:
- `Dict[str, Any]`: Dictionary containing results of the feedback loop execution

##### `calculate_modulated_activation_probability`

```python
calculate_modulated_activation_probability(base_probability: float, state: List[float], subset_weight: float = 1.0, memory_factor: Optional[float] = None, environment_factor: Optional[float] = None) -> float
```

Calculate activation probability modulated by stability metrics.

**Parameters**:
- **base_probability** (`float`): Base activation probability
- **state** (`List[float]`): Current state vector
- **subset_weight** (`float`, default=1.0): Weight of the state subset (W')
- **memory_factor** (`float`, optional): Memory factor (M(t))
- **environment_factor** (`float`, optional): Environment factor (E(t))

**Returns**:
- `float`: Modulated activation probability

### `CircuitTracerMonitorAdapter`

```python
class CircuitTracerMonitorAdapter(cd_monitor_instance: Any, threshold_config: Optional[Dict[str, float]] = None, history_window_size: int = 10)
```

Adapter class that connects Constitutional Dynamics' monitoring capabilities with the Circuit Tracer Bridge.

**Parameters**:
- **cd_monitor_instance** (`Any`): An instance of a Constitutional Dynamics monitor (typically AlignmentVectorSpace or similar)
- **threshold_config** (`Dict[str, float]`, optional): Dictionary mapping content types to threshold values
- **history_window_size** (`int`, default=10): Number of recent scores to keep for trend analysis

#### Methods

##### `detect_alignment_issues`

```python
detect_alignment_issues(embedding: List[float], content_type: str = "default", prompt: Optional[str] = None, response: Optional[str] = None) -> Dict[str, Any]
```

Detect potential alignment issues in the model output.

**Parameters**:
- **embedding** (`List[float]`): Vector representation of model output
- **content_type** (`str`, default="default"): Type of content being evaluated
- **prompt** (`str`, optional): Original prompt text (optional, for context-aware detection)
- **response** (`str`, optional): Model response text (optional, for text-based heuristics)

**Returns**:
- `Dict[str, Any]`: Dictionary containing detection results including:
  - `score`: Alignment score of the embedding
  - `threshold`: Threshold used for evaluation
  - `below_threshold`: Boolean indicating if score is below threshold
  - `drift_detected`: Boolean indicating if drift was detected
  - `issue_types`: List of detected issue types (e.g., "jailbreak", "harmful_content")
  - `requires_intervention`: Boolean indicating if intervention is needed

### `JailbreakDetectionMonitor`

```python
class JailbreakDetectionMonitor(cd_monitor_instance: Any, **kwargs)
```

Specialized monitor for detecting jailbreak attempts.

**Parameters**:
- **cd_monitor_instance** (`Any`): An instance of a Constitutional Dynamics monitor
- **\*\*kwargs**: Additional arguments for the base adapter

#### Methods

##### `detect_jailbreak_attempt`

```python
detect_jailbreak_attempt(embedding: List[float], prompt: Optional[str] = None) -> Dict[str, Any]
```

Specialized method for detecting jailbreak attempts.

**Parameters**:
- **embedding** (`List[float]`): Vector representation of model output
- **prompt** (`str`, optional): Original prompt text (optional, for heuristic detection)

**Returns**:
- `Dict[str, Any]`: Dictionary with comprehensive jailbreak detection results including:
  - `jailbreak_detected`: Boolean indicating if a jailbreak attempt was detected
  - `jailbreak_score`: Numerical score indicating jailbreak likelihood (0-1)
  - `jailbreak_type`: Type of jailbreak attempt detected (if any)
  - `confidence`: Confidence level in the detection (0-1)
  - `requires_intervention`: Boolean indicating if intervention is needed
  - `suggested_intervention`: Suggested intervention type if applicable

### `DriftDetectionMonitor`

```python
class DriftDetectionMonitor(cd_monitor_instance: Any, drift_threshold: float = 0.1, window_sizes: List[int] = [5, 10, 20], **kwargs)
```

Specialized monitor for detecting alignment drift over time.

**Parameters**:
- **cd_monitor_instance** (`Any`): An instance of a Constitutional Dynamics monitor
- **drift_threshold** (`float`, default=0.1): Threshold for considering a change as drift
- **window_sizes** (`List[int]`, default=[5, 10, 20]): List of window sizes for multi-scale drift detection
- **\*\*kwargs**: Additional arguments for the base adapter

#### Methods

##### `detect_drift`

```python
detect_drift(current_embedding: List[float], content_type: str = "default") -> Dict[str, Any]
```

Detect alignment drift across multiple time scales.

**Parameters**:
- **current_embedding** (`List[float]`): Vector representation of current model output
- **content_type** (`str`, default="default"): Type of content being evaluated

**Returns**:
- `Dict[str, Any]`: Dictionary with comprehensive drift detection results including:
  - `drift_detected`: Boolean indicating if drift was detected
  - `current_score`: Current alignment score
  - `max_drift_magnitude`: Maximum magnitude of drift across all time windows
  - `window_results`: Dictionary with drift results for each time window
  - `basic_drift_detected`: Boolean indicating if basic drift detection found drift
  - `advanced_drift_detected`: Boolean indicating if advanced metrics detected drift
  - `requires_intervention`: Boolean indicating if intervention is needed
  - `advanced_metrics`: Additional metrics from stability analysis

### `CircuitTracerIntervention`

```python
class CircuitTracerIntervention(model_interface: Any, circuit_tracer_instance: Any, intervention_strength: float = 0.5, max_features_to_modify: int = 5, stability_aware: bool = True)
```

Base class for interventions that leverage Circuit Tracer's mechanistic insights.

**Parameters**:
- **model_interface** (`Any`): Interface to the model for applying interventions
- **circuit_tracer_instance** (`Any`): Instance of Circuit Tracer for mechanistic analysis
- **intervention_strength** (`float`, default=0.5): Strength of the intervention (0.0 to 1.0)
- **max_features_to_modify** (`int`, default=5): Maximum number of features to modify in a single intervention
- **stability_aware** (`bool`, default=True): Whether to take stability metrics into account when applying interventions

#### Methods

##### `apply`

```python
apply(circuit_analysis_result: Dict[str, Any], issue_type: Optional[str] = None, stability_metrics: Optional[Dict[str, Any]] = None, modulated_probabilities: Optional[List[float]] = None) -> Dict[str, Any]
```

Apply an intervention based on circuit analysis results.

**Parameters**:
- **circuit_analysis_result** (`Dict[str, Any]`): Results from Circuit Tracer analysis
- **issue_type** (`str`, optional): Type of alignment issue detected
- **stability_metrics** (`Dict[str, Any]`, optional): Stability metrics from the AlignmentThermostat
- **modulated_probabilities** (`List[float]`, optional): Activation probabilities modulated by stability

**Returns**:
- `Dict[str, Any]`: Dictionary with intervention results including:
  - `success`: Boolean indicating if the intervention was successfully applied
  - `intervention_type`: Type of intervention applied
  - `modified_features`: List of features that were modified
  - `modification_strengths`: List of strengths used for each modification
  - `stability_modulated`: Boolean indicating if stability metrics were used
  - `expected_impact`: Estimated impact of the intervention on alignment
  - `metadata`: Additional intervention-specific metadata

### `InterventionFactory`

```python
class InterventionFactory
```

Factory class for creating appropriate interventions based on issue type.

#### Methods

##### `create_intervention`

```python
@staticmethod
create_intervention(issue_type: str, model_interface: Any, circuit_tracer_instance: Any, **kwargs) -> CircuitTracerIntervention
```

Create an appropriate intervention based on the issue type.

**Parameters**:
- **issue_type** (`str`): Type of alignment issue detected (e.g., "jailbreak", "harmful_content", "drift", "bias")
- **model_interface** (`Any`): Interface to the model for applying interventions
- **circuit_tracer_instance** (`Any`): Instance of Circuit Tracer for mechanistic analysis
- **\*\*kwargs**: Additional configuration for the intervention, including:
  - `intervention_strength` (`float`, optional): Strength of the intervention (0.0 to 1.0)
  - `max_features_to_modify` (`int`, optional): Maximum number of features to modify
  - `stability_aware` (`bool`, optional): Whether to consider stability metrics
  - `intervention_specific_params` (`Dict[str, Any]`, optional): Parameters specific to the intervention type

**Returns**:
- `CircuitTracerIntervention`: Configured intervention instance specialized for the detected issue type

## Attribution

This integration acknowledges and builds upon the Circuit Tracer work by Anthropic. The original Circuit Tracer can be found at [safety-research/circuit-tracer](https://github.com/safety-research/circuit-tracer).

Core feedback loop architecture inspired by Anthropic, circuit-tracer team, and recent interpretability research (Ameisen et al. 2025, Lindsey et al. 2025).
