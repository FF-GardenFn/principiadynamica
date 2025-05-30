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
