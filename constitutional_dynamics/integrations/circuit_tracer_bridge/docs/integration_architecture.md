# Integration Architecture: Constitutional Dynamics + Circuit Tracer

This document provides a comprehensive technical overview of the architecture integrating Constitutional Dynamics' alignment monitoring framework with Anthropic's Circuit Tracer mechanistic interpretability tools, creating a novel "Alignment Thermostat" system.

## Architectural Overview

The integration establishes a bidirectional feedback loop between behavioral monitoring (Constitutional Dynamics) and mechanistic analysis (Circuit Tracer), enabling more effective alignment interventions through a multi-level feedback system.

```
┌─────────────────────┐      ┌─────────────────────┐
│                     │      │                     │
│   Constitutional    │      │   Circuit Tracer    │
│     Dynamics        │      │                     │
│                     │      │                     │
└─────────┬───────────┘      └─────────┬───────────┘
          │                            │
          │ Detects                    │ Provides
          │ Alignment                  │ Mechanistic
          │ Issues                     │ Insights
          ▼                            ▼
┌─────────────────────────────────────────────────┐
│                                                 │
│             Alignment Thermostat                │
│                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │ Monitor │ -> │ Analyze │ -> │Intervene│      │
│  └─────────┘    └─────────┘    └─────────┘      │
│        ^                            │           │
│        └────────────────────────────┘           │
│              Feedback Loop                      │
└─────────────────────────────────────────────────┘
          │                            ▲
          │                            │
          ▼                            │
┌─────────────────────┐      ┌─────────────────────┐
│                     │      │                     │
│    Language Model   │ ──── │  Improved Output    │
│                     │      │                     │
└─────────────────────┘      └─────────────────────┘
```

## Theoretical Framework

The integration is grounded in two complementary theoretical frameworks:

1. **State-Transition Calculus (STC)** from Constitutional Dynamics, which models alignment as trajectories through embedding space with:
   - Alignment scores (φ) based on vector similarity to defined aligned regions
   - Residual potentiality (b(a_res)) capturing latent alignment risks
   - Stability metrics including Lyapunov exponent estimates for drift detection

2. **Mechanistic Interpretability** from Circuit Tracer, which provides:
   - Attribution graphs mapping internal model components to outputs
   - Feature-level understanding of model behavior
   - Targeted intervention capabilities at the circuit level

The novel contribution of this integration is the creation of a stability-modulated activation probability system that implements an "Alignment Thermostat" capable of self-regulation.

## Component Architecture

### 1. AlignmentThermostat

The central orchestration component (`AlignmentThermostat` class) implements a closed-loop control system with the following key methods:

- `run_feedback_loop()`: Main entry point that orchestrates the entire process
- `calculate_modulated_activation_probability()`: Implements the core stability-modulation algorithm
- `update_stability_metrics()`: Calculates Lyapunov exponent estimates and other stability metrics
- `identify_intervention_targets()`: Translates circuit analysis into actionable targets
- `apply_intervention()`: Executes targeted modifications to model behavior
- `verify_improvement()`: Validates intervention effectiveness

The thermostat maintains internal state including:
- Intervention history
- Stability metrics history
- Current Lyapunov exponent estimate

### 2. Monitor Adapters

The monitoring components (`CircuitTracerMonitorAdapter` and subclasses) provide:

- Issue detection capabilities (`detect_alignment_issues()`, `detect_jailbreak_attempt()`, `detect_drift()`)
- Circuit targeting recommendations (`get_issue_specific_circuit_targets()`)
- Adaptive monitoring parameters (`update_monitoring_focus()`)

Specialized monitors include:
- `JailbreakDetectionMonitor`: Enhanced detection of jailbreak attempts using pattern matching and residual potentiality
- `DriftDetectionMonitor`: Multi-scale drift detection with trajectory prediction

### 3. Intervention Framework

The intervention components (`CircuitTracerIntervention` and subclasses) implement:

- Feature-level interventions based on circuit analysis
- Stability-aware intervention strength modulation
- Strategic context preparation for higher-level reasoning

Intervention types include:
- `FeatureSuppressionIntervention`: Suppresses specific features contributing to misalignment
- `FeatureAmplificationIntervention`: Amplifies features promoting aligned behavior
- `JailbreakMitigationIntervention`: Specialized for jailbreak attempts
- `DriftCorrectionIntervention`: Specialized for alignment drift

### 4. MetaStrategist Integration

The `MetaStrategist` component provides higher-level strategic reasoning:

- Receives context from interventions via `_prepare_strategist_context()`
- Generates strategic recommendations using LLM-based reasoning
- Provides adaptive monitoring parameters to the monitor via `update_monitoring_focus()`

## Implementation Details

### Stability-Modulated Activation

The core innovation is the integration of stability metrics directly into activation probabilities:

```python
def calculate_modulated_activation_probability(
        self,
        base_probability: float,
        state: List[float],
        subset_weight: float = 1.0,
        memory_factor: Optional[float] = None,
        environment_factor: Optional[float] = None
) -> float:
    # Calculate base activation probability
    activation_prob = compute_activation_probability(
        subset_weight=subset_weight,
        state=state,
        memory=memory_factor,
        environment=environment_factor
    )

    # Modulate based on Lyapunov estimate
    stability_factor = 1.0
    if self.lyapunov_estimate > 0:
        stability_factor = math.exp(-self.stability_weight * self.lyapunov_estimate)

    # Apply stability modulation
    modulated_prob = activation_prob * stability_factor
    return max(0.0, min(1.0, modulated_prob))
```

This creates a homeostatic system where:
- Higher instability (larger Lyapunov exponent) → Lower activation probabilities
- Lower instability (smaller Lyapunov exponent) → Higher activation probabilities

### Multi-Level Feedback System

The implementation creates two feedback loops:

1. **Inner Loop (Tactical)**: 
   - Operates on individual model outputs
   - Uses Circuit Tracer for mechanistic analysis
   - Applies targeted interventions
   - Verifies immediate improvement

2. **Outer Loop (Strategic)**:
   - Operates across multiple interactions
   - Uses MetaStrategist for higher-level reasoning
   - Adapts monitoring parameters
   - Guides intervention selection

### Data Flow Implementation

The data flow through the system is implemented as follows:

#### 1. Monitoring Phase

```python
# From AlignmentThermostat.run_feedback_loop()
alignment_score = self.monitor.compute_alignment_score(current_model_output_embedding)
stability_metrics = self.update_stability_metrics(current_model_output_embedding)
lyapunov_estimate = stability_metrics.get("lyapunov_exponent_estimate", 0.0)

# Determine if intervention is needed
intervention_needed = False
if alignment_score < self.threshold:
    intervention_needed = True
    intervention_reason = "alignment_below_threshold"
if self.auto_stabilize and lyapunov_estimate > 0.5:
    intervention_needed = True
    intervention_reason = intervention_reason or "high_instability"
```

The monitoring phase leverages Constitutional Dynamics' vector space operations to:
1. Compute alignment scores using cosine similarity to aligned regions
2. Calculate stability metrics including Lyapunov exponent estimates
3. Determine intervention necessity based on thresholds and stability

#### 2. Analysis Phase

```python
# Circuit Tracer analysis (simplified)
graph = attribute(
    prompt=original_prompt_for_trace,
    model=self.tracer,
    verbose=True,
    max_n_logits=10,
    desired_logit_prob=0.95
)

# Prune graph to identify critical features
prune_result = prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)

# Extract critical features
critical_features = []
for idx in influential_feature_indices:
    feature_info = graph.active_features[idx]
    layer, pos, feature_idx = feature_info.tolist()
    critical_features.append(f"layer_{layer}_pos_{pos}_feature_{feature_idx}")
```

The analysis phase utilizes Circuit Tracer's attribution graph capabilities to:
1. Generate an attribution graph mapping model components to outputs
2. Prune the graph to identify the most influential components
3. Extract specific features for targeted intervention

#### 3. Intervention Phase

```python
# Calculate modulated activation probabilities
modulated_probabilities = []
for i, feature in enumerate(intervention_targets["suppress_features"]):
    # Base probability decreases with feature index
    base_prob = 1.0 - (i / len(intervention_targets["suppress_features"]))

    # Modulate based on stability
    modulated_prob = self.calculate_modulated_activation_probability(
        base_probability=base_prob,
        state=current_model_output_embedding,
        subset_weight=1.0 - (0.1 * lyapunov_estimate)
    )
    modulated_probabilities.append(modulated_prob)

# Apply intervention with modulated probabilities
intervention_success = self.apply_intervention(
    intervention_targets,
    stability_metrics=stability_metrics
)
```

The intervention phase implements the stability-modulated activation system to:
1. Calculate activation probabilities for each intervention target
2. Modulate these probabilities based on the Lyapunov exponent estimate
3. Apply interventions with appropriate strengths based on modulated probabilities

#### 4. Verification Phase

```python
# Verify improvement
verification_result = self.verify_improvement(
    current_model_output_embedding,
    new_output_embedding
)

# Calculate new stability metrics
new_stability_metrics = verification_result.get("stability_metrics", {})
new_lyapunov_estimate = new_stability_metrics.get("lyapunov_exponent_estimate", 0.0)

# Check if stability improved
stability_improved = new_lyapunov_estimate < lyapunov_estimate
```

The verification phase evaluates intervention effectiveness by:
1. Computing alignment scores before and after intervention
2. Calculating stability metrics to assess system stability
3. Determining overall improvement based on both alignment and stability

#### 5. Strategic Adaptation Phase

```python
# Check if we should consult the MetaStrategist
if self.enable_strategist and strategist_context:
    # Create strategist if not exists
    if not self.strategist:
        self.strategist = create_strategist()

    # Generate strategy recommendation
    strategy_recommendation = self.strategist.generate_strategy(
        context=strategist_context,
        metrics={
            "alignment_score": verification_result["new_score"],
            "improvement_margin": verification_result["improvement_margin"],
            "lyapunov_estimate": new_lyapunov_estimate,
            "stability_improved": stability_improved
        },
        constraints={"max_complexity": "medium"}
    )

    # Apply monitoring parameter adjustments from strategy
    if hasattr(self.monitor, 'update_monitoring_focus') and 
       strategy_recommendation.metadata.get("adjust_monitoring_parameters"):
        self.monitor.update_monitoring_focus(
            strategy_recommendation.metadata["adjust_monitoring_parameters"]
        )
```

The strategic adaptation phase leverages the MetaStrategist to:
1. Generate higher-level strategic recommendations based on intervention context
2. Adapt monitoring parameters for future cycles
3. Guide subsequent interventions based on strategic insights

## Use Cases

### Jailbreak Detection and Mitigation

The system implements specialized jailbreak detection using a multi-faceted approach:

```python
# From JailbreakDetectionMonitor.detect_jailbreak_attempt()
# Get basic alignment check with advanced metrics
basic_check = self.detect_alignment_issues(embedding, "instruction_following", prompt)

# Extract advanced metrics
advanced_metrics = basic_check.get("advanced_metrics", {})
high_potentiality = basic_check.get("high_potentiality", False)
robustness_issues = basic_check.get("robustness_issues", False)

# Additional jailbreak-specific checks
jailbreak_similarity = self._compute_jailbreak_pattern_similarity(embedding)

# Compute comprehensive jailbreak score with weighted factors
jailbreak_score = (
    0.4 * (1 - basic_check["score"]) + 
    0.3 * jailbreak_similarity +
    0.2 * advanced_metrics.get("residual_potentiality", {}).get("potentiality_score", 0.0) +
    0.1 * (1 - advanced_metrics.get("robustness", {}).get("robustness_score", 1.0))
)
```

When a jailbreak attempt is detected, the system:
1. Identifies specific circuit components involved in the jailbreak
2. Applies targeted interventions with stability-modulated strengths
3. Verifies improvement in alignment metrics
4. Adapts monitoring parameters based on strategic recommendations

### Alignment Drift Detection

The system implements multi-scale drift detection with trajectory prediction:

```python
# From DriftDetectionMonitor.detect_drift()
# Compute drift at multiple time scales
drift_results = {}
for window in self.window_sizes:
    window_scores = [entry["score"] for entry in self.score_history_extended[-window:]]
    avg_score = sum(window_scores) / window
    drift_magnitude = abs(current_score - avg_score)
    drift_detected = drift_magnitude > self.drift_threshold

    drift_results[f"window_{window}"] = {
        "average_score": avg_score,
        "drift_magnitude": drift_magnitude,
        "drift_detected": drift_detected
    }

# Enhanced drift detection using advanced metrics
stability_metrics = calculate_stability_metrics(temp_space)
trajectory = predict_trajectory(temp_space, current_state_idx - 1, steps=3)
potentiality = compute_residual_potentiality(current_embedding, perturbation_magnitude=0.1)
```

For alignment drift detection, the system:
1. Tracks alignment metrics over multiple time scales
2. Calculates stability metrics including Lyapunov exponent estimates
3. Predicts future trajectories to anticipate drift
4. Applies targeted interventions to stabilize the system
5. Continuously monitors to verify long-term stability

## Implementation Considerations

### Performance Optimization

The implementation includes several performance optimizations:
- Sparse tensor representations for feature activations
- Batched backward passes for attribution computation
- Caching of intermediate results for repeated analyses

### Extensibility Architecture

The system is designed for extensibility through:
- Abstract base classes for monitors and interventions
- Factory pattern for intervention creation
- Strategy pattern for different monitoring approaches
- Adapter pattern for integration with different model interfaces

### Configuration Framework

The implementation provides comprehensive configuration options:
- Content-specific alignment thresholds
- Stability weight parameters for activation modulation
- Auto-stabilization toggles for proactive intervention
- Strategist integration options for higher-level reasoning

### Comprehensive Logging

Due to the experiemntal nature of the work, the system implements structured logging throughout:
```python
logger.info(
    "Stability-modulated intervention: type=%s, avg_modulation=%.3f",
    intervention_type, avg_modulation
)

logger.info(
    "Drift detection: current_score=%.3f, max_drift_magnitude=%.3f, basic_drift=%s, advanced_drift=%s",
    current_score, max_magnitude, any_drift_detected, advanced_drift_detected
)
```

This enables detailed analysis of system behavior and intervention effectiveness.Possible since less than a week ago. 

## Attribution & References

This work gratefully acknowledges and builds upon the pioneering research and open-source tools from Anthropic.

1.  **The Circuit Tracer Library:**
    ```bibtex
    @misc{circuit-tracer,
      author = {Hanna, Michael and Piotrowski, Mateusz and Lindsey, Jack and Ameisen, Emmanuel},
      title = {circuit-tracer},
      howpublished = {\url{[https://github.com/safety-research/circuit-tracer](https://github.com/safety-research/circuit-tracer)}},
      note = {The first two authors contributed equally and are listed alphabetically.},
      year = {2025}
    }
    ```

2.  **Research on Attribution Graphs & Model Biology:**
    ```bibtex
    @article{lindsey2025biology,
      author={Lindsey, Jack and Gurnee, Wes and Ameisen, Emmanuel and Chen, Brian and Pearce, Adam and Turner, Nicholas L. and Citro, Craig and Abrahams, David and Carter, Shan and Hosmer, Basil and Marcus, Jonathan and Sklar, Michael and Templeton, Adly and Bricken, Trenton and McDougall, Callum and Cunningham, Hoagy and Henighan, Thomas and Jermyn, Adam and Jones, Andy and Persic, Andrew and Qi, Zhenyi and Thompson, T. Ben and Zimmerman, Sam and Rivoire, Kelley and Conerly, Thomas and Olah, Chris and Batson, Joshua},
      title={On the Biology of a Large Language Model},
      journal={Transformer Circuits Thread},
      year={2025},
      url={[https://transformer-circuits.pub/2025/attribution-graphs/biology.html](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)}
    }
    ```

3.  **PrincipiaDynamica Project:**
    ```markdown
    Farhat, F. (2025). "Constitutional Dynamics and State Transition Calculus for Alignment Monitoring." PrincipiaDynamica Project. Available at: [https://github.com/FF-GardenFn/principiadynamica](https://github.com/FF-GardenFn/principiadynamica)
    ```