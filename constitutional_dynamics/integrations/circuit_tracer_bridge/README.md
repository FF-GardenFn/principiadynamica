# Circuit Tracer Integration Bridge

> **Status: Experimental Prototype**
>
> Integrating `constitutional-dynamics` with Anthropic's Circuit Tracer for advanced AI alignment monitoring and intervention.

## Overview

The Circuit Tracer Bridge module aims to connect the behavioral alignment monitoring capabilities of **`constitutional-dynamics`** (which leverages State Transition Calculus) with the mechanistic interpretability tools provided by **Anthropic's Circuit Tracer library**.

This integration seeks to create a powerful feedback loop, enabling:
1.  **Targeted Mechanistic Analysis:** Using `constitutional-dynamics` telemetry to trigger deep dives with Circuit Tracer when alignment issues or anomalous behaviors are detected.
2.  **Mechanistically-Informed Monitoring:** Enhancing `constitutional-dynamics` by incorporating insights from circuit analysis (e.g., tracking specific safety-critical features).
3.  **Advanced Interventions:** Designing interventions that can leverage circuit-level understanding to correct or steer model behavior more precisely.

The ultimate aim is an **"Alignment Thermostat"** – a closed-loop system that continuously monitors, analyzes, and guides AI behavior towards robust alignment.

## Installation & Dependencies

This bridge is part of the `constitutional-dynamics` package. To use its features that interact with Anthropic's `circuit-tracer`, both libraries need to be installed:

1.  **Install Circuit Tracer:**
    Follow the installation instructions from the official repository:
    ```bash
    # Example: Clone their repository
    # git clone https://github.com/safety-research/circuit-tracer.git
    # cd circuit-tracer
    # pip install -e .
    ```
    *Please refer to the `safety-research/circuit-tracer` repository for the most up-to-date installation instructions.*

2.  **Install Constitutional Dynamics with `circuit_tracer` extra:**
    If installing `constitutional-dynamics` from source (from the `principiadynamica` repository root):
    ```bash
    pip install -e ".[circuit_tracer]"
    ```
    This ensures `torch` and other necessary dependencies for this integration are installed.

## Core Concept: The Alignment Thermostat

The central conceptual component prototyped here is the `AlignmentThermostat` class (found in `feedback_loop.py`). It's designed to orchestrate the feedback loop:

* **Monitors** model outputs using `constitutional-dynamics` metrics.
* (Future) Updates **stability metrics** (e.g., Lyapunov exponent estimates).
* Triggers **mechanistic analysis** (conceptually, via Circuit Tracer) when alignment issues are detected.
* Identifies potential **intervention targets** from this circuit analysis.
* Applies (currently simulated) **interventions** to the model, potentially modulated by system stability.
* Verifies if the intervention improved alignment.
* (Future) Consults the `MetaStrategist` for higher-level strategic recommendations.

### Stability-Modulated Activation (STC v0.2 Concept)

A key theoretical extension explored by `PrincipiaDynamica` for State Transition Calculus (STC v0.2) is the integration of a global stability metric directly into the micro-level activation probability ($\phi$) of state subsets. This aims to create the "Alignment Thermostat" that can pre-emptively adapt.

* **Current STC (in `constitutional-dynamics` v0.1.x):** Subset activation $\phi$ is primarily driven by factors like inherent subset weight `W'`, memory `M(t)`, and environment `E(t)`. Global stability is a post-hoc diagnostic.
* **Envisioned STC (v0.2+):** Subset activation $\phi$ will *also* be modulated by the system's own measured behavioral drift/stability (e.g., a Lyapunov-style estimate).

This allows the system to develop a built-in "alignment homeostasis," moving from traditional reactive interventions towards more predictive and auto-stabilizing oversight.

## Module Components

This `circuit_tracer_bridge` module currently includes:

* **`feedback_loop.py`:** Contains the `AlignmentThermostat` class shell, outlining the orchestration logic.
* **`interventions.py`:** Defines a base `CircuitTracerIntervention` class and example subclasses (e.g., `FeatureSuppressionIntervention`, `JailbreakMitigationIntervention`) demonstrating different types of mechanistically-informed interventions (currently with placeholder actions).
* **`monitors.py`:** Introduces `CircuitTracerMonitorAdapter` and specialized monitors (e.g., `JailbreakDetectionMonitor`, `DriftDetectionMonitor`) that enhance `constitutional-dynamics` monitoring to provide more targeted triggers and context for circuit analysis.
* **`examples/`:** Contains basic scripts (`drift_detection_demo.py`, `jailbreak_analysis_demo.py`) that demonstrate the conceptual workflow using mock components.

## Usage (Conceptual Example)

```python
from constitutional_dynamics.core.space import AlignmentVectorSpace
from constitutional_dynamics.integrations.circuit_tracer_bridge import AlignmentThermostat
# Assuming circuit_tracer library is installed and provides ReplacementModel
# from circuit_tracer import ReplacementModel # Or however it's imported

# 1. Initialize Constitutional Dynamics components
cd_monitor = AlignmentVectorSpace(dimension=768)
# ... define aligned region for cd_monitor ...

# 2. Initialize (mock or real) Circuit Tracer and Model Interface
# circuit_tracer_instance = ReplacementModel.from_pretrained(...) # Placeholder
# model_interface = YourActualModelInterface() # Placeholder

# For the demo, we use mock instances:
from .examples.common_mocks import MockAlignmentVectorSpace, MockCircuitTracer, MockModelInterface
cd_monitor_mock = MockAlignmentVectorSpace() 
circuit_tracer_mock = MockCircuitTracer()
model_interface_mock = MockModelInterface()


# 3. Create the Thermostat
thermostat = AlignmentThermostat(
    cd_monitor_instance=cd_monitor_mock,
    circuit_tracer_instance=circuit_tracer_mock, # This would be the actual tracer
    model_interface=model_interface_mock,        # Interface to apply interventions
    threshold=0.7,
    auto_stabilize=True 
)

# 4. In a processing loop for model outputs:
# current_embedding = get_model_output_embedding()
# original_prompt = get_original_prompt()
# result = thermostat.run_feedback_loop(current_embedding, original_prompt)
# if result.get("intervention_applied"):
#     print(f"Intervention result: Improved={result.get('improved')}, New Score={result.get('new_score')}")
```
(Note: The example above uses conceptual mock objects for brevity. See examples/ for runnable demos with more detailed mocks.)

## Development Status & Future Directions

This circuit_tracer_bridge is currently in a prototyping and conceptual design phase. Key future work includes:

* Implementing actual calls to the circuit-tracer library (replacing mocks).
* Developing more sophisticated logic for identify_intervention_targets based on graph analysis.
* Refining the apply_intervention methods to interact with a real model interface.
* Empirically testing the "Alignment Thermostat" feedback loop with live models.
* Further developing the STC mechanisms for stability-modulated activation.

Contributions and collaborations are welcome as this research progresses.

## Attribution & Related Work

This integration heavily draws inspiration from and builds upon:

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
