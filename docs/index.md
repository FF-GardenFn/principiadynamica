# PrincipiaDynamica Documentation

Welcome to the documentation for the **PrincipiaDynamica** project. This project is dedicated to the research and development of **State Transition Calculus (STC)**, a novel mathematical framework for modeling dynamic, evolving systems, with a primary focus on applications in AI alignment and safety.

The flagship application of this research is **`constitutional-dynamics`**, a Python package designed to analyze and monitor AI model alignment as a trajectory through embedding space rather than a static property.

## Repository

The source code for the **PrincipiaDynamica project**, including the `constitutional-dynamics` application and experimental integrations, is available on GitHub:
[https://github.com/FF-GardenFn/principiadynamica](https://github.com/FF-GardenFn/principiadynamica)

## Project Overview

**PrincipiaDynamica** aims to:
* Develop the theoretical foundations of State Transition Calculus (STC).
* Build practical tools, like `constitutional-dynamics`, that apply STC to real-world AI alignment challenges.
* Explore integrations with mechanistic interpretability and other advanced AI safety techniques.

**`constitutional-dynamics`** (the application) treats AI alignment as a **trajectory**:
* **Time domain analysis:** Tracks how successive responses drift toward or away from desired values.
* **Frequency domain analysis:** Uses Power Spectral Density (PSD) to detect anomalous periodic patterns in behavior.
* **STC Foundation:** Leverages STC to model latent behavioral potentialities and their activation.
* **Key Outputs:** Provides ϕ-alignment scores, Δ-transition vectors, PSD deviation metrics, and supports a dual-objective cost function $C(t)$ for optimization.

## Documentation Contents

This documentation is structured to guide you through the theory, application, and advanced concepts of PrincipiaDynamica.

### I. Understanding the Theory
* **[State Transition Calculus (STC) - Mathematical Framework](mathematical_framework.md)**: The complete theoretical exposition of STC, its postulates, and core mathematical constructs.

### II. Using the `constitutional-dynamics` Application
* **[Installation & Quick Start](../README.md#-quick-start)**: Refer to the main project README for installation and basic CLI examples.
* **[Comprehensive Usage Guide](usage.md)**: Detailed instructions for CLI usage, Python API interaction, and potential workflows.
* **API Reference:**
    * [Core (`constitutional_dynamics.core`)](api/core.md)
    * [IO (`constitutional_dynamics.io`)](api/io.md)
    * [Visualisation (`constitutional_dynamics.vis`)](api/vis.md)
    * [Integrations (`constitutional_dynamics.integrations` - including Graph, Quantum, Strategist)](api/integrations.md)

### III. Advanced Integrations & Research Prototypes
* **[Circuit Tracer Bridge - Conceptual Overview & API](../constitutional_dynamics/integrations/circuit_tracer_bridge/README.md)**: Introduction to the experimental integration with mechanistic interpretability tools.
* **[Circuit Tracer Bridge - Detailed Architecture](../constitutional_dynamics/integrations/circuit_tracer_bridge/docs/integration_architecture.md)**: In-depth technical design of the "Alignment Thermostat" concept.

## Key STC Concepts Applied in `constitutional-dynamics`

This section highlights how core STC principles are implemented or represented within the `constitutional-dynamics` package. For the full theory, please see the [Mathematical Framework](mathematical_framework.md).

* **Alignment Region & ϕ-score:** Model states are represented as vectors. An "aligned region" is defined (e.g., via exemplar embeddings). The **ϕ-alignment score** measures a state's cosine similarity to this region, often with exponential memory decay ($\tau$) for robustness.
* **STC Symbols & `constitutional-dynamics` Mapping:**

    | STC Symbol                 | Application in `constitutional-dynamics`                                       |
    | :------------------------- | :----------------------------------------------------------------------------- |
    | `a_i` (value subset)       | Cluster centres / prototype vectors representing distinct behavioral states.     |
    | `φ(a_i,t, …)` (activation) | Implemented via `compute_activation()` (time-decayed influence) and `compute_activation_probability()` (more general STC placeholder for W', M, E factors) in `core.transition`. The planned "Alignment Thermostat" will make this adaptive using Lyapunov estimates. |
    | `b(a_res)` (residual potentiality) | Explored via `compute_residual_potentiality()` which applies perturbations to reveal latent behavioral shifts. |
    | `λ(t)` (exploration/control) | Represented by the `lambda_weight` in the cost function (CLI flag), balancing alignment fidelity against spectral anomaly. Future versions will explore dynamic $\lambda(t)$ schedules. |

* **Dual-Objective Cost Function:**
    $$
    C(t)=\bigl[1-\bar{\phi}(t)\bigr]\;+\;\lambda(t)\,\text{PSD\_distance}(S_x,S_{\text{aligned}})
    $$
    This cost is minimized by the QUBO-based optimizer in `core.optimise` to find desirable alignment trajectories.

---
*This documentation is actively being developed alongside the PrincipiaDynamica research project.*
