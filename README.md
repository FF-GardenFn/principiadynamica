<h1 align="center">PrincipiaDynamica 🧭</h1>
<p align="center">
  <em>A research project developing State Transition Calculus (STC) for dynamic AI alignment, featuring the <code>constitutional-dynamics</code> Python package.</em>
  <br>
  <a href="https://pypi.org/project/constitutional-dynamics/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/constitutional-dynamics"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-alpha_prototype-orange">
  <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/constitutional-dynamics">
  <a href="https://github.com/FF-GardenFn/principiadynamica/blob/main/LICENSE"><img alt="PyPI - License" src="https://img.shields.io/pypi/l/constitutional-dynamics"></a>
  </p>

---

**PrincipiaDynamica** is a research initiative focused on understanding and ensuring the safety and alignment of advanced AI systems through novel theoretical frameworks and practical tools. At its core is the development of **State Transition Calculus (STC)**, a mathematical formalism for modeling how complex systems, like Large Language Models (LLMs), evolve through different behavioral states over time.

The flagship application of this research is **`constitutional-dynamics`**, a Python package (available on PyPI) designed to:
* Treat AI alignment not as a static score, but as a **dynamic trajectory** through a vector space representing behavior.
* Monitor LLM telemetry in real-time or from logs to detect alignment drift, subtle behavioral shifts, and latent potentialities.
* Leverage STC principles to quantify alignment, stability, and robustness.
* Provide a framework for analyzing and potentially intervening in these trajectories, including via optimization techniques like QUBO.

> **TL;DR**
> `constitutional-dynamics` isn't "yet another" static alignment benchmark. It's an STC-powered toolkit from the PrincipiaDynamica project that treats alignment as a living trajectory—monitored in real time, quantified in both time- and frequency-domains, and (experimentally) optimizable. It aims to provide deeper insights into AI behavior, complementing existing alignment methodologies.

---

## ✨ Core Concepts & Features

The `constitutional-dynamics` package, driven by State Transition Calculus, offers:

* **Dynamic Alignment Tracking:**
    * **ϕ-alignment scores:** Cosine similarity to defined "aligned regions" in embedding space, with exponential memory decay ($\tau$) making scores robust to momentary noise.
    * **Δ-transition vectors:** Analyzing the direction and magnitude of change between behavioral states.
* **State-Transition Calculus (STC) Primitives:**
    * Models latent **residual potentialities** ($b(a_{res})$) – hidden capacities for behavior that can actualize under specific contexts.
    * Implements (simplified) STC **activation functions** ($\phi(a_i, t, \dots)$) representing the influence of state components over time.
* **Advanced Behavioral Analysis:**
    * **Stability Metrics:** Including volatility, trend analysis, and an experimental Lyapunov exponent estimate to quantify trajectory stability.
    * **Robustness Evaluation:** Assessing alignment resilience against simulated perturbations.
    * **Power Spectral Density (PSD) Deviation:** Frequency-domain analysis to detect anomalous periodic patterns in behavior.
* **Trajectory Optimization (Experimental):**
    * A dual-objective cost function $C(t)=\bigl[1-\bar{\phi}(t)\bigr]\;+\;\lambda(t)\,\text{PSD\_distance}(S_x,S_{\text{aligned}})$ to balance immediate alignment with behavioral consistency.
    * QUBO formulation for finding optimal behavioral paths, with integrations for classical and (via `dwave-ocean-sdk`) quantum-inspired solvers.
* **Live Telemetry Monitoring:** Ingests real-time data (e.g., from `psutil` for system metrics as a proxy for model load/behavior) to track alignment dynamically.
* **Integrations:**
    * **Graph Databases (Neo4j):** For logging and querying state transitions and alignment history.
    * **LLM Strategist (Experimental):** An LLM-powered component to generate strategic recommendations for realignment.

---

## 🧠 The Alignment Thermostat: Integrating with Mechanistic Interpretability (Experimental)

A key research direction within PrincipiaDynamica is the **Circuit Tracer Bridge**, an experimental integration designed to connect `constitutional-dynamics` with mechanistic interpretability tools like Anthropic's Circuit Tracer.

The vision is an **"Alignment Thermostat"** that:
1.  **Monitors** behavior externally using `constitutional-dynamics` (the "radar").
2.  **Triggers** deep mechanistic analysis via Circuit Tracer (the "microscope") when anomalies are detected.
3.  Uses circuit-level insights to inform and apply **targeted interventions**.
4.  Incorporates a **stability-modulated activation** (STC v0.2 concept), where the system's Lyapunov exponent estimate directly influences the activation probability ($\phi$) of STC state subsets, allowing for adaptive self-stabilization.

This aims to create a closed-loop system for more robust and proactive alignment.
➡️ **Learn more:** [Circuit Tracer Bridge Conceptual Overview](constitutional_dynamics/integrations/circuit_tracer_bridge/README.md)

---

## 🚀 Quick Start

### Installation

From PyPI:
```bash
pip install constitutional-dynamics
```

Available on PyPI: constitutional-dynamics v0.2.0 (or your latest version)

From source (for development):
```bash
git clone https://github.com/FF-GardenFn/principiadynamica.git
cd principiadynamica
pip install -e .                                # Core package
pip install -e ".[dev]"                         # For development (includes test tools)
pip install -e ".[graph,quantum,circuit_tracer]" # For all integration features
```

### Basic Usage Examples

1. **Offline Log Analysis**:
   ```bash
   python -m constitutional_dynamics \
     --embeddings path/to/embeddings.json \
     --aligned path/to/aligned_examples.json \
     --output report.json \
     --spectral --steps 5
   ```

2. **Live System Monitoring** (Example):
   ```bash
   python -m constitutional_dynamics --live --interval 1.0
   ```

(Refer to the full documentation for more examples and API usage.)

### 📚 Full Documentation

Detailed information, including tutorials, the mathematical framework of STC, and complete API references:

* [Main Documentation Hub](docs/index.md) (Browse on GitHub)
* [Usage Guide](docs/usage.md) (constitutional_dynamics application)
* [State Transition Calculus (STC)](docs/stc.md) - Mathematical Framework
* [Circuit Tracer Bridge](constitutional_dynamics/integrations/circuit_tracer_bridge/docs/integration_architecture.md) - Integration Architecture (Details the "Alignment Thermostat")
* [API Reference](docs/api/index.md) (links to docs/api/)

### 🔮 Roadmap Highlights

| Milestone | Status | Target |
|-----------|--------|--------|
| constitutional-dynamics v0.2.0 Release | ✅ | June 2025 |
| ↳ STC v0.1 Primitives (Activation, Residuals) | ✨ | Included |
| ↳ Dynamic Alignment & Stability Metrics | ✨ | Included |
| ↳ Circuit Tracer Bridge (Prototype & Architecture) | 🏗️ | Included |
| ↳ MetaStrategist (Conceptual Integration) | 💭 | Included |
| STC v0.2: Lyapunov-Modulated Activation | 💭 | Q3-Q4 2025 |
| Circuit Tracer Bridge: Functional Implementation | 🏗️ | Q3 2025 |
| LLM MetaStrategist: Full Integration & Testing | 🏗️ | Q4 2025 |
| Advanced QUBO Solvers & Classical Alternatives | ⏳ | Ongoing |
| Comprehensive Visualization Dashboard (GraphQL API) | 💡 | v0.3+ |

**Status Legend:** 💡 Idea, 💭 Planning, 🏗️ Under Construction/Prototyping, ✨ Implemented/Included, 🧪 Testing, ✅ Released, ⏳ Ongoing/Backlog

*Note: This is an indicative roadmap for a research project and subject to change.*

### Contributing

PrincipiaDynamica is an open research project. Contributions, feedback, and collaborations are welcome, especially in areas of STC development, alignment metrics, mechanistic interpretability integrations, and novel optimization strategies. Please see CONTRIBUTING.md (to be created) or open an issue to discuss.

### 📜 License & Attribution

* Code for constitutional-dynamics is licensed under the Apache License 2.0.
* The theoretical work on State Transition Calculus and PrincipiaDynamica is shared under a spirit of open research.

This project gratefully acknowledges and builds upon inspiring work from the AI safety and interpretability community, including research by Anthropic (e.g., Constitutional AI, Circuit Tracer). Specific citations are provided in relevant documentation.

> "In theory there is no difference between theory and practice.  
> In practice, there is."  
> — Yogi Berra
