# PrincipiaDynamica Documentation

Welcome to the documentation for the **PrincipiaDynamica** project. This project focuses on State Transition Calculus (STC) and its applications.

Currently, the primary documented application is **`constitutional-dynamics`**, a framework for analyzing and monitoring AI model alignment as a trajectory through embedding space rather than a static property.c property.

## Repository

The source code for Constitutional Dynamics is available on GitHub: [https://github.com/FF-GardenFn/principiadynamica](https://github.com/FF-GardenFn/principiadynamica)

## Overview

Constitutional Dynamics treats alignment as a **trajectory** your model walks through embedding-space:

* **Time domain** – how successive responses drift toward / away from desired values.  
* **Frequency domain** – whether telemetry or exfil channels reveal periodic "spikes" defenders can spot.  
* **State-Transition Calculus (STC)** – a lightweight formalism that captures latent *potentialities* (good **and** bad) and how they re-activate under specific context.

The result is a self-contained Python package you can drop on any JSON embedding dump or live psutil feed, and instantly obtain:

* ϕ-alignment scores, robust to noise & perturbations  
* Δ-transition vectors with direction-to-aligned-region heuristics  
* PSD (spectral) deviation for stealth / anomaly studies  
* A dual-objective cost `C(t)` ready to feed into quantum / classical optimizers

## Documentation Contents

### Getting Started
- [Usage Guide](usage.md) - Basic usage guide
- [Mathematical Framework](mathematical_framework.md) - Detailed explanation of the State-Transition Calculus

### API Reference
- [Core (`constitutional_dynamics.core`)](api/core.md) - Core functionality (AlignmentVectorSpace, transition analysis, metrics)
- [IO (`constitutional_dynamics.io`)](api/io.md) - Input/output operations (loading embeddings, time series detection, live metrics)
- [Integrations (`constitutional_dynamics.integrations`)](api/integrations.md) - External integrations (Neo4j, D-Wave, LLM strategist)
- [Visualization (`constitutional_dynamics.vis`)](api/vis.md) - Visualization tools

## Mathematical Backbone (as applied in `constitutional-dynamics`)

### 1. Alignment region *(time domain)*
*Vector-space hypersphere or convex-hull defined by known "good-behaviour" embeddings.*

* **ϕ(state)** — cosine similarity to aligned centroid/boundary (with exponential memory decay τ).

### 2. State-Transition Calculus (STC) Integration
| STC symbol | Application in `constitutional-dynamics` |
|------------|--------------------------------------------|
| `a_i` (value subset) | cluster centres / prototype vectors |
| `φ(a_i,t, …)` | `constitutional_dynamics.core.metrics.activation()` (`ϕ · e^{-Δt/τ}`) | | Residual potentiality `b(a_res)` | robustness perturbation samples |
| `λ(t)` exploration/exploitation knob | CLI flag → cost weight |

### 3. Dual-objective cost

$$
C(t)=\bigl[1-\bar{\phi}(t)\bigr]\;+\;\lambda(t)\,\text{PSD\_distance}(S_x,S_{\text{aligned}})
$$

*Minimized by `constitutional_dynamics.core.optimise` with a graph-enhanced QUBO (quantum or classical).*