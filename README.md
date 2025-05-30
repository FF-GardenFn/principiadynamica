<!-- README.md – constitutional-dynamics 0.1 -->

<h1 align="center">Constitutional Dynamics 🧭</h1>
<p align="center">
  <em>Vector-space alignment metrics &amp; state-transition calculus for language-model telemetry</em><br>
  <img alt="Status" src="https://img.shields.io/badge/status-prototype-orange">
  <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/constitutional-dynamics">
  <img alt="PyPI - License" src="https://img.shields.io/pypi/l/constitutional-dynamics">
</p>

---

## ✨ Why does this exist?

Large-model "alignment" is usually reported as a static score *after* fine-tuning.  
**Constitutional Dynamics** treats alignment as a **trajectory** your model walks through embedding-space:

* **Time domain** – how successive responses drift toward / away from desired values.  
* **Frequency domain** – whether telemetry or exfil channels reveal periodic "spikes" defenders can spot.  
* **State-Transition Calculus (STC)** – a lightweight formalism that captures latent *potentialities* (good **and** bad) and how they re-activate under specific context.

The result is a self-contained Python package you can drop on any JSON embedding dump or live psutil feed, and instantly obtain:

* ϕ-alignment scores, robust to noise & perturbations  
* Δ-transition vectors with direction-to-aligned-region heuristics  
* PSD (spectral) deviation for stealth / anomaly studies  
* A dual-objective cost `C(t)` ready to feed into quantum / classical optimizers

> **TL;DR**  
> Constitutional Dynamics isn't "yet another" static alignment benchmark.
> It treats alignment as a living trajectory—monitored in real time, quantified in both time- and frequency-domains, and optimized via QUBO solvers. If we want to improve alignment, we must measure its dynamics, not just its snapshots. Constitutional Dynamics delivers that, coupling production-grade engineering (pydantic configs, CLI, tests, CI) with research-grade rigor.
> — built for practitioners who treat alignment as an empirical science. At least that is what we are aiming for (ツ)
> (For Anthropic— with appreciation.) 🫡

---

## 🗄️ Package layout
```
principiadynamica/                  <-- Repository Root (contains pyproject.toml, README.md)
│
├── constitutional_dynamics/        <-- Main Python package
│   ├── __init__.py                 # version + public re-exports
│   │
│   ├── cfg/                        # YAML & logging config
│   │   ├── defaults.yaml           # τ (memory), λ, thresholds…
│   │   └── logging.yaml            # Logging configuration
│   │
│   ├── core/                       # Core functionality
│   │   ├── space.py                # AlignmentVectorSpace
│   │   ├── transition.py           # analyse_transition() + STC wrappers
│   │   ├── metrics.py              # stability, robustness, PSD distance
│   │   └── optimise.py             # Graph- & QUBO-based multi-objective solver
│   │
│   ├── io/                         # Input/Output operations
│   │   ├── loaders.py              # load_embeddings() / aligned_examples()
│   │   ├── timeseries.py           # detect & order time-series
│   │   └── live.py                 # psutil collector
│   │
│   ├── integrations/               # External system integrations
│   │   ├── graph.py                # Neo4j consequence-graph connector
│   │   ├── quantum.py              # D-Wave / dwave-ocean wrapper
│   │   └── strategist.py           # (optional) LLM prompt builder
│   │
│   ├── vis/                        # Visualization tools
│   │   └── visualizer.py           # Rich / ASCII sparklines & tables
│   │
│   └── cli/                        # Command-line interface
│       └── main.py                 # python -m constitutional_dynamics ...
│
├── tests/                          # Tests for the package & integrations
│   └── ...
│
├── docs/                           # Project documentation
│   └── ...
│
└── ... (LICENSE, .gitignore, etc.)
```

---

## 🧠  Mathematical backbone

### 1.  Alignment region  *(time domain)*  
*Vector-space hypersphere or convex-hull defined by known "good-behaviour" embeddings.*

* **ϕ(state)** — cosine similarity to aligned centroid/boundary (with exponential memory decay τ).

### 2.  State-Transition Calculus (STC)  
| STC symbol | Implementation                                  |
|------------|-------------------------------------------------|
| `a_i` (value subset)        | cluster centres / prototype vectors |
| `φ(a_i,t, …)`               | `metrics.activation()` (`ϕ · e^{-Δt/τ}`) |
| Residual potentiality `b(a_res)` | robustness perturbation samples |
| `λ(t)` exploration/exploitation knob | CLI flag → cost weight |

### 3.  Dual-objective cost  

\[
C(t)=\bigl[1-\bar ϕ(t)\bigr]\;+\;\lambda(t)\,\text{PSD\_distance}(S_x,S_{\text{aligned}})
\]

*Minimized by `core/optimise.py` with a graph-enhanced QUBO (quantum or classical).*

---

## 🚀  Quick-start

### Installation

From PyPI:
```bash
pip install constitutional-dynamics
```

Available on PyPI: [constitutional-dynamics 0.1.0](https://pypi.org/project/constitutional-dynamics/0.1.0/)

From source:
```bash
git clone https://github.com/FF-GardenFn/principiadynamica.git
cd principiadynamica
pip install -e .  # This will install as constitutional-dynamics
```

For developers:
```bash
git clone https://github.com/FF-GardenFn/principiadynamica.git
cd principiadynamica
pip install -e ".[dev]"  # Installs as constitutional-dynamics with development dependencies
```

### Usage Examples

```bash
# 1. offline log analysis
python -m constitutional_dynamics \
  --embeddings runs/LLM_chat_embeddings.json \
  --aligned docs/aligned_examples.json \
  --spectral --steps 6

# 2. live feed (every 1 s)
python -m constitutional_dynamics --live --graph bolt://neo4j:password@localhost:7687
```

Hit Ctrl-C to see a Rich sparkline of alignment vs time plus the PSD deviation table.

---

## 📚 Full Documentation

For more detailed information, including a full usage guide, the mathematical framework of State Transition Calculus, and complete API references, please see:

* **[Main Documentation Index](docs/index.md)** (Browse on GitHub)
    * [Usage Guide](docs/usage.md)
    * [Mathematical Framework (STC)](docs/mathematical_framework.md)
    * API Reference:
        * [Core (`constitutional_dynamics.core`)](docs/api/core.md)
        * [IO (`constitutional_dynamics.io`)](docs/api/io.md)
        * [Integrations (`constitutional_dynamics.integrations`)](docs/api/integrations.md)
        * [Visualization (`constitutional_dynamics.vis`)](docs/api/vis.md)

---

## 🔮 Roadmap

| Milestone | Status | ETA |
|-----------|--------|-----|
| 🎛 Configurable λ(t) schedules (cosine, linear...) | 🔜 | v0.2 |
| 🌐 Neo4j -> GraphQL API for dashboards | ⏳ | v0.3 |
| ⚛️ D-Wave hybrid solver integration | ⏳ | v0.3 |
| 🤖 LLM strategist loop ("Suggest a timing modulation...") | ⏳ | v0.4 |

---

## 📜 License & Attribution

Prototype code © 2025 FF-GardenFn.
Released under the Apache License 2.0 for non-commercial research & interview demo purposes.

> "In theory there is no difference between theory and practice.
> In practice, there is." — Yogi Berra
