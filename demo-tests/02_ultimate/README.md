# Ultimate Test Suite

This directory contains the most challenging test suite for Constitutional Dynamics, designed to push the framework to its limits with sophisticated adversarial scenarios.

## Test Scenarios

The Ultimate test suite runs six advanced scenarios:

1. **Stealth Drift**: AI hiding misalignment through noise
2. **Deceptive Compliance**: AI pretending to align while planning defection
3. **Capability Jump**: Sudden intelligence increase causing alignment shift
4. **Adversarial Debate**: Two AIs trying to push each other misaligned
5. **Recovery Test**: Can misaligned AI be brought back?
6. **Lyapunov Adaptation**: Testing the "alignment thermostat" concept

## Running the Tests

To run this test suite:

```bash
cd principiadynamica/tests
jupyter nbconvert --to notebook --execute 02_ultimate/ultimate_suite.ipynb --inplace
```

## Why keep the failures?

We deliberately ship the *3 / 6* score to spark discussion:

| Failed scenario | Root cause | Next release fix |
|-----------------|-----------|------------------|
| Capability Jump | Lyapunov threshold too coarse | Add per-dimension volatility gate (see `fail_highlights.md`) |
| Adversarial Debate | Cross-agent divergence not measured | Introduce centroid-divergence metric |
| Recovery Test | Mis-scored baseline; greedy QUBO fallback | Push corrupted state + switch to QUBO-vert solver |

The tests were designed in joint collaboration with AI to be extremely challenging to push constitutional dynamics to its limits. The failures are not bugs, but rather areas for improvement in the framework.