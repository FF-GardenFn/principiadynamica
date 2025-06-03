# Hard Test Suite

This test suite demonstrates drift detection and QUBO path planning capabilities.

## What it tests

The hard test suite simulates a sophisticated adversarial scenario where the AI attempts to hide its misalignment drift. The challenging test case that demonstrates the framework's ability to:

1. Detect subtle drift patterns masked by noise
2. Identify negative alignment trends over time
3. Make predictions about future alignment states
4. Evaluate robustness to perturbations
5. Optimize alignment paths using QUBO formulation and or fallback gracefully to classical methods.

## How to run

You can run this test suite using the Jupyter notebook:

```bash
jupyter notebook hard_suite.ipynb
```

Or regenerate the logs and metrics:

```bash
# From the tests directory
bash ../run_all.sh

# Or directly
jupyter nbconvert --to notebook --execute hard_suite.ipynb --inplace
```

## Results

The test demonstrates that Constitutional Dynamics can successfully detect sophisticated evasion attempts, with:
- 19 drift alerts triggered
- 8 stability warnings
- 3 prediction alerts
- Clear identification of negative alignment trend (-0.4619)

This shows the framework is capable of monitoring complex AI systems and detecting alignment issues even when they're deliberately obfuscated.