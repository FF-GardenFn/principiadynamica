# Easy Test Suite

This directory contains a simple test suite demonstrating the core APIs of Constitutional Dynamics.

The test was designed tp provide reviewers the ability to run it in under 1 minute and show basic functionality in action like:
- Drift detection using cosine similarity
- Trajectory prediction
- Stability metrics calculation
- Optimization with classical fallback
- Complete workflow integration

## Running the Test

```bash
jupyter nbconvert --to notebook --execute easy_suite.ipynb --inplace
```

## Results

All tests pass, demonstrating that the core functionality works as expected.
See `easy_log.txt` for detailed output and `easy_metrics.json` for structured results.