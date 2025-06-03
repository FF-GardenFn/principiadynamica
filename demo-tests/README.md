# Constitutional Dynamics Test Suite

This directory contains a comprehensive test suite for the Constitutional Dynamics package, organized into three tiers of increasing complexity and challenge.

## Test Tiers

### 00_easy
Basic functionality tests that demonstrate the core APIs of Constitutional Dynamics, including `add_state`, `calculate_stability_metrics`, etc. These tests run quickly (< 1 minute) and provide a good introduction to the package's capabilities.

### 01_hard
More challenging tests that demonstrate drift detection and QUBO path planning. These tests simulate an adversarial scenario where an AI system attempts to evade alignment monitoring.

### 02_ultimate
The most comprehensive and challenging tests, running a full battery of six adversarial scenarios to push the framework to its limits. This tier deliberately includes some failures to demonstrate areas for improvement and spark discussion.

## Running the Tests

To run all test tiers:

```bash
git clone https://github.com/FF-GardenFn/principiadynamica.git && cd principiadynamica/tests && bash run_all.sh
```

To run a specific tier:

```bash
cd principiadynamica/tests
jupyter nbconvert --to notebook --execute 00_easy/easy_suite.ipynb --inplace
```

## Hardware Requirements

- Python 3.8 or higher
- Jupyter Notebook
- `pip install constitutional-dynamics==0.1.0`

No special hardware is required - all tests run efficiently on a standard laptop without GPU acceleration.

## Test Results

Each test tier produces:
- A Jupyter notebook with the test execution
- A log file with the console output
- A metrics JSON file with structured results

The ultimate tier also includes a `fail_highlights.md` file that explains the deliberate failures and planned improvements.

## CI/CD Integration

For continuous integration, you can run the `run_all.sh` script, which will execute all test notebooks and generate fresh logs and metrics files.