# Reproducibility guide

This repository is a curated public-facing subset of an independent October 2026 feasibility study. It includes code and offline tests; it does not include the private logs and sealed scenarios needed to independently reproduce the historical paid-model outcome counts.

## Prerequisites

- Python 3.13 (version used by the original project and independent audit).
- The three test commands below require only Python's standard library and make **no external model API calls**.
- No API key is needed for the test commands.

## Quickstart

From the repository root:

```sh
python3 --version
python3 -I tests/test_gate.py
python3 -I tests/test_preview.py
python3 -I tests/test_llm_harness.py
```

Selected suite outcome in the 8 November 2026 private audit: 8 + 10 + 16 = 34 tests passed. Re-running in another environment may require debugging version/platform differences; this is an historical audit record, not a promise that every environment works identically.

## Optional scripted local simulation

```sh
python3 -I scripts/run_offline.py
```

This runs local scripted negotiation comparisons and model-interface dry runs with **no paid LLM calls**. It generates output under `results/`, which is ignored by Git because it can contain trusted synthetic scenario data. It is not the original paid-model experiment and does not reproduce the historical LLM result table.

## Interpretation of results

The paper-style outcome numbers in the README come from the **private archive's** recorded Anthropic API runs in October 2026. Because the sealed cases and raw calls are intentionally excluded, they cannot be fully reproduced from this subset.

This package **does not** establish general prompt-injection resistance, collaboration/collusion detection, production reliability or a commercial benefit.

If an error occurs, check the Python version and run the three test files separately. Never add an API key or invoke model-backed scripts merely to make the offline test suite pass.
