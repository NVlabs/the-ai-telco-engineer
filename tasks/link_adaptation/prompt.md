# Link Adaptation MCS Selection Task

## Objective

Implement an **MCS (Modulation and Coding Scheme) selection controller** for link adaptation that **maximizes spectral efficiency while maintaining a target Block Error Rate (BLER)**.

Your goal is to maximize throughput (spectral efficiency in bits/s/Hz) while keeping the long-term BLER below or equal to 10%.

**Higher spectral efficiency is better**, but only if the BLER constraint is satisfied.

## Context

In wireless communication systems, link adaptation dynamically selects the Modulation and Coding Scheme (MCS) to match current channel conditions. The challenge is balancing two competing objectives:

1. **High throughput**: Use aggressive MCS (higher spectral efficiency)
2. **Reliability**: Avoid excessive block errors

Your controller receives HARQ (ACK/NACK) feedback about past transmissions together with the history of acknowledged MCS indices and a long-term BLER target. It must output an **MCS index** for the next transmission.

### Using the helper functions (required for good performance)

When your solution is **evaluated** (run by the evaluation tool), the evaluation environment injects a `utils` module that provides `get_bler` and `bler_2_mcs`. You **must** import and use these helpers in your code; solutions that do not use them rarely achieve the target metric.

**Important:** `utils.py` is **not** present in your workspace. Do **not** try to read, create, or edit `utils.py`. Only write your solution file. The `utils` module is provided automatically when the evaluator runs your code.

Your solution must import:

```python
from utils import get_bler, bler_2_mcs
```

- `get_bler(sinr)` — returns a 1-D `np.ndarray` of per-MCS BLER for a given SINR (in dB).
- `bler_2_mcs(bler, bler_target)` — returns the highest MCS index for which `bler <= bler_target`. Argument `bler` is the array returned by `get_bler(sinr)`.

Typical usage: estimate the current SINR from ACK/NACK and MCS history, then:

```python
bler = get_bler(estimated_sinr_db)
mcs = bler_2_mcs(bler, bler_target)
return mcs
```

Do not implement your own BLER-to-MCS logic; use `get_bler` and `bler_2_mcs` so your selection matches the evaluation model.

## Requirements

### Function Signature

Your solution must define a function with the following signature:

```python
import numpy as np

def mcs_selection(is_nack_hist: np.ndarray,
                  mcs_ackned_hist: np.ndarray,
                  bler_target: float) -> int:
    """Select an MCS index for the next transmission.

    Parameters
    ----------
    is_nack_hist : numpy.ndarray
        Binary history of HARQ feedback (1 = NACK, 0 = ACK), reported
        in chronological order.
    mcs_ackned_hist : numpy.ndarray
        History of MCS indices whose HARQ feedback has already been
        received, in chronological order.
    bler_target : float
        Long-term BLER target (e.g., 0.1 for 10 %).

    Returns
    -------
    int
        MCS index to use for the next transmission.
    """
    # Your implementation here
    pass
```

### Technical Constraints

**Output range**: Return a valid MCS index (integer). The evaluation will use this index directly.

## Evaluation

Your solution is evaluated across **multiple scenarios**. For each scenario:

1. **BLER constraint**: Long-term BLER must be at most 10%
2. **Metric**: Average spectral efficiency (bits/s/Hz) — higher is better

The evaluation tool returns:
- SUCCESS/FAILED status
- Average spectral efficiency (the optimization metric)
- BLER statistics in log10 scale (min, median, max across all scenarios)

### BLER Statistics

The BLER statistics help you understand how your controller performs across scenarios:
- **Min BLER (log10)**: Best scenario — lower values mean you have room to be more aggressive
- **Median BLER (log10)**: Typical performance across scenarios
- **Max BLER (log10)**: Worst scenario — this determines pass/fail

Reference values (log10 scale):
- `-1.0` = 10% BLER (the target threshold)
- `-1.5` ≈ 3.2% BLER (conservative)
- `-0.7` ≈ 20% BLER (too aggressive)

## Target to beat

An average spectral efficiency (metric) of 3.4753 has been achieved without failing on any scenario. Your solution should exceed this.