# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Utility functions and constants for link adaptation evaluation.

Provides BLER sigmoid curve fitting utilities, BLER-to-MCS mapping, and
pre-loaded lookup tables (sigmoid parameters, spectral efficiency per MCS)
used by the evaluation pipeline.
"""

import numpy as np


def get_bler_sigmoid_params(bler_table,
                            cbs,
                            table_index,
                            category='PDSCH',
                            return_mcs_min_available=False):
    """Extract BLER sigmoid parameters from a BLER lookup table.

    Filters the table by physical channel category, MCS table index, and code
    block size, then returns the sigmoid center and scale parameters that
    approximate each MCS's BLER-vs-SINR curve.

    Parameters
    ----------
    bler_table : numpy structured array
        BLER lookup table with fields ``category``, ``table_index``,
        ``CBS_num_info_bits``, ``MCS``, ``sigmoid_center_db``, and
        ``sigmoid_scale_db``.
    cbs : int
        Code block size (number of information bits).
    table_index : int
        MCS table index (e.g., 1 for 64-QAM).
    category : str, optional
        Physical channel category. Default is ``'PDSCH'``.
    return_mcs_min_available : bool, optional
        If True, also return the lowest MCS index available for the
        selected configuration. Default is False.

    Returns
    -------
    bler_sigmoid_params : dict
        Dictionary with keys ``'center'`` and ``'scale'``, each mapping to
        a 1-D numpy array of per-MCS sigmoid parameters (in dB).
    mcs_min_available : int
        Minimum available MCS index. Only returned when
        *return_mcs_min_available* is True.

    Raises
    ------
    ValueError
        If the requested *cbs* is not present in the table.
    """

    # Select PDSCH and appropriate table index
    bler_table = bler_table[(bler_table['category'] == category) &
                            (bler_table['table_index'] == table_index)]

    available_cbs = np.unique(bler_table['CBS_num_info_bits'])
    if cbs not in available_cbs:
        raise ValueError(f'Code block size {cbs} not found in the data. ' +
                         f'Available CBS: {available_cbs}')

    # Select appropriate code block size, sorted by MCS
    rows = bler_table[bler_table['CBS_num_info_bits'] == cbs]
    rows = rows[np.argsort(rows['MCS'])]

    # Extract sigmoid parameters to approximate BLER tables
    bler_sigmoid_params = {
        'center': rows['sigmoid_center_db'],
        'scale': rows['sigmoid_scale_db'],
    }

    if return_mcs_min_available:
        mcs_min_available = rows['MCS'].min()
        return bler_sigmoid_params, mcs_min_available
    else:
        return bler_sigmoid_params


def sigmoid(x, center, scale):
    """Evaluate a logistic (sigmoid) function.

    Computes ``1 / (1 + exp(-(x - center) / scale))``.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s) (e.g., SINR in dB).
    center : float or numpy.ndarray
        Inflection point(s) of the sigmoid.
    scale : float or numpy.ndarray
        Steepness parameter(s); larger values produce a gentler slope.

    Returns
    -------
    float or numpy.ndarray
        Sigmoid output in the range (0, 1).
    """
    return 1 / (1 + np.exp(-(x - center) / scale))

# =============================================================================
# BLER Table Configuration
# =============================================================================

TABLE_INDEX = 1       # MCS table index (Table 1 for 64QAM)
CATEGORY = "PDSCH"    # Physical channel category

# Load BLER sigmoid fit parameters from CSV
BLER_TABLE = np.genfromtxt(
    "data/bler_sigmoid_fit_extended.csv",
    delimiter=',',
    dtype=None,
    names=True,
    encoding='utf-8',
)

# Extract sigmoid parameters to approximate BLER curves per MCS
BLER_SIGMOID_PARAMS, _ = get_bler_sigmoid_params(
    BLER_TABLE,
    100,  # CBS (Code Block Size) info bits
    TABLE_INDEX,
    category=CATEGORY,
    return_mcs_min_available=True,
)

# Spectral efficiency lookup table indexed by MCS
MCS_TO_SE = BLER_TABLE[
    (BLER_TABLE["category"] == CATEGORY)
    & (BLER_TABLE["table_index"] == TABLE_INDEX)
    & (BLER_TABLE["CBS_num_info_bits"] == 100)
]["spectral_efficiency"]


# =============================================================================
# Utilities
# =============================================================================

def get_bler(sinr: float) -> np.ndarray:
    """Compute the BLER for every MCS at a given SINR.

    Uses the module-level ``BLER_SIGMOID_PARAMS`` to evaluate the sigmoid
    approximation of each MCS's BLER curve.

    Parameters
    ----------
    sinr : float
        Signal-to-Interference-plus-Noise Ratio in dB.

    Returns
    -------
    numpy.ndarray
        Per-MCS BLER values (one entry per MCS index).
    """
    ack = sigmoid(sinr, BLER_SIGMOID_PARAMS['center'], BLER_SIGMOID_PARAMS['scale'])
    return 1. - ack


def bler_2_mcs(bler, target_bler):
    """Select the highest MCS whose BLER does not exceed a target.

    Parameters
    ----------
    bler : numpy.ndarray
        Per-MCS BLER values (as returned by :func:`get_bler`).
    target_bler : float
        Maximum acceptable BLER.

    Returns
    -------
    int
        Highest MCS index with ``bler <= target_bler``, or 0 if no MCS
        satisfies the constraint.
    """
    cond = bler <= target_bler
    valid_mcs = np.where(cond)[0]
    if len(valid_mcs) == 0:
        return 0
    else:
        return valid_mcs[-1]
