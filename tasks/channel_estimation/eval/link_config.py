# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Link configuration for MIMO detector evaluation.

This file defines the ResourceGrid and StreamManagement used by both
the evaluation script and the detector.
"""
import numpy as np
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid

# System constants
NUM_TX_ANT = 1
NUM_UT = 4
NUM_RX_ANT = 16
NUM_BITS_PER_SYMBOL = 2
NUM_OFDM_SYMBOLS = 14
FFT_SIZE = 72
SUBCARRIER_SPACING = 30e3
CYCLIC_PREFIX_LENGTH = 0
NUM_GUARD_CARRIERS = [0, 0]
DC_NULL = False
PILOT_PATTERN = "kronecker"
PILOT_OFDM_SYMBOL_INDICES = [2, 11]
CODERATE = 0.5

# Carrier and channel parameters
CARRIER_FREQUENCY = 3.5e9

# Stream management for single TX with NUM_TX_ANT streams
SM = StreamManagement(np.ones([1, NUM_UT]), NUM_TX_ANT)

# Resource grid
RG = ResourceGrid(
    num_ofdm_symbols=NUM_OFDM_SYMBOLS,
    fft_size=FFT_SIZE,
    subcarrier_spacing=SUBCARRIER_SPACING,
    num_tx=NUM_UT,
    num_streams_per_tx=NUM_TX_ANT,
    cyclic_prefix_length=CYCLIC_PREFIX_LENGTH,
    num_guard_carriers=NUM_GUARD_CARRIERS,
    dc_null=DC_NULL,
    pilot_pattern=PILOT_PATTERN,
    pilot_ofdm_symbol_indices=PILOT_OFDM_SYMBOL_INDICES
)

# Derived constants
NUM_EFFECTIVE_SUBCARRIERS = RG.num_effective_subcarriers
NUM_DATA_SYMBOLS = RG.num_data_symbols
