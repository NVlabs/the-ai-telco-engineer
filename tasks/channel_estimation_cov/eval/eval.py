# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
MIMO Detector Evaluation Script for Sionna.

This script evaluates a MIMO detector for a 4x16 MIMO uplink transmission
using the 3GPP CDL-B channel model.

The detector must be implemented in a Python file (default 'draft.py') in
the same directory, with a function called 'mimo_detector(y, no)'.

The detector performs channel estimation, equalization, and demapping.

Usage: python eval.py [source_file]
  source_file: path to the detector module (default: draft.py).
"""
import os
import sys
import importlib.util
from contextlib import redirect_stdout

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

import torch
import numpy as np
import pickle
import sionna.phy
sionna.phy.config.seed = 42

import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

from sionna.phy import Block
from sionna.phy.ofdm import ResourceGridMapper
from sionna.phy.channel.tr38901 import AntennaArray, UMi, Antenna
from sionna.phy.channel import OFDMChannel, gen_single_sector_topology
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, BinarySource
from sionna.phy.utils import ebnodb2no, sim_ber

# Import link configuration
from link_config import (
    NUM_TX_ANT, NUM_UT, NUM_RX_ANT, NUM_BITS_PER_SYMBOL, NUM_OFDM_SYMBOLS,
    CODERATE,
    CARRIER_FREQUENCY,
    SM, RG
)

SNR_RANGE = (-9., -2., 2.0)  # (start, stop, step) in dB
BATCH_SIZE = 10
MAX_MC_ITER = 1000
NUM_TARGET_BLOCK_ERRORS = 1000
TARGET_BLER = 1e-3

SPEED = 3.0


class MIMOModel(Block):
    """MIMO OFDM transmission model with CDL channel for detector evaluation.

    This model simulates a 4x16 MIMO uplink transmission using:
    - 4 TX antennas (UT) and 16 RX antennas (BS)
    - CDL-B channel model (good for spatial multiplexing)
    - 5G NR LDPC coding with rate 0.5
    - QPSK modulation
    - Custom MIMO detector that performs channel estimation, equalization, and demapping
    """

    def __init__(self, detector_fn):
        super().__init__()

        # Store detector function
        self._detector_fn = detector_fn

        # Use imported configuration
        self._num_tx_ant = NUM_TX_ANT
        self._num_rx_ant = NUM_RX_ANT
        self._num_ut = NUM_UT
        self._num_streams_per_tx = NUM_TX_ANT
        self._num_ofdm_symbols = NUM_OFDM_SYMBOLS
        self._num_bits_per_symbol = NUM_BITS_PER_SYMBOL
        self._coderate = CODERATE

        # Use imported resource grid and stream management
        self._rg = RG
        self._sm = SM

        # Coding parameters
        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        self._k = int(self._n * self._coderate)

        # Antenna arrays - UT with 4 antennas, BS with 16 antennas
        self._ut_array = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=CARRIER_FREQUENCY
        )
        self._bs_array = AntennaArray(
            num_rows=1,
            num_cols=int(self._num_rx_ant / 2),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=CARRIER_FREQUENCY
        )

        # CDL channel model
        self._umi = UMi(carrier_frequency=CARRIER_FREQUENCY,
                    o2i_model='low',
                    ut_array=self._ut_array,
                    bs_array=self._bs_array,
                    direction='uplink')

        # Channel application
        self._channel = OFDMChannel(self._umi, self._rg, return_channel=False, normalize_channel=True)

        # Transmitter components
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        # Receiver - only decoder, detector handles the rest
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    @torch.compile
    def call(self, batch_size, ebno_db):
        # Compute noise power
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)

        # Transmitter
        b = self._binary_source([batch_size, self._num_ut, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        # Channel - generate CIR and convert to frequency domain
        topology = gen_single_sector_topology(batch_size, self._num_ut, 'umi', max_ut_velocity=SPEED)
        self._umi.set_topology(*topology)

        # Apply channel
        y = self._channel(x_rg, no)

        # Call the detector
        # Input: y [batch, num_rx_ant, num_ofdm_symbols, num_effective_subcarriers]
        # Output: llr [batch, 1, num_streams_per_tx, n] where n = num_data_symbols * num_bits_per_symbol
        llr = self._detector_fn(y, no)

        # Reshape LLRs if needed to match expected shape [batch, 1, num_streams, n]
        llr = torch.reshape(llr, [batch_size, self._num_ut, self._num_streams_per_tx, self._n])

        # Decoding
        b_hat = self._decoder(llr)

        return b, b_hat


def _load_detector_module(source_file: str):
    """Load the mimo_detector callable from a Python file (e.g. draft.py or solution.py)."""
    module_name = os.path.splitext(os.path.basename(source_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {source_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_detector(source_file: str = "draft.py"):
    """Main evaluation function.

    Loads the mimo_detector function from the given source file and evaluates it.
    Output format: first line "SUCCESS, <metric>" or "FAILURE," then optional text.
    Evaluation always succeeds unless the code crashed (import/runtime error).
    """
    # Load the agent's detector module (same pattern as link_adaptation eval)
    try:
        module = _load_detector_module(source_file)
    except ImportError as e:
        import traceback
        return f"FAILURE,\nERROR: Could not import from {source_file}: {e}\n\n{traceback.format_exc()}"
    except SyntaxError as e:
        import traceback
        return f"FAILURE,\nERROR: Syntax error in {source_file}: {e}\n\n{traceback.format_exc()}"
    except Exception as e:
        import traceback
        return f"FAILURE,\nERROR: Failed to load {source_file}: {e}\n\n{traceback.format_exc()}"

    if not callable(getattr(module, "mimo_detector", None)):
        return f"FAILURE,\nERROR: {source_file} must define a callable 'mimo_detector' function.\n\n"

    try:
        # Create model with agent's detector
        model = MIMOModel(module.mimo_detector)

        # Run simulation over a coarse grid to find approximate range
        snr_points = np.arange(SNR_RANGE[0], SNR_RANGE[1], SNR_RANGE[2])

        # Suppress verbose output from sim_ber to avoid context window overflow
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull):
                _, bler = sim_ber(model, snr_points, batch_size=BATCH_SIZE, max_mc_iter=MAX_MC_ITER,
                                num_target_block_errors=NUM_TARGET_BLOCK_ERRORS, target_bler=TARGET_BLER)
        bler = bler.cpu().numpy()

        # Load BLER for perfect CSI
        with open("bler_perf_csi.pkl", "rb") as f:
            bler_perf_csi = pickle.load(f)
        
        # Discard SNR points where perfect CSI BLER is 0 (avoid division by 0)
        nonzero = bler_perf_csi > 0
        bler_perf_csi = bler_perf_csi[nonzero]
        bler = bler[nonzero]

        # Compute the BLER of the agent's detector
        nve = np.mean(bler / bler_perf_csi)
        if np.isnan(nve) or np.isinf(nve):
            return f"FAILURE,\nERROR: NaN or Inf"

        return f"SUCCESS, {nve:.4f}"

    except Exception as e:
        import traceback
        return f"FAILURE,\nERROR: Runtime error during evaluation: {e}\n\n{traceback.format_exc()}"


if __name__ == "__main__":
    source_file = sys.argv[1] if len(sys.argv) > 1 else "draft.py"
    result = evaluate_detector(source_file)
    print(result)
