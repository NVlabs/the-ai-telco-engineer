# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Generate reference BLER curve under perfect CSI.

Simulates a MIMO OFDM uplink transmission with perfect channel knowledge
(no estimation error) and saves the resulting BLER vs SNR curve to a pickle
file. This reference is used by eval.py to compute the Normalized Validation
Error (NVE) of agent-designed channel estimators.

Usage: python bler_perfect_csi.py
Output: bler_perf_csi.pkl
"""
import os
import pickle

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np

import sionna.phy
sionna.phy.config.seed = 42

from sionna.phy import Block
from sionna.phy.ofdm import (
    ResourceGridMapper, LMMSEEqualizer, RemoveNulledSubcarriers,
)
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMi
from sionna.phy.channel import OFDMChannel, gen_single_sector_topology
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, BinarySource, Demapper
from sionna.phy.utils import ebnodb2no, sim_ber

from link_config import (
    NUM_TX_ANT, NUM_UT, NUM_RX_ANT, NUM_BITS_PER_SYMBOL, NUM_OFDM_SYMBOLS,
    CODERATE,
    CARRIER_FREQUENCY,
    SM, RG,
)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SNR_RANGE = (-9., -2., 2.0)  # (start, stop, step) in dB
BATCH_SIZE = 10
MAX_MC_ITER = 1000
NUM_TARGET_BLOCK_ERRORS = 1000
SPEED = 3.0  # UE speed in m/s


# ---------------------------------------------------------------------------
# End-to-end MIMO model with perfect CSI
# ---------------------------------------------------------------------------
class PerfectCSIModel(Block):
    """MIMO OFDM uplink model that uses the true channel for equalization.

    The channel frequency response is returned by the OFDMChannel block and
    fed directly into the LMMSE equalizer (error variance = 0), providing
    an upper-bound on detection performance.
    """

    def __init__(self):
        super().__init__()

        self._num_ut = NUM_UT
        self._num_streams_per_tx = NUM_TX_ANT
        self._num_bits_per_symbol = NUM_BITS_PER_SYMBOL
        self._coderate = CODERATE

        self._rg = RG
        self._sm = SM

        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        self._k = int(self._n * self._coderate)

        # Antenna arrays
        ut_array = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=CARRIER_FREQUENCY,
        )
        bs_array = AntennaArray(
            num_rows=1,
            num_cols=int(NUM_RX_ANT / 2),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=CARRIER_FREQUENCY,
        )

        # UMi channel model
        self._channel_model = UMi(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
        )
        self._channel = OFDMChannel(
            self._channel_model, self._rg,
            normalize_channel=True,
            return_channel=True,
        )

        # Transmitter chain
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        # Receiver chain (perfect CSI path)
        self._remove_nulled_subcarriers = RemoveNulledSubcarriers(self._rg)
        self._equalizer = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", NUM_BITS_PER_SYMBOL)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    @torch.compile
    def call(self, batch_size, ebno_db):
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)

        # Transmitter
        b = self._binary_source([batch_size, self._num_ut, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        # Channel (returns both the received signal and the true channel)
        topology = gen_single_sector_topology(
            batch_size, self._num_ut, "umi", max_ut_velocity=SPEED,
        )
        self._channel_model.set_topology(*topology)
        y, h = self._channel(x_rg, no)

        # Receiver with perfect CSI (err_var = 0)
        h_hat = self._remove_nulled_subcarriers(h)
        x_hat, no_eff = self._equalizer(y, h_hat, 0.0, no)
        llr = self._demapper(x_hat, no_eff)
        llr = torch.reshape(llr, [batch_size, self._num_ut, self._num_streams_per_tx, self._n])
        b_hat = self._decoder(llr)

        return b, b_hat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = PerfectCSIModel()

    snr_points = np.arange(SNR_RANGE[0], SNR_RANGE[1], SNR_RANGE[2])
    _, bler = sim_ber(
        model, snr_points,
        batch_size=BATCH_SIZE,
        max_mc_iter=MAX_MC_ITER,
        num_target_block_errors=NUM_TARGET_BLOCK_ERRORS,
    )
    bler = bler.cpu().numpy()

    with open("bler_perf_csi.pkl", "wb") as f:
        pickle.dump(bler, f)

    print(f"Saved BLER curve ({len(snr_points)} SNR points) to bler_perf_csi.pkl")
