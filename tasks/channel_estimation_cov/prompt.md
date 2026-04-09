# MIMO Channel Estimator Implementation Task

## Objective

Implement a **channel estimator** for a MIMO detector in a wireless communication system that **minimizes the Normalized Validation Error (NVE)** — the average ratio between the BLER achieved by your estimator and the BLER under perfect channel knowledge (perfect CSI).

## Context

In the physical layer of wireless communication systems, a MIMO detector processes received signals to recover transmitted data. The detector receives the noisy signal from multiple receive antennas and must output soft bit information (log-likelihood ratios) for the channel decoder.

Your task is to improve the **channel estimation** stage. Accurate channel estimation is critical for equalization and detection performance, especially in challenging channel conditions with high mobility or delay spread.

**Key advantage of this task:** You are provided with pre-computed **channel covariance matrices** that capture the true second-order statistics of the evaluation channel. These matrices describe how the channel is correlated across subcarriers, OFDM symbols, and receive antennas.

## Requirements

### Function Signature

Your solution must define a function with the following signature:

```python
def mimo_detector(y, no):
    """
    MIMO detector function.

    Args:
        y: Received resource grid tensor of shape
           [batch_size, num_rx_ant, num_ofdm_symbols, num_subcarriers] (complex64)
        no: Noise variance (scalar, float32)

    Returns:
        llr: Log-likelihood ratios of shape
             [batch_size, num_ut, num_tx_ant, num_data_symbols * num_bits_per_symbol] (float32)
    """
```

### Configuration

**IMPORTANT: `link_config.py` and `channel_cov.pkl` are NOT available in your workspace. Do NOT try to read or find these files before running evaluation.**

Both files will be **automatically injected when you call the evaluation tool**. Write your solution using the documented imports and file reads; they will resolve correctly when the evaluation tool runs.

**Do NOT:**
- Try to read `link_config.py` or `channel_cov.pkl` from the workspace before calling the evaluation tool
- Create your own `link_config.py` or `channel_cov.pkl`
- Wait for these files to appear

**DO:**
- Call the evaluation tool to test your implementation
- The evaluation tool will provide both files automatically

`link_config.py` contains the following (for reference only - do not try to access this file):

**Objects:**
- `RG`: A `sionna.phy.ofdm.ResourceGrid` object (contains pilot pattern, resource grid structure)
- `SM`: A `sionna.phy.mimo.StreamManagement` object (manages MIMO streams)

**System Parameters:**
- `NUM_UT`: Number of user terminals (UTs)
- `NUM_TX_ANT`: Number of transmit antennas per UT
- `NUM_RX_ANT`: Number of receive antennas at the base station
- `NUM_BITS_PER_SYMBOL`: Bits per modulation symbol (defines constellation)
- `NUM_OFDM_SYMBOLS`: OFDM symbols per slot
- `FFT_SIZE`: FFT size for OFDM
- `SUBCARRIER_SPACING`: Subcarrier spacing in Hz
- `CYCLIC_PREFIX_LENGTH`: Cyclic prefix length in samples
- `NUM_GUARD_CARRIERS`: Guard carriers [left, right]
- `DC_NULL`: Whether DC subcarrier is nulled
- `PILOT_PATTERN`: Pilot pattern type (e.g., "kronecker")
- `PILOT_OFDM_SYMBOL_INDICES`: OFDM symbols containing pilots
- `CODERATE`: LDPC code rate

**Channel Parameters:**
- `CARRIER_FREQUENCY`: Carrier frequency in Hz

**Derived Constants:**
- `NUM_EFFECTIVE_SUBCARRIERS`: Subcarriers after removing guards/DC
- `NUM_DATA_SYMBOLS`: Data symbols per resource grid

### Channel Covariance Matrices

`channel_cov.pkl` contains pre-computed **second-order channel statistics** for the evaluation channel, estimated from many Monte Carlo channel samples.

**Loading the covariance matrices:**

Load `channel_cov.pkl` at **module level** (outside `mimo_detector`), before any `torch.compile` tracing:

```python
import pickle
import torch

with open("channel_cov.pkl", "rb") as f:
    channel_cov = pickle.load(f)

freq_cov_mat  = torch.tensor(channel_cov["freq"],  dtype=torch.complex64)  # [FFT_SIZE, FFT_SIZE]
time_cov_mat  = torch.tensor(channel_cov["time"],  dtype=torch.complex64)  # [NUM_OFDM_SYMBOLS, NUM_OFDM_SYMBOLS]
space_cov_mat = torch.tensor(channel_cov["space"], dtype=torch.complex64)  # [NUM_RX_ANT, NUM_RX_ANT]
```

**What each matrix represents:**

| Matrix | Shape | Meaning |
|--------|-------|---------|
| `freq_cov_mat` | `[FFT_SIZE, FFT_SIZE]` | Frequency-domain correlation: `E[h(f) h(f')^H]` averaged over time and space. Captures how correlated the channel is across subcarriers. |
| `time_cov_mat` | `[NUM_OFDM_SYMBOLS, NUM_OFDM_SYMBOLS]` | Time-domain correlation: `E[h(t) h(t')^H]` averaged over frequency and space. Captures how fast the channel varies across OFDM symbols. |
| `space_cov_mat` | `[NUM_RX_ANT, NUM_RX_ANT]` | Spatial correlation: `E[h_rx h_rx^H]` averaged over frequency and time. Captures antenna correlation at the base station. |

### Channel Estimate Tensor Layout

The channel estimate and error variance tensors passed to `LMMSEEqualizer` must have shapes:

```
h_hat:   [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]  complex64
err_var: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]  float32
           axis 0      axis 1   axis 2    axis 3       axis 4             axis 5          axis 6
```

**This is a 7-dimensional tensor.** Any channel estimate you compute must match this layout exactly.

### Technical Constraints

1. **`torch.compile` Compatibility**: Your implementation must be compatible with `torch.compile`. Avoid operations that prevent compilation.

2. **Practical Complexity**: Keep computational complexity practical. Avoid algorithms with exponential complexity in the number of antennas or constellation points.

3. **No Hardcoded Values**: Import all system parameters from `link_config.py`. Your detector must work for any valid configuration.

4. **Use LMMSE equalization and APP demapping**: Use the LMMSE equalizer and APP demapper included in Sionna. Focus on improving the channel estimator.

5. **Must use the covariance matrices**: Your channel estimator **must** use `freq_cov_mat`, `time_cov_mat`, and/or `space_cov_mat` to improve `h_hat` beyond the LS baseline. Loading the matrices but not applying them to the channel estimate is **not acceptable**. Any approach that ignores the covariance matrices is incomplete.

## Design Freedom

Your channel estimator **should not** simply replicate algorithms already available in Sionna (LS, LMMSE, etc.). You should go beyond what is available in Sionna.

The only requirements are:
1. Your detector takes the received signal `y` and noise variance `no` as input
2. It outputs LLRs with the correct shape
3. It uses the LMMSE equalizer and APP demapper from Sionna
4. It minimizes the NVE (normalized validation error)
5. **It actively uses the provided covariance matrices** to compute a better `h_hat` — not just loads them

## Evaluation

Use the provided evaluation tool to test your detector. The tool returns the **Normalized Validation Error (NVE)**: the mean ratio of your detector's BLER to the BLER under perfect CSI, computed across a range of SNR points.

\[
\text{NVE} = \frac{1}{N} \sum_{i=1}^{N} \frac{\text{BLER}_{\text{agent}}(\text{SNR}_i)}{\text{BLER}_{\text{perfect CSI}}(\text{SNR}_i)}
\]

**Lower is better.**

The LS baseline (code above) achieves approximately **NVE ≈ 94**. Your implementation must achieve **substantially lower NVE** than this — a result near or above the baseline means your estimator is not providing meaningful improvement and needs to be rethought.

## Hints

- The `RG.pilot_pattern` object contains pilot positions and symbols
- **Load `channel_cov.pkl` at module level**, not inside `mimo_detector`. The evaluation wraps the detector with `torch.compile`, so file I/O must complete before tracing begins.

## Baseline

```python
import pickle
import torch
from sionna.phy.ofdm import LSChannelEstimator, LMMSEEqualizer
from sionna.phy.mapping import Demapper
from link_config import RG, SM, NUM_BITS_PER_SYMBOL

# Load covariance matrices at module level (before any torch.compile tracing)
with open("channel_cov.pkl", "rb") as f:
    _cov = pickle.load(f)
_freq_cov  = torch.tensor(_cov["freq"],  dtype=torch.complex64)  # [FFT_SIZE, FFT_SIZE]
_time_cov  = torch.tensor(_cov["time"],  dtype=torch.complex64)  # [NUM_OFDM_SYMBOLS, NUM_OFDM_SYMBOLS]
_space_cov = torch.tensor(_cov["space"], dtype=torch.complex64)  # [NUM_RX_ANT, NUM_RX_ANT]

_ls_est = LSChannelEstimator(RG, interpolation_type="lin_time_avg")
_lmmse_equ = LMMSEEqualizer(RG, SM)
_demapper = Demapper("app", "qam", NUM_BITS_PER_SYMBOL)

def mimo_detector(y, no):
    # h_hat shape: [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
    # err_var: same shape, float32
    h_hat, err_var = _ls_est(y, no)
    # TODO: Improve h_hat using _freq_cov, _time_cov, _space_cov

    x_hat, no_eff = _lmmse_equ(y, h_hat, err_var, no)
    llr = _demapper(x_hat, no_eff)
    return llr
```

The baseline detector above uses a Least Squares (LS) channel estimator with linear interpolation and time averaging. Your task is to design a channel estimator that outperforms this LS baseline, while remaining compatible with `torch.compile` and adhering to the provided constraints.
