# MIMO Channel Estimator Implementation Task

## Objective

Implement a **channel estimator** for a MIMO detector in a wireless communication system that **minimizes the Normalized Validation Error (NVE)** — the average ratio between the BLER achieved by your estimator and the BLER under perfect channel knowledge (perfect CSI).

## Context

In the physical layer of wireless communication systems, a MIMO detector processes received signals to recover transmitted data. The detector receives the noisy signal from multiple receive antennas and must output soft bit information (log-likelihood ratios) for the channel decoder.

Your task is to improve the **channel estimation** stage. Accurate channel estimation is critical for equalization and detection performance, especially in challenging channel conditions with high mobility or delay spread.

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

**IMPORTANT: `link_config.py` is NOT available in your workspace. Do NOT try to read or find this file.**

The file `link_config.py` will be **automatically injected when you call the evaluation tool**. You must write your solution based on the documentation below, trusting that the imports will work during evaluation.

Simply write your code with the documented imports (e.g., `from link_config import RG, SM, NUM_BITS_PER_SYMBOL`) and they will resolve correctly when the evaluation tool runs.

**Do NOT:**
- Try to read `link_config.py` from the workspace
- Create your own `link_config.py`
- Wait for `link_config.py` to appear

**DO:**
- Call the evaluation tool to test your implementation
- The evaluation tool will provide `link_config.py` automatically

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

## Design Freedom

Your channel estimator **should not** simply replicate algorithms already available in Sionna (LS, LMMSE, etc.). You should go beyond what is available in Sionna.

The only requirements are:
1. Your detector takes the received signal `y` and noise variance `no` as input
2. It outputs LLRs with the correct shape
3. It uses the LMMSE equalizer and APP demapper from Sionna
4. It minimizes the NVE (normalized validation error)

## Evaluation

Use the provided evaluation tool to test your detector. The tool returns the **Normalized Validation Error (NVE)**: the mean ratio of your detector's BLER to the BLER under perfect CSI, computed across a range of SNR points.

\[
\text{NVE} = \frac{1}{N} \sum_{i=1}^{N} \frac{\text{BLER}_{\text{agent}}(\text{SNR}_i)}{\text{BLER}_{\text{perfect CSI}}(\text{SNR}_i)}
\]

**Lower is better.**

The LS baseline (code above) achieves approximately **NVE ≈ 94**. Your implementation must achieve **substantially lower NVE** than this — a result near or above the baseline means your estimator is not providing meaningful improvement and needs to be rethought.

## Hints

- The `RG.pilot_pattern` object contains pilot positions and symbols

## Baseline

```python
from sionna.phy.ofdm import LSChannelEstimator, LMMSEEqualizer
from sionna.phy.mapping import Demapper
from link_config import RG, SM, NUM_BITS_PER_SYMBOL

_ls_est = LSChannelEstimator(RG, interpolation_type="lin_time_avg")
_lmmse_equ = LMMSEEqualizer(RG, SM)
_demapper = Demapper("app", "qam", NUM_BITS_PER_SYMBOL)

def mimo_detector(y, no):
    h_hat, err_var = _ls_est(y, no) # Improve the estimator
    x_hat, no_eff = _lmmse_equ(y, h_hat, err_var, no)
    llr = _demapper(x_hat, no_eff)
    return llr
```

The baseline detector above uses a Least Squares (LS) channel estimator with linear interpolation and time averaging. Your task is to design a channel estimator that outperforms this LS baseline, while remaining compatible with `torch.compile` and adhering to the provided constraints.