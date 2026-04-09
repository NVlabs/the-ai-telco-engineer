# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Link adaptation evaluation pipeline.

Evaluates an agent's MCS selection algorithm by replaying pre-generated
SINR trajectory scenarios through a simulated link adaptation loop.
Two metrics are computed per scenario:

1. **Long-term BLER** -- must remain below the target threshold
   (with a configurable tolerance).
2. **Spectral efficiency** -- the optimisation objective (higher is better).

The agent is expected to supply a ``draft.py`` module that exposes a
``mcs_selection(is_nack_hist, mcs_ackned_hist, bler_target) -> int`` callable.
"""

import importlib.util
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from utils import BLER_SIGMOID_PARAMS, sigmoid, MCS_TO_SE


def _load_mcs_selection_module(source_file: str):
    """Load the mcs_selection callable from a Python file. source_file is the filename (e.g. draft.py or solution.py)."""
    module_name = os.path.splitext(os.path.basename(source_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {source_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Configuration
# =============================================================================

BLER_TARGET = 0.1
NACK_REPORT_BATCH_SIZE = 5

BLER_TARGET_TOLERANCE = 0.1  # 10% tolerance
NUM_WORKERS = 10  # Number of parallel workers for evaluation


# =============================================================================
# Class for calling the link adaptation algorithm
# =============================================================================

class LinkAdaptation:
    """Stateful wrapper around an MCS selection function.

    Maintains internal ACK/NACK and MCS histories, and delegates the
    actual MCS decision to *mcs_selection_func*.

    Parameters
    ----------
    bler_target : float
        Long-term BLER target (e.g., 0.1 for 10 %).
    mcs_selection_func : callable
        Function with signature
        ``(is_nack_hist, mcs_ackned_hist, bler_target) -> int``
        that returns the MCS index for the next transmission.
    """

    def __init__(
        self,
        bler_target: float,
        mcs_selection_func: callable,
    ):
        self.bler_target = bler_target
        self.mcs_selection_func = mcs_selection_func
        self.mcs_unackned_hist = []
        self.mcs_ackned_hist = []
        self.is_nack_hist = []

    def __call__(self,
                 is_nack=None,
                 **args):
        """Record new HARQ feedback and select the next MCS.

        Parameters
        ----------
        is_nack : int, list of int, or None, optional
            Latest HARQ feedback (1 = NACK, 0 = ACK).  May be a single
            value, a list/array of values, or None (no new feedback).
        **args
            Additional keyword arguments (currently unused; reserved for
            future extensions such as CQI-based SINR estimates).

        Returns
        -------
        int
            MCS index chosen for the next transmission.
        """
        # Convert is_nack to list
        if is_nack is None:
            is_nack = []
        elif not hasattr(is_nack, '__len__'):
            is_nack = [is_nack]

        # Record ACK/NACK history
        self.is_nack_hist.extend(is_nack)

        # Update (un-)acknowledged MCS history
        self.mcs_ackned_hist.extend(self.mcs_unackned_hist[:len(is_nack)])
        self.mcs_unackned_hist = self.mcs_unackned_hist[len(is_nack):]

        # Select the MCS index for the next transmission
        mcs = self.mcs_selection_func(np.array(self.is_nack_hist),
                                      np.array(self.mcs_ackned_hist),
                                      self.bler_target)

        # Update list of unacknowledged MCS
        self.mcs_unackned_hist.append(mcs)

        return mcs


# =============================================================================
# Loads the trajectories for evaluation
# =============================================================================

# SINR range for trajectory normalization (in dB and linear scale)
SINR_BOUNDS_DB = (0, 20)
SINR_BOUNDS_LIN = (
    10 ** (SINR_BOUNDS_DB[0] / 10), 10 ** (SINR_BOUNDS_DB[1] / 10))

# Load pre-generated training trajectories
with open("data/trajectories_training.pkl", "rb") as f:
    TRAJECTORIES_TRAINING, TRAJECTORIES_GAINS_TRAINING = pickle.load(f)

# Normalize gains to [0, 1] and scale to SINR bounds
TRAJECTORIES_GAINS_TRAINING = (
    TRAJECTORIES_GAINS_TRAINING
    / np.max(TRAJECTORIES_GAINS_TRAINING, axis=-1, keepdims=True)
)
TRAJECTORIES_GAINS_TRAINING = (
    TRAJECTORIES_GAINS_TRAINING * (SINR_BOUNDS_LIN[1] - SINR_BOUNDS_LIN[0])
    + SINR_BOUNDS_LIN[0]
)


# =============================================================================
# Evaluation
# =============================================================================

def generate_is_nack(sinr,
                     mcs,
                     bler_sigmoid_params,
                     random_generator,
                     return_bler=False):
    """Stochastically generate HARQ ACK/NACK feedback.

    For each (SINR, MCS) pair the BLER is computed via the sigmoid
    approximation, and a Bernoulli draw determines whether the
    transmission is a NACK.

    Parameters
    ----------
    sinr : float or array-like of float
        SINR value(s) (linear scale).
    mcs : int or array-like of int
        MCS index/indices (must match the length of *sinr*).
    bler_sigmoid_params : dict
        Sigmoid parameters (keys ``'center'`` and ``'scale'``) used to
        approximate the BLER-vs-SINR curve for each MCS.
    random_generator : numpy.random.RandomState
        Pseudo-random number generator used for the Bernoulli draw.
    return_bler : bool, optional
        If True, also return the computed BLER value(s).
        Default is False.

    Returns
    -------
    is_nack : int or numpy.ndarray of int
        HARQ feedback (1 = NACK, 0 = ACK).  Scalar when *sinr* is scalar,
        array otherwise.
    bler : float or numpy.ndarray of float
        Per-transmission BLER. Only returned when *return_bler* is True.
    """
    if not hasattr(sinr, '__len__'):
        sinr = np.array([sinr])
        return_scalar = True
    else:
        sinr = np.array(sinr)
        return_scalar = False
    mcs = np.array([mcs]) if not hasattr(mcs, '__len__') else np.array(mcs)
    assert len(sinr) == len(mcs)

    # Compute BLER
    bler = 1 - sigmoid(sinr, center=BLER_SIGMOID_PARAMS['center'][mcs], scale=BLER_SIGMOID_PARAMS['scale'][mcs])

    # Generate ACK/NACK
    rand01 = np.array([random_generator.rand() for _ in range(len(sinr))])
    is_nack = (rand01 < bler).astype(int)
    is_nack = is_nack[0] if return_scalar else is_nack
    if return_bler:
        return is_nack, bler
    else:
        return is_nack


def run_la(la_algo: object,
           sinr_hist: list,
           bler_sigmoid_params: dict,
           sinr_from_cqi_hist: list | None = None,
           cqi_hist: list | None = None,
           sinr_estimator_from_cqi: object | None = None,
           nack_report_batch_size=1,
           mcs_to_se=None,
           random_generator=None,
           seed=42):
    """Run a full link adaptation simulation over a SINR trajectory.

    At each slot the function:

    1. Generates stochastic HARQ feedback via :func:`generate_is_nack`.
    2. Optionally estimates SINR from CQI feedback.
    3. Delivers batched ACK/NACK reports to *la_algo* every
       *nack_report_batch_size* slots.
    4. Records the selected MCS and achieved rate.

    Parameters
    ----------
    la_algo : LinkAdaptation
        Callable link adaptation object (see :class:`LinkAdaptation`).
    sinr_hist : array-like of float
        Ground-truth SINR trajectory (one value per slot, linear scale).
    bler_sigmoid_params : dict
        Sigmoid parameters (keys ``'center'``, ``'scale'``) for BLER
        approximation (see :func:`utils.get_bler_sigmoid_params`).
    sinr_from_cqi_hist : array-like of float or None, optional
        Pre-computed SINR estimates derived from CQI.  Mutually exclusive
        with *cqi_hist*.  Default is None (no CQI side-information).
    cqi_hist : array-like of int or None, optional
        Raw CQI feedback per slot (``np.nan`` where unavailable).
        Requires *sinr_estimator_from_cqi* to be set.  Default is None.
    sinr_estimator_from_cqi : callable or None, optional
        Online CQI-to-SINR mapping learner.  Required when *cqi_hist* is
        provided.  Default is None.
    nack_report_batch_size : int, optional
        Number of slots between consecutive HARQ feedback reports.
        Default is 1 (report every slot).
    mcs_to_se : array-like of float or None, optional
        Spectral efficiency lookup table indexed by MCS.  When provided,
        the achieved rate per slot is recorded.  Default is None.
    random_generator : numpy.random.RandomState or None, optional
        PRNG instance.  If None, a new one is created with *seed*.
    seed : int, optional
        Random seed used to initialise or re-seed *random_generator*.
        Default is 42.

    Returns
    -------
    is_nack_hist : numpy.ndarray of int
        HARQ feedback history (1 = NACK, 0 = ACK) for every slot.
    rate_hist : numpy.ndarray of float
        Achieved spectral efficiency per slot (NaN when *mcs_to_se* is
        None).
    la_algo : LinkAdaptation
        The link adaptation object after the full simulation (carries
        updated internal state).
    mcs_hist : numpy.ndarray of int
        Selected MCS index per slot.
    sinr_from_cqi_hist : numpy.ndarray of float
        SINR estimates derived from CQI feedback (NaN where unavailable).

    Raises
    ------
    ValueError
        If *cqi_hist* and *sinr_estimator_from_cqi* are not both provided
        or both omitted, or if *sinr_from_cqi_hist* and *cqi_hist* are
        supplied simultaneously.
    """
    n_slots = len(sinr_hist)
    if (cqi_hist is None) != (sinr_estimator_from_cqi is None):
        raise ValueError(
            'Either both sinr_from_cqi_hist and cqi_hist or neither must be provided.')
    if (sinr_from_cqi_hist is not None) & (cqi_hist is not None):
        raise ValueError('sinr_from_cqi_hist and cqi_hist cannot be both provided. ' +
                         'Use either sinr_from_cqi_hist or cqi_hist, not both.')
    if (cqi_hist is not None) & (sinr_estimator_from_cqi is not None):
        # Learn CQI-to-SINR mapping
        learn_sinr_from_cqi = True
        sinr_from_cqi_hist = np.full(n_slots, np.nan)
        cqi_hist = np.array(cqi_hist)
    else:
        learn_sinr_from_cqi = False
        if sinr_from_cqi_hist is None:
            # No SINR from CQI feedback
            sinr_from_cqi_hist = np.full(n_slots, np.nan)
        else:
            # Use pre-computed SINR from CQI feedback
            sinr_from_cqi_hist = np.array(sinr_from_cqi_hist)

    if random_generator is None:
        random_generator = np.random.RandomState(seed=seed)
    else:
        random_generator.seed(seed)

    # Initialize history
    is_nack_hist = np.zeros(n_slots)

    mcs_hist = np.zeros(n_slots)
    if mcs_to_se is not None:
        rate_hist = np.zeros(n_slots)
    else:
        rate_hist = np.full(n_slots, np.nan)

    # MCS for the first slot
    mcs_i = la_algo()
    mcs_hist[0] = mcs_i

    # Initialize counter and wait time
    n_iter_since_last_feedback = 0

    for ii in range(n_slots):
        n_iter_since_last_feedback += 1

        # ACK/NACK observation
        is_nack_i = generate_is_nack(sinr_hist[ii],
                                     mcs_i,
                                     bler_sigmoid_params,
                                     random_generator)
        is_nack_hist[ii] = is_nack_i

        # Estimate SINR from CQI feedback
        if learn_sinr_from_cqi:
            sinr_cqi_i = sinr_estimator_from_cqi(cqi_hist[ii],
                                                 is_nack_i,
                                                 mcs_i)
            sinr_from_cqi_hist[ii] = sinr_cqi_i

        # Achieved rate
        if mcs_to_se is not None:
            rate_hist[ii] = (is_nack_i == 0) * mcs_to_se[mcs_i]

        # Feedback is reported in batch every `nack_report_batch_size` slots
        if n_iter_since_last_feedback == nack_report_batch_size:
            # Report CQI and ACK/NACK in batch
            sinr_cqi_reported = sinr_from_cqi_hist[ii -
                                                   nack_report_batch_size + 1: ii+1]
            is_nack_reported = is_nack_hist[ii -
                                            nack_report_batch_size + 1: ii+1]

            # Reinitialize counter and wait time
            n_iter_since_last_feedback = 0

            # Select new MCS
            mcs_i = la_algo(is_nack=is_nack_reported,
                            sinr_cqi=sinr_cqi_reported)
        else:
            mcs_i = la_algo()

        if ii < n_slots - 1:
            mcs_hist[ii+1] = mcs_i

    return is_nack_hist, rate_hist, la_algo, mcs_hist, sinr_from_cqi_hist

def _evaluate_single_scenario(scenario_idx: int, scenario: np.ndarray, source_file: str) -> dict:
    """Evaluate the agent's MCS selection on a single SINR scenario.

    Intended as a worker function for parallel execution via
    :class:`~concurrent.futures.ProcessPoolExecutor`.  Each worker
    loads the module from source_file, runs the simulation, and
    returns aggregated metrics.

    Parameters
    ----------
    scenario_idx : int
        Index of the scenario (used to identify results).
    scenario : numpy.ndarray
        1-D SINR gains trajectory for the scenario (linear scale).
    source_file : str
        Python file to load (e.g. draft.py or solution.py).

    Returns
    -------
    dict
        Result dictionary with keys:

        * ``scenario_idx`` -- echo of the input index.
        * ``long_term_bler`` -- average BLER over the trajectory
          (None on error).
        * ``success`` -- whether the BLER constraint is met.
        * ``metric`` -- mean spectral efficiency (None on error).
        * ``error`` -- traceback string, or None on success.
    """
    try:
        module = _load_mcs_selection_module(source_file)

        # Create MCS selection algorithm with agent's function
        la_algo = LinkAdaptation(
            BLER_TARGET,
            mcs_selection_func=module.mcs_selection,
        )

        # Run link adaptation simulation
        is_nack_hist, rate_hist, _, _, _ = run_la(
            la_algo,
            scenario,
            BLER_SIGMOID_PARAMS,
            nack_report_batch_size=NACK_REPORT_BATCH_SIZE,
            mcs_to_se=MCS_TO_SE,
        )

        # Compute metrics for this scenario
        long_term_bler = np.mean(is_nack_hist)
        success = long_term_bler < BLER_TARGET * (1.0 + BLER_TARGET_TOLERANCE)
        metric = np.mean(rate_hist)

        return {
            "scenario_idx": scenario_idx,
            "long_term_bler": long_term_bler,
            "success": success,
            "metric": metric,
            "error": None,
        }

    except SyntaxError as e:
        import traceback
        return {
            "scenario_idx": scenario_idx,
            "long_term_bler": None,
            "success": False,
            "metric": None,
            "error": f"SyntaxError: {e}\n{traceback.format_exc()}",
        }

    except Exception as e:
        import traceback
        return {
            "scenario_idx": scenario_idx,
            "long_term_bler": None,
            "success": False,
            "metric": None,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }

def evaluate_mcs_selection() -> str:
    """Evaluate the agent's MCS selection algorithm end-to-end.

    Imports the agent's ``mcs_selection`` module, validates its interface,
    then runs the algorithm in parallel across every training trajectory.
    The output report includes:

    * **Pass / fail status** -- whether the BLER constraint is met on all
      scenarios.
    * **Average spectral efficiency** -- the optimisation metric.
    * **BLER statistics** (min / median / max in log10 scale).

    Returns
    -------
    str
        Human-readable evaluation report.
    """
    source_file = sys.argv[1] if len(sys.argv) > 1 else "draft.py"

    # Try to import the agent's MCS selection module (validate before parallel execution)
    try:
        mcs_selection = _load_mcs_selection_module(source_file)
    except ImportError as e:
        import traceback
        return f"FAILURE,\nERROR: Could not import from {source_file}: {e}\n\n{traceback.format_exc()}"
    except SyntaxError as e:
        import traceback
        return f"FAILURE,\nERROR: Syntax error in {source_file}: {e}\n\n{traceback.format_exc()}"
    except Exception as e:
        import traceback
        return f"FAILURE,\nERROR: Failed to load {source_file}: {e}\n\n{traceback.format_exc()}"

    # Validate the function signature
    if not callable(getattr(mcs_selection, "mcs_selection", None)):
        return f"FAILURE,\nERROR: {source_file} must define a callable 'mcs_selection' function.\n\n"

    # Run evaluation across all scenarios in parallel
    long_term_bler = {}
    success = {}
    metric = {}
    errors = []

    try:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all scenarios for parallel evaluation
            futures = {
                executor.submit(_evaluate_single_scenario, idx, scenario, source_file): idx
                for idx, scenario in enumerate(TRAJECTORIES_GAINS_TRAINING)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                scenario_idx = result["scenario_idx"]

                if result["error"] is not None:
                    errors.append(f"Scenario {scenario_idx}: {result['error']}")
                    success[scenario_idx] = False
                    metric[scenario_idx] = 0.0
                else:
                    long_term_bler[scenario_idx] = result["long_term_bler"]
                    success[scenario_idx] = result["success"]
                    metric[scenario_idx] = result["metric"]

    except Exception as e:
        import traceback
        return f"FAILURE,\nERROR: Runtime error during evaluation: {e}\n\n{traceback.format_exc()}"

    # Check for errors during parallel execution
    if errors:
        return f"FAILURE,\nERROR: Runtime errors during evaluation:\n" + "\n".join(errors[:3])

    # Compute BLER statistics in log10 domain
    bler_values = np.array(list(long_term_bler.values()))
    bler_min_log = np.log10(np.min(bler_values))
    bler_median_log = np.log10(np.median(bler_values))
    bler_max_log = np.log10(np.max(bler_values))
    # Reference: log10(0.1) = -1.0, log10(0.01) = -2.0

    # Generate output report: first line is "SUCCESS, <metric>" or "FAILURE, <metric>", then optional lines
    all_success = all(success.values())
    average_metric = np.mean(list(metric.values()))
    first_line = f"SUCCESS, {average_metric:.4f}" if all_success else f"FAILURE, {average_metric:.4f}"
    output_lines = [
        first_line,
        "",
        "BLER Statistics (log10 scale, target is -1.0 i.e. 10%):",
        f"  Min BLER:    {bler_min_log:.2f} (best scenario)",
        f"  Median BLER: {bler_median_log:.2f} (typical scenario)",
        f"  Max BLER:    {bler_max_log:.2f} (worst scenario)",
    ]
    if not all_success:
        failed_scenarios = [s for s, passed in success.items() if not passed]
        output_lines.insert(
            2,
            f"FAILED: {len(failed_scenarios)}/{TRAJECTORIES_GAINS_TRAINING.shape[0]} "
            "scenarios did not meet BLER target.",
        )
    return "\n".join(output_lines)


if __name__ == "__main__":
    result = evaluate_mcs_selection()
    print(result)
