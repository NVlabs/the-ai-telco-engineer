# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Trajectory of gains for link adaptation evaluation.

This script generates realistic UE (User Equipment) trajectories within a 3D urban
scene using Sionna's ray-tracer. For each trajectory, it computes
the channel frequency response and corresponding path gains across multiple slots.

Output:
    Two pickle files (training and evaluation sets, 50/50 split):
    - <output>_training.pkl and <output>_evaluation.pkl, each containing:
        - trajectories: Array of shape (num_trajectories/2, num_slots, 3) with 3D positions
        - trajectories_gains: Array of shape (num_trajectories/2, num_slots) with path gains
"""

import argparse
import pickle

import numpy as np
import sionna.rt as rt
from sionna.rt import (
    PathSolver,
    PlanarArray,
    RadioMapSolver,
    Receiver,
    Transmitter,
    load_scene,
    subcarrier_frequencies,
)


# =============================================================================
# Configuration Constants
# =============================================================================

# Radio map computation parameters
RM_CELL_SIZE = 0.5              # Cell size in meters for radio map resolution
RM_SAMPLES_PER_TX = 10**8       # Number of Monte Carlo samples for radio map
RM_MAX_DEPTH = 5                # Maximum number of reflections for radio map

# Trajectory and mobility parameters
SLOT_DURATION = 0.5e-3          # Duration of one slot in seconds (0.5 ms)
SPEED_MIN = 3.0                     # UE minimum speed in m/s
SPEED_MAX = 14.0                     # UE maximum speed in m/s

# Path solver parameters
PS_BATCH_SIZE = 100             # Number of receiver positions processed per batch
PS_NUM_SAMPLES_PER_TX = 10**6   # Monte Carlo samples for path computation
PS_MAX_DEPTH = 5                # Maximum reflections for path solver

# Channel frequency response parameters
NUM_SUBCARRIERS = 128           # Number of OFDM subcarriers
SUBCARRIER_SPACING = 30e3       # Subcarrier spacing in Hz (30 kHz for 5G NR)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for trajectory generation.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - num_slots: Number of time slots per trajectory
            - num_trajectories: Number of trajectories to generate
            - output: Path to output pickle file
    """
    parser = argparse.ArgumentParser(
        description="Generate UE trajectories with channel gains using ray tracing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-slots",
        type=int,
        default=3000,
        help="Number of time slots per trajectory",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=100,
        help="Number of trajectories to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/trajectories",
        help="Base path for output files (with or without .pkl extension)",
    )
    return parser.parse_args()


def setup_scene() -> rt.Scene:
    """
    Load and configure the 3D scene with transmitter and receiver arrays.

    The scene uses the Munich urban environment from Sionna's built-in scenes.
    A single transmitter (base station) is placed at a fixed location with
    a directional antenna pattern (3GPP TR 38.901).

    Returns:
        rt.Scene: Configured Sionna scene with TX array and placeholder RX array.
    """
    scene = load_scene(rt.scene.munich)

    # Configure transmitter antenna array (base station)
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        polarization="V",
        pattern="tr38901",  # 3GPP TR 38.901 antenna pattern
    )

    # Configure receiver antenna array (UE with isotropic pattern)
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        polarization="V",
        pattern="iso",  # Isotropic pattern for mobile UE
    )

    # Add transmitter at fixed position (base station location)
    tx = Transmitter(name="tx", position=(46, -217, 6.0))
    scene.add(tx)
    tx.look_at((45.0, -233.0, 1.5))  # Point antenna toward coverage area

    return scene


def compute_radio_map(scene: rt.Scene) -> rt.RadioMap:
    """
    Compute the radio coverage map for the scene.

    The radio map provides path gain information across the scene,
    which is used to sample valid initial positions for trajectories.

    Args:
        scene: Configured Sionna scene with transmitter.

    Returns:
        rt.RadioMap: Computed radio map with path gain information.
    """
    rm = RadioMapSolver()(
        scene,
        cell_size=RM_CELL_SIZE,
        samples_per_tx=RM_SAMPLES_PER_TX,
        refraction=False,
        max_depth=RM_MAX_DEPTH,
    )
    return rm


def add_receivers_to_scene(scene: rt.Scene, batch_size: int) -> None:
    """
    Add placeholder receivers to the scene for batch processing.

    Args:
        scene: Sionna scene to add receivers to.
        batch_size: Number of receivers to add (determines batch processing size).
    """
    for i in range(batch_size):
        scene.add(Receiver(name=f"rx-{i}", position=np.array([0.0, 0.0, 0.0])))


def generate_trajectories(
    scene: rt.Scene,
    radio_map: rt.RadioMap,
    num_trajectories: int,
    num_slots: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate UE trajectories with corresponding channel gains.

    For each trajectory:
    1. Sample a valid starting position from the radio map
    2. Choose a random direction of movement
    3. Compute positions for all time slots along the trajectory
    4. Calculate channel gains at each position using ray tracing
    5. Validate that all positions have valid paths (non-zero gain)

    Args:
        scene: Configured scene with transmitter and receivers.
        radio_map: Pre-computed radio map for position sampling.
        num_trajectories: Target number of valid trajectories to generate.
        num_slots: Number of time slots (positions) per trajectory.

    Returns:
        tuple: (trajectories, trajectories_gains)
            - trajectories: Array of shape (num_trajectories, num_slots, 3)
            - trajectories_gains: Array of shape (num_trajectories, num_slots)
    """
    # Validate that num_slots is divisible by batch size
    assert num_slots % PS_BATCH_SIZE == 0, (
        f"num_slots ({num_slots}) must be divisible by PS_BATCH_SIZE ({PS_BATCH_SIZE})"
    )

    # Compute derived parameters
    num_iterations = num_slots // PS_BATCH_SIZE
    frequencies = subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING)

    # Initialize path solver
    path_solver = PathSolver()

    # Storage for valid trajectories
    trajectories = []
    trajectories_gains = []
    num_invalid = 0

    while len(trajectories) < num_trajectories:
        # Sample initial positions with minimum path gain threshold
        start, _ = radio_map.sample_positions(
            num_pos=num_trajectories,
            min_val_db=-120.0,  # Minimum path gain in dB
            min_dist=3.0,       # Minimum distance from TX in meters
        )
        start = start.numpy()
        start = np.squeeze(start, axis=0)

        # Process each candidate trajectory
        for i in range(num_trajectories):
            start_i = start[i]

            # Sample random direction of displacement (uniform in [0, 2π])
            angle = np.random.uniform(0, 2 * np.pi)

            # Compute displacement vector
            speed = np.random.uniform(SPEED_MIN, SPEED_MAX)
            distance_between_slots = speed * SLOT_DURATION
            total_displacement = distance_between_slots * (num_slots - 1)
            displacement = total_displacement * np.array([np.cos(angle), np.sin(angle), 0.0])

            # Compute end position for full trajectory
            end_i = start_i + displacement
            positions = np.linspace(start_i, end_i, num_slots, endpoint=True)

            # Storage for gains at each position
            gains = []
            successful = True

            # Process trajectory in batches
            for j in range(num_iterations):
                start_idx = j * PS_BATCH_SIZE
                end_idx = start_idx + PS_BATCH_SIZE

                # Update receiver positions for this batch
                for k, pos in enumerate(positions[start_idx:end_idx]):
                    scene.get(f"rx-{k}").position = pos

                # Compute propagation paths using ray tracing
                paths = path_solver(
                    scene,
                    refraction=False,
                    max_depth=PS_MAX_DEPTH,
                    samples_per_src=PS_NUM_SAMPLES_PER_TX,
                )

                # Extract number of valid paths per receiver
                num_paths = paths.valid.numpy().sum(axis=-1)[0]

                # Compute channel frequency response and average gain over subcarriers
                cfr = paths.cfr(frequencies, out_type="numpy")
                gain = np.squeeze(np.mean(np.square(np.abs(cfr)), axis=-1))

                gains.append(gain)

                # Validate: trajectory is invalid if any position has no path or zero gain
                success = (num_paths > 0) & (gain > 0)
                if not np.all(success):
                    successful = False
                    break

            if not successful:
                num_invalid += 1
                continue

            # Store valid trajectory
            gains = np.concatenate(gains, axis=0)
            trajectories.append(positions)
            trajectories_gains.append(gains)

            print(
                f"Valid trajectories: {len(trajectories)}/{num_trajectories} | "
                f"Invalid: {num_invalid}",
                end="\r",
            )

            if len(trajectories) >= num_trajectories:
                break

    print()  # New line after progress updates
    return np.array(trajectories), np.array(trajectories_gains)


def get_output_paths(output_path: str) -> tuple[str, str]:
    """
    Compute training and evaluation output file paths from base path.

    Handles both cases where the user provides a path with or without
    the '.pkl' extension.

    Args:
        output_path: Base path (e.g., 'data/trajectories' or 'data/trajectories.pkl').

    Returns:
        tuple: (training_path, evaluation_path) with '_training.pkl' and
            '_evaluation.pkl' suffixes.
    """
    if output_path.endswith(".pkl"):
        base_path = output_path[:-4]  # Remove '.pkl'
    else:
        base_path = output_path

    return f"{base_path}_training.pkl", f"{base_path}_evaluation.pkl"


def save_trajectories(
    trajectories: np.ndarray,
    trajectories_gains: np.ndarray,
    output_path: str,
) -> None:
    """
    Save generated trajectories and gains to pickle files.

    The data is split 50/50 into training and evaluation sets, saved as
    separate files with '_training.pkl' and '_evaluation.pkl' suffixes.

    Args:
        trajectories: Array of trajectory positions, shape (N, num_slots, 3).
        trajectories_gains: Array of path gains, shape (N, num_slots).
        output_path: Base path for output files (e.g., 'data/trajectories' or
            'data/trajectories.pkl' will produce 'data/trajectories_training.pkl'
            and 'data/trajectories_evaluation.pkl').
    """
    training_set = trajectories[:int(0.5 * len(trajectories))]
    training_set_gains = trajectories_gains[:int(0.5 * len(trajectories_gains))]
    evaluation_set = trajectories[int(0.5 * len(trajectories)):]
    evaluation_set_gains = trajectories_gains[int(0.5 * len(trajectories_gains)):]

    output_path_training, output_path_evaluation = get_output_paths(output_path)

    with open(output_path_training, "wb") as f:
        pickle.dump([training_set, training_set_gains], f)
    print(f"Saved {len(training_set)} trajectories to {output_path_training}")

    with open(output_path_evaluation, "wb") as f:
        pickle.dump([evaluation_set, evaluation_set_gains], f)
    print(f"Saved {len(evaluation_set)} trajectories to {output_path_evaluation}")


def main() -> None:
    """Main entry point for trajectory generation."""
    args = parse_arguments()

    # Compute actual output file paths
    output_training, output_evaluation = get_output_paths(args.output)

    print("=" * 60)
    print("Trajectory Generator for Link Adaptation Evaluation")
    print("=" * 60)
    print(f"  Number of slots:        {args.num_slots}")
    print(f"  Number of trajectories: {args.num_trajectories}")
    print(f"  Output files:")
    print(f"    - Training:   {output_training}")
    print(f"    - Evaluation: {output_evaluation}")
    print("=" * 60)

    # Setup scene and compute radio map
    print("\n[1/4] Setting up scene...")
    scene = setup_scene()

    print("[2/4] Computing radio map (this may take a while)...")
    radio_map = compute_radio_map(scene)

    print("[3/4] Adding receivers to scene...")
    add_receivers_to_scene(scene, PS_BATCH_SIZE)

    print("[4/4] Generating trajectories...")
    trajectories, trajectories_gains = generate_trajectories(
        scene=scene,
        radio_map=radio_map,
        num_trajectories=args.num_trajectories,
        num_slots=args.num_slots,
    )

    # Save results
    save_trajectories(trajectories, trajectories_gains, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
