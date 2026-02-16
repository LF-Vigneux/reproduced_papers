import sys
from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import perceval as pcvl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import math
import torch
from papers.photonic_QCNN.lib.src.qcnn_paper import (  # noqa: E402
    generate_all_fock_states_list,
)


MZI = (
    pcvl.components.BS()
    // (0, pcvl.components.PS(pcvl.Parameter("phi1")))
    // pcvl.components.BS()
    // (0, pcvl.components.PS(pcvl.Parameter("phi2")))
)


def marginalize_photon_presence(keys, probs):
    """
    Marginalize Fock state probabilities to get per-mode occupation probabilities.

    Computes the probability that each mode contains at least one photon
    by summing over all Fock states where that mode is occupied.

    Args:
        keys (list): List of Fock state tuples, e.g., [(0,1,0,2), (1,0,1,0), ...]
        probs (torch.Tensor): Tensor of shape (N, num_keys) with probabilities
            for each Fock state, with requires_grad=True

    Returns:
        torch.Tensor: Shape (N, num_modes) with marginal probability that
            each mode has at least one photon
    """
    device = probs.device
    keys_tensor = torch.tensor(
        keys, dtype=torch.long, device=device
    )  # shape: (num_keys, num_modes)
    keys_tensor.shape[1]

    # Create mask of shape (num_modes, num_keys)
    # Each mask[i] is a binary vector indicating which Fock states have >=1 photon in mode i
    mask = (keys_tensor >= 1).T  # shape: (num_modes, num_keys)

    # Convert to float to allow matrix multiplication
    mask = mask.float()

    # Now do: (N, num_keys) @ (num_keys, num_modes) → (N, num_modes)
    marginalized = probs @ mask.T  # shape: (N, num_modes)
    return marginalized


def generate_partial_fock_states(subset, n, m):
    """
    Generate all the possible Fock state considering a subset of modes.

    Args:
    :param subset: Number of modes to consider. Has to be smaller or equal to m (number of modes)
    :param n: Number of photons
    :param m: Total number of modes
    :return: List of all possible Fock states considering the subset of modes
    """
    reduced_states = []
    # Account for when subset == m or subset + 1 == m. There cannot have 1 or 0 photon
    for i in range(max(0, subset - m + n), n + 1):
        reduced_states += generate_all_fock_states_list(subset, i, true_order=True)
    return reduced_states


def partial_measurement_output_size(subset: int, n: int, total_modes: int) -> int:
    """
    Compute number of possible measurement outcomes when measuring a subset
    of modes in Fock space, constrained by total photon number.

    Args:
        subset (int): Number of measured modes
        n (int): Total number of photons
        total_modes (int): Total number of modes (m)

    Returns:
        int: Number of reduced Fock states consistent with measurement
    """
    if subset == total_modes:
        # Full measurement: all photons must be in measured modes
        return math.comb(subset + n - 1, n)
    else:
        # Partial measurement: sum over all valid photon counts in measured modes
        return sum(math.comb(subset + i - 1, i) for i in range(n + 1))


def generate_fourrier_sub_matrix(feature: float, num_photons: int) -> NDArray:
    """
    The / in exp is just j+1 insted of n-j to take for account the end swap
    """

    def _generate_fourrier_sub_matrix(
        num_photons_done: int = 0, matrix: NDArray | None = None
    ):
        if num_photons_done == num_photons:
            return matrix

        if num_photons_done == 0:
            matrix = np.array(
                [
                    [1, 1],
                    [
                        np.exp(1.0j * np.pi * feature),
                        (-1) * np.exp(1.0j * np.pi * feature),
                    ],
                ]
            )
            return _generate_fourrier_sub_matrix(num_photons_done=1, matrix=matrix)

        matrix_to_tensor = np.array(
            [
                [1, 1],
                [
                    np.exp(1.0j * 2 * np.pi * feature / (2 ** (num_photons_done + 1))),
                    (-1)
                    * np.exp(
                        1.0j * 2 * np.pi * feature / (2 ** (num_photons_done + 1))
                    ),
                ],
            ]
        )
        return _generate_fourrier_sub_matrix(
            num_photons_done=num_photons_done + 1,
            matrix=np.kron(matrix, matrix_to_tensor),
        )

    return (1 / (2 ** (num_photons / 2))) * _generate_fourrier_sub_matrix()


def generate_fourrier_sub_matrix_v2(feature: float, photon_index: int) -> NDArray:
    return (1 / np.sqrt(2)) * np.array(
        [
            [1, 1],
            [
                np.exp(1.0j * 2 * np.pi * feature / (2 ** (photon_index + 1))),
                (-1) * np.exp(1.0j * 2 * np.pi * feature / (2 ** (photon_index + 1))),
            ],
        ]
    )
