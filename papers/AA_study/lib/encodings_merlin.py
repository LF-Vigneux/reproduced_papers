import numpy as np
import merlin as ml
import perceval as pcvl
import math
from numpy.typing import NDArray
import scipy as sp
from pathlib import Path
import sys
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.AA_study.utils.qlayers_utils import generate_fourrier_sub_matrix, MZI


# TODO

# All the encodings must return a density matrix after a forward.

"""
Classes:
- The normal amplitude
- Angle encoding
- Dense angle encoding
- Dense amplitude
- Hamiltonian evolution
- Fourier unitary
- Angle re-uploading
"""


def dense_angle_encodding_circuit(
    num_features: int, param_prefix: str = "phi"
) -> pcvl.Circuit:
    """
    Needs to be Dual rail...
    """
    width = len(str(num_features - 1))
    params = [
        pcvl.Parameter(f"{param_prefix}{i:0{width}d}") for i in range(num_features)
    ]
    circuit = pcvl.Circuit(m=int(np.ceil(num_features / 2)) * 2)

    mode_index = 0
    bs_angle_done = False
    for param in params:
        if not bs_angle_done:
            circuit.add([mode_index, mode_index + 1], pcvl.BS(theta=param))
            bs_angle_done = True
        else:
            circuit.add(mode_index, pcvl.PS(phi=param))
            bs_angle_done = False
            mode_index += 2

    return circuit


def dense_encoding_of_features(
    features: list[float],
    num_modes: int,
    num_photons: int = 0,
    computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
) -> pcvl.Circuit:
    if computation_space == ml.ComputationSpace.UNBUNCHED:
        state = np.zeros(math.comb(num_modes, num_photons), dtype=complex)
    elif computation_space == ml.ComputationSpace.FOCK:
        state = np.zeros(
            math.comb(num_modes + num_photons - 1, num_photons), dtype=complex
        )
    elif computation_space == ml.ComputationSpace.DUAL_RAIL:
        state = np.zeros(2**num_modes, dtype=complex)
    else:
        raise ValueError("Invalid computation space")

    for state_index, feature_index in enumerate(range(0, len(features), 2)):
        if feature_index == len(features) - 1:
            state[state_index] = features[feature_index]
        else:
            state[state_index] = (
                features[feature_index] + 1.0j * features[feature_index + 1]
            )
    state /= np.linalg.norm(state)
    return state


# def unitary_evoution(
#     features_matrix: NDArray[np.float128],
#     num_modes: int,
#     time: float,
#     num_photons: int = 0,
#     computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
# ) -> pcvl.Circuit:
#     if computation_space == ml.ComputationSpace.UNBUNCHED:
#         size = math.comb(num_modes, num_photons)
#     elif computation_space == ml.ComputationSpace.FOCK:
#         size = math.comb(num_modes + num_photons - 1, num_photons)
#     elif computation_space == ml.ComputationSpace.DUAL_RAIL:
#         size = 2**num_modes
#     else:
#         raise ValueError("Invalid computation space")

#     unitary_to_apply = np.zeros([size, size], dtype=complex)

#     feature_size = np.shape(features_matrix)[0]

#     unitary_to_apply[size - feature_size :, 0:feature_size] = features_matrix
#     unitary_to_apply[0:feature_size, size - feature_size :] = (
#         features_matrix.conjugate().T
#     )
#     unitary_to_apply = pcvl.Matrix(unitary_to_apply)

#     circuit = pcvl.Circuit(m=num_modes)
#     pass


def unitary_evolution(
    features_matrix: NDArray[np.float128],
    time: float,
) -> pcvl.Circuit:

    feature_size = np.shape(features_matrix)[0]

    unitary_to_apply = np.zeros([feature_size * 2, feature_size * 2], dtype=complex)

    unitary_to_apply[feature_size:, :feature_size] = features_matrix
    unitary_to_apply[:feature_size, feature_size:] = features_matrix.conjugate().T
    unitary_to_apply = pcvl.Matrix(sp.linalg.expm(time * 1.0j * unitary_to_apply))

    return pcvl.Circuit.decomposition(
        unitary_to_apply, MZI, shape=pcvl.InterferometerShape.TRIANGLE
    )


def fourier_basis(features: list[float], num_qubits_per_feature: int):

    main_circuit = pcvl.Circuit(m=len(features) * num_qubits_per_feature * 2)
    mode_index = 0
    for feature in features:
        unitary_to_apply = pcvl.Matrix(
            generate_fourrier_sub_matrix(
                feature=feature, num_photons=num_qubits_per_feature
            )
        )
        main_circuit.add(
            [i for i in range(mode_index, mode_index + num_qubits_per_feature * 2)],
            pcvl.Circuit.decomposition(
                unitary_to_apply, MZI, shape=pcvl.InterferometerShape.TRIANGLE
            ),
        )

        mode_index += num_qubits_per_feature * 2

    return main_circuit


class AngleEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_photons: int,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_photons = num_photons
        circuit = ml.CircuitBuilder(n_modes=num_features)
        circuit.add_entangling_layer()
        circuit.add_angle_encoding()
        circuit.add_entangling_layer()
        self.qlayer = ml.QuantumLayer(
            builder=circuit,
            n_photons=num_photons,
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
            computation_space=ml.ComputationSpace.UNBUNCHED,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        amplitudes_output = self.qlayer(x)

        output_tensors = torch.empty(
            (x.shape[0], amplitudes_output.shape[1] * amplitudes_output.shape[1]),
            dtype=complex,
        )
        for amplitude in amplitudes_output:
            output_tensors[0] = torch.outer(amplitude, amplitude.resolve_conj())

        return output_tensors

    def __repr__(self):
        return "OneHotEncoder()"
