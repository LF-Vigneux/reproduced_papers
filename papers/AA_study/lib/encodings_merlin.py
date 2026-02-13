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
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.AA_study.utils.qlayers_utils import generate_fourrier_sub_matrix, MZI


def dense_angle_encoding_circuit(
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


def amplitude_encoding(
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

    for state_index, feature_index in enumerate(range(len(features))):
        state[state_index] = features[feature_index]
    state /= np.linalg.norm(state)
    return state


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


def unitary_evolution(
    features_matrix: NDArray[np.float16],
    time: float,
) -> pcvl.Circuit:
    """
    One mode per feature
    """

    feature_size = np.shape(features_matrix)[0]

    unitary_to_apply = np.zeros([feature_size * 2, feature_size * 2], dtype=complex)

    unitary_to_apply[feature_size:, :feature_size] = features_matrix.conj().T
    unitary_to_apply[:feature_size, feature_size:] = features_matrix
    unitary_to_apply = pcvl.Matrix(
        sp.linalg.expm((-1) * time * 1.0j * unitary_to_apply)
    )

    return pcvl.Circuit.decomposition(
        unitary_to_apply,
        MZI,
        phase_shifter_fn=pcvl.PS,
        shape=pcvl.InterferometerShape.TRIANGLE,
        allow_error=True,
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
                unitary_to_apply,
                MZI,
                phase_shifter_fn=pcvl.PS,
                shape=pcvl.InterferometerShape.TRIANGLE,
                allow_error=True,
            ),
        )

        mode_index += num_qubits_per_feature * 2

    return main_circuit


class AngleEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_photons: int,
        computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
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
            computation_space=computation_space,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        amplitudes_output = self.qlayer(x)

        output_tensors = torch.empty(
            (x.shape[0], amplitudes_output.shape[1], amplitudes_output.shape[1]),
            dtype=complex,
        )
        for i, amplitude in enumerate(amplitudes_output):
            output_tensors[i, :, :] = torch.outer(amplitude, amplitude.conj())

        return output_tensors

    def __repr__(self):
        return "AngleEncoder()"


class AmplitudeEncoder(nn.Module):
    def __init__(
        self,
        num_modes: int,
        computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
        num_photons: int = 0,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.num_photons = num_photons
        self.computation_space = computation_space

        if self.computation_space is ml.ComputationSpace.UNBUNCHED:
            self.output_size = math.comb(self.num_modes, self.num_photons)
        elif self.computation_space is ml.ComputationSpace.FOCK:
            self.output_size = math.comb(
                self.num_modes + self.num_photons - 1, self.num_photons
            )
        elif self.computation_space is ml.ComputationSpace.DUAL_RAIL:
            self.output_size = 2 ** (num_modes / 2)
        else:
            raise ValueError("Wrong computation space")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        output_tensors = torch.empty(
            (x.shape[0], self.output_size, self.output_size),
            dtype=complex,
        )

        for i, tensor in enumerate(x):
            state = torch.tensor(
                amplitude_encoding(
                    tensor,
                    self.num_modes,
                    computation_space=self.computation_space,
                    num_photons=self.num_photons,
                )
            )
            output_tensors[i, :, :] = torch.outer(state, state.conj())

        return output_tensors

    def __repr__(self):
        return "AmplitudeEncoder()"


class DenseAngleEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
    ):
        super().__init__()
        width = len(str(num_features - 1))

        self.num_features = num_features
        perceval_circuit = dense_angle_encoding_circuit(num_features=num_features)

        input_state = [
            1 if i % 2 == 0 else 0 for i in range(int(np.ceil(num_features / 2)) * 2)
        ]

        self.qlayer = ml.QuantumLayer(
            circuit=perceval_circuit,
            input_state=input_state,
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
            computation_space=ml.ComputationSpace.DUAL_RAIL,
            input_parameters=[f"phi{i:0{width}d}" for i in range(num_features)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        amplitudes_output = self.qlayer(x)

        output_tensors = torch.empty(
            (x.shape[0], amplitudes_output.shape[1], amplitudes_output.shape[1]),
            dtype=complex,
        )
        for i, amplitude in enumerate(amplitudes_output):
            output_tensors[i, :, :] = torch.outer(amplitude, amplitude.conj())

        return output_tensors

    def __repr__(self):
        return "DenseAngleEncoder()"


class DenseAmplitudeEncoder(nn.Module):
    def __init__(
        self,
        num_modes: int,
        computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
        num_photons: int = 0,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.num_photons = num_photons
        self.computation_space = computation_space

        if self.computation_space is ml.ComputationSpace.UNBUNCHED:
            self.output_size = math.comb(self.num_modes, self.num_photons)
        elif self.computation_space is ml.ComputationSpace.FOCK:
            self.output_size = math.comb(
                self.num_modes + self.num_photons - 1, self.num_photons
            )
        elif self.computation_space is ml.ComputationSpace.DUAL_RAIL:
            self.output_size = 2 ** (num_modes / 2)
        else:
            raise ValueError("Wrong computation space")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        output_tensors = torch.empty(
            (x.shape[0], self.output_size, self.output_size),
            dtype=complex,
        )

        for i, tensor in enumerate(x):
            state = torch.tensor(
                dense_encoding_of_features(
                    tensor,
                    self.num_modes,
                    computation_space=self.computation_space,
                    num_photons=self.num_photons,
                )
            )
            output_tensors[i, :, :] = torch.outer(state, state.conj())

        return output_tensors

    def __repr__(self):
        return "DenseAmplitudeEncoder()"


class TimeEvolutionEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_photons: int,
        time: float = 0.1,
        computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
    ):
        """
        image_size is one size of the image
        """
        super().__init__()
        self.time = time
        self.n_photons = num_photons
        self.image_size = image_size
        self.computation_space = computation_space

        base_circuit = ml.CircuitBuilder(n_modes=2 * image_size)
        base_circuit.add_entangling_layer(trainable=False)
        self.base_perceval = base_circuit.to_pcvl_circuit()

        if self.computation_space is ml.ComputationSpace.UNBUNCHED:
            self.output_size = math.comb(self.image_size * 2, self.n_photons)
        elif self.computation_space is ml.ComputationSpace.FOCK:
            self.output_size = math.comb(
                (self.image_size * 2) + self.n_photons - 1, self.n_photons
            )
        elif self.computation_space is ml.ComputationSpace.DUAL_RAIL:
            self.output_size = 2**image_size
        else:
            raise ValueError("Wrong computation space")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        output_tensors = torch.empty(
            (x.shape[0], self.output_size, self.output_size),
            dtype=complex,
        )

        for i, tensor in enumerate(x):
            total_circuit = self.base_perceval.copy()
            total_circuit.add(
                list(range(2 * self.image_size)), unitary_evolution(tensor, self.time)
            )
            qlayer = ml.QuantumLayer(
                circuit=total_circuit,
                n_photons=self.n_photons,
                measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
                computation_space=self.computation_space,
            )
            state = qlayer().flatten()

            output_tensors[i, :, :] = torch.outer(state, state.conj())

        return output_tensors

    def __repr__(self):
        return "TimeEvolutionEncoder()"


class FourierEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        n_photon_per_feature: int,
    ):
        """
        n_modes is one size of the image
        """
        super().__init__()
        self.num_features = num_features
        self.n_photon_per_feature = n_photon_per_feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        output_tensors = torch.empty(
            (x.shape[0], x.shape[1], x.shape[1]),
            dtype=complex,
        )

        for i, tensor in enumerate(x):
            total_circuit = ml.CircuitBuilder.from_circuit(
                fourier_basis(tensor, num_qubits_per_feature=self.n_photon_per_feature)
            )
            qlayer = ml.QuantumLayer(
                builder=total_circuit,
                measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
                computation_space=ml.ComputationSpace.DUAL_RAIL,
            )
            state = qlayer(tensor)

            output_tensors[i, :, :] = torch.outer(state, state.conj())

        return output_tensors

    def __repr__(self):
        return "FourierEncoder()"
