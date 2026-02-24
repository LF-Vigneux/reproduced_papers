import numpy as np
import merlin as ml
import perceval as pcvl
import math
from pathlib import Path
import sys
import torch
import torch.nn as nn
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.AA_study.utils.qlayers_utils import (
    generate_fourrier_sub_matrix,
    MZI,
    generate_fourrier_sub_matrix_v2,
    find_upper_even_square,
)


def dense_angle_encoding_circuit(
    num_features: int, param_prefix: str = "phi", num_modes: int | None = None
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
            if num_modes is not None:
                if mode_index == num_modes:
                    mode_index = 0

    return circuit


def amplitude_encoding(
    features: list[float],
    num_modes: int,
    num_photons: int = 0,
    computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
    indexes_perm: list[int] | None = None,
) -> pcvl.Circuit:
    if computation_space == ml.ComputationSpace.UNBUNCHED:
        state = np.zeros(math.comb(num_modes, num_photons), dtype=np.complex128)
    elif computation_space == ml.ComputationSpace.FOCK:
        state = np.zeros(
            math.comb(num_modes + num_photons - 1, num_photons), dtype=np.complex128
        )
    elif computation_space == ml.ComputationSpace.DUAL_RAIL:
        state = np.zeros(2**num_modes, dtype=np.complex128)
    else:
        raise ValueError("Invalid computation space")
    state[: len(features)] = features
    state /= np.linalg.norm(state)
    if indexes_perm is not None:
        state = state[indexes_perm]
    return state


def dense_encoding_of_features(
    features: list[float],
    num_modes: int,
    num_photons: int = 0,
    computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
    indexes_perm: list[int] | None = None,
) -> pcvl.Circuit:
    if computation_space == ml.ComputationSpace.UNBUNCHED:
        state = np.zeros(math.comb(num_modes, num_photons), dtype=np.complex128)
    elif computation_space == ml.ComputationSpace.FOCK:
        state = np.zeros(
            math.comb(num_modes + num_photons - 1, num_photons), dtype=np.complex128
        )
    elif computation_space == ml.ComputationSpace.DUAL_RAIL:
        state = np.zeros(2**num_modes, dtype=np.complex128)
    else:
        raise ValueError("Invalid computation space")

    feature_shuffler = np.zeros(len(state) * 2)
    feature_shuffler[: len(features)] = features

    if indexes_perm is not None:
        feature_shuffler = feature_shuffler[indexes_perm]

    for state_index, feature_index in enumerate(range(0, len(feature_shuffler), 2)):
        state[state_index] = (
            feature_shuffler[feature_index] + 1.0j * feature_shuffler[feature_index + 1]
        )
    state /= np.linalg.norm(state)
    return state


def unitary_evolution(
    features_matrix: torch.Tensor,
    time: float,
) -> pcvl.Circuit:
    """
    One mode per feature
    """

    feature_size = np.shape(features_matrix)[0]

    unitary_to_apply = torch.zeros(
        [feature_size * 2, feature_size * 2], dtype=torch.complex128
    )

    unitary_to_apply[feature_size:, :feature_size] = features_matrix.conj().transpose(
        0, 1
    )
    unitary_to_apply[:feature_size, feature_size:] = features_matrix
    unitary_to_apply = pcvl.Matrix(
        torch.matrix_exp((-1) * time * 1.0j * unitary_to_apply).detach().cpu().numpy()
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
                feature=float(feature), num_photons=num_qubits_per_feature
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


def fourier_basis_v2(features: list[float], num_qubits_per_feature: int):

    main_circuit = pcvl.Circuit(m=len(features) * num_qubits_per_feature * 2)
    mode_index = 0
    for feature in features:
        for qubit in range(num_qubits_per_feature):
            unitary_to_apply = pcvl.Matrix(
                generate_fourrier_sub_matrix_v2(
                    feature=float(feature), photon_index=qubit
                )
            )
            main_circuit.add(
                [
                    i
                    for i in range(
                        mode_index + (qubit * 2), mode_index + (qubit * 2) + 2
                    )
                ],
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


class OneHotEncoder(nn.Module):
    """
    One Hot Encoder

    Converts an image `x` to density matrix in the One Hot Amplitude
    basis. For a given d by d image, the density matrix will be of
    size d^2 by d^2.
    """

    def __init__(self, image_size: int, return_sv: bool = True):
        super().__init__()
        self.output_size = image_size**2
        self.return_sv = return_sv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        norm = torch.sqrt(torch.square(torch.abs(x)).sum(dim=(1, 2, 3)))
        x = x / norm.view(-1, 1, 1, 1)

        # Flatten each image and multiply by transpose to get density matrix
        x_flat = x.reshape(x.shape[0], -1)
        if self.return_sv:
            return x_flat
        rho = x_flat.unsqueeze(2) @ x_flat.unsqueeze(1)
        rho = rho.to(torch.complex128)

        return rho

    def __repr__(self):
        return "OneHotEncoder()"


class AngleEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_photons: int,
        computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
        num_modes: int | None = None,
        return_sv: bool = True,
        change_output_size_even_square: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_photons = num_photons
        self.computation_space = computation_space
        self.return_sv = return_sv
        circuit = ml.CircuitBuilder(
            n_modes=num_features if num_modes is None else num_modes
        )
        features_assigned = 0
        while features_assigned < self.num_features:
            circuit.add_entangling_layer(trainable=False)
            if features_assigned + circuit.n_modes < self.num_features:
                circuit.add_angle_encoding()
                features_assigned += circuit.n_modes
            else:
                circuit.add_angle_encoding(
                    modes=[i for i in range(self.num_features - features_assigned)]
                )
                features_assigned += self.num_features - features_assigned
            circuit.add_entangling_layer(trainable=False)

        self.qlayer = ml.QuantumLayer(
            builder=circuit,
            n_photons=num_photons,
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
            computation_space=computation_space,
            dtype=torch.complex128,
            amplitude_encoding=False,
        )
        if change_output_size_even_square:
            self.output_size = find_upper_even_square(self.qlayer.output_size)
            self.encoder = nn.Sequential(
                self.qlayer, ml.LexGrouping(self.qlayer.output_size, self.output_size)
            )
        else:
            self.output_size = self.qlayer.output_size
            self.encoder = self.qlayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        if x.dim() > 2:
            x = torch.reshape(x, (x.shape[0], np.prod(x.shape[1:])))

        # Normalize for inputs
        x /= x.max().item()
        amplitudes_output = self.encoder(x)

        if self.return_sv:
            return amplitudes_output.to(torch.complex128)
        else:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size, self.output_size),
                dtype=torch.complex128,
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
        num_photons: int = 0,
        computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
        shuffle_amplitude: bool = False,
        return_sv: bool = True,
        change_output_size_even_square: bool = False,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.num_photons = num_photons
        self.computation_space = computation_space
        self.shuffle_amplitude = shuffle_amplitude
        self.return_sv = return_sv

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

        if self.shuffle_amplitude:
            self.indexes_perm = np.arange(self.output_size)
            np.random.shuffle(self.indexes_perm)
        else:
            self.indexes_perm = None

        if change_output_size_even_square:
            corrected_output_size = find_upper_even_square(self.output_size)
            self.encoder = ml.LexGrouping(self.output_size, corrected_output_size)
            self.output_size = corrected_output_size
        else:
            self.encoder = ml.LexGrouping(self.output_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        if self.return_sv:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                output_tensors[i, :] = self.encoder(
                    torch.tensor(
                        amplitude_encoding(
                            tensor,
                            self.num_modes,
                            computation_space=self.computation_space,
                            num_photons=self.num_photons,
                            indexes_perm=self.indexes_perm,
                        )
                    )
                )
        else:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size, self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                state = self.encoder(
                    torch.tensor(
                        amplitude_encoding(
                            tensor,
                            self.num_modes,
                            computation_space=self.computation_space,
                            num_photons=self.num_photons,
                            indexes_perm=self.indexes_perm,
                        )
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
        num_modes: int | None = None,
        return_sv: bool = True,
        change_output_size_even_square: bool = False,
    ):
        super().__init__()
        width = len(str(num_features - 1))

        self.num_features = num_features
        self.return_sv = return_sv
        perceval_circuit = dense_angle_encoding_circuit(
            num_features=num_features,
            num_modes=num_features if num_modes is None else num_modes,
        )

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
        if change_output_size_even_square:
            self.output_size = find_upper_even_square(self.qlayer.output_size)
            self.encoder = nn.Sequential(
                self.qlayer, ml.LexGrouping(self.qlayer.output_size, self.output_size)
            )
        else:
            self.output_size = self.qlayer.output_size
            self.encoder = self.qlayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        x /= x.max().item()

        amplitudes_output = self.encoder(x)

        if self.return_sv:
            return amplitudes_output.to(torch.complex128)
        else:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size, self.output_size),
                dtype=torch.complex128,
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
        num_photons: int = 0,
        computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
        shuffle_amplitude: bool = False,
        return_sv: bool = True,
        change_output_size_even_square: bool = False,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.num_photons = num_photons
        self.computation_space = computation_space
        self.shuffle_amplitude = shuffle_amplitude
        self.return_sv = return_sv

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

        if self.shuffle_amplitude:
            self.indexes_perm = np.arange(self.output_size)
            np.random.shuffle(self.indexes_perm)
        else:
            self.indexes_perm = None

        if change_output_size_even_square:
            corrected_output_size = find_upper_even_square(self.output_size)
            self.encoder = ml.LexGrouping(self.output_size, corrected_output_size)
            self.output_size = corrected_output_size
        else:
            self.encoder = ml.LexGrouping(self.output_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        if self.return_sv:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                output_tensors[i, :] = self.encoder(
                    torch.tensor(
                        dense_encoding_of_features(
                            tensor,
                            self.num_modes,
                            computation_space=self.computation_space,
                            num_photons=self.num_photons,
                            indexes_perm=self.indexes_perm,
                        )
                    )
                )
        else:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size, self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                state = self.encoder(
                    torch.tensor(
                        dense_encoding_of_features(
                            tensor,
                            self.num_modes,
                            computation_space=self.computation_space,
                            num_photons=self.num_photons,
                            indexes_perm=self.indexes_perm,
                        )
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
        return_sv: bool = True,
        change_output_size_even_square: bool = False,
    ):
        """
        image_size is one size of the image
        """
        super().__init__()
        self.time = time
        self.n_photons = num_photons
        self.image_size = image_size
        self.computation_space = computation_space
        self.return_sv = return_sv

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

        if change_output_size_even_square:
            corrected_output_size = find_upper_even_square(self.output_size)
            self.encoder = ml.LexGrouping(self.output_size, corrected_output_size)
            self.output_size = corrected_output_size
        else:
            self.encoder = ml.LexGrouping(self.output_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if self.return_sv:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                total_circuit = self.base_perceval.copy()
                total_circuit.add(
                    list(range(2 * self.image_size)),
                    unitary_evolution(tensor, self.time),
                )
                qlayer = ml.QuantumLayer(
                    circuit=total_circuit,
                    n_photons=self.n_photons,
                    measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
                    computation_space=self.computation_space,
                )
                output_tensors[i, :] = self.encoder(qlayer().flatten())
        else:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size, self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                total_circuit = self.base_perceval.copy()
                total_circuit.add(
                    list(range(2 * self.image_size)),
                    unitary_evolution(tensor, self.time),
                )
                qlayer = ml.QuantumLayer(
                    circuit=total_circuit,
                    n_photons=self.n_photons,
                    measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
                    computation_space=self.computation_space,
                )
                state = self.encoder(qlayer().flatten())

                output_tensors[i, :, :] = torch.outer(state, state.conj())

        return output_tensors

    def __repr__(self):
        return "TimeEvolutionEncoder()"


class FourierEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        n_photon_per_feature: int,
        return_sv: bool = True,
        change_output_size_even_square: bool = False,
    ):
        """
        n_modes is one size of the image
        """
        super().__init__()
        self.num_features = num_features
        self.n_photon_per_feature = n_photon_per_feature
        self.return_sv = return_sv
        self.input_state = [
            1 if i % 2 == 0 else 0
            for i in range(num_features * n_photon_per_feature * 2)
        ]
        self.output_size = 2 ** (num_features * n_photon_per_feature)

        if change_output_size_even_square:
            corrected_output_size = find_upper_even_square(self.output_size)
            self.encoder = ml.LexGrouping(self.output_size, corrected_output_size)
            self.output_size = corrected_output_size
        else:
            self.encoder = ml.LexGrouping(self.output_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        if self.return_sv:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                total_circuit = fourier_basis_v2(
                    tensor, num_qubits_per_feature=self.n_photon_per_feature
                )
                qlayer = ml.QuantumLayer(
                    circuit=total_circuit,
                    input_state=self.input_state,
                    measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
                    computation_space=ml.ComputationSpace.DUAL_RAIL,
                )

                output_tensors[i, :] = self.encoder(qlayer().flatten())
        else:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size, self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                total_circuit = fourier_basis_v2(
                    tensor, num_qubits_per_feature=self.n_photon_per_feature
                )
                qlayer = ml.QuantumLayer(
                    circuit=total_circuit,
                    input_state=self.input_state,
                    measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
                    computation_space=ml.ComputationSpace.DUAL_RAIL,
                )
                state = self.encoder(qlayer().flatten())

                output_tensors[i, :, :] = torch.outer(state, state.conj())

        return output_tensors

    def __repr__(self):
        return "FourierEncoder()"


def choose_encoding(
    encoding_name: str,
    return_sv: bool = True,
    change_output_size_even_square: bool = False,
    num_photons: int | None = None,
    num_modes: int | None = None,
    num_features: int | None = None,
    time: float = 0.01,
    computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
    shuffle_amplitude: bool = False,
) -> tuple[
    AngleEncoder
    | DenseAngleEncoder
    | AmplitudeEncoder
    | DenseAmplitudeEncoder
    | TimeEvolutionEncoder
    | FourierEncoder
    | OneHotEncoder,
    int,
]:
    if encoding_name == "OneHot":
        return OneHotEncoder(int(np.sqrt(num_features)), return_sv=return_sv)
    elif encoding_name == "Angle":
        return AngleEncoder(
            num_features=num_features,
            num_photons=num_photons,
            computation_space=computation_space,
            num_modes=num_modes,
            return_sv=return_sv,
            change_output_size_even_square=change_output_size_even_square,
        )
    elif encoding_name == "DenseAngle":
        return DenseAngleEncoder(
            num_features=num_features,
            num_modes=num_modes,
            return_sv=return_sv,
            change_output_size_even_square=change_output_size_even_square,
        )
    elif encoding_name == "Amplitude":
        return AmplitudeEncoder(
            num_photons=num_photons,
            num_modes=num_modes,
            computation_space=computation_space,
            shuffle_amplitude=shuffle_amplitude,
            return_sv=return_sv,
            change_output_size_even_square=change_output_size_even_square,
        )
    elif encoding_name == "DenseAmplitude":
        return DenseAmplitudeEncoder(
            num_photons=num_photons,
            num_modes=num_modes,
            computation_space=computation_space,
            shuffle_amplitude=shuffle_amplitude,
            return_sv=return_sv,
            change_output_size_even_square=change_output_size_even_square,
        )
    elif encoding_name == "TimeEvolution":
        return TimeEvolutionEncoder(
            image_size=num_features,
            num_photons=num_photons,
            time=time,
            computation_space=computation_space,
            return_sv=return_sv,
            change_output_size_even_square=change_output_size_even_square,
        )
    elif encoding_name == "Fourier":
        return FourierEncoder(
            num_features=num_features,
            n_photon_per_feature=num_photons,
            return_sv=return_sv,
            change_output_size_even_square=change_output_size_even_square,
        )
    else:
        raise ValueError("No encoding method associates with that name")
