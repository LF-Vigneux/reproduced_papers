import sys
from pathlib import Path
import numpy as np
import merlin as ml
import perceval as pcvl
import scipy as sp
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.AA_study.utils.qlayers_utils import generate_fourrier_sub_matrix_v2

from papers.AA_study.lib.encodings_merlin import (  # noqa: E402
    dense_angle_encoding_circuit,
    dense_encoding_of_features,
    unitary_evolution,
    amplitude_encoding,
    fourier_basis_v2,
    fourier_basis_v3,
    AngleEncoder,
    AmplitudeEncoder,
    DenseAngleEncoder,
    DenseAmplitudeEncoder,
    TimeEvolutionEncoder,
    FourierEncoder,
    FourierEncoderV2,
)


def test_dense_angle_encoding_circuit():
    for num_features in [10, 11, 21]:
        circuit = dense_angle_encoding_circuit(num_features=num_features)
        assert circuit.m == int(np.ceil(num_features / 2) * 2)
        assert len(list(circuit.params)) == num_features


def test_amplitude_encoding():
    features = np.ones(11)
    for computation_space, n_modes, n_photons in zip(
        [
            ml.ComputationSpace.UNBUNCHED,
            ml.ComputationSpace.FOCK,
            ml.ComputationSpace.DUAL_RAIL,
        ],
        [6, 5, 6],
        [2, 2, 0],
    ):
        state = amplitude_encoding(
            features,
            num_modes=n_modes,
            num_photons=n_photons,
            computation_space=computation_space,
        )
        assert np.linalg.norm(state) == 1
        assert np.sum(state[11:]) == 0
        for i in range(11):
            assert state[i] == np.sqrt(1 / 11, dtype=np.complex128)


def test_dense_encoding_of_features():
    features = np.ones(11)
    for computation_space, n_modes, n_photons in zip(
        [
            ml.ComputationSpace.UNBUNCHED,
            ml.ComputationSpace.FOCK,
            ml.ComputationSpace.DUAL_RAIL,
        ],
        [5, 4, 6],
        [2, 2, 0],
    ):
        state = dense_encoding_of_features(
            features,
            num_modes=n_modes,
            num_photons=n_photons,
            computation_space=computation_space,
        )
        assert np.linalg.norm(state) == 1
        assert np.sum(state[6:]) == 0
        assert np.imag(state[5]) == 0
        for i in range(5):
            assert np.imag(state[i]) == np.real(state[i])


def test_unitary_evolution():
    Z_matrix = np.array([[1, 0], [0, -1]])
    Z_matrix_encoded = np.zeros((4, 4))
    Z_matrix_encoded[:2, 2:] = Z_matrix
    Z_matrix_encoded[2:, :2] = Z_matrix

    unit_circuit = unitary_evolution(torch.tensor(Z_matrix), time=0.1)
    target = np.array(pcvl.Matrix(sp.linalg.expm(-0.1j * Z_matrix_encoded)))
    computed = np.array(unit_circuit.compute_unitary())

    # Fix global phase so (0, 0) entry is real and positive.
    target_phase = np.exp(-1.0j * np.angle(target[0, 0]))
    computed_phase = np.exp(-1.0j * np.angle(computed[0, 0]))
    target = target * target_phase
    computed = computed * computed_phase

    assert np.allclose(torch.tensor(target), computed, rtol=0.01)


# TODO: Change once I can really understand the qubit--> mode maping
def test_fourier_basis_v2():
    features = [0.3, 0.67]
    target = np.zeros((8, 8), dtype=np.complex128)
    target[0:2, 0:2] = generate_fourrier_sub_matrix_v2(
        feature=features[0], photon_index=0
    )
    target[2:4, 2:4] = generate_fourrier_sub_matrix_v2(
        feature=features[0], photon_index=1
    )
    target[4:6, 4:6] = generate_fourrier_sub_matrix_v2(
        feature=features[1], photon_index=0
    )
    target[6:, 6:] = generate_fourrier_sub_matrix_v2(
        feature=features[1], photon_index=1
    )

    output_circuit = fourier_basis_v2(features=features, num_qubits_per_feature=2)
    computed = np.array(output_circuit.compute_unitary())

    # Fix global phase so (0, 0) entry is real and positive.
    target_phase = np.exp(-1.0j * np.angle(target[0, 0]))
    computed_phase = np.exp(-1.0j * np.angle(computed[0, 0]))
    target = target * target_phase
    computed = computed * computed_phase

    assert np.allclose(target, computed, rtol=0.01)


def test_AngleEncoder():
    encoder = AngleEncoder(
        num_features=10,
        num_photons=2,
        change_output_size_even_square=True,
        return_sv=False,
    )

    features = torch.rand((10, 10))

    output_state = encoder(features)

    assert output_state.shape == (10, 64, 64)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)

    encoder = AngleEncoder(
        num_features=10,
        num_photons=2,
        computation_space=ml.ComputationSpace.FOCK,
        change_output_size_even_square=True,
        return_sv=False,
    )

    features = torch.rand((10, 10))

    output_state = encoder(features)

    assert output_state.shape == (10, 64, 64)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)


def test_AmplitudeEncoder():
    encoder = AmplitudeEncoder(
        num_modes=10,
        num_photons=2,
        change_output_size_even_square=True,
        return_sv=False,
    )

    features = torch.rand((10, 30))

    output_state = encoder(features)

    assert output_state.shape == (10, 64, 64)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)

    encoder = AmplitudeEncoder(
        num_modes=10,
        num_photons=2,
        computation_space=ml.ComputationSpace.FOCK,
        change_output_size_even_square=True,
        return_sv=False,
    )

    features = torch.rand((10, 45))

    output_state = encoder(features)

    assert output_state.shape == (10, 64, 64)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)


def test_DenseAngleEncoder():
    encoder = DenseAngleEncoder(
        num_features=10,
        change_output_size_even_square=True,
        return_sv=False,
    )

    features = torch.rand((10, 10))

    output_state = encoder(features)

    assert output_state.shape == (10, 36, 36)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)


def test_DenseAmplitudeEncoder():
    encoder = DenseAmplitudeEncoder(
        num_photons=2,
        num_modes=8,
        change_output_size_even_square=True,
        return_sv=False,
    )

    features = torch.rand((10, 54))

    output_state = encoder(features)

    assert output_state.shape == (10, 36, 36)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)

    encoder = DenseAmplitudeEncoder(
        num_modes=7,
        num_photons=2,
        computation_space=ml.ComputationSpace.FOCK,
        change_output_size_even_square=True,
        return_sv=False,
    )

    features = torch.rand((10, 54))

    output_state = encoder(features)

    assert output_state.shape == (10, 36, 36)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)


def test_TimeEvolutionEncoder():
    encoder = TimeEvolutionEncoder(
        num_photons=2,
        image_size=5,
        change_output_size_even_square=True,
        return_sv=False,
    )

    features = torch.rand((10, 5, 5), dtype=torch.complex128)

    output_state = encoder(features)

    assert output_state.shape == (10, 64, 64)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)

    encoder = TimeEvolutionEncoder(
        image_size=4,
        num_photons=2,
        computation_space=ml.ComputationSpace.FOCK,
        change_output_size_even_square=True,
        return_sv=False,
    )

    features = torch.rand((10, 4, 4), dtype=torch.complex128)

    output_state = encoder(features)

    assert output_state.shape == (10, 36, 36)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)


def test_FourierEncoder():
    encoder = FourierEncoder(
        num_features=3,
        n_photon_per_feature=3,
        change_output_size_even_square=True,
        return_sv=False,
    )

    features = torch.rand((10, 3))

    output_state = encoder(features)

    assert output_state.shape == (10, 576, 576)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)


def test_FourierEncoderV2():
    encoder = FourierEncoderV2(
        num_features=3,
        n_photon_per_feature=3,
        change_output_size_even_square=False,
        return_sv=False,
    )

    features = torch.rand((10, 3))

    output_state = encoder(features)

    assert output_state.shape == (10, 2**9, 2**9)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)


def test_FourierEncoderV2_deeper():
    for _ in range(5):
        encoder = FourierEncoderV2(
            num_features=3,
            n_photon_per_feature=3,
            change_output_size_even_square=False,
            return_sv=True,
        )

        features = torch.rand((1, 3))

        output_state = encoder(features).flatten()
        feature_states = []
        for x in features.flatten():
            feature_state = torch.zeros(8, dtype=torch.complex128)
            for k in range(8):
                feature_state[k] = torch.exp(1j * torch.pi * x * k * 0.25)
            feature_states.append(feature_state)
        expected_state = 2 ** (-(4.5)) * np.kron(
            feature_states[0], np.kron(feature_states[1], feature_states[2])
        )

        ov = output_state.detach().cpu().numpy()
        ev = np.asarray(expected_state)
        fidelity = np.abs(np.vdot(ov, ev)) ** 2
        assert np.allclose(fidelity, 1, rtol=1e-4)


def test_param_circuit():
    param_circuit = fourier_basis_v3(num_features=1, num_qubits_per_feature=1)

    for x in [0.37, 1.2, 3]:
        matrix_one_photon = (1 / (2 ** (0.5))) * np.array(
            [
                [1, 1],
                [np.exp(1.0j * np.pi * x), (-1) * np.exp(1.0j * np.pi * x)],
            ]
        )

        computed = param_circuit.compute_unitary(assign={"phi0": np.pi * x})
        computed_phase = np.exp(-1.0j * np.angle(computed[0, 0]))
        computed = computed * computed_phase

        assert np.allclose(matrix_one_photon, computed, rtol=0.001)
