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

from papers.AA_study.utils.qlayers_utils import generate_fourrier_sub_matrix

from papers.AA_study.lib.encodings_merlin import (  # noqa: E402
    dense_angle_encoding_circuit,
    dense_encoding_of_features,
    unitary_evolution,
    amplitude_encoding,
    fourier_basis,
    AngleEncoder,
    AmplitudeEncoder,
    DenseAngleEncoder,
    DenseAmplitudeEncoder,
    TimeEvolutionEncoder,
    FourierEncoder,
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
            assert state[i] == np.sqrt(1 / 11)


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

    unit_circuit = unitary_evolution(Z_matrix, time=0.1)
    target = np.array(pcvl.Matrix(sp.linalg.expm(-0.1j * Z_matrix_encoded)))
    computed = np.array(unit_circuit.compute_unitary())

    # Fix global phase so (0, 0) entry is real and positive.
    target_phase = np.exp(-1.0j * np.angle(target[0, 0]))
    computed_phase = np.exp(-1.0j * np.angle(computed[0, 0]))
    target = target * target_phase
    computed = computed * computed_phase

    assert np.allclose(target, computed, rtol=0.01)


# TODO: Change once I can really understand the qubit--> mode maping
"""
def test_fourier_basis():
    features = [0.3, 0.67]
    target = np.kron(
        generate_fourrier_sub_matrix(feature=features[0], num_photons=2),
        generate_fourrier_sub_matrix(feature=features[1], num_photons=2),
    )

    output_circuit = fourier_basis(features=features, num_qubits_per_feature=2)
    computed = np.array(output_circuit.compute_unitary())

    # Fix global phase so (0, 0) entry is real and positive.
    target_phase = np.exp(-1.0j * np.angle(target[0, 0]))
    computed_phase = np.exp(-1.0j * np.angle(computed[0, 0]))
    target = target * target_phase
    computed = computed * computed_phase

    assert np.allclose(target, computed, rtol=0.01)
"""


def test_AngleEncoder():
    encoder = AngleEncoder(num_features=10, num_photons=2)

    features = torch.rand((10, 10))

    output_state = encoder(features)

    assert output_state.shape == (10, 45, 45)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)

    encoder = AngleEncoder(
        num_features=10, num_photons=2, computation_space=ml.ComputationSpace.FOCK
    )

    features = torch.rand((10, 10))

    output_state = encoder(features)

    assert output_state.shape == (10, 55, 55)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)


def test_AmplitudeEncoder():
    encoder = AmplitudeEncoder(
        num_modes=10,
        num_photons=2,
    )

    features = torch.rand((10, 30))

    output_state = encoder(features)

    assert output_state.shape == (10, 45, 45)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)

    encoder = AmplitudeEncoder(
        num_modes=10, num_photons=2, computation_space=ml.ComputationSpace.FOCK
    )

    features = torch.rand((10, 45))

    output_state = encoder(features)

    assert output_state.shape == (10, 55, 55)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)


def test_DenseAngleEncoder():
    encoder = DenseAngleEncoder(num_features=10)

    features = torch.rand((10, 10))

    output_state = encoder(features)

    assert output_state.shape == (10, 2**5, 2**5)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)


def test_DenseAmplitudeEncoder():
    encoder = DenseAmplitudeEncoder(num_photons=2, num_modes=8)

    features = torch.rand((10, 54))

    output_state = encoder(features)

    assert output_state.shape == (10, 28, 28)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)

    encoder = DenseAmplitudeEncoder(
        num_modes=7, num_photons=2, computation_space=ml.ComputationSpace.FOCK
    )

    features = torch.rand((10, 54))

    output_state = encoder(features)

    assert output_state.shape == (10, 28, 28)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)


def test_TimeEvolutionEncoder():
    encoder = TimeEvolutionEncoder(num_photons=2, image_size=5)

    features = torch.rand((10, 5, 5))

    output_state = encoder(features)

    assert output_state.shape == (10, 45, 45)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)

    encoder = TimeEvolutionEncoder(
        image_size=4, num_photons=2, computation_space=ml.ComputationSpace.FOCK
    )

    features = torch.rand((10, 4, 4))

    output_state = encoder(features)

    assert output_state.shape == (10, 36, 36)
    assert output_state.dtype == torch.complex128
    for i in output_state:
        assert np.allclose(torch.trace(i).detach().numpy(), [1.0 + 0.0j], rtol=0.01)


# TODO: Change once I can really understand the qubit--> mode maping
"""
def test_FourierEncoder():
    pass
"""
