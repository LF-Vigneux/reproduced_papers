import sys
from pathlib import Path
import numpy as np
import merlin as ml
import perceval as pcvl
import scipy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.AA_study.lib.encodings_merlin import (  # noqa: E402
    dense_angle_encoding_circuit,
    dense_encoding_of_features,
    unitary_evolution,
    fourier_basis,
    AngleEncoder,
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


def test_dense_encoding_of_features():
    features = np.ones(11)
    for computation_space, n_modes, n_photons in zip(
        [
            ml.ComputationSpace.UNBUNCHED,
            ml.ComputationSpace.FOCK,
            ml.ComputationSpace.DUAL_RAIL,
        ],
        [4, 4, 6],
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
    print(target)
    print(computed)

    assert np.allclose(target, computed, rtol=0.01)
