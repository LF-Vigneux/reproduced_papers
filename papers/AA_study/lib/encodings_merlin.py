import numpy as np
import merlin as ml
import perceval as pcvl
import math


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
