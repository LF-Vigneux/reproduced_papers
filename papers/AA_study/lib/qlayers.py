import sys
from pathlib import Path
import merlin as ml
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.AA_study.utils.utils import find_mode_photon_config  # noqa: E402
from papers.photonic_QCNN.lib.src.qcnn_paper import (  # noqa: E402
    Measure,
    QConv2d,
    QDense,
    QPooling,
    generate_all_fock_states_list,
)
from papers.AA_study.utils.qlayers_utils import (  # noqa: E402
    marginalize_photon_presence,
    generate_partial_fock_states,
    partial_measurement_output_size,
)
from papers.AA_study.lib.encodings_merlin import choose_encoding


def angle_encoding_simple(
    num_features: int,
    num_layers: int = 3,
    measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.PROBABILITIES,
    num_classes: int = 2,
) -> ml.QuantumLayer:
    """
    Build a simple angle-encoding photonic quantum layer.

    Parameters
    ----------
    num_features : int
        Number of input features / modes.
    num_layers : int, optional
        Number of entangling/rotation layers.
    measurement_strategy : merlin.MeasurementStrategy, optional
        Measurement strategy for the quantum layer.
    num_classes : int, optional
        Number of output classes after lexicographic grouping.

    Returns
    -------
    torch.nn.Sequential
        Quantum layer followed by lexicographic grouping.
    """
    input_state = [0] * num_features
    for i in range(num_features // 2):
        input_state[(i * 2) + 1] = 1
    circuit = ml.CircuitBuilder(n_modes=num_features)
    if num_features == 1:
        circuit.add_rotations(trainable=True)
    else:
        circuit.add_entangling_layer()
    circuit.add_angle_encoding()
    for _ in range(num_layers):
        if num_features == 1:
            circuit.add_rotations(trainable=True)
        else:
            circuit.add_entangling_layer()
    qlayer = ml.QuantumLayer(
        input_size=num_features,  # Follow the convention?
        builder=circuit,
        input_state=input_state,
        n_photons=num_features // 2,
        measurement_strategy=measurement_strategy,
    )
    return nn.Sequential(qlayer, ml.LexGrouping(qlayer.output_size, num_classes))


def amplitude_encoding_simple(
    num_features: int,
    num_layers: int = 3,
    measurement_strategy: ml.MeasurementStrategy = ml.MeasurementStrategy.PROBABILITIES,
    num_classes: int = 2,
) -> ml.QuantumLayer:
    """
    Build a simple amplitude-encoding photonic quantum layer.

    Parameters
    ----------
    num_features : int
        Number of input features.
    num_layers : int, optional
        Number of entangling/rotation layers.
    measurement_strategy : merlin.MeasurementStrategy, optional
        Measurement strategy for the quantum layer.
    num_classes : int, optional
        Number of output classes after lexicographic grouping.

    Returns
    -------
    torch.nn.Sequential
        Quantum layer followed by lexicographic grouping.
    """
    n_modes, n_photons = find_mode_photon_config(num_features=num_features)
    circuit = ml.CircuitBuilder(n_modes=n_modes)
    for _ in range(num_layers):
        if n_modes == 1:
            circuit.add_rotations(trainable=True)
        else:
            circuit.add_entangling_layer()
    qlayer = ml.QuantumLayer(
        builder=circuit,
        amplitude_encoding=True,
        n_photons=n_photons,
        measurement_strategy=measurement_strategy,
    )
    return nn.Sequential(qlayer, ml.LexGrouping(qlayer.output_size, num_classes))


class PhotonicQCNN(nn.Module):
    """
    Hybrid photonic quantum CNN model using MerLin framework.

    Combines quantum convolution, pooling, dense layers and measurement
    with classical post-processing for binary classification tasks.

    Args:
        dims (tuple): Input image dimensions (height, width)
        conv_circuit (str): Circuit type for convolution layer ('MZI', 'BS', etc.)
        dense_circuit (str): Circuit type for dense layer
        measure_subset (int): Number of modes to measure (None for all)
        dense_added_modes (int): Additional modes to insert in dense layer
        output_proba_type (str): 'state' for Fock state probabilities, 'mode' for per-mode
        output_formatting (str): Output mapping strategy ('Train_linear', 'Lex_grouping', etc.)
        num_classes (int): Number of output classes (default: 2)
    """

    def __init__(
        self,
        conv_circuit,
        dense_circuit,
        measure_subset,
        dense_added_modes,
        output_proba_type,
        output_formatting,
        num_classes=2,
        encoding_name: str = "OneHot",
        num_photons: int = 0,
        num_modes: int = 0,
        num_features: int = 0,
        time: float = 0.0,
        computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
        shuffle_amplitude: bool = False,
    ):
        super().__init__()
        self.num_modes_end = dims[0] + dense_added_modes
        self.num_modes_measured = (
            measure_subset if measure_subset is not None else dims[0]
        )

        self.encoding = choose_encoding(
            encoding_name=encoding_name,
            num_photons=num_photons,
            num_modes=num_modes,
            num_features=num_features,
            time=time,
            computation_space=computation_space,
            shuffle_amplitude=shuffle_amplitude,
        )
        dims = self.encoding.output_size

        self.conv2d = QConv2d(dims, kernel_size=2, stride=2, circuit=conv_circuit)
        self.pooling = QPooling(dims, kernel_size=2)
        self.dense = QDense(
            (int(dims[0] / 2), int(dims[1] / 2)),
            circuit=dense_circuit,
            add_modes=dense_added_modes,
        )
        self.measure = Measure(
            m=dims[0] + dense_added_modes, n=2, subset=measure_subset
        )

        self.qcnn = nn.Sequential(
            self.encoding, self.conv2d, self.pooling, self.dense, self.measure
        )

        # Output dimension of the QCNN
        # Depends on whether we consider the probability of each Fock state or of each mode separately
        self.output_proba_type = output_proba_type
        if output_proba_type == "state":
            if measure_subset is not None:
                qcnn_output_dim = partial_measurement_output_size(
                    self.num_modes_measured, 2, self.num_modes_end
                )
            else:
                states = generate_all_fock_states_list(
                    self.num_modes_end, 2, true_order=True
                )
                qcnn_output_dim = len(states)
            print(f"Number of Fock states: {qcnn_output_dim}")

        elif output_proba_type == "mode":
            if measure_subset is not None:
                qcnn_output_dim = measure_subset
            else:
                qcnn_output_dim = self.num_modes_end  # Number of modes
        else:
            raise NotImplementedError(
                f"Output probability type {output_proba_type} not implemented"
            )
        self.qcnn_output_dim = qcnn_output_dim

        # Output mapping strategy
        if output_formatting == "Train_linear":
            self.output_mapping = nn.Linear(qcnn_output_dim, num_classes)
        elif output_formatting == "No_train_linear":
            self.output_mapping = nn.Linear(qcnn_output_dim, num_classes)
            self.output_mapping.weight.requires_grad = False
            self.output_mapping.bias.requires_grad = False
        elif output_formatting == "Lex_grouping":
            self.output_mapping = ml.LexGrouping(qcnn_output_dim, num_classes)
        elif output_formatting == "Mod_grouping":
            self.output_mapping = ml.ModGrouping(qcnn_output_dim, num_classes)
        else:
            raise NotImplementedError

        if measure_subset is not None:
            self.keys = generate_partial_fock_states(
                measure_subset, 2, self.num_modes_end
            )
        else:
            self.keys = generate_all_fock_states_list(
                self.num_modes_end, 2, true_order=True
            )

    def forward(self, x):
        probs = self.qcnn(x)

        if self.output_proba_type == "mode":
            probs = marginalize_photon_presence(self.keys, probs)

        output = self.output_mapping(probs)
        output = output * 66

        return output
