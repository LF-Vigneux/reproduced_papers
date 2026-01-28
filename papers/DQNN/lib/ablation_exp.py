"""
Ablation Experiment Module

This module contains functions to run experiments evaluating the importance
of the quantum layer in the Quantum Train algorithm.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import merlin as ML
import perceval as pcvl
import time
from math import comb
import torch.nn as nn
import numpy as np
import torch
from typing import List
import torch.optim as optim
import json
from papers.DQNN.lib.photonic_qt_utils import (
    create_boson_samplers,
    calculate_qubits,
    generate_qubit_states_torch,
)
from papers.DQNN.lib.classical_utils import (
    evaluate_classical_model,
    CNNModel,
    CIFARModel,
    build_parameter_dict,
)
from papers.DQNN.lib.model import (
    PhotonicQuantumTrain,
    train_quantum_model,
    evaluate_model,
)
from torch.func import functional_call
from papers.DQNN.lib.boson_sampler import BosonSampler
from papers.DQNN.utils.utils import plot_ablation_exp, create_datasets


from papers.DQNN.lib.torchmps import MPS

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def create_ablation_class(
    bond,
    bs: List[BosonSampler] = None,
    classical_model: nn.Module = CNNModel(),
    with_general_interferometer: bool = False,
    groupping: bool = False,
    Haar_matrix_init: bool = False,
):
    """
    Create an lone MPS model for the experiment.

    This function defines a CNN model and an ablation module that mimics the quantum train behavior
    using only the classical MPS, along with data loaders for training and validation.

    Parameters
    -----------
    bond : int
        The bond dimension for the MPS in the ablation module.
    bs: List[BosonSampler]
        Boson sampler to use untrained in the ablation Model. If None, just used a randomized tensor
        Default is 1.
    classical_model: nn.Module
        The classical model whose parameters we want to optimize. Default is CNNModel().

    Returns
    --------
    tuple
        A tuple containing the ablation model (AblationModule)
    """
    n_qubit, nw_list_normal = calculate_qubits(classical_model)
    bs = create_boson_samplers(
        nw_list_normal, with_general_interferometer=with_general_interferometer
    )
    num_params = 1
    for i in bs:
        num_params *= comb(i.m, i.n)
        if Haar_matrix_init is True:
            input_state = i.m * [0]
            places = torch.linspace(0, i.m - 1, i.n)
            for photon in places:
                input_state[int(photon)] = 1
            input_state = pcvl.BasicState(input_state)
            i.quantum_layer = ML.QuantumLayer(
                input_size=0,
                n_photons=i.n,
                circuit=pcvl.Circuit(i.m)
                // pcvl.Unitary(pcvl.Matrix.random_unitary(i.m)),
                input_state=input_state,
                computation_space=ML.ComputationSpace.UNBUNCHED,
            )

    random_tensor = torch.randn(
        num_params, 1
    )  # TODO FOr generalization, change to arbitrary size, here is the number of combinations of each BS (4 in 9) and (4 in 8)

    class AblationModule(nn.Module):
        """
        Ablation study module that uses MPS to process weights instead of quantum boson samplers.
        """

        def __init__(
            self,
            groupping: bool = False,
            nw_list_normal: List[float] = None,
            embedding_size: int = None,
        ):
            """
            Initialize the AblationModule with an MPS for weight processing.
            """
            super().__init__()

            self.MappingNetwork = MPS(
                input_dim=n_qubit + 1, output_dim=1, bond_dim=bond
            )
            self.embedding_size = embedding_size
            if groupping is None:
                self.grouper = None
            else:
                self.grouper = ML.ModGrouping(embedding_size, len(nw_list_normal))

        def forward(self, x, classical_model_: nn.Module):
            """
            Forward pass through the ablation module.

            This method processes the input through a classical MPS-based weight generation
            and then applies the CNN layers.

            Parameters
            -----------
            x : torch.Tensor
                Input tensor of shape (batch_size, 1, 28, 28).
            classical_model: nn.Module
                The classical model whose parameters we want to optimize. Default is CNNModel().

            Returns
            --------
            torch.Tensor
                Output tensor of shape (batch_size, 10).
            """

            with torch.no_grad():
                if bs is None:
                    probs_ = random_tensor
                else:
                    probs_ = torch.flatten(bs[0].quantum_layer())
                    new_size = bs[0].embedding_size
                    for i in range(1, len(bs)):
                        new_size *= bs[i].embedding_size
                        probs_ = (
                            torch.outer(probs_, torch.flatten(bs[i].quantum_layer()))
                            .flatten()
                            .reshape(new_size, 1)
                        )
            if self.grouper is None:
                probs_ = probs_[: len(nw_list_normal)]
                probs_ = probs_.reshape(len(nw_list_normal), 1)
            else:
                probs_ = probs_.reshape(1, self.embedding_size)
                probs_ = self.grouper(probs_)
                probs_ = probs_.reshape(len(nw_list_normal), 1)

            # Generate qubit states using PyTorch
            qubit_states_torch = generate_qubit_states_torch(n_qubit)[
                : len(nw_list_normal)
            ]
            qubit_states_torch = qubit_states_torch.to(device)

            # Combine qubit states with probability values using PyTorch
            combined_data_torch = torch.cat((qubit_states_torch, probs_), dim=1)
            combined_data_torch = combined_data_torch.reshape(
                len(nw_list_normal), n_qubit + 1
            )

            prob_val_post_processed = self.MappingNetwork(combined_data_torch)
            prob_val_post_processed = (
                prob_val_post_processed - prob_val_post_processed.mean()
            )

            ########
            param_dict = build_parameter_dict(prob_val_post_processed, classical_model_)
            output = functional_call(classical_model_, param_dict, (x,))
            return output

    return AblationModule(
        groupping=groupping,
        nw_list_normal=nw_list_normal,
        embedding_size=num_params,
    ).to(device)


def run_ablation_exp(
    bond_dimensions_to_test: List[int] = np.arange(2, 17),
    num_training_rounds: int = 2,
    num_epochs: int = 5,
    qu_train_with_cobyla: bool = False,
    num_qnn_train_step: int = 12,
    generate_graph: bool = True,
    run_dir: Path = None,
    with_general_interferometer: bool = False,
    groupping: bool = False,
    use_fashion: bool = False,
    use_cifar: bool = False,
    Haar_matrix_init: bool = False,
):
    """
    Run ablation experiments to evaluate the importance of the quantum layer in the Quantum Train.

    This function iterates over a list of bond dimensions, trains both the photonic QT and an ablation
    study model, collects training loss and accuracy metrics, saves the results to a JSON file
    (/results/ablation_data.json), and can generate a plot comparing the performance across models.

    Parameters
    -----------
    bond_dimensions_to_test : List[int], optional
        List of bond dimension values to test. Default is [2, 3, ..., 16].
    num_training_rounds : int, optional
        Number of training rounds (epochs) of MPS and quantum training. Default is 200.
    num_epochs : int, optional
        Number of epochs for per training round for the MPS. Default is 5.
    qu_train_with_cobyla : bool, optional
        Whether to use COBYLA optimizer for quantum training. Default is False.
    num_qnn_train_step : int, optional
        Number of training steps for the boson samplers per training round. Default is 12. If COBYLA
        is to be used, 1000 is the suggested value.
    generate_graph : bool, optional
        Whether to plot a the resulting graph of the experiment.
        Default .
    run_dir : pathlib.Path, optional
        Output directory for the PDF when running via the shared runtime. If None,
        the plot is saved under the local results folder.

    Returns
    --------
    None
    """
    current_dir = str(Path(__file__).parent.parent.resolve()) + "/results/"

    loss_ablation = []
    accuracy_ablation = []
    params_ablation = []
    loss_qt = []
    accuracy_qt = []
    params_qt = []
    _, _, train_loader, val_loader = create_datasets(
        batch_size=1000, use_fashion=use_fashion, use_CIFAR=use_cifar
    )

    for bond in bond_dimensions_to_test:
        ### QTrain
        if use_cifar is True:

            classical_model = CIFARModel()
        else:

            classical_model = CNNModel()
        n_qubit, nw_list_normal = calculate_qubits(classical_model)
        bs = create_boson_samplers(
            nw_list_normal, with_general_interferometer=with_general_interferometer
        )
        embedding_size = bs[0].embedding_size
        for i in range(1, len(bs)):
            embedding_size *= bs[i].embedding_size
        qt_model = PhotonicQuantumTrain(
            n_qubit,
            bond_dim=bond,
            groupping=groupping,
            nw_list_normal=nw_list_normal,
            embedding_size=embedding_size,
        ).to(device)

        ### Ablation
        ablation_model = create_ablation_class(
            bond=bond,
            bs=bs,
            classical_model=CIFARModel() if use_cifar is True else CNNModel(),
            with_general_interferometer=with_general_interferometer,
            groupping=groupping,
            Haar_matrix_init=Haar_matrix_init,
        )

        params_ablation.append(
            sum(p.numel() for p in ablation_model.parameters() if p.requires_grad)
        )

        # Training setting
        step = 1e-3  # Learning rate

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(ablation_model.parameters(), lr=step)

        # Training loop for the ablation
        for round_ in range(num_training_rounds):
            print("-----------------------")
            print("Ablation")

            acc_list = []
            acc_best = 0
            for epoch in range(num_epochs):
                ablation_model.train()
                train_loss = 0
                for i, (images, labels) in enumerate(train_loader):
                    correct = 0
                    total = 0
                    since_batch = time.time()

                    images, labels = (
                        images.to(device),
                        labels.to(device),
                    )  # Move data to GPU
                    optimizer.zero_grad()
                    # Forward pass
                    outputs = ablation_model(images, classical_model)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # Compute loss
                    loss = criterion(outputs, labels)

                    acc = 100 * correct / total
                    acc_list.append(acc)
                    train_loss += loss.cpu().detach().numpy()

                    if acc > acc_best:
                        acc_best = acc
                    loss.backward()

                    optimizer.step()
                    if (i + 1) * 4 % len(train_loader) == 0:
                        print(
                            f"Training round [{round_ + 1}/{num_training_rounds}], Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, batch time: {time.time() - since_batch:.2f}, accuracy:  {(acc):.2f}%"
                        )
                    loss_ab = loss.item()
                    acc_ab = acc

                train_loss /= len(train_loader)

        acc_ab, loss_ab = evaluate_classical_model(
            ablation_model, val_loader, classical_model=classical_model
        )

        ################################################################################################################################
        print("QTrain")

        qt_model, qnn_parameters_qt, _, _ = train_quantum_model(
            qt_model,
            classical_model,
            train_loader,
            train_loader,
            bs,
            n_qubit,
            nw_list_normal,
            num_training_rounds=num_training_rounds,
            num_epochs=num_epochs,
            qu_train_with_cobyla=qu_train_with_cobyla,
            num_qnn_train_step=num_qnn_train_step,
        )

        accuracy_test, loss_test, _ = evaluate_model(
            qt_model,
            classical_model,
            train_loader,
            val_loader,
            bs,
            n_qubit,
            nw_list_normal,
            qnn_parameters_qt,
        )

        num_trainable_params_qt = sum(
            p.numel() for p in qt_model.parameters() if p.requires_grad
        )

        # Save results
        params_qt.append(
            num_trainable_params_qt + np.sum([i.num_effective_params for i in bs])
        )
        loss_qt.append(loss_test)
        accuracy_qt.append(accuracy_test)

        loss_ablation.append(loss_ab)
        accuracy_ablation.append(acc_ab)

        json_payload = {
            "loss_qt": [float(v) for v in loss_qt],
            "accuracy_qt": [float(v) for v in accuracy_qt],
            "params_qt": [int(v) for v in params_qt],
            "loss_ablation": [float(v) for v in loss_ablation],
            "accuracy_ablation": [float(v) for v in accuracy_ablation],
            "params_ablation": [int(v) for v in params_ablation],
        }
        json_str = json.dumps(json_payload, indent=4)
        with open(current_dir + "ablation_data.json", "w") as f:
            f.write(json_str)
    if generate_graph is True:
        plot_ablation_exp(
            params_qt=params_qt,
            accuracy_qt=accuracy_qt,
            params_ablation=params_ablation,
            accuracy_ablation=accuracy_ablation,
            run_dir=run_dir,
        )


# run_ablation_exp(num_training_rounds=50)
