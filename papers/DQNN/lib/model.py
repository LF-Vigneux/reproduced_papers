"""
Models and training utilities for the DQNN photonic quantum train.

This module defines the quantum train model, along with training and evaluation
helpers used by the experiment runners.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import merlin as ML
from scipy.optimize import minimize
from papers.DQNN.lib.photonic_qt_utils import (
    generate_qubit_states_torch,
)
from papers.DQNN.lib.classical_utils import build_parameter_dict
from papers.DQNN.lib.boson_sampler import BosonSampler
from typing import List, Tuple


from papers.DQNN.lib.torchmps import MPS

from torch.func import functional_call

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class PhotonicQuantumTrain(nn.Module):
    """
    Photonic quantum train model that maps quantum probabilities to CNN weights.

    Parameters
    ----------
    n_qubit : int
        Number of qubits used to generate the quantum states.
    bond_dim : int, optional
        Bond dimension for the MPS mapping network. Default is 7.
    """

    def __init__(
        self,
        n_qubit: int,
        bond_dim: int = 7,
        groupping: bool = False,
        nw_list_normal: List[float] = None,
        embedding_size: int = None,
    ):
        """
        Initialize the mapping network based on an MPS.

        Parameters
        ----------
        n_qubit : int
            Number of qubits used to generate the quantum states.
        bond_dim : int, optional
            Bond dimension for the MPS mapping network. Default is 7.
        """
        super().__init__()
        self.MappingNetwork = MPS(
            input_dim=n_qubit + 1, output_dim=1, bond_dim=bond_dim
        )
        self.embedding_size = embedding_size
        if groupping is None:
            print("No")
            self.grouper = None
        else:
            print("Yes")
            self.grouper = ML.ModGrouping(embedding_size, len(nw_list_normal))

    def extract_parameters(
        self,
        bs: List[BosonSampler],
        n_qubit: int,
        nw_list_normal: List[float],
    ):
        """
        Convert quantum-layer probabilities into a CNN-compatible state dict.

        Parameters
        ----------
        bs : List[BosonSampler]
            The list of BosonSamplers composing the quantum layer.
        n_qubit : int
            Number of qubits used to generate the quantum states.
        nw_list_normal : List[float]
            Indices of network weights to keep from the generated probabilities.

        Returns
        -------
        dict
            State dict reshaped to match the CNN template.
        """

        # Generate the probabilities from the quantum layers
        probs_ = torch.flatten(bs[0].quantum_layer())
        new_size = bs[0].embedding_size
        for i in range(1, len(bs)):
            new_size *= bs[i].embedding_size
            probs_ = (
                torch.outer(probs_, torch.flatten(bs[i].quantum_layer()))
                .flatten()
                .reshape(new_size, 1)
            )
        # Get the necessary probabilities
        if self.grouper is None:
            probs_ = probs_[: len(nw_list_normal)]
            probs_ = probs_.reshape(len(nw_list_normal), 1)
        else:
            probs_ = probs_.reshape(1, self.embedding_size)
            probs_ = self.grouper(probs_)
            probs_ = probs_.reshape(len(nw_list_normal), 1)

        # Generate all qubit states
        qubit_states_torch = generate_qubit_states_torch(n_qubit)[: len(nw_list_normal)]
        qubit_states_torch = qubit_states_torch.to(device)

        # Combining the data to prepare the MPS
        combined_data_torch = torch.cat((qubit_states_torch, probs_), dim=1)
        combined_data_torch = combined_data_torch.reshape(
            len(nw_list_normal), n_qubit + 1
        )

        # MPS transforms the probs to weights
        prob_val_post_processed = self.MappingNetwork(combined_data_torch)
        prob_val_post_processed = (
            prob_val_post_processed - prob_val_post_processed.mean()
        )

        # Always use standard CNN architecture for quantum training regardless of classical training method
        return prob_val_post_processed

    def forward(
        self,
        x: torch.Tensor,
        bs: List[BosonSampler],
        n_qubit: int,
        nw_list_normal: List[float],
        classical_model: nn.Module,
    ) -> torch.Tensor:
        """
        Run the forward pass by mapping quantum probabilities to CNN weights.

        Parameters
        ----------
        x : torch.Tensor
            Input images tensor.
        bs : List[BosonSampler]
            The list of BosonSamplers composing the quantum layer.
        n_qubit : int
            Number of qubits used to generate the quantum states.
        nw_list_normal : List[float]
            Indices of network weights to keep from the generated probabilities.
        classical_model: nn.Module
            The classical model whose parameters we want to optimize.

        Returns
        -------
        torch.Tensor
            Model logits for classification.
        """
        params = self.extract_parameters(bs, n_qubit, nw_list_normal)

        param_dict = build_parameter_dict(params, classical_model)
        output = functional_call(classical_model, param_dict, (x,))
        return output

        # params = self.extract_parameters(bs, n_qubit, nw_list_normal)
        # params.to(torch.float32)
        # assign_parameters(params, classical_model)
        # return classical_model(x)


def train_quantum_model(
    qt_model: PhotonicQuantumTrain,
    classical_model: nn.Module,
    train_loader: DataLoader,
    train_loader_qnn: DataLoader,
    bs: List[BosonSampler],
    n_qubit: int,
    nw_list_normal: List[float],
    num_training_rounds: int,
    num_epochs: int,
    num_qnn_train_step: int = 12,
    qu_train_with_cobyla: bool = False,
) -> Tuple[PhotonicQuantumTrain, List[float], List[float], List[float]]:
    """
    Train the quantum train model and quantum layer parameters.

    Parameters
    ----------
    qt_model : PhotonicQuantumTrain
        Model to train.
    classical_model: nn.Module
        The classical model whose parameters we want to optimize.
    train_loader : DataLoader
        Loader for standard training batches.
    train_loader_qnn : DataLoader
        Loader for QNN parameter training batches.
    bs : List[BosonSampler]
        The list of BosonSamplers composing the quantum layer.
    n_qubit : int
        Number of qubits used to generate the quantum states.
    nw_list_normal : List[float]
        Indices of network weights to keep from the generated probabilities.
    num_training_rounds : int
        Number of training rounds.
    num_epochs : int
        Number of epochs per training round for the MPS mapping network.
    num_qnn_train_step : int, optional
        Number of optimization steps for the QNN parameters. Default is 12. If the
        COBYLA optimizer is to be used, 1000 is the suggested value.
    qu_train_with_cobyla : bool, optional
        Whether to use COBYLA for QNN optimization. Default is False.

    Returns
    -------
    Tuple[PhotonicQuantumTrain, List[float], List[float], List[float]]
        Trained model, final QNN parameters, epoch losses, and epoch accuracies.
    """
    step = 1e-3
    q_delta = 2 * np.pi

    init_qnn_parameters = q_delta * np.random.rand(
        np.sum([i.num_effective_params for i in bs])
    )
    print(f"\n ---- QNN parameters of shape {init_qnn_parameters.shape} \n ----")

    qnn_parameters = init_qnn_parameters

    criterion = nn.CrossEntropyLoss()
    optimizer_mapping = optim.Adam(qt_model.parameters(), lr=step)
    if not qu_train_with_cobyla:
        optimizers = []
        for i in bs:
            optimizers.append(optim.Adam(i.quantum_layer.parameters(), lr=step))

    num_trainable_params = sum(
        p.numel() for p in qt_model.parameters() if p.requires_grad
    )
    print("# of trainable parameter in Mapping model: ", num_trainable_params)
    print(
        "# of trainable parameter in QNN model: ",
        len(init_qnn_parameters),
    )
    print(
        "# of trainable parameter in full model: ",
        num_trainable_params + len(init_qnn_parameters),
    )

    loss_list = []
    loss_list_epoch = []
    acc_list_epoch = []

    for round_ in range(num_training_rounds):
        print("-----------------------")

        acc_list = []
        acc_best = 0
        epoch_loss = 0.0
        epoch_acc = 0.0

        # Training on regular batches
        for epoch in range(num_epochs):
            qt_model.train()
            train_loss = 0
            for i, (images, labels) in enumerate(train_loader):
                correct = 0
                total = 0
                since_batch = time.time()

                images, labels = images.to(device), labels.to(device)
                optimizer_mapping.zero_grad()
                outputs = qt_model(
                    images,
                    bs=bs,
                    n_qubit=n_qubit,
                    nw_list_normal=nw_list_normal,
                    classical_model=classical_model,
                )
                # labels_one_hot = F.one_hot(labels, num_classes=10).float()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)

                loss_list.append(loss.cpu().detach().numpy())
                acc = 100 * correct / total
                acc_list.append(acc)
                train_loss += loss.cpu().detach().numpy()

                if acc > acc_best:
                    acc_best = acc

                loss.backward()
                optimizer_mapping.step()

                if (i + 1) * 4 % len(train_loader) == 0:
                    print(
                        f"Training round [{round_ + 1}/{num_training_rounds}], Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, batch time: {time.time() - since_batch:.2f}, accuracy:  {(acc):.2f}%"
                    )

            train_loss /= len(train_loader)

        # QNN parameter optimization using scipy minimize (like in ref.ipynb)
        if qu_train_with_cobyla:
            train_iter = iter(train_loader_qnn)
            images, labels = next(train_iter)

            global qnn_train_step
            qnn_train_step = 0

            def qnn_minimize_loss(qnn_parameters_=None):
                global qnn_train_step

                correct = 0
                total = 0

                images_gpu, labels_gpu = images.to(device), labels.to(device)

                param_index = 0
                for i in bs:
                    i.set_params(
                        qnn_parameters_[
                            param_index : param_index + i.num_effective_params
                        ]
                    )
                    param_index += i.num_effective_params

                outputs = qt_model(
                    images_gpu,
                    bs=bs,
                    n_qubit=n_qubit,
                    nw_list_normal=nw_list_normal,
                    classical_model=classical_model,
                )
                _, predicted = torch.max(outputs.data, 1)
                total += labels_gpu.size(0)
                correct += (predicted == labels_gpu).sum().item()
                loss = criterion(outputs, labels_gpu)
                loss_val = loss.cpu().detach().numpy()
                acc = 100 * correct / total

                qnn_train_step += 1
                if qnn_train_step % 100 == 0:
                    print(
                        f"Training round [{round_ + 1}/{num_training_rounds}], qnn_train_step: [{qnn_train_step}/{1000}], loss: {loss_val}, accuracy: {acc} %"
                    )

                return loss_val

            # Use scipy minimize like in ref.ipynb
            init_param = qnn_parameters
            result = minimize(
                qnn_minimize_loss,
                init_param,
                method="COBYLA",
                options={"maxiter": num_qnn_train_step, "adaptive": True},
            )
            qnn_parameters = result.x
            param_index = 0
            for i in bs:
                i.set_params(
                    qnn_parameters[param_index : param_index + i.num_effective_params]
                )
                param_index += i.num_effective_params

            # Update epoch metrics with final values
            epoch_loss = result.fun
            # Calculate final accuracy for this batch
            images_gpu, labels_gpu = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = qt_model(
                    images_gpu,
                    bs=bs,
                    n_qubit=n_qubit,
                    nw_list_normal=nw_list_normal,
                    classical_model=classical_model,
                )
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels_gpu).sum().item()
                total = labels_gpu.size(0)
                epoch_acc = 100 * correct / total

            loss_list_epoch.append(epoch_loss)
            acc_list_epoch.append(epoch_acc)

        else:
            for train_s in range(num_qnn_train_step):
                for i in bs:
                    i.quantum_layer.train()

                for i, (images, labels) in enumerate(train_loader_qnn):
                    correct = 0
                    total = 0
                    since_batch = time.time()

                    images, labels = images.to(device), labels.to(device)
                    for optimizer in optimizers:
                        optimizer.zero_grad()

                    outputs = qt_model(
                        images,
                        bs=bs,
                        n_qubit=n_qubit,
                        nw_list_normal=nw_list_normal,
                        classical_model=classical_model,
                    )
                    # labels_one_hot = F.one_hot(labels, num_classes=10).float()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)

                    loss_list.append(loss.cpu().detach().numpy())
                    acc = 100 * correct / total
                    acc_list.append(acc)
                    train_loss += loss.cpu().detach().numpy()

                    if acc > acc_best:
                        acc_best = acc

                    loss.backward()
                    for optimizer in optimizers:
                        optimizer.step()

                    if ((i + 1) * 4) % len(train_loader_qnn) == 0:
                        print(
                            f"Training round [{round_ + 1}/{num_training_rounds}], Q-Epoch [{train_s + 1}/{num_qnn_train_step}], Step [{i + 1}/{len(train_loader_qnn)}], Loss: {loss.item():.4f}, batch time: {time.time() - since_batch:.2f}, accuracy:  {(acc):.2f}%"
                        )

            loss_list_epoch.append(float(loss.item()))
            acc_list_epoch.append(float(acc))

    return qt_model, qnn_parameters, loss_list_epoch, acc_list_epoch


def evaluate_model(
    qt_model: PhotonicQuantumTrain,
    classical_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    bs: List[BosonSampler],
    n_qubit: int,
    nw_list_normal: List[float],
    qnn_parameters: List[float] = None,
) -> Tuple[float, float, float]:
    """
    Evaluate the model on train and validation sets.

    Parameters
    ----------
    qt_model : PhotonicQuantumTrain
        Trained model to evaluate.
    classical_model: nn.Module
        The classical model whose parameters we want to optimize.
    train_loader : DataLoader
        Loader for the training set.
    val_loader : DataLoader
        Loader for the validation set.
    bs : List[BosonSampler]
        The list of BosonSamplers composing the quantum layer.
    n_qubit : int
        Number of qubits used to generate the quantum states.
    nw_list_normal : List[float]
        Indices of network weights to keep from the generated probabilities.
    qnn_parameters : List[float] | None, optional
        Parameters to temporarily set for evaluation. If None, uses current
        boson-sampler parameters.

    Returns
    -------
    tuple[float, float, float]
        Test accuracy, test loss, and generalization error.
    """
    if qnn_parameters is not None:
        original_params = []
        for i in bs:
            for param in i.quantum_layer.parameters():
                original_params.extend(param.detach().cpu().flatten().tolist())

        param_index = 0
        for i in bs:
            i.set_params(
                qnn_parameters[param_index : param_index + i.num_effective_params]
            )
            param_index += i.num_effective_params

    criterion = nn.CrossEntropyLoss()

    qt_model.eval()
    correct = 0
    total = 0
    loss_train_list = []
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = qt_model(
                images,
                bs=bs,
                n_qubit=n_qubit,
                nw_list_normal=nw_list_normal,
                classical_model=classical_model,
            )
            loss_train = criterion(outputs, labels).cpu().detach().numpy()
            loss_train_list.append(loss_train)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc_train = 100 * correct / total
    loss_train = np.mean(loss_train_list)

    print(f"Accuracy on the train set: {acc_train:.2f}%")
    print(f"Loss on the train set: {loss_train:.2f}")

    qt_model.eval()
    correct = 0
    total = 0
    loss_test_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = qt_model(
                images,
                bs=bs,
                n_qubit=n_qubit,
                nw_list_normal=nw_list_normal,
                classical_model=classical_model,
            )
            loss_test = criterion(outputs, labels).cpu().detach().numpy()
            loss_test_list.append(loss_test)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc_test = 100 * correct / total
    loss_test = np.mean(loss_test_list)
    gen_error = np.mean(loss_test_list) - np.mean(loss_train_list)

    print(f"Accuracy on the test set: {acc_test:.2f}%")
    print(f"Loss on the test set: {loss_test:.2f}")
    print("Generalization error:", gen_error)

    if qnn_parameters is not None:
        param_index = 0
        for i in bs:
            i.set_params(
                original_params[param_index : param_index + i.num_effective_params]
            )
            param_index += i.num_effective_params

    return (
        acc_test,
        loss_test,
        gen_error,
    )
