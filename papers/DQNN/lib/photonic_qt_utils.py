"""
Photonic quantum train utilities for DQNN experiments.

This module provides helpers for boson sampler creation,
qubit calculations, and probability-to-weight mapping utilities.
"""

import numpy as np
import torch.nn as nn
from math import comb
import torch
import sys
import os
from typing import List, Tuple
from papers.DQNN.lib.boson_sampler import BosonSampler

sys.path.append(os.path.join(os.path.dirname(__file__), "torchmps"))

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def create_boson_samplers(nw_list_normal: List[float]) -> BosonSampler:
    """
    Create the boson samplers used in photonic quantum training.

    Parameters
    ----------
    nw_list_normal : List[float]
        Indices of network weights to keep from the generated probabilities.

    Returns
    -------
    tuple
        The list of BosonSamplers composing the quantum layer.
    """
    nw_list_normal_len = len(nw_list_normal)
    bs = []

    num_bs = int(
        np.floor(np.log(nw_list_normal_len) / np.log(252))
    )  # 252 is comb(10,5)

    for _ in range(num_bs):
        bs.append(BosonSampler(m=10, n=5))
    num_params_filled = 256**num_bs
    if num_params_filled == nw_list_normal_len:
        return bs
    for i in range(1, 11):
        if num_params_filled * comb(i, i // 2) >= nw_list_normal_len:
            bs.append(BosonSampler(m=i, n=i // 2))
            return bs
    raise SyntaxError("Function create_boson_samplers failed")


def calculate_qubits(model: nn.Module) -> Tuple[int, List[float]]:
    """
    Compute the number of qubits required for the CNN weight mapping.

    Parameters
    ----------
    model: nn.Module
        The classical model whose parameters we want to optimize.

    Returns
    -------
    Tuple[int, List[float]]
        Number of qubits and flattened list of CNN parameters.
    """
    numpy_weights = {}
    nw_list = []
    nw_list_normal = []
    for name, param in model.state_dict().items():
        numpy_weights[name] = param.cpu().numpy()
    for i in numpy_weights:
        nw_list.append(list(numpy_weights[i].flatten()))
    for i in nw_list:
        for j in i:
            nw_list_normal.append(j)
    print("# of NN parameters for quantum circuit: ", len(nw_list_normal))
    n_qubits = int(np.ceil(np.log2(len(nw_list_normal))))
    print("Required qubit number: ", n_qubits)
    n_qubit = n_qubits
    return n_qubit, nw_list_normal


def probs_to_weights(probs_: torch.Tensor, model_template: torch.nn.Module) -> dict:
    """
    Convert a flat probability tensor into a model state dict.

    Parameters
    ----------
    probs_ : torch.Tensor
        Flattened probability values.
    model_template : torch.nn.Module
        Model whose state dict shapes are used for reshaping.

    Returns
    -------
    dict
        State dict with tensors reshaped to match the template model.
    """
    new_state_dict = {}
    data_iterator = probs_.view(-1)

    for name, param in model_template.state_dict().items():
        shape = param.shape
        num_elements = param.numel()
        chunk = data_iterator[:num_elements].reshape(shape)
        new_state_dict[name] = chunk
        data_iterator = data_iterator[num_elements:]

    return new_state_dict


def generate_qubit_states_torch(n_qubit: int) -> torch.Tensor:
    """
    Generate all qubit states in the {-1, 1} basis.

    Parameters
    ----------
    n_qubit : int
        Number of qubits.

    Returns
    -------
    torch.Tensor
        Tensor of shape (2**n_qubit, n_qubit) with all basis states.
    """
    all_states = torch.cartesian_prod(*[torch.tensor([-1, 1]) for _ in range(n_qubit)])
    return all_states
