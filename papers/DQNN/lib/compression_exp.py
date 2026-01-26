"""
Compression Experiment Module

This module contains functions to run experiments evaluating the importance
of the quantum layer in the Quantum Train algorithm.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import os
import numpy as np
import torch
from typing import List
import json
from papers.DQNN.lib.photonic_qt_utils import (
    create_boson_samplers,
    calculate_qubits,
)
from papers.DQNN.lib.model import (
    PhotonicQuantumTrain,
    train_quantum_model,
    evaluate_model,
)
from papers.DQNN.lib.classical_utils import (
    train_classical_cnn,
    evaluate_classical_model,
    CNNModel,
)
from papers.DQNN.utils.utils import create_datasets, plot_compression_exp


from papers.DQNN.lib.torchmps import MPS

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def run_compression_exp(
    bond_dimensions_to_test: List[int] = np.arange(2, 17),
    num_training_rounds: int = 2,
    classical_epochs: int = 50,
    num_epochs: int = 5,
    qu_train_with_cobyla: bool = False,
    num_qnn_train_step: int = 12,
    generate_graph: bool = True,
    run_dir: Path = None,
    with_general_interferometer: bool = False,
    groupping: bool = False,
    use_fashion: bool = False,
):
    """
    Run compression experiments across bond dimensions and baselines.

    This compares the quantum train model against classical weight sharing and
    pruning baselines while tracking accuracy, parameter counts, and QTrain
    generalization error.

    Parameters
    ----------
    bond_dimensions_to_test : List[int], optional
        Bond dimensions (or shared rows for weight sharing) to evaluate.
        Default is np.arange(2, 17).
    num_training_rounds : int, optional
        Number of quantum training rounds. Default is 2.
    classical_epochs : int, optional
        Number of epochs for the classical baselines. Default is 50.
    num_epochs : int, optional
        Number of epochs per quantum training round for the mapping network.
        Default is 5.
    qu_train_with_cobyla : bool, optional
        Whether to use COBYLA for QNN optimization. Default is False.
    num_qnn_train_step : int, optional
        Number of QNN training steps per round. Default is 12.
    generate_graph : bool, optional
        Whether to generate the compression plot. Default is True.
    run_dir : pathlib.Path, optional
        Output directory for the plot when running via the shared runtime.
        If None, the plot is saved under the local results folder.

    Returns
    -------
    None
    """

    current_dir = str(Path(__file__).parent.parent.resolve()) + "/results/"

    accuracy_ws = []
    params_ws = []
    accuracy_prun = []
    params_prun = []
    gen_error_qt = []
    accuracy_qt = []
    params_qt = []

    _, _, train_loader, val_loader = create_datasets(
        batch_size=1000, use_fashion=use_fashion
    )
    pruning_iterator = 0
    for bond in bond_dimensions_to_test:
        ### QTrain
        print(
            "---------------------------------------------------------------------------------"
        )
        print("QTrain")
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
        qt_model, qnn_parameters, _, _ = train_quantum_model(
            qt_model,
            classical_model,
            train_loader,
            train_loader,
            bs,
            n_qubit,
            nw_list_normal,
            num_training_rounds=num_training_rounds,
            num_epochs=num_epochs,
            num_qnn_train_step=num_qnn_train_step,
            qu_train_with_cobyla=qu_train_with_cobyla,
        )
        num_trainable_params_qt = sum(
            p.numel() for p in qt_model.parameters() if p.requires_grad
        )
        params_qt.append(num_trainable_params_qt)
        acc_qt, _, gen_error = evaluate_model(
            qt_model,
            classical_model,
            train_loader,
            val_loader,
            bs,
            n_qubit,
            nw_list_normal,
            qnn_parameters,
        )
        accuracy_qt.append(acc_qt)
        gen_error_qt.append(gen_error)

        ### WS
        print(
            "---------------------------------------------------------------------------------"
        )
        print("WS")
        ws_model = train_classical_cnn(
            train_loader,
            val_loader,
            classical_epochs,
            use_pruning=False,
            use_weight_sharing=True,
            shared_rows=bond,
        )
        num_trainable_params_ws = sum(
            p.numel() for p in ws_model.parameters() if p.requires_grad
        )
        params_ws.append(num_trainable_params_ws)
        acc_ws, _ = evaluate_classical_model(ws_model, val_loader)
        accuracy_ws.append(acc_ws)

        ### Pruning
        print(
            "---------------------------------------------------------------------------------"
        )
        print("Pruning")
        pruning_iterator += 1
        prun_model = train_classical_cnn(
            train_loader,
            val_loader,
            classical_epochs,
            use_pruning=True,
            pruning_amount=pruning_iterator / len(bond_dimensions_to_test) * 0.7,
        )
        num_trainable_params_prun = int(
            (1 - (pruning_iterator / len(bond_dimensions_to_test) * 0.7))
            * sum(p.numel() for p in prun_model.parameters() if p.requires_grad)
        )
        params_prun.append(num_trainable_params_prun)
        acc_prun, _ = evaluate_classical_model(prun_model, val_loader)
        accuracy_prun.append(acc_prun)

        json_payload = {
            "accuracy_ws": [float(v) for v in accuracy_ws],
            "params_ws": [int(v) for v in params_ws],
            "accuracy_prun": [float(v) for v in accuracy_prun],
            "params_prun": [int(v) for v in params_prun],
            "gen_error_qt": [float(v) for v in gen_error_qt],
            "accuracy_qt": [float(v) for v in accuracy_qt],
            "params_qt": [int(v) for v in params_qt],
        }
        json_str = json.dumps(json_payload, indent=4)
        with open(current_dir + "compression_data.json", "w") as f:
            f.write(json_str)
    if generate_graph:
        plot_compression_exp(
            accuracy_ws,
            params_ws,
            accuracy_prun,
            params_prun,
            accuracy_qt,
            gen_error_qt,
            params_qt,
            run_dir=run_dir,
        )
