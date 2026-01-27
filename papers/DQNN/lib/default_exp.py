"""
Default experiment runner for the DQNN photonic quantum train model.

This module trains a classical CNN baseline, then trains and evaluates the
photonic quantum train model using the specified hyperparameters.
"""

import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
import warnings

warnings.filterwarnings("ignore")

from papers.DQNN.lib.photonic_qt_utils import (
    create_boson_samplers,
    calculate_qubits,
)
from papers.DQNN.lib.model import (
    PhotonicQuantumTrain,
    train_quantum_model,
    evaluate_model,
)
from papers.DQNN.lib.classical_utils import train_classical_cnn, CNNModel, CIFARModel
from papers.DQNN.utils.utils import plot_training_metrics, create_datasets

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def run_default_exp(
    bond_dim: int = 7,
    num_training_rounds: int = 50,
    num_epochs: int = 5,
    num_qnn_train_step: int = 12,
    classical_epochs: int = 5,
    pruning: bool = False,
    pruning_amount: float = 0.5,
    weight_sharing: bool = False,
    shared_rows: int = 10,
    qu_train_with_cobyla: bool = False,
    generate_graph: bool = True,
    run_dir: Path = None,
    with_general_interferometer: bool = False,
    groupping: bool = False,
    use_fashion: bool = False,
    use_cifar: bool = False,
):
    """
    Run the default experiment workflow.

    Parameters
    ----------
    bond_dim : int, optional
        Bond dimension for the photonic quantum train model. Default is 7.
    num_training_rounds : int, optional
        Number training rounds: the number of MPS and then the quantum layer training epochs. Default is 50.
    num_epochs : int, optional
        Number of epochs per training round for the MPS. Default is 5.
    num_qnn_train_step : int, optional
        Number of QNN training steps per round. Default is 12.
    classical_epochs : int, optional
        Number of epochs for the classical CNN baseline. Default is 5.
    pruning : bool, optional
        Whether to enable pruning during classical training. Default is False.
    pruning_amount : float, optional
        Fraction of weights to prune if pruning is enabled. Default is 0.5.
    weight_sharing : bool, optional
        Whether to enable weight sharing in the classical model. Default is False.
    shared_rows : int, optional
        Number of rows to share if weight sharing is enabled. Default is 10.
    qu_train_with_cobyla : bool, optional
        Whether to use COBYLA optimizer for quantum training. Default is False. If COBYLA
        is to be used, 1000 is the suggested value.
    generate_graph : bool, optional
        Whether to plot a summary of train/test metrics after evaluation.
        Default .
    run_dir : pathlib.Path, optional
        Output directory for the PDF when running via the shared runtime. If None,
        the plot is saved under the local results folder.

    Returns
    -------
    None
    """

    print(f"Running experiment with:")
    print(f"  Pruning: {pruning} (amount: {pruning_amount if pruning else 'N/A'})")
    print(
        f"  Weight sharing: {weight_sharing} (shared rows: {shared_rows if weight_sharing else 'N/A'})"
    )

    if use_cifar is True:
        model = CIFARModel()
    else:
        model = CNNModel(use_weight_sharing=weight_sharing, shared_rows=shared_rows)

    train_dataset, _, train_loader, val_loader = create_datasets(
        use_fashion=use_fashion, use_CIFAR=use_cifar
    )

    _ = train_classical_cnn(
        model,
        train_loader,
        val_loader,
        classical_epochs,
        use_pruning=pruning,
        pruning_amount=pruning_amount,
    )
    if use_cifar is True:
        classical_model = CIFARModel()
    else:
        classical_model = CNNModel(use_weight_sharing=False)
    n_qubit, nw_list_normal = calculate_qubits(classical_model)
    bs = create_boson_samplers(
        nw_list_normal, with_general_interferometer=with_general_interferometer
    )

    embedding_size = bs[0].embedding_size
    for i in range(1, len(bs)):
        embedding_size *= bs[i].embedding_size

    qt_model = PhotonicQuantumTrain(
        n_qubit,
        bond_dim=bond_dim,
        groupping=groupping,
        nw_list_normal=nw_list_normal,
        embedding_size=embedding_size,
    ).to(device)

    batch_size_qnn = 1000
    train_loader_qnn = DataLoader(train_dataset, batch_size_qnn, shuffle=True)

    qt_model, _, loss_list_epoch, acc_list_epoch = train_quantum_model(
        qt_model,
        classical_model,
        train_loader,
        train_loader_qnn,
        bs,
        n_qubit,
        nw_list_normal,
        num_training_rounds,
        num_epochs,
        qu_train_with_cobyla=qu_train_with_cobyla,
        num_qnn_train_step=num_qnn_train_step,
    )

    evaluate_model(
        qt_model,
        classical_model,
        train_loader,
        val_loader,
        bs,
        n_qubit,
        nw_list_normal,
    )

    if generate_graph is True:
        plot_training_metrics(loss_list_epoch, acc_list_epoch, run_dir=run_dir)
