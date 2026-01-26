"""
Utility functions and datasets for DQNN experiments.

This module provides dataset loaders, plotting helpers, and common metrics.
"""

import argparse
from datasets import load_dataset
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import regex as re
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
from torch.utils.data import DataLoader
import pathlib
import os

script_dir = Path(__file__).parent.parent.parent.parent
DATA_PATH = (script_dir / "data/DQNN").resolve()


class MNIST_fashion(Dataset):
    def __init__(self, data=DATA_PATH, transform=None, split="train"):
        """
        Args:
            data: path to dataset folder which contains train.csv and val.csv
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., data augmentation or normalization)
            split: 'train' or 'val' to determine which set to download
        """
        self.data_dir = data
        self.transform = transform
        self.data = []

        if split == "train":
            filename = os.path.join(self.data_dir, "fashion-mnist_train_formatted.csv")
        elif split == "val":
            filename = os.path.join(self.data_dir, "fashion-mnist_test_formatted.csv")
        else:
            raise AttributeError(
                "split!='train' and split!='val': split must be train or val"
            )

        self.df = pd.read_csv(filename)

    def __len__(self):
        l = len(self.df["image"])
        return l

    def __getitem__(self, idx):
        img = self.df["image"].iloc[idx]
        label = self.df["label"].iloc[idx]
        # string to list
        img_list = re.split(r",", img)
        # remove '[' and ']'
        img_list[0] = img_list[0][1:]
        img_list[-1] = img_list[-1][:-1]
        # convert to float
        img_float = [float(el) for el in img_list]
        # convert to image
        img_square = torch.unflatten(torch.tensor(img_float), 0, (1, 28, 28))
        if self.transform is not None:
            img_square = self.transform(img_square)
        return img_square, label


class HFImageDataset(Dataset):
    """
    Torch Dataset wrapper for HF datasets with image/label pairs.
    """

    def __init__(self, dataset, transform: callable = None):
        self.dataset = pd.DataFrame(dataset)
        self.transform = transform

    def __len__(self) -> int:
        l = len(self.dataset["image"])
        return l

    def __getitem__(self, idx: int):
        """
        Return one sample from the dataset.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[torch.Tensor, int]
            Image tensor and its label.
        """
        img = self.dataset["image"].iloc[idx]
        label = self.dataset["label"].iloc[idx]
        # string to list
        img_list = re.split(r",", img)
        # remove '[' and ']'
        img_list[0] = img_list[0][1:]
        img_list[-1] = img_list[-1][:-1]
        # convert to float
        img_float = [float(el) for el in img_list]
        # convert to image
        img_square = torch.unflatten(torch.tensor(img_float), 0, (1, 28, 28))
        if self.transform is not None:
            img_square = self.transform(img_square)
        return img_square, label


def create_datasets(
    batch_size: int = 128, use_fashion: bool = False
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    """
    Create MNIST train/validation datasets and data loaders.
    Parameters
    ----------
    batch_size: int
        The number of elements per batches. Default is 128

    Returns
    -------
    Tuple[Dataset, Dataset, DataLoader, DataLoader]
        Train dataset, validation dataset, train loader and validation loader.
    """
    if use_fashion:
        train_dataset = MNIST_fashion(split="train")
        val_dataset = MNIST_fashion(split="val")
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        return train_dataset, val_dataset, train_loader, val_loader

    train_dataset = HFImageDataset(
        load_dataset("Quandela/PercevalQuest-MNIST", split="train")
    )
    val_dataset = HFImageDataset(
        load_dataset("Quandela/PercevalQuest-MNIST", split="validation")
    )
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    return train_dataset, val_dataset, train_loader, val_loader


# compute the accuracy of the model
def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute classification accuracy.

    Parameters
    ----------
    outputs : torch.Tensor
        Model logits of shape (batch, num_classes).
    labels : torch.Tensor
        Ground-truth labels of shape (batch,).

    Returns
    -------
    torch.Tensor
        Accuracy as a scalar tensor.
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def int_list(arg):
    """
    Parse a comma-separated string into a list of integers.

    Parameters
    ----------
    arg : str
        Comma-separated integers (e.g., "2,4,8").

    Returns
    -------
    list[int]
        Parsed list of integers.
    """
    return list(map(int, arg.split(",")))


def parse_args():
    """
    Parse command-line arguments for the experiment runner.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Photonic Quantum Training")
    parser.add_argument(
        "--exp_to_run",
        type=str,
        default="DEFAULT",
        help="Which experiment to run between 'DEFAULT', 'BOND', 'ABLATION' and COMPRESSION (default: 'DEFAULT')",
    )
    parser.add_argument(
        "--bond_dim",
        type=int,
        default=7,
        help="ONLY FOR THE DEFAULT EXPERIMENT! Bond dimension for MPS (default: 7)",
    )
    parser.add_argument(
        "--bond_dimensions_to_test",
        type=int_list,
        default=np.arange(1, 11),
        help="ONLY FOR THE ABLATION AND BOND DIMENSION EXPERIMENT! Bond dimension to test for MPS. Each number needs to be seperated by commas. (default: np.arange(1, 11))",
    )
    parser.add_argument(
        "--num_training_rounds",
        type=int,
        default=2,
        help="Number of training rounds for quantum model (default: 2)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs for training (default: 5)",
    )
    parser.add_argument(
        "--num_qnn_train_step",
        type=int,
        default=12,
        help="Number of epochs for training the quantum layer (default: 12). If the COBYLA is to be runned, 1000 is the suggested value",
    )
    parser.add_argument(
        "--classical_epochs",
        type=int,
        default=1,
        help="Number of epochs for classical CNN training (default: 1)",
    )
    parser.add_argument(
        "--qu_train_with_cobyla",
        action="store_true",
        help="Enable optimize the quantum layer with COBYLA",
    )
    parser.add_argument(
        "--pruning", action="store_true", help="Enable pruning experiment"
    )
    parser.add_argument(
        "--pruning_amount",
        type=float,
        default=0.5,
        help="Pruning amount (default: 0.5)",
    )
    parser.add_argument(
        "--weight_sharing", action="store_true", help="Enable weight sharing experiment"
    )
    parser.add_argument(
        "--shared_rows",
        type=int,
        default=10,
        help="Number of shared rows for weight sharing (default: 10)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file to override CLI arguments",
    )
    parser.add_argument(
        "--dont_generate_graph",
        action="store_true",
        help="Disable graph generation",
    )
    return parser.parse_args()


def plot_training_metrics(
    loss_list_epoch: List[float],
    acc_list_epoch: List[float],
    run_dir: Path = None,
):
    """
    Plot training accuracy and loss curves side by side. The plot is
    saved as a PDF file: results/training_metrics_graph.pdf (or to run_dir if set).

    Parameters
    ----------
    loss_list_epoch : list[float]
        Training loss per epoch.
    acc_list_epoch : list[float]
        Training accuracy per epoch.
    run_dir : pathlib.Path, optional
        Output directory for the PDF when running via the shared runtime. If None,
        the plot is saved under the local results folder.

    Returns
    -------
    None
    """
    loss_values = [
        loss_i if isinstance(loss_i, (int, float)) else loss_i.cpu().detach()
        for loss_i in loss_list_epoch
    ]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    fig.suptitle("Training Metrics", fontsize=12, fontweight="bold")

    ax_loss.plot(loss_values, lw=2)
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    ax_acc.plot(acc_list_epoch, lw=2, color="tab:green")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")

    plt.tight_layout()
    if run_dir is None:
        output_path = (
            pathlib.Path(__file__).parent.parent.resolve()
            / "results"
            / "training_metrics_graph.pdf"
        )
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
    else:
        plt.savefig(
            run_dir / "training_metrics_graph.pdf", format="pdf", bbox_inches="tight"
        )


def plot_bond_exp(
    bond_dimensions: List[int],
    epochs: List[int],
    loss_list_epoch: List[float],
    acc_list_epoch: List[float],
    run_dir: Path = None,
):
    """
    Plot the training loss and accuracy over epochs for different bond dimensions.

    This function creates a two-panel plot showing the evolution of training loss
    and accuracy across epochs for various bond dimension values. The plots are
    saved as a PDF file: results/bond_dimension_graph.pdf (or to run_dir if set).

    Parameters
    -----------
    bond_dimensions : List[int]
        List of bond dimension values tested in the experiment.
    epochs : List[int]
        List of epoch numbers for the x-axis.
    loss_list_epoch : List[float]
        List of loss values for each bond dimension, where each sublist contains
        loss values per epoch.
    acc_list_epoch : List[float]
        List of accuracy values for each bond dimension, where each sublist contains
        accuracy values per epoch.
    run_dir : pathlib.Path, optional
        Output directory for the PDF when running via the shared runtime. If None,
        the plot is saved under the local results folder.

    Returns
    --------
    None
    """
    rng = np.random.default_rng(0)

    # ----------------------------
    # Plot styling
    # ----------------------------
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    cmap = plt.cm.magma
    colors = [cmap(i) for i in reversed(np.linspace(0.15, 0.95, len(bond_dimensions)))]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 3.2), sharex=True, gridspec_kw={"wspace": 0.18}
    )

    for i, (bd, col) in enumerate(zip(bond_dimensions, colors)):
        ax1.plot(epochs, loss_list_epoch[i], color=col, lw=2, alpha=0.95)
        ax2.plot(
            epochs,
            acc_list_epoch[i],
            color=col,
            lw=2,
            alpha=0.95,
            label=f"bond_dim = {bd}",
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_xlim(1, epochs[-1] + 1)
    ax1.set_ylim(0, 2.2)
    ax1.text(
        -0.12, 1.02, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Acc (%)")
    ax2.set_xlim(1, epochs[-1] + 1)
    ax2.set_ylim(20, 100)
    ax2.text(
        -0.12, 1.02, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold"
    )

    # Legend outside the right subplot (like your example)
    leg = ax2.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9
    )

    plt.tight_layout()
    if run_dir is None:
        plt.savefig(
            str(pathlib.Path(__file__).parent.parent.resolve())
            + "/results/bond_dimension_graph.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            run_dir / "bond_dimension_graph.pdf",
            format="pdf",
            bbox_inches="tight",
        )


def plot_ablation_exp(
    params_qt: List[int],
    accuracy_qt: List[float],
    params_ablation: List[int],
    accuracy_ablation: List[float],
    run_dir: Path = None,
):
    """
    Plot the ablation experiment results comparing photonic QT and the model with a lone MPS layer.

    This function creates a plot showing the testing accuracy versus the number of trainable parameters
    for both the photonic Quantum Train and the ablation study models. The plot is saved as a PDF file:
    results/ablation_graph.pdf (or to run_dir if set).


    Parameters
    -----------
    params_qt : List[int]
        List of number of trainable parameters for the photonic QT models.
    accuracy_qt : list
        List of testing accuracies for the photonic QT models.
    params_ablation : List[int]
        List of number of trainable parameters for the ablation study models.
    accuracy_ablation : List[float]
        List of testing accuracies for the ablation study models.
    run_dir : pathlib.Path, optional
        Output directory for the PDF when running via the shared runtime. If None,
        the plot is saved under the local results folder.

    Returns
    --------
    None
    """
    plt.plot(
        params_qt,
        accuracy_qt,
        linewidth=1.4,
        markersize=6,
        color="#6a1b9a",
        markerfacecolor="#6a1b9a",
        markeredgecolor="#4a148c",
        label="photonic QT",
    )
    plt.plot(
        params_ablation,
        accuracy_ablation,
        linewidth=1.4,
        markersize=6,
        color="#d65a73",
        markerfacecolor="#d65a73",
        markeredgecolor="#b23a4e",
        label="ablation study",
    )

    # Axes / styling to match the figure
    plt.xlabel("# Trainable Parameters")
    plt.ylabel("Testing Accuracy (%)")

    plt.grid(True, which="major", linestyle=":", linewidth=0.9, alpha=0.6)

    leg = plt.legend(loc="center right", frameon=True)
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()
    if run_dir is None:
        plt.savefig(
            str(pathlib.Path(__file__).parent.parent.resolve())
            + "/results/ablation_graph.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            run_dir / "ablation_graph.pdf",
            format="pdf",
            bbox_inches="tight",
        )


def plot_compression_exp(
    accuracy_ws,
    params_ws,
    accuracy_prun,
    params_prun,
    accuracy_qt,
    gen_error_qt,
    params_qt,
    classical_target_acc=(6700, 96.890),
    classical_target_gen_error=0.1690,
    run_dir: Path = None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # ---------- Panel (a): Accuracy ----------
    ax1.plot(
        params_qt,
        accuracy_qt,
        marker="s",
        label="photonic QT",
    )

    ax1.plot(
        params_ws,
        accuracy_ws,
        marker="^",
        label="weight sharing",
    )

    ax1.plot(
        params_prun,
        accuracy_prun,
        marker="o",
        label="pruning",
    )

    # Classical target (single star)
    if classical_target_acc is not None:
        ax1.plot(
            classical_target_acc[0],
            classical_target_acc[1],
            marker="*",
            markersize=12,
            linestyle="None",
            label="classical target",
        )

    ax1.set_xlim(0, 7000)
    ax1.set_ylim(20, 100)
    ax1.set_xlabel("# Trainable Parameters")
    ax1.set_ylabel("Testing Accuracy (%)")
    ax1.text(
        0.02,
        0.98,
        "(a)",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )
    ax1.grid(True, linestyle=":")
    ax1.legend()

    # ---------- Panel (b): Generalization error ----------
    if classical_target_gen_error is not None:
        ax2.axhline(
            classical_target_gen_error,
            label="classical target",
        )

    ax2.plot(
        params_qt,
        gen_error_qt,
        marker="s",
        label="photonic QT",
    )

    ax2.set_ylim(0, 0.32)
    ax2.set_xlabel("# Trainable Parameters")
    ax2.set_ylabel("Gen. error")
    ax2.text(
        0.02,
        0.98,
        "(b)",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )
    ax2.grid(True, linestyle=":")
    ax2.legend()

    plt.tight_layout()
    if run_dir is None:
        plt.savefig(
            str(pathlib.Path(__file__).parent.parent.resolve())
            + "/results/compression_exp_graph.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            run_dir / "compression_exp_graph.pdf",
            format="pdf",
            bbox_inches="tight",
        )
