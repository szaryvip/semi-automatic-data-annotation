import click
import torch
import sys
import torchvision
import warnings

from torch.utils.data import Subset
from experiments import reduction_experiment, splitting_experiment, quantity_experiment, _calculate_metrics

warnings.filterwarnings("ignore", category=RuntimeWarning)

@click.command()
@click.option(
    "-p",
    "--show-plots",
    required=True,
    type=bool,
    help="If plot results",
)
@click.option(
    "-e",
    "--experiment",
    required=True,
    type=click.Choice(
        ["reduction", "splitting", "quantity"]
    ),
    help="Experiment type",
)
@click.option(
    "-i",
    "--iterations",
    required=True,
    type=int,
    help="How many times experiment should run to calculate metrics",
)
def cli(
    show_plots: bool,
    experiment: str,
    iterations: int,
) -> None:
    if "--help" in sys.argv:
        return
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    datasets = {}
    mnist = torchvision.datasets.MNIST(root="../datasets", train=False, download=True, transform=transform)
    cifar10 = torchvision.datasets.CIFAR10(root="../datasets", train=False, download=True, transform=transform)
    cifar100 = torchvision.datasets.CIFAR100(root="../datasets", train=False, download=True, transform=transform)
    datasets["Mnist"] = mnist
    datasets["Cifar10"] = cifar10
    datasets["Cifar100"] = cifar100
    match experiment:
        case "reduction":
            reduction_experiment(datasets, show_plots, iterations)
        case "splitting":
            splitting_experiment(datasets, show_plots, iterations)
        case "quantity":
            quantity_experiment(datasets, show_plots, iterations)
            # labels_for_param = {"mnist": {1: [[1,1,1,1], [1,1,0,0]], 2: [[1,0,0,1], [1,1,0,0]], 3:[[1,1,1,1], [1,1,1,1]]},
            #                     "cifar10": {1: [[1,1,0,0], [0,0,0,0]], 2: [[1,1,0,1], [0,1,0,0]], 3:[[1,1,1,1], [1,1,1,1]]},
            #                     "cifar100": {1: [[1,1,0,0], [0,0,0,0]], 2: [[1,1,0,1], [0,1,0,0]], 3:[[1,1,1,1], [1,1,1,1]]}}
            # true_labels = [1,1,1,1]
            # _calculate_metrics(labels_for_param, true_labels, True, "quantity")


if __name__ == "__main__":
    cli()
