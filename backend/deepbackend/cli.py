import click
import torch
import sys
import torchvision
import warnings

from torch.utils.data import Subset
from experiments import reduction_experiment, splitting_experiment, quantity_experiment

warnings.filterwarnings("ignore", category=RuntimeWarning)

@click.command()
@click.option(
    "-d",
    "--data",
    required=True,
    type=click.Choice(
        ["MNIST", "CIFAR10", "CIFAR100"]
    ),
    help="Dataset for experiments",
)
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
    data: str,
    show_plots: bool,
    experiment: str,
    iterations: int,
) -> None:
    if "--help" in sys.argv:
        return
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    dataset = None
    match data:
        case "MNIST":
            dataset = torchvision.datasets.MNIST(root="datasets", train=True, download=True, transform=transform)
        case "CIFAR10":
            dataset = torchvision.datasets.CIFAR10(root="datasets", train=True, download=True, transform=transform)
        case "CIFAR100":
            dataset = torchvision.datasets.CIFAR100(root="datasets", train=True, download=True, transform=transform)
    indices = torch.arange(10000)
    dataset = Subset(dataset, indices)
    match experiment:
        case "reduction":
            reduction_experiment(dataset, show_plots, iterations)
        case "splitting":
            splitting_experiment(dataset, show_plots, iterations)
        case "quantity":
            quantity_experiment(dataset, show_plots, iterations)


if __name__ == "__main__":
    cli()
