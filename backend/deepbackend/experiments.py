import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

from collections import defaultdict
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from vae import VariationalAutoencoder


def reduction_experiment(dataset: datasets.VisionDataset, show_plot: bool) -> None:
    pass


def splitting_experiment(dataset: datasets.VisionDataset, show_plot: bool) -> None:
    pass


def quantity_experiment(dataset: datasets.VisionDataset, show_plot: bool) -> None:
    pass
