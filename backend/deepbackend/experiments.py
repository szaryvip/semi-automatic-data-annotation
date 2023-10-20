import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
from collections import defaultdict
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from torchvision import datasets
from sklearn.cluster import DBSCAN

from vae import VariationalAutoencoder
from modules import train_epoch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = VariationalAutoencoder(1, 4, device)
model.load_state_dict(torch.load("model.pt"))
model.to(device)


def _calculate_metrics(labels: list, true_labels: list) -> None:
    # TODO, labels: list[list for each iteration]
    # calculate mean, stddev, min, max
    pass


def _prepare_vae(dataset) -> VariationalAutoencoder:
    d = 64
    lr = 1e-3 
    vae = VariationalAutoencoder(len(dataset[0][0]), latent_dims=d, device=device)
    vae.to(device)
    optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_epoch(vae, device, dataset, optim)
        print(epoch, " epoch: ",train_loss)
    return vae


def _process_through_vae(dataset) -> tuple:
    encoded_samples = []
    true_labels = []
    model = _prepare_vae(dataset)
    model.eval()
    for sample in tqdm(dataset):
        img = sample[0].unsqueeze(0).to(device)
        label = sample[1]
        with torch.no_grad():
            encoded_img  = model.encoder(img)
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
        encoded_samples.append(encoded_sample)
        true_labels.append(label)
    return encoded_samples, true_labels


def reduction_experiment(dataset: datasets.VisionDataset, show_plot: bool, iterations: int) -> None:
    # =================With T-SNE reduction=======================
    encoded_samples, true_labels = _process_through_vae(dataset)
    encoded_samples = pd.DataFrame(encoded_samples)
    all_labels = []
    
    for _ in range(iterations):
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
        
        ms = DBSCAN(eps=3, min_samples=20).fit(tsne_results)
        labels = ms.labels_
        labels_unique = np.unique(labels)
        print("Number of clusters: ",len(labels_unique))
        
        label_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
            
        num_samples_per_label = 10 # IMPORTANT TODO EXPERIMENT IN QUANTITY

        selected_indices = []
        for label in label_indices.keys():
            selected_indices.extend(random.sample(label_indices[label], num_samples_per_label))
        
        labels = []
        for idx, label in enumerate(true_labels):
            if idx in selected_indices:
                labels.append(label)
            else:
                labels.append(-1)
        
        label_prop_model = LabelSpreading()
        label_prop_model.fit(tsne_results, labels)
        labels = label_prop_model.predict(tsne_results)
        all_labels.append(labels)

    print("Metrics for reduction experiment - with reduction:")
    _calculate_metrics(all_labels, true_labels)
    
    # =================Without reduction=======================
    encoded_samples, true_labels = _process_through_vae(dataset)
    encoded_samples = pd.DataFrame(encoded_samples)
    all_labels = []
    
    for _ in range(iterations):
        ms = DBSCAN(eps=3, min_samples=20).fit(encoded_samples.drop(['label'],axis=1))
        labels = ms.labels_
        labels_unique = np.unique(labels)
        print("Number of clusters: ",len(labels_unique))
        
        label_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
            
        num_samples_per_label = 10 # IMPORTANT TODO EXPERIMENT IN QUANTITY

        selected_indices = []
        for label in label_indices.keys():
            selected_indices.extend(random.sample(label_indices[label], num_samples_per_label))
        
        labels = []
        for idx, label in enumerate(true_labels):
            if idx in selected_indices:
                labels.append(label)
            else:
                labels.append(-1)
        
        label_prop_model = LabelSpreading()
        label_prop_model.fit(encoded_samples.drop(['label'],axis=1), labels)
        labels = label_prop_model.predict(encoded_samples.drop(['label'],axis=1))
        all_labels.append(labels)

    print("Metrics for reduction experiment - without reduction:")
    _calculate_metrics(all_labels, true_labels)


def splitting_experiment(dataset: datasets.VisionDataset, show_plot: bool, iterations: int) -> None:
    # =================Clustering=======================
    encoded_samples, true_labels = _process_through_vae(dataset)
    encoded_samples = pd.DataFrame(encoded_samples)
    all_labels = []
    
    for _ in range(iterations):
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
        
        ms = DBSCAN(eps=3, min_samples=20).fit(tsne_results)
        labels = ms.labels_
        labels_unique = np.unique(labels)
        print("Number of clusters: ",len(labels_unique))
        
        label_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
            
        num_samples_per_label = 10 # IMPORTANT TODO EXPERIMENT IN QUANTITY

        selected_indices = []
        for label in label_indices.keys():
            selected_indices.extend(random.sample(label_indices[label], num_samples_per_label))
        
        labels = []
        for idx, label in enumerate(true_labels):
            if idx in selected_indices:
                labels.append(label)
            else:
                labels.append(-1)
        
        label_prop_model = LabelSpreading()
        label_prop_model.fit(tsne_results, labels)
        labels = label_prop_model.predict(tsne_results)
        all_labels.append(labels)

    print("Metrics for splitting experiment - with clustering:")
    _calculate_metrics(all_labels, true_labels)
    
    # =================Random Split=======================
    encoded_samples = []
    true_labels = []

    m=len(dataset)
    dataset_label, dataset_unlabel = random_split(dataset, [int(m*0.005), int(m*0.995)], torch.Generator().manual_seed(42))
    print("Len of labeled: ", len(dataset_label), " Len of unlabeled: ", len(dataset_unlabel))
    
    encoded_samples_labeled, true_labels_labeled = _process_through_vae(dataset_label)
    encoded_samples.extend(encoded_samples_labeled)
    true_labels.extend(true_labels_labeled)
    encoded_samples_unlabeled, true_labels_unlabeled = _process_through_vae(dataset_unlabel)
    encoded_samples.extend(encoded_samples_unlabeled)
    true_labels.extend(true_labels_unlabeled)
    encoded_samples = pd.DataFrame(encoded_samples)
    
    all_labels = []
    
    for _ in range(iterations):
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
        
        label_prop_model = LabelSpreading()
        label_prop_model.fit(tsne_results, encoded_samples["label"])
        labels = label_prop_model.predict(tsne_results)
        all_labels.append(labels)
    
    print("Metrics for splitting experiment - with random split:")
    _calculate_metrics(all_labels, true_labels)


def quantity_experiment(dataset: datasets.VisionDataset, show_plot: bool, iterations: int) -> None:
    encoded_samples, true_labels = _process_through_vae(dataset)
    encoded_samples = pd.DataFrame(encoded_samples)
    all_labels = []
    
    for _ in range(iterations):
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
        
        ms = DBSCAN(eps=3, min_samples=20).fit(tsne_results)
        labels = ms.labels_
        labels_unique = np.unique(labels)
        print("Number of clusters: ",len(labels_unique))
        
        label_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
            
        num_samples_per_label = 10 # IMPORTANT TODO EXPERIMENT IN QUANTITY

        selected_indices = []
        for label in label_indices.keys():
            selected_indices.extend(random.sample(label_indices[label], num_samples_per_label))
        
        labels = []
        for idx, label in enumerate(true_labels):
            if idx in selected_indices:
                labels.append(label)
            else:
                labels.append(-1)
        
        label_prop_model = LabelSpreading()
        label_prop_model.fit(tsne_results, labels)
        labels = label_prop_model.predict(tsne_results)
        all_labels.append(labels)

    print("Metrics for quantity experiment:")
    _calculate_metrics(all_labels, true_labels)
