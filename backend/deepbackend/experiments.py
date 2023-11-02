import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

from typing import Literal
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
LATENT_DIM = 8

def _calculate_metrics(labels_for_param: dict, true_labels: list, show_plot: bool, experiment: Literal["reduction", "splitting", "quantity"]) -> None:
    # labels_for_param: dict[param_value: list for each iteration]
    # mean, stddev, min, max in each metric
    accuracies = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    for param in labels_for_param.keys():
        accs = []
        f1s = []
        recalls = []
        precs = []
        for labels in labels_for_param[param]:
            accs.append(accuracy_score(true_labels, labels)*100)
            f1s.append(f1_score(true_labels, labels, average="macro")*100)
            recalls.append(recall_score(true_labels, labels, average="macro")*100)
            precs.append(precision_score(true_labels, labels, average="macro")*100)
        for metric, results in zip([accs, f1s, recalls, precs], [accuracies, f1_scores, recall_scores, precision_scores]):
            mean_ = np.mean(metric)
            stddev_ = np.std(metric)
            min_ = np.min(metric)
            max_ = np.max(metric)
            results.append((mean_, stddev_, min_, max_))
    best_acc = max(accuracies, key=lambda x: x[0])
    best_f1 = max(f1_scores, key=lambda x: x[0])
    best_recall = max(recall_scores, key=lambda x: x[0])
    best_prec = max(precision_scores, key=lambda x: x[0])
    best_index = accuracies.index(best_acc)
    print(f"Best results in {experiment} experiment are with parameter = {list(labels_for_param.keys())[best_index]}:")
    print(f"Accuracy:\nmean: {best_acc[0]}, stddev: {best_acc[1]}, min: {best_acc[2]}, max: {best_acc[3]}")
    print(f"Precision:\nmean: {best_prec[0]}, stddev: {best_prec[1]}, min: {best_prec[2]}, max: {best_prec[3]}")
    print(f"F1 score:\nmean: {best_f1[0]}, stddev: {best_f1[1]}, min: {best_f1[2]}, max: {best_f1[3]}")
    print(f"Recall:\nmean: {best_recall[0]}, stddev: {best_recall[1]}, min: {best_recall[2]}, max: {best_recall[3]}")
    if list(labels_for_param.keys()) != [None] and show_plot:
        x = np.arange(len(labels_for_param.keys()))
        width = 0.15
        
        for i, param in enumerate(labels_for_param.keys()):
            plt.bar(x[i] + i*width, accuracies[i][0], width, label=param)
            plt.errorbar(x[i] + i*width, accuracies[i][0], yerr=[[np.array(accuracies[i][0])-np.array(accuracies[i][2])], [np.array(accuracies[i][3])-np.array(accuracies[i][0])]], fmt='none', color='k', capsize=5)

        plt.xlabel('Parameter', fontsize=24)
        plt.ylabel('Accuracy', fontsize=24)
        plt.xticks(x + width*1.5, labels_for_param.keys(), fontsize=22)
        plt.yticks([0,20,40,80,90,95], fontsize=22)
        plt.legend(fontsize=18, loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        # Show percentages on Y-axis
        plt.gca().set_yticklabels(['{:.0f}%'.format(y) for y in plt.gca().get_yticks()])
        plt.show()


def _prepare_vae(dataset) -> VariationalAutoencoder:
    d = LATENT_DIM
    lr = 1e-3
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64)
    vae = VariationalAutoencoder(len(dataset[0][0]), latent_dims=d, device=device)
    vae.to(device)
    optim = torch.optim.Adam(vae.parameters(), lr=lr)
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_epoch(vae, device, data_loader, optim)
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


def _select_indices(label_indices: dict, num_samples_per_label: int) -> list:
    selected_indices = []
    for label in label_indices:
        if len(label_indices[label]) < num_samples_per_label:
            selected_indices.extend(label_indices[label])
        else:
            selected_indices.extend(random.sample(label_indices[label], num_samples_per_label))
    return selected_indices


def reduction_experiment(dataset: datasets.VisionDataset, show_plot: bool, iterations: int) -> None:
    # =================With T-SNE reduction=======================
    encoded_samples, true_labels = _process_through_vae(dataset)
    encoded_samples = pd.DataFrame(encoded_samples)
    results = {}
    for param_value in range(4, 0, -1):
        all_labels = []
        for _ in range(iterations):
            if param_value < 4:
                tsne = TSNE(n_components=param_value)
                tsne_results = tsne.fit_transform(encoded_samples)
            else:
                tsne_results = encoded_samples

            ms = DBSCAN(eps=1, min_samples=20).fit(tsne_results)
            labels = ms.labels_
            labels_unique = np.unique(labels)
            print("Number of clusters: ",len(labels_unique))
            
            label_indices = {}
            for idx, label in enumerate(labels):
                if label not in label_indices.keys():
                    label_indices[label] = [idx]
                else:
                    label_indices[label].append(idx)
                
            annotated_quantity = 0.05*len(true_labels)
            num_samples_per_label = int(annotated_quantity//len(labels_unique))

            selected_indices = _select_indices(label_indices, num_samples_per_label)
            
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
        results[param_value if param_value < 4 else LATENT_DIM] = all_labels

    print("Metrics for reduction experiment:")
    _calculate_metrics(results, true_labels, show_plot, "reduction")


def splitting_experiment(dataset: datasets.VisionDataset, show_plot: bool, iterations: int) -> None:
    # =================Clustering=======================
    encoded_samples, true_labels = _process_through_vae(dataset)
    encoded_samples = pd.DataFrame(encoded_samples)
    results = {}
    all_labels = []
    for _ in range(iterations):
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(encoded_samples)
        
        ms = DBSCAN(eps=1, min_samples=20).fit(tsne_results)
        labels = ms.labels_
        labels_unique = np.unique(labels)
        print("Number of clusters: ",len(labels_unique))
        
        label_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
            
        annotated_quantity = (5/100)*len(true_labels)
        num_samples_per_label = int(annotated_quantity//len(labels_unique))

        selected_indices = _select_indices(label_indices, num_samples_per_label)
        
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
    results[None] = all_labels
    print("Metrics for splitting experiment - with clustering:")
    _calculate_metrics(results, true_labels, show_plot, "splitting")
    
    # =================Random Split=======================
    encoded_samples = []
    true_labels = []

    annotated_percent = 0.05
    labeled_quantity = int(annotated_percent * len(dataset))
    print("Len of labeled: ", labeled_quantity, " Len of unlabeled: ", len(dataset)- labeled_quantity)

    encoded_samples, true_labels = _process_through_vae(dataset)
    encoded_samples = pd.DataFrame(encoded_samples)
    
    results = {}
    all_labels = []
    for _ in range(iterations):
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(encoded_samples)
        
        random_indices = random.sample(range(len(true_labels)), labeled_quantity)
        labels = []
        for index, true_label in enumerate(true_labels):
            if index in random_indices:
                labels.append(true_label)
            else:
                labels.append(-1)
        
        label_prop_model = LabelSpreading()
        label_prop_model.fit(tsne_results, labels)
        labels = label_prop_model.predict(tsne_results)
        all_labels.append(labels)
    results[None] = all_labels
    
    print("Metrics for splitting experiment - with random split:")
    _calculate_metrics(results, true_labels, show_plot, "splitting")


def quantity_experiment(dataset: datasets.VisionDataset, show_plot: bool, iterations: int) -> None:
    encoded_samples, true_labels = _process_through_vae(dataset)
    encoded_samples = pd.DataFrame(encoded_samples)
    results = {}
    for param_value in range(30, 0, -5):
        all_labels = []
        for _ in range(iterations):
            tsne = TSNE(n_components=2)
            tsne_results = tsne.fit_transform(encoded_samples)
            
            ms = DBSCAN(eps=1, min_samples=20).fit(tsne_results)
            labels = ms.labels_
            labels_unique = np.unique(labels)
            print("Number of clusters: ",len(labels_unique))
            
            label_indices = defaultdict(list)
            for idx, label in enumerate(labels):
                label_indices[label].append(idx)
            
            annotated_quantity = (param_value/100)*len(true_labels)
            num_samples_per_label = int(annotated_quantity//len(labels_unique))

            selected_indices = _select_indices(label_indices, num_samples_per_label)
            
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
        results[param_value] = all_labels
    print("Metrics for quantity experiment:")
    _calculate_metrics(results, true_labels, show_plot, "quantity")
