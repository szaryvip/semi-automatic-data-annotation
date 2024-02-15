import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

from typing import Literal, Dict
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
from collections import defaultdict
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from torchvision import datasets

from vae import VariationalAutoencoder
from modules import train_epoch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LATENT_DIM = 64

def _calculate_metrics(all_results_datasets: dict, true_labels_datasets: Dict[str, list], show_plot: bool, experiment: Literal["reduction", "splitting", "quantity"]) -> None:
    # labels_for_param: dict[dataset_name, dict[param_value: list for each iteration]]
    colors = {-1: "blue", 0: "green", 1:"red"}
    conts = []
    legend_labels = []
    d = -1
    for key in all_results_datasets.keys():
        labels_for_param = all_results_datasets[key]
        true_labels = true_labels_datasets[key]
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
        print(key)
        print(f"Best results in {experiment} experiment are with parameter = {list(labels_for_param.keys())[best_index]}:")
        print(f"Accuracy:\nmean: {best_acc[0]:.3f}, stddev: {best_acc[1]:.3f}, min: {best_acc[2]:.3f}, max: {best_acc[3]:.3f}")
        print(f"Precision:\nmean: {best_prec[0]:.3f}, stddev: {best_prec[1]:.3f}, min: {best_prec[2]:.3f}, max: {best_prec[3]:.3f}")
        print(f"F1 score:\nmean: {best_f1[0]:.3f}, stddev: {best_f1[1]:.3f}, min: {best_f1[2]:.3f}, max: {best_f1[3]:.3f}")
        print(f"Recall:\nmean: {best_recall[0]:.3f}, stddev: {best_recall[1]:.3f}, min: {best_recall[2]:.3f}, max: {best_recall[3]:.3f}")
        if list(labels_for_param.keys()) != ["no_parameter"] and show_plot:
            experiment_to_parameter_name = {
                "quantity": "Labeled data (%)",
                "reduction": "Encoded data dimension"
            }
            x = np.arange(len(labels_for_param.keys()))
            width = 0.30
            
            xtick_positions = []
            for i, param in enumerate(labels_for_param.keys()):
                xtick_positions.append(x[i] + i*width/2)
                cont = plt.bar(x[i] + i*width/2+d*width, accuracies[i][0], width, edgecolor='black', linewidth=1, color=colors[d])
                plt.errorbar(x[i] + i*width/2+d*width, accuracies[i][0], yerr=[[np.array(accuracies[i][0])-np.array(accuracies[i][2])], [np.array(accuracies[i][3])-np.array(accuracies[i][0])]], fmt='none', color='k', capsize=5)
                if i == 0:
                    conts.append(cont)
                    legend_labels.append(key)
        d += 1
    if list(labels_for_param.keys()) != ["no_parameter"] and show_plot:
        plt.xlabel(experiment_to_parameter_name[experiment], fontsize=18)
        plt.ylabel('Accuracy (%)', fontsize=18)
        plt.xticks(xtick_positions, labels_for_param.keys(), fontsize=14)
        plt.yticks([0,40,80,90,95], fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.gca().set_yticklabels(['{:.0f}%'.format(y) for y in plt.gca().get_yticks()])
        plt.legend(conts, legend_labels,fontsize=12, loc='lower left')
        plt.savefig(f"{experiment}_plot_{random.randint(0,1000)}.png")
        plt.show()


def _prepare_vae(dataset) -> VariationalAutoencoder:
    d = LATENT_DIM
    lr = 1e-3
    batch_size = 128
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    vae = VariationalAutoencoder(len(dataset[0][0]), latent_dims=d, device=device)
    mono_iter = len(dataset)/batch_size*4
    vae.set_monotonic_iterations(mono_iter)
    vae.to(device)
    optim = torch.optim.Adam(vae.parameters(), lr=lr)
    num_epochs = 20
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
            encoded_img  = model.encoder.encode(img)
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


def reduction_experiment(datasets: datasets.VisionDataset, show_plot: bool, iterations: int) -> None:
    all_results = {}
    all_true_labels = {}
    for name, dataset in datasets.items():
        encoded_samples, true_labels = _process_through_vae(dataset)
        all_true_labels[name] = true_labels
        encoded_samples = pd.DataFrame(encoded_samples)
        results = {}
        for param_value in range(4, 0, -1):
            all_labels = []
            for _ in range(iterations):
                if param_value < 4:
                    tsne = TSNE(n_components=param_value, perplexity=50)
                    tsne_results = tsne.fit_transform(encoded_samples)
                else:
                    tsne_results = encoded_samples

                # ms = DBSCAN(eps=3, min_samples=20).fit(tsne_results)
                # labels = ms.labels_
                # labels_unique = np.unique(labels)
                # print("Number of clusters: ",len(labels_unique))
                
                # label_indices = {}
                # for idx, label in enumerate(labels):
                #     if label not in label_indices.keys():
                #         label_indices[label] = [idx]
                #     else:
                #         label_indices[label].append(idx)
                    
                # annotated_quantity = 0.20*len(true_labels)
                # num_samples_per_label = int(annotated_quantity//len(labels_unique))

                # selected_indices = _select_indices(label_indices, num_samples_per_label)
                
                # labels = []
                # for idx, label in enumerate(true_labels):
                #     if idx in selected_indices:
                #         labels.append(label)
                #     else:
                #         labels.append(-1)
                
                annotated_percent = 0.20
                labeled_quantity = int(annotated_percent * len(dataset))
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
            results[param_value if param_value < 4 else LATENT_DIM] = all_labels
        all_results[name] = results
    print("Metrics for reduction experiment:")
    _calculate_metrics(all_results, all_true_labels, show_plot, "reduction")


def splitting_experiment(datasets: datasets.VisionDataset, show_plot: bool, iterations: int) -> None:
    # =================Clustering=======================
    all_results = {}
    all_true_labels = {}
    for name, dataset in datasets.items():
        encoded_samples, true_labels = _process_through_vae(dataset)
        all_true_labels[name] = true_labels
        encoded_samples = pd.DataFrame(encoded_samples)
        results = {}
        all_labels = []
        for _ in range(iterations):
            tsne = TSNE(n_components=2, perplexity=50)
            tsne_results = tsne.fit_transform(encoded_samples)
            
            ms = DBSCAN(eps=3, min_samples=20).fit(tsne_results)
            labels = ms.labels_
            labels_unique = np.unique(labels)
            print("Number of clusters: ",len(labels_unique))
            
            label_indices = defaultdict(list)
            for idx, label in enumerate(labels):
                label_indices[label].append(idx)
                
            annotated_quantity = (20/100)*len(true_labels)
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
        results["no_parameter"] = all_labels
        all_results[name] = results
    print("Metrics for splitting experiment - with clustering:")
    _calculate_metrics(all_results, all_true_labels, show_plot, "splitting")
    
    # =================Random Split=======================
    all_results = {}
    all_true_labels = {}
    for name, dataset in datasets.items():
        results = {}
        encoded_samples = []
        true_labels = []

        annotated_percent = 0.20
        labeled_quantity = int(annotated_percent * len(dataset))

        encoded_samples, true_labels = _process_through_vae(dataset)
        all_true_labels[name] = true_labels
        encoded_samples = pd.DataFrame(encoded_samples)
        
        results = {}
        all_labels = []
        for _ in range(iterations):
            tsne = TSNE(n_components=2, perplexity=50)
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
        results["no_parameter"] = all_labels
        all_results[name] = results
    print("Metrics for splitting experiment - with random split:")
    _calculate_metrics(all_results, all_true_labels, show_plot, "splitting")


def quantity_experiment(datasets: Dict[str, datasets.VisionDataset], show_plot: bool, iterations: int) -> None:
    all_results = {}
    all_true_labels = {}
    for name, dataset in datasets.items():
        all_results[name] = {}
        encoded_samples, true_labels = _process_through_vae(dataset)
        all_true_labels[name] = true_labels
        encoded_samples = pd.DataFrame(encoded_samples)
        for param_value in [30, 20, 10, 5, 2, 1]:
            all_labels = []
            for _ in range(iterations):
                tsne = TSNE(n_components=2, perplexity=50)
                tsne_results = tsne.fit_transform(encoded_samples)
                
                # ms = DBSCAN(eps=3, min_samples=20).fit(tsne_results)
                # labels = ms.labels_
                # labels_unique = np.unique(labels)
                # print("Number of clusters: ",len(labels_unique))
                
                # label_indices = defaultdict(list)
                # for idx, label in enumerate(labels):
                #     label_indices[label].append(idx)
                
                annotated_quantity = int((param_value/100)*len(true_labels))
                # num_samples_per_label = int(annotated_quantity//len(labels_unique))

                # selected_indices = _select_indices(label_indices, num_samples_per_label)
                
                # labels = []
                # for idx, label in enumerate(true_labels):
                #     if idx in selected_indices:
                #         labels.append(label)
                #     else:
                #         labels.append(-1)
                random_indices = random.sample(range(len(true_labels)), annotated_quantity)
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
            all_results[name][param_value] = all_labels.copy()
    print("Metrics for quantity experiment:")
    _calculate_metrics(all_results, all_true_labels, show_plot, "quantity")
