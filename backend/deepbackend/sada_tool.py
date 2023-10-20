import torch
import os
import numpy as np
import random
import pandas as pd
import shutil

from collections import defaultdict
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from vae import VariationalAutoencoder
from modules import train_epoch


class SADATool:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.selected_idx = []
        self.map_from_names = {}

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.all_data = datasets.ImageFolder("media/")
        self.all_data.transform = transform
        self.data_loader = DataLoader(self.all_data)

    def _annotate_files(self, predictions):
        key_list = list(self.map_from_names.keys())
        val_list = list(self.map_from_names.values())
        already_created_classes = os.listdir("annotated_data/")
        files = os.listdir("media/unlabeled/")
        for file, pred_class in zip(files, predictions):
            pred_class_name = key_list[val_list.index(pred_class)]
            if pred_class_name not in already_created_classes:
                os.mkdir(f"annotated_data/{pred_class_name}")
                already_created_classes.append(str(pred_class_name))
            shutil.move(f"media/unlabeled/{file}", f"annotated_data/{pred_class_name}/{file}")

    def prepare_vae(self) -> None:
        lr = 1e-3 
        self.model = VariationalAutoencoder(len(self.all_data[0][0]), 32, self.device)
        self.model.to(self.device)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        num_epochs = 10
        for epoch in range(num_epochs):
            train_loss = train_epoch(self.model, self.device, self.data_loader, optim)
            print(epoch, " epoch: ",train_loss)

    def select_to_manual_annotation(self) -> list:
        self.model.eval()
        encoded_samples = []
        for sample, _ in self.data_loader:
            sample.to(self.device)
            with torch.no_grad():
                encoded_img  = self.model.encoder(sample)
            encoded_img = encoded_img.flatten().cpu().numpy()
            encoded_sample = {f"Enc. Var. {i}": enc for i, enc in enumerate(encoded_img)}
            encoded_samples.append(encoded_sample)

        encoded_samples = pd.DataFrame(encoded_samples)
        
        tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=30, n_iter=1000)
        self.tsne_results = tsne.fit_transform(encoded_samples)

        ms = DBSCAN(eps=3, min_samples=20).fit(self.tsne_results)
        labels = ms.labels_
        print(len(np.unique(labels)))

        label_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
            
        num_samples_per_label = 10

        self.selected_idx = []
        for label in label_indices.keys():
            self.selected_idx.extend(random.sample(label_indices[label], num_samples_per_label))
        return self.selected_idx

    def annotate_data(self, answers: list):
        labels = []
        for idx, _ in enumerate(self.all_data):
            if idx in self.selected_idx:
                if answers[0] not in self.map_from_names.keys():
                    value = len(self.map_from_names.keys())
                    self.map_from_names[answers[0]] = value
                labels.append(self.map_from_names[answers[0]])
                answers.remove(answers[0])
            else:
                labels.append(-1)
        label_prop_model = LabelSpreading()
        label_prop_model.fit(self.tsne_results, labels)
        predictions = label_prop_model.predict(self.tsne_results)
        self._annotate_files(predictions)
