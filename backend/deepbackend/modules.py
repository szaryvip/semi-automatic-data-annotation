import torch
import io
import os
import numpy as np
import matplotlib.pyplot as plt
import base64
import random
import pandas as pd
import shutil

from collections import defaultdict
from PIL import Image
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from vae import VariationalAutoencoder


def np_image_to_base64(im_matrix) -> str:
    im_matrix = im_matrix * 255
    im_matrix = im_matrix.astype(np.uint8)
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def train_epoch(vae, device, dataloader, optimizer) -> float:
    vae.train()
    train_loss = 0.0
    for x, _ in dataloader: 
        x = x.to(device)
        x_hat = vae(x)
        loss = ((x_hat-x)**2).sum() + vae.encoder.kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)


def test_epoch(vae, device, dataloader) -> float:
    vae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            encoded_data = vae.encoder(x)
            x_hat = vae(x)

            loss = ((x_hat-x)**2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)


def plot_ae_outputs(encoder,decoder,testset,device,n=10):
    plt.figure(figsize=(16,4.5))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        img = testset[i][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img  = decoder(encoder(img), encoder.conv_shape, img.shape[2:])
        plt.imshow(img.cpu().squeeze().permute(1,2,0).numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().permute(1,2,0).numpy())  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Reconstructed images')
    plt.show()  


class SADATool:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.selected_idx = []

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.all_data = datasets.ImageFolder("../media/")
        self.all_data.transform = transform

    def _annotate_files(self, predictions):
        already_created_classes = os.listdir("../annotated_data/")
        files = os.listdir("../media/unlabeled/")
        for file, pred_class in zip(files, predictions):
            if pred_class not in already_created_classes:
                os.mkdir(f"../annotated_data/{pred_class}")
                shutil.move(f"../media/unlabeled/{file}", f"../annotated_data/{pred_class}/{file}")
                already_created_classes.append(pred_class)

    def prepare_vae(self) -> None:
        lr = 1e-3 
        self.model = VariationalAutoencoder(len(self.all_data[0][0]), 4, self.device)
        self.model.to(self.device)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        train_loader = DataLoader(self.all_data)
        num_epochs = 100
        for epoch in range(num_epochs):
            train_loss = train_epoch(self.model,self.device,train_loader,optim)
            print(epoch, " epoch: ",train_loss)

    def select_to_manual_annotation(self) -> list:
        self.model.eval()
        for sample in self.all_data:
            img = sample[0].unsqueeze(0).to(self.device)
            label = sample[1]
            with torch.no_grad():
                encoded_img  = self.model.encoder(img)
            encoded_img = encoded_img.flatten().cpu().numpy()
            encoded_sample = {f"Enc. Var. {i}": enc for i, enc in enumerate(encoded_img)}
            encoded_samples.append(encoded_sample)

        encoded_samples = pd.DataFrame(encoded_samples)
        encoded_samples = None
        
        tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=30, n_iter=1000)
        self.tsne_results = tsne.fit_transform(encoded_samples)

        ms = DBSCAN(eps=3, min_samples=20).fit(self.tsne_results)
        labels = ms.labels_

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
                labels.append(answers[0])
                answers.remove(answers[0])
            else:
                labels.append(-1)
        label_prop_model = LabelSpreading()
        label_prop_model.fit(self.tsne_results, labels)
        predictions = label_prop_model.predict(self.tsne_results)
        self._annotate_files(predictions)
