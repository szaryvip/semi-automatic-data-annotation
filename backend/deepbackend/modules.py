import torch
import io
import numpy as np
import matplotlib.pyplot as plt
import base64

from torch.nn import functional as F
from PIL import Image


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
        # loss = F.mse_loss(x_hat, x) + vae.encoder.kl

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
            _ = vae.encoder(x)
            x_hat = vae(x)
            loss = ((x_hat-x)**2).sum() + vae.encoder.kl
            # loss = F.mse_loss(x_hat, x) + vae.encoder.kl
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
