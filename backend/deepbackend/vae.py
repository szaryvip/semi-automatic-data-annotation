import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def sobel(tensor_image):
    tensor_image = tensor_image
    # Define the Sobel filter kernels
    sobel_x = torch.tensor([[1/8, 0, -1/8], [2/8, 0, -2/8], [1/8, 0, -1/8]], dtype=torch.float32)
    sobel_y = torch.tensor([[1/8, 2/8, 1/8], [0, 0, 0], [-1/8, -2/8, -1/8]], dtype=torch.float32)

    # Add a batch and channel dimension to the kernels
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).expand(1, 3, 3, 3)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).expand(1, 3, 3, 3)

    # Apply the Sobel filters to the image
    gradient_x = F.conv2d(tensor_image, sobel_x, padding=1)
    gradient_y = F.conv2d(tensor_image, sobel_y, padding=1)

    # Compute the magnitude of the gradients
    gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_magnitude

class VariationalEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims, device):  
        super(VariationalEncoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8*in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(8*in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8*in_channels, 16*in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(16*in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16*in_channels, 32*in_channels, 3, stride=2, padding=0)
        )
        self.encoder_linear = nn.Sequential(
            nn.Linear(32*3*3*in_channels, 1024),
            nn.LeakyReLU(inplace=True),
            )
        
        self.fc_mu = nn.Linear(1024, latent_dims)
        self.fc_sig = nn.Linear(1024, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        self.monotonic_iterations = 500
        self.i = 0
        self.j = 0
        
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder_conv(x)
        self.conv_shape = x.shape[2:]
        x = F.adaptive_avg_pool2d(x, 3)
        x = torch.flatten(x, start_dim=1)
        x = self.encoder_linear(x)
        mu =  self.fc_mu(x)
        sigma = torch.exp(0.5 * self.fc_sig(x))
        z = mu + sigma*self.N.sample(mu.shape)

        self.kl = (self.i/self.monotonic_iterations) * 0.5 * torch.sum(-1 - sigma + torch.exp(sigma) + mu**2)
        # self.kl = 0.5 * torch.sum(-1 - sigma + torch.exp(sigma) + mu**2)
        
        if self.i >= self.monotonic_iterations:
            self.i = self.monotonic_iterations
            self.j += 1
            if self.j >= self.monotonic_iterations:
                self.j = 0
                self.i = 0
        else:
            self.i += 1
        return z
    
    def encode(self, x):
        x = x.to(self.device)
        x = self.encoder_conv(x)
        self.conv_shape = x.shape[2:]
        x = F.adaptive_avg_pool2d(x, 3)
        x = torch.flatten(x, start_dim=1)
        x = self.encoder_linear(x)
        mu =  self.fc_mu(x)
        return mu

class Decoder(nn.Module):
    
    def __init__(self, in_channels, latent_dims, device):
        super().__init__()
        self.device = device

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 32*3*3*in_channels),
            nn.LeakyReLU(inplace=True),
            
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32*in_channels, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32*in_channels, 16*in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16*in_channels),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16*in_channels, 8*in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8*in_channels),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(8*in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x, conv_shape, image_shape):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = F.adaptive_avg_pool2d(x, conv_shape)
        x = self.decoder_conv(x)
        x = F.adaptive_avg_pool2d(x, image_shape)
        x = torch.sigmoid(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_dims, device):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(in_channels, latent_dims, device)
        self.decoder = Decoder(in_channels, latent_dims, device)
        self.device = device
        def init_conv(m):
            if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d:
                m.reset_parameters()
        self.encoder.encoder_conv.apply(init_conv)
        self.decoder.decoder_conv.apply(init_conv)

    def set_monotonic_iterations(self, iter):
        self.encoder.monotonic_iterations = iter

    def forward(self, x):
        # x = x.to("cpu")
        # x = sobel(x)
        x = x.to(self.device)
        z = self.encoder(x)
        return self.decoder(z, self.encoder.conv_shape, x.shape[2:])
