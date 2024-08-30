import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder, nonlinearity, Normalize
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config


class Codebook(nn.Module):
    def __init__(self, num_codebook_vectors, latent_dim, beta=0.25):
        super(Codebook, self).__init__()
        """
        codebook embedding: (num_codebook_vectors, latent_dim)
        beta: the factor for vq_loss
        input: (B, latent_dim, h, w)
        output: (B, latent_dim, h, w)
        """
        # self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_codebook_vectors, latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codebook_vectors, 1.0 / num_codebook_vectors)

    def forward(self, z):

        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        q_loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, q_loss


class AutoencoderVQ(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 scale_factor=1,
                 num_codebook_vectors=8192
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor

        self.codebook = Codebook(num_codebook_vectors, latent_dim=embed_dim, beta=0.25)


    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.sample() * self.scale_factor

    def decode(self, z):
        z = 1. / self.scale_factor * z
        z_q, q_loss = self.codebook(z)
        z = self.post_quant_conv(z_q)
        dec = self.decoder(z)
        return dec, q_loss


class DUALAutoencoderVQ(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 scale_factor=1,
                 num_codebook_vectors=8192
                 ):
        super().__init__()
        self.conv_in_RGB = torch.nn.Conv2d(3, ddconfig["in_channels"], 3, 1, 1)
        self.norm_RGB = Normalize(ddconfig["in_channels"])

        self.conv_in_TIR = torch.nn.Conv2d(3, ddconfig["in_channels"], 3, 1, 1)
        self.norm_TIR = Normalize(ddconfig["in_channels"])

        self.encoder = Encoder(**ddconfig)
        self.decoder_RGB = Decoder(**ddconfig)
        self.decoder_TIR = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor

        self.codebook = Codebook(num_codebook_vectors, latent_dim=embed_dim, beta=0.25)


    def encode(self, image_RGB, image_TIR):
        h_RGB = nonlinearity(self.norm_RGB(self.conv_in_RGB(image_RGB)))
        h_TIR = nonlinearity(self.norm_TIR(self.conv_in_TIR(image_TIR)))

        h = h_RGB + h_TIR

        h = self.encoder(h)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.sample() * self.scale_factor

    def decode(self, z):
        z = 1. / self.scale_factor * z
        z_q, q_loss = self.codebook(z)
        z = self.post_quant_conv(z_q)
        dec_RGB = self.decoder_RGB(z)
        dec_TIR = self.decoder_TIR(z)
        return dec_RGB, dec_TIR, q_loss