import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder, nonlinearity, Normalize
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config



class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 scale_factor=1
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor



    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.sample() * self.scale_factor

    def decode(self, z):
        z = 1. / self.scale_factor * z
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec


class DUALAutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 scale_factor=1,
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
        z = self.post_quant_conv(z) 
        dec_RGB = self.decoder_RGB(z)
        dec_TIR = self.decoder_TIR(z)
        return dec_RGB, dec_TIR