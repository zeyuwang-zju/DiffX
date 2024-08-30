import torch
import torch.nn as nn
#import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Decoder, nonlinearity, Normalize, ResnetBlock, make_attn, Downsample
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, pyramid_feat):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

                h = h + pyramid_feat[i_level] # wzy

                hs.append(h)

            # torch.Size([batch_size, 128, 512, 512])
            # torch.Size([batch_size, 256, 256, 256])
            # torch.Size([batch_size, 512, 128, 128])
            # torch.Size([batch_size, 512, 64, 64])


            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class ForthAutoencoderLP(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 scale_factor=1,
                 ):
        super().__init__()

        self.conv_in_1 = torch.nn.Conv2d(3, ddconfig["in_channels"], 3, 1, 1)
        self.norm_1 = Normalize(ddconfig["in_channels"])

        self.conv_in_2 = torch.nn.Conv2d(3, ddconfig["in_channels"], 3, 1, 1)
        self.norm_2 = Normalize(ddconfig["in_channels"])

        self.conv_in_3 = torch.nn.Conv2d(3, ddconfig["in_channels"], 3, 1, 1)
        self.norm_3 = Normalize(ddconfig["in_channels"])

        self.conv_in_4 = torch.nn.Conv2d(3, ddconfig["in_channels"], 3, 1, 1)
        self.norm_4 = Normalize(ddconfig["in_channels"])

        self.encoder = Encoder(**ddconfig)
        self.decoder_1 = Decoder(**ddconfig)
        self.decoder_2 = Decoder(**ddconfig)
        self.decoder_3 = Decoder(**ddconfig)
        self.decoder_4 = Decoder(**ddconfig)

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor

        self.num_resolutions = len(ddconfig["ch_mult"])
        # self.codebook = Codebook(num_codebook_vectors, latent_dim=embed_dim, beta=0.25)  # wzy

        for i in range(self.num_resolutions):
            # setattr(self, f'pyr_RGB{i}', torch.nn.Conv2d(3, ddconfig["ch"] * ddconfig["ch_mult"][i], 3, 1, 1))
            # setattr(self, f'pyr_TIR{i}', torch.nn.Conv2d(3, ddconfig["ch"] * ddconfig["ch_mult"][i], 3, 1, 1))
            setattr(self, f'pyr_conv{i}', torch.nn.Conv2d(12, ddconfig["ch"] * ddconfig["ch_mult"][i], 3, 1, 1))
            setattr(self, f'pyr_norm{i}', Normalize(ddconfig["ch"] * ddconfig["ch_mult"][i]))

    def encode(self, image_1, image_2, image_3, image_4):
        pyramid_1 = self.pyramid_decom(image_1)  # wzy
        pyramid_2 = self.pyramid_decom(image_2)  # wzy
        pyramid_3 = self.pyramid_decom(image_3)  # wzy
        pyramid_4 = self.pyramid_decom(image_4)  # wzy
        pyramid_feat = []

        for i in range(len(pyramid_1)):  # wzy
            # pyramid_RGB[i] = getattr(self, f'pyr_RGB{i}')(pyramid_RGB[i])  # wzy
            # pyramid_TIR[i] = getattr(self, f'pyr_TIR{i}')(pyramid_TIR[i])  # wzy
            temp_feat = getattr(self, f'pyr_conv{i}')(torch.cat((pyramid_1[i], pyramid_2[i], pyramid_3[i], pyramid_4[i]), dim=1))  # wzy
            temp_feat = getattr(self, f'pyr_norm{i}')(temp_feat)
            temp_feat = nonlinearity(temp_feat)
            pyramid_feat.append(temp_feat)

        h_1 = nonlinearity(self.norm_1(self.conv_in_1(image_1)))
        h_2 = nonlinearity(self.norm_2(self.conv_in_2(image_2)))
        h_3 = nonlinearity(self.norm_3(self.conv_in_3(image_3)))
        h_4 = nonlinearity(self.norm_4(self.conv_in_4(image_4)))

        h = h_1 + h_2 + h_3 + h_4

        # h = self.encoder(torch.cat([image_RGB, image_TIR], dim=1))
        # h = self.encoder(h)
        h = self.encoder(h, pyramid_feat) # wzy
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        # print((posterior.sample() * self.scale_factor).shape)  # torch.Size([batch_size, 4, 64, 64])
        return posterior.sample() * self.scale_factor    # torch.Size([batch_size, 4, 64, 64])

    def decode(self, z):
        z = 1. / self.scale_factor * z
        # z_q, q_loss = self.codebook(z)  # wzy
        # q_loss = 0
        z = self.post_quant_conv(z)   # wzy
        dec_1 = self.decoder_1(z)
        dec_2 = self.decoder_2(z)
        dec_3 = self.decoder_3(z)
        dec_4 = self.decoder_4(z)

        return dec_1, dec_2, dec_3, dec_4

    def pyramid_decom(self, image):
        current = image
        pyr = []
        for _ in range(self.num_resolutions - 1):
            down = F.interpolate(current, size=(current.shape[2] // 2, current.shape[3] // 2), mode='bicubic', align_corners=True)
            up = F.interpolate(down, size=(current.shape[2], current.shape[3]), mode='bicubic', align_corners=True)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr