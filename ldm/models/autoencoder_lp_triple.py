import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

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

                h = h + pyramid_feat[i_level]
                hs.append(h)

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


class TripleAutoencoderLP(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 scale_factor=1,
                 ):
        super().__init__()

        self.conv_in_RGB = torch.nn.Conv2d(3, ddconfig["in_channels"], 3, 1, 1)
        self.norm_RGB = Normalize(ddconfig["in_channels"])

        self.conv_in_D = torch.nn.Conv2d(3, ddconfig["in_channels"], 3, 1, 1)
        self.norm_D = Normalize(ddconfig["in_channels"])

        self.conv_in_Sobel = torch.nn.Conv2d(3, ddconfig["in_channels"], 3, 1, 1)
        self.norm_Sobel = Normalize(ddconfig["in_channels"])

        self.encoder = Encoder(**ddconfig)
        self.decoder_RGB = Decoder(**ddconfig)
        self.decoder_D = Decoder(**ddconfig)
        self.decoder_Sobel = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor

        self.num_resolutions = len(ddconfig["ch_mult"])

        for i in range(self.num_resolutions):
            setattr(self, f'pyr_conv{i}', torch.nn.Conv2d(9, ddconfig["ch"] * ddconfig["ch_mult"][i], 3, 1, 1))
            setattr(self, f'pyr_norm{i}', Normalize(ddconfig["ch"] * ddconfig["ch_mult"][i]))

    def encode(self, image_RGB, image_D, image_Sobel):
        pyramid_RGB = self.pyramid_decom(image_RGB)
        pyramid_D = self.pyramid_decom(image_D)
        pyramid_Sobel = self.pyramid_decom(image_Sobel)
        pyramid_feat = []

        for i in range(len(pyramid_RGB)):
            temp_feat = getattr(self, f'pyr_conv{i}')(torch.cat((pyramid_RGB[i], pyramid_D[i], pyramid_Sobel[i]), dim=1))
            temp_feat = getattr(self, f'pyr_norm{i}')(temp_feat)
            temp_feat = nonlinearity(temp_feat)
            pyramid_feat.append(temp_feat)

        h_RGB = nonlinearity(self.norm_RGB(self.conv_in_RGB(image_RGB)))
        h_D = nonlinearity(self.norm_D(self.conv_in_D(image_D)))
        h_Sobel = nonlinearity(self.norm_Sobel(self.conv_in_Sobel(image_Sobel)))

        h = h_RGB + h_D + h_Sobel

        h = self.encoder(h, pyramid_feat)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.sample() * self.scale_factor

    def decode(self, z):
        z = 1. / self.scale_factor * z
        z = self.post_quant_conv(z)
        dec_RGB = self.decoder_RGB(z)
        dec_D = self.decoder_D(z)
        dec_Sobel = self.decoder_Sobel(z)

        return dec_RGB, dec_D, dec_Sobel

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