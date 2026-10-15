"""C64 Binary Autoencoder for image encoding and decoding.

The pretrained C64 checkpoint is not redistributed here. See README.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class BinaryQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, use_tanh=False):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        act = nn.Sigmoid
        if use_tanh:
            act = nn.Tanh
        self.proj = nn.Sequential(
            nn.Conv2d(num_hiddens, codebook_size, 1),  # projects last encoder layer to quantized logits
            act(),
            )
        self.embed = nn.Embedding(codebook_size, emb_dim)
        self.use_tanh = use_tanh

    def quantizer(self, x, deterministic=False):
        if self.use_tanh:
            x = x * 0.5 + 0.5
            if deterministic:
                x = (x > 0.5) * 1.0 
            else:
                x = torch.bernoulli(x)
            x = (x - 0.5) * 2.0
            return x
            
        else:
            if deterministic:
                x = (x > 0.5) * 1.0 
                return x
            else:
                return torch.bernoulli(x)

    def forward(self, h, deterministic=False):

        z = self.proj(h)

        code_book_loss = (torch.sigmoid(z) * (1 - torch.sigmoid(z))).mean()

        z_b = self.quantizer(z, deterministic=deterministic)

        z_flow = z_b.detach() + z - z.detach()

        z_q = torch.einsum("b n h w, n d -> b d h w", z_flow, self.embed.weight)

        return z_q,  code_book_loss, {
            "binary_code": z_b.detach()
        }, z_b.detach()


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class Encoder(nn.Module):
    def __init__(self, in_channels, nf, out_channels, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        blocks = []
        # initial convultion
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Generator(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.nf = H.nf
        self.ch_mult = H.ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = int(H.res_blocks * H.gen_mul)
        self.resolution = H.img_size
        self.attn_resolutions = H.attn_resolutions
        self.in_channels = H.emb_dim
        self.out_channels = H.n_channels
        self.norm_first = H.norm_first
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions-1)

        blocks = []
        # initial conv
        if self.norm_first:
            blocks.append(normalize(self.in_channels))
        blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)

        self.logsigma = nn.Sequential(
                            nn.Conv2d(block_in_ch, block_in_ch, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(block_in_ch, H.n_channels, kernel_size=1, stride=1, padding=0)
                        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class BinaryAutoEncoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.in_channels = H.n_channels
        self.nf = H.nf
        self.n_blocks = H.res_blocks
        self.codebook_size = H.codebook_size
        self.embed_dim = H.emb_dim
        self.ch_mult = H.ch_mult
        self.resolution = H.img_size
        self.attn_resolutions = H.attn_resolutions
        self.quantizer_type = H.quantizer
        self.beta = H.beta
        self.gumbel_num_hiddens = H.emb_dim
        self.deterministic = H.deterministic
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions
        )

        self.quantize = BinaryQuantizer(self.codebook_size, self.embed_dim, self.embed_dim, use_tanh=H.use_tanh)
        self.generator = Generator(H)
    def forward(self, x, code_only=False, code=None):
        if code is None:
            x = self.encoder(x)
            quant, codebook_loss, quant_stats, binary = self.quantize(x, deterministic=self.deterministic)
            if code_only:
                return binary
        else:
            quant = torch.einsum("b n h w, n d -> b d h w", code, self.quantize.embed.weight)
            codebook_loss, quant_stats = None, None
        x = self.generator(quant)
        return x, codebook_loss, quant_stats

