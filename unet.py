from abc import abstractmethod
import numbers
import numpy as np
import torch as th
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import checkpoint, conv_nd, linear, avg_pool_nd, zero_module, normalization, timestep_embedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TimestepBlock(nn.Module):

    @abstractmethod
    def forward(self, x, emb):
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):

    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), conv_nd(dims, channels, self.out_channels, 3, padding=1))
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout), zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)))
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = (self.in_layers[:-1], self.in_layers[-1])
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = (self.out_layers[0], self.out_layers[1:])
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-05) * self.weight

class WithBias_LayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-05) * self.weight + self.bias

class LayerNorm(nn.Module):

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
import torch
import torch.nn as nn
import torch.nn.functional as F

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class SEModule(nn.Module):

    def __init__(self, in_dim, reduction=4, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.global_pool = nn.AdaptiveAvgPool2d(1).half()
        self.dropout = dropout
        self.fc = nn.Sequential(nn.Linear(in_dim, in_dim // reduction, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout) if dropout > 0 else nn.Identity(), nn.Linear(in_dim // reduction, in_dim, bias=False), nn.Sigmoid()).half()

    def forward(self, x):
        B, C, H, W = x.shape
        weight = self.global_pool(x).view(B, C)
        weight = self.fc(weight).view(B, self.in_dim, 1, 1)
        return x * weight + x

class _MSFFChannelFusion(nn.Module):

    def __init__(self, base_channel, bias=False, dropout=0.0):
        super().__init__()
        self.base_channel = base_channel
        self.in_channels = base_channel * 4
        self.out_high_channels = self.in_channels * 4
        self.dropout = dropout
        self.scale_attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(base_channel, base_channel // 4, 1, bias=bias), nn.ReLU(inplace=True), nn.Dropout(self.dropout) if dropout > 0 else nn.Identity(), nn.Conv2d(base_channel // 4, 1, 1, bias=bias), nn.Sigmoid())
        self.groups = 4
        assert self.in_channels % self.groups == 0 and self.out_high_channels % self.groups == 0, f'groups={self.groups} must divide both in_channels={self.in_channels} and out_channels={self.out_high_channels}'
        self.conv3x3_group_in = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_high_channels, kernel_size=3, padding=1, groups=self.groups, bias=bias)
        self.conv3x3_group_middle = nn.Conv2d(in_channels=self.out_high_channels, out_channels=self.in_channels, kernel_size=3, padding=1, groups=self.groups, bias=bias)
        self.conv3x3_group_out = nn.Conv2d(in_channels=self.in_channels, out_channels=self.base_channel, kernel_size=3, padding=1, bias=bias)
        self.se_high = SEModule(in_dim=self.out_high_channels, dropout=self.dropout)
        self.se_low = SEModule(in_dim=self.in_channels, dropout=self.dropout)

    def channel_shuffle(self, x):
        B, C, H, W = x.shape
        g = self.groups
        n = C // g
        x = x.reshape(B, g, n, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return x.reshape(B, C, H, W)

    def forward(self, x):
        concat_input = x
        x = self.conv3x3_group_in(concat_input)
        x = self.channel_shuffle(x)
        x = self.se_high(x)
        x = self.conv3x3_group_middle(x)
        x = self.se_low(x)
        x = self.conv3x3_group_out(x)
        return x

class MSFF(nn.Module):

    def __init__(self, channel, heads, bias, patch_sizes=None, LayerNorm_type='WithBias', dropout=0.0):
        super(MSFF, self).__init__()
        if patch_sizes is None:
            patch_sizes = [4, 8, 16]
        self.patch_sizes = [int(s) for s in patch_sizes.split(',')]
        self.feature_noise_std = 0.0
        self.heads = heads
        self.dropout = dropout
        self.num_patch_sizes = len(self.patch_sizes)
        self.output_channels = channel * (self.num_patch_sizes + 1)
        self.norm1 = LayerNorm(channel, LayerNorm_type)
        self.project_to_qkv = nn.Conv2d(channel, channel * 3, kernel_size=1, groups=channel, bias=bias)
        self.DeepConv = nn.Conv2d(channel * 3, channel * 3, kernel_size=3, padding=1, groups=channel, bias=bias)
        self.proj_dropout = nn.Dropout2d(self.dropout) if self.dropout > 0 else nn.Identity()
        self.freq_fft_weights = nn.ParameterList([nn.Parameter(torch.ones(49152, patch_size, patch_size // 2 + 1)) for patch_size in self.patch_sizes])
        self.MultiHeadsAttention = QKVFlashAttention(channel, num_heads=self.heads, attention_dropout=self.dropout)
        self.norm2 = LayerNorm(self.output_channels, LayerNorm_type)
        self.feature_fusion = _MSFFChannelFusion(channel, dropout=self.dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        spatial = [h, w]
        x_norm = self.norm1(x)
        if hasattr(self.project_to_qkv, 'weight'):
            x_norm = x_norm.to(self.project_to_qkv.weight.dtype)
        qkv = self.project_to_qkv(x_norm)
        qkv = self.DeepConv(qkv)
        m_att_out = self.MultiHeadsAttention(qkv.view(b, -1, np.prod(spatial)))
        m_att_out = m_att_out.reshape(b, c, h, w)
        assert m_att_out.shape == (b, c, h, w), f'm_att_out shape error: {m_att_out.shape} vs {(b, c, h, w)}'
        q, k, v = qkv.chunk(3, dim=1)
        multi_scale_outputs = []
        for idx, patch_size in enumerate(self.patch_sizes):
            if h % patch_size == 0 and w % patch_size == 0:
                q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=patch_size, patch2=patch_size)
                k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=patch_size, patch2=patch_size)
                q_fft = torch.fft.rfft2(q_patch.float())
                k_fft = torch.fft.rfft2(k_patch.float())
                b, c, h_patch, w_patch, ps1, ps2 = q_fft.shape
                total = h_patch * w_patch
                weight = self.freq_fft_weights[idx][:total]
                q_fft = q_fft.reshape(b, c, -1, ps1, ps2)
                k_fft = k_fft.reshape(b, c, -1, ps1, ps2)
                out_fft = q_fft * k_fft * weight
                out = torch.fft.irfft2(out_fft, s=(patch_size, patch_size))
                out = out.reshape(b, c, h_patch, w_patch, patch_size, patch_size)
                out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=patch_size, patch2=patch_size)
                output = v * out
                output = self.proj_dropout(output)
                if self.training and self.feature_noise_std > 0.0:
                    noise = torch.randn_like(output) * self.feature_noise_std
                    output = output + noise
                multi_scale_outputs.append(output)
        final_output = torch.cat([m_att_out, torch.cat(multi_scale_outputs, dim=1)], dim=1)
        final_output = self.norm2(final_output)
        final_output = self.feature_fusion(final_output.half())
        return final_output

class QKVFlashAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, batch_first=True, attention_dropout=0.0, causal=False, device=None, dtype=None, **kwargs) -> None:
        from einops import rearrange
        from flash_attn.flash_attention import FlashAttention
        assert batch_first
        factory_kwargs = {}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        assert self.embed_dim % num_heads == 0, 'self.kdim must be divisible by num_heads'
        self.head_dim = self.embed_dim // num_heads
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout)
        self.rearrange = rearrange

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, need_weights=False):
        qkv = self.rearrange(qkv, 'b (three h d) s -> b s three h d', three=3, h=self.num_heads)
        qkv = qkv.half()
        qkv = qkv.contiguous()
        qkv, _ = self.inner_attn(qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal)
        return self.rearrange(qkv, 'b s h d -> b (h d) s')

class QKVAttention_fft(nn.Module):

    def __init__(self, channels, num_heads, patch_size=None, bias=False, LayerNorm_type='WithBias', dropout=0.0):
        super().__init__()
        if patch_size is None:
            patch_size = [4, 8, 16]
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.patch_size = patch_size
        self.msff = MSFF(channel=channels, heads=self.num_heads, bias=bias, patch_sizes=self.patch_size, LayerNorm_type=LayerNorm_type, dropout=dropout)

    def forward(self, x):
        att_out = self.msff(x)
        output = att_out + x
        return output

class fft_AttentionBlock(nn.Module):

    def __init__(self, channels, num_heads=1, use_checkpoint=False, patch_size=None, dropout=0.0):
        super().__init__()
        if patch_size is None:
            patch_size = [4, 8, 16]
        self.channels = channels
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.use_attention_checkpoint = not self.use_checkpoint
        self.attention = QKVAttention_fft(channels, num_heads, patch_size=self.patch_size, bias=False, dropout=dropout)

    def forward(self, x, encoder_out=None):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        h = checkpoint(self.attention, (x,), (), self.use_attention_checkpoint)
        return h

def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * num_spatial ** 2 * c
    model.total_ops += th.DoubleTensor([matmul_ops])

class UNetModel(nn.Module):

    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, patch_size=[4, 8, 16], dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.cond_processor = nn.Sequential(nn.Upsample(size=(720, 1440), mode='bilinear', align_corners=False))
        self.in_channels = in_channels + 1
        self.model_channels = model_channels
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim))
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, self.in_channels, ch, 3, padding=1))])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=int(mult * model_channels), dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(fft_AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, patch_size=self.patch_size, dropout=dropout))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm), fft_AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, patch_size=self.patch_size, dropout=dropout), ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=int(model_channels * mult), dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(fft_AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample, patch_size=self.patch_size, dropout=dropout))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True) if resblock_updown else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        self.out = nn.Sequential(normalization(ch), nn.LeakyReLU(negative_slope=0.01, inplace=True), zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)))

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, cond=None):
        if cond is not None:
            cond = cond.float()
            cond_processed = self.cond_processor(cond)
        else:
            cond_processed = None
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert cond.shape == (x.shape[0],)
            emb = emb + self.label_emb(cond)
        x = th.cat([x, cond_processed], dim=1)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
