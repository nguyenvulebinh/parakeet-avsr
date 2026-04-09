"""Conformer encoder for AV-ASR.

Pure-PyTorch conformer encoder adapted for this AVASR pipeline.
"""
from __future__ import annotations

import logging
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def avoid_float16_autocast_context():
    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return torch.amp.autocast("cuda", dtype=torch.float32)
        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            return torch.amp.autocast("cuda", dtype=torch.float32)
    else:
        return nullcontext()


def compute_stochastic_depth_drop_probs(
    num_layers, stochastic_depth_drop_prob=0.0, stochastic_depth_mode="linear", stochastic_depth_start_layer=1,
):
    if not (0 <= stochastic_depth_drop_prob < 1.0):
        raise ValueError("stochastic_depth_drop_prob has to be in [0, 1).")
    layer_drop_probs = [0.0] * stochastic_depth_start_layer
    if (L := num_layers - stochastic_depth_start_layer) > 0:
        if stochastic_depth_mode == "linear":
            layer_drop_probs += [l / L * stochastic_depth_drop_prob for l in range(1, L + 1)]
        elif stochastic_depth_mode == "uniform":
            layer_drop_probs += [stochastic_depth_drop_prob] * L
        else:
            raise ValueError(f"Unknown stochastic_depth_mode: {stochastic_depth_mode}")
    return layer_drop_probs


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for _ in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


class Swish(nn.SiLU):
    pass


@dataclass
class _StreamingCfg:
    drop_extra_pre_encoded: int = 0
    last_channel_cache_size: int = 0
    cache_drop_size: int = 0
    chunk_size: int = 0
    shift_size: int = 0
    valid_out_len: int = 0
    pre_encode_cache_size: int = 0


# ---------------------------------------------------------------------------
# CausalConv1D  (used by ConformerConvolution's depthwise_conv)
# ---------------------------------------------------------------------------

class CausalConv1D(nn.Conv1d):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0,
        dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None,
    ):
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif isinstance(padding, list) and len(padding) == 2 and padding[0] + padding[1] == kernel_size - 1:
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")
        self._max_cache_len = self._left_padding
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype,
        )

    def update_cache(self, x, cache=None):
        if cache is None:
            new_x = F.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = cache
        else:
            new_x = F.pad(x, pad=(0, self._right_padding))
            new_x = torch.cat([cache, new_x], dim=-1)
            if self.cache_drop_size > 0:
                next_cache = new_x[:, :, : -self.cache_drop_size]
            else:
                next_cache = new_x
            next_cache = next_cache[:, :, -cache.size(-1) :]
        return new_x, next_cache

    def forward(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        if cache is None:
            return x
        else:
            return x, cache


# ---------------------------------------------------------------------------
# ConvSubsampling helpers
# ---------------------------------------------------------------------------

def apply_channel_mask(tensor, mask):
    batch_size, channels, time, features = tensor.shape
    expanded_mask = mask.unsqueeze(1).expand(batch_size, channels, time, features)
    return tensor * expanded_mask


def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    return (input_size + padding[0] + padding[1] - kernel_size) // stride + 1


class MaskedConvSequential(nn.Sequential):
    def forward(self, x, lengths):
        x = x.unsqueeze(1)
        current_lengths = lengths.clone().float()
        mask = self._create_mask(x, current_lengths.long())
        for layer in self:
            x = apply_channel_mask(x, mask)
            x = layer(x)
            if hasattr(layer, "stride") and layer.stride != (1, 1):
                if hasattr(layer, "_left_padding"):
                    padding = (layer._left_padding, layer._right_padding)
                else:
                    padding = layer.padding
                current_lengths = calculate_conv_output_size(
                    current_lengths, layer.kernel_size[0], layer.stride[0], padding
                )
                mask = self._create_mask(x, current_lengths.long())
        x = apply_channel_mask(x, mask)
        return x, current_lengths.long()

    def _create_mask(self, tensor, lengths):
        batch_size, channels, time, features = tensor.shape
        time_mask = torch.arange(time, device=tensor.device).expand(batch_size, time) < lengths.unsqueeze(1)
        return time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)


class ConvSubsampling(nn.Module):
    def __init__(
        self, subsampling, subsampling_factor, feat_in, feat_out, conv_channels,
        subsampling_conv_chunking_factor=1, activation=nn.ReLU(True), is_causal=False,
    ):
        super().__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out
        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        if subsampling == "dw_striding":
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False
            if is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
            self._max_cache_len = 0

            if is_causal:
                layers.append(
                    torch.nn.Conv2d(
                        in_channels, conv_channels, self._kernel_size,
                        stride=self._stride, padding=0,
                    )
                )
            else:
                layers.append(
                    torch.nn.Conv2d(
                        in_channels, conv_channels, self._kernel_size,
                        stride=self._stride, padding=self._left_padding,
                    )
                )
            in_channels = conv_channels
            layers.append(activation)

            for _ in range(self._sampling_num - 1):
                if is_causal:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels, in_channels, self._kernel_size,
                            stride=self._stride, padding=0, groups=in_channels,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels, in_channels, self._kernel_size,
                            stride=self._stride, padding=self._left_padding, groups=in_channels,
                        )
                    )
                layers.append(
                    torch.nn.Conv2d(in_channels, conv_channels, 1, stride=1, padding=0, groups=1)
                )
                layers.append(activation)
                in_channels = conv_channels
        elif subsampling == "striding":
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False
            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2
            self._max_cache_len = 0

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels, conv_channels, self._kernel_size,
                        stride=self._stride, padding=self._left_padding,
                    )
                )
                layers.append(activation)
                in_channels = conv_channels
        else:
            raise ValueError(f"Unsupported subsampling: {subsampling}")

        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = calc_length(
            in_length, self._left_padding + self._right_padding,
            self._kernel_size, self._stride, self._ceil_mode, self._sampling_num,
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv2d_subsampling = True
        self.conv = MaskedConvSequential(*layers)

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, (self._kernel_size - 1) * (2 ** (self._sampling_num - 1))]

    def forward(self, x, lengths):
        out_lengths = calc_length(
            lengths, self._left_padding + self._right_padding,
            self._kernel_size, self._stride, self._ceil_mode, self._sampling_num,
        )
        x, _ = self.conv(x, lengths)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        return x, out_lengths


# ---------------------------------------------------------------------------
# Positional Encodings
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000, xscale=None, dropout_rate_emb=0.0):
        super().__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        self.dropout_emb = nn.Dropout(dropout_rate_emb) if dropout_rate_emb > 0 else None

    def create_pe(self, positions, dtype):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        if hasattr(self, "pe"):
            self.pe = pe
        else:
            self.register_buffer("pe", pe, persistent=False)

    def extend_pe(self, length, device, dtype):
        if hasattr(self, "pe") and self.pe.size(1) >= length:
            return
        positions = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x, cache_len=0):
        if self.xscale:
            x = x * self.xscale
        pe = self.pe[:, : x.size(1)]
        if self.dropout_emb:
            pe = self.dropout_emb(pe)
        x = x + pe
        return self.dropout(x)


class RelPositionalEncoding(PositionalEncoding):
    def extend_pe(self, length, device, dtype):
        needed_size = 2 * length - 1
        if hasattr(self, "pe") and self.pe.size(1) >= needed_size:
            return
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x, cache_len=0):
        if self.xscale:
            x = x * self.xscale
        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(
        self, n_head, n_feat, dropout_rate, max_cache_len=0, use_bias=True,
        use_pytorch_sdpa=False, use_pytorch_sdpa_backends=None,
    ):
        super().__init__()
        self.cache_drop_size = None
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout_rate)
        self._max_cache_len = max_cache_len

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -10000.0)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def update_cache(self, key, value, query, cache):
        if cache is not None:
            key = value = torch.cat([cache, key], dim=1)
            q_keep_size = query.shape[1] - self.cache_drop_size
            cache = torch.cat([cache[:, q_keep_size:, :], query[:, :q_keep_size, :]], dim=1)
        return key, value, query, cache

    def forward(self, query, key, value, mask, pos_emb=None, cache=None):
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
        out = self.forward_attention(v, scores, mask)
        if cache is None:
            return out
        else:
            return out, cache


class RelPositionMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self, n_head, n_feat, dropout_rate, pos_bias_u, pos_bias_v,
        max_cache_len=0, use_bias=True, use_pytorch_sdpa=False, use_pytorch_sdpa_backends=None,
    ):
        super().__init__(
            n_head=n_head, n_feat=n_feat, dropout_rate=dropout_rate,
            max_cache_len=max_cache_len, use_bias=use_bias,
            use_pytorch_sdpa=use_pytorch_sdpa, use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
        )
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        b, h, qlen, pos_len = x.size()
        x = F.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        x = x[:, :, 1:].view(b, h, qlen, pos_len)
        return x

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)

            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)

            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            matrix_bd = self.rel_shift(matrix_bd)
            matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]

            scores = (matrix_ac + matrix_bd) / self.s_d_k
            out = self.forward_attention(v, scores, mask)

        if cache is None:
            return out
        else:
            return out, cache


# ---------------------------------------------------------------------------
# Conformer Modules
# ---------------------------------------------------------------------------

class ConformerFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation=None, use_bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = activation if activation is not None else Swish()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ConformerConvolution(nn.Module):
    def __init__(
        self, d_model, kernel_size, norm_type="batch_norm",
        conv_context_size=None, pointwise_activation="glu_", use_bias=True,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.use_bias = use_bias

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2

        # 'glu_' is not in activation_registry -> stored as string
        self.pointwise_activation = pointwise_activation
        dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            d_model, d_model * 2, kernel_size=1, stride=1, padding=0, bias=use_bias,
        )
        self.depthwise_conv = CausalConv1D(
            in_channels=dw_conv_input_dim, out_channels=dw_conv_input_dim,
            kernel_size=kernel_size, stride=1, padding=conv_context_size,
            groups=dw_conv_input_dim, bias=use_bias,
        )
        if norm_type == "batch_norm":
            self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)
        elif norm_type == "layer_norm":
            self.batch_norm = nn.LayerNorm(dw_conv_input_dim)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            dw_conv_input_dim, d_model, kernel_size=1, stride=1, padding=0, bias=use_bias,
        )

    def forward(self, x, pad_mask=None, cache=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # pointwise_activation == 'glu_'

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, cache=cache)
        if cache is not None:
            x, cache = x

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        if cache is None:
            return x
        else:
            return x, cache


class ConformerLayer(nn.Module):
    def __init__(
        self, d_model, d_ff, self_attention_model="rel_pos",
        global_tokens=0, global_tokens_spacing=1, global_attn_separate=False,
        n_heads=4, conv_kernel_size=31, conv_norm_type="batch_norm",
        conv_context_size=None, dropout=0.1, dropout_att=0.1,
        pos_bias_u=None, pos_bias_v=None, att_context_size=None,
        use_bias=True, use_pytorch_sdpa=False, use_pytorch_sdpa_backends=None,
    ):
        super().__init__()
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model, kernel_size=conv_kernel_size,
            norm_type=conv_norm_type, conv_context_size=conv_context_size, use_bias=use_bias,
        )

        self.norm_self_att = nn.LayerNorm(d_model)
        if att_context_size is None:
            att_context_size = [-1, -1]
        MHA_max_cache_len = att_context_size[0]

        if self_attention_model == "rel_pos":
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u, pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len, use_bias=use_bias,
                use_pytorch_sdpa=use_pytorch_sdpa, use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
            )
        elif self_attention_model == "abs_pos":
            self.self_attn = MultiHeadAttention(
                n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att,
                max_cache_len=MHA_max_cache_len, use_bias=use_bias,
                use_pytorch_sdpa=use_pytorch_sdpa, use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
            )
        else:
            raise ValueError(f"Unsupported self_attention_model: {self_attention_model}")

        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)
        if self.self_attention_model == "rel_pos":
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb, cache=cache_last_channel)
        else:
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, cache=cache_last_channel)

        if x is not None and cache_last_channel is not None:
            (x, cache_last_channel) = x
        residual = residual + self.dropout(x)

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time)
        if cache_last_time is not None:
            (x, cache_last_time) = x
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)

        if cache_last_channel is None:
            return x
        else:
            return x, cache_last_channel, cache_last_time


# ---------------------------------------------------------------------------
# FiLM
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    def __init__(self, d_model, generator_hidden=None, dropout=0.0,
                 zero_init_delta_gamma=True, clamp_gamma=None):
        super().__init__()
        self.d_model = d_model
        self.clamp_gamma = clamp_gamma
        if generator_hidden is None:
            self.gamma_head = nn.Linear(d_model, d_model, bias=True)
            self.beta_head = nn.Linear(d_model, d_model, bias=True)
        else:
            self.gamma_head = nn.Sequential(
                nn.Linear(d_model, generator_hidden), nn.ReLU(inplace=True),
                nn.Linear(generator_hidden, d_model),
            )
            self.beta_head = nn.Sequential(
                nn.Linear(d_model, generator_hidden), nn.ReLU(inplace=True),
                nn.Linear(generator_hidden, d_model),
            )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if zero_init_delta_gamma:
            def zero_out(m):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            if isinstance(self.gamma_head, nn.Sequential):
                self.gamma_head.apply(zero_out)
            else:
                zero_out(self.gamma_head)
            if isinstance(self.beta_head, nn.Sequential):
                self.beta_head.apply(zero_out)
            else:
                zero_out(self.beta_head)

    def film_parameters(self, c):
        if c.dim() == 2:
            c = c.unsqueeze(1)
        delta_gamma = self.gamma_head(c)
        beta = self.beta_head(c)
        gamma = 1.0 + delta_gamma
        if self.clamp_gamma is not None:
            gamma = torch.clamp(gamma, self.clamp_gamma[0], self.clamp_gamma[1])
        gamma = self.dropout(gamma)
        beta = self.dropout(beta)
        return gamma, beta

    def forward(self, x, c):
        if c.dim() == 2:
            gamma, beta = self.film_parameters(c)
            if gamma.size(1) == 1 and x.size(1) > 1:
                gamma = gamma.expand(-1, x.size(1), -1)
                beta = beta.expand(-1, x.size(1), -1)
        else:
            gamma, beta = self.film_parameters(c)
        return gamma * x + beta


# ---------------------------------------------------------------------------
# Visual modules
# ---------------------------------------------------------------------------

class ResNetLikeBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(dim), nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(dim), nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(dim), nn.SiLU(),
        )
        self.residual_downsample = nn.Conv1d(dim, dim, kernel_size=1, stride=2)

    def forward(self, x):
        x_in = x.transpose(1, 2)
        residual = self.residual_downsample(x_in)
        out = self.net(x_in)
        out = (out + residual).transpose(1, 2)
        return out


class NonLinCrossFadeFusion(nn.Module):
    def __init__(self, d_model, use_film_fusion=False):
        super().__init__()
        self.d_model = d_model
        self.use_film_fusion = use_film_fusion
        self.vis_adapter = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.LayerNorm(d_model * 2),
            nn.SiLU(), nn.Dropout(0.1), nn.Linear(d_model * 2, d_model),
        )
        if self.use_film_fusion:
            self.concat_net = self._build(2 * d_model, d_model, d_model)
            self.film = FiLM(d_model)
        else:
            self.gate_net = self._build(2 * d_model, d_model // 2, d_model, add_sigmoid=True)
        self.audio_ln = nn.LayerNorm(d_model)

    def _build(self, inp, bottleneck, out, add_sigmoid=False):
        layers = [nn.Linear(inp, bottleneck), nn.SiLU(), nn.Linear(bottleneck, out)]
        if add_sigmoid:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, audio, visual):
        vis_feat = self.vis_adapter(visual)
        audio_norm = self.audio_ln(audio)
        gate_input = torch.cat([audio_norm, vis_feat], dim=-1)
        if self.use_film_fusion:
            concat_proj = self.concat_net(gate_input)
            out = self.film(audio, concat_proj)
        else:
            gate = self.gate_net(gate_input)
            out = (audio * gate) + (vis_feat * (1 - gate))
        return out


class VisualProcessingModule(nn.Module):
    def __init__(self, d_visual_embeds, d_model, visual_downsampling_factor,
                 visual_preprocessing_model="base", conditioning_embed_aggr_method="avg",
                 num_conditioning_embeds=1):
        super().__init__()
        self.d_visual_embeds = d_visual_embeds
        self.d_model = d_model
        self.visual_downsampling_factor = visual_downsampling_factor
        self.visual_preprocessing_model = visual_preprocessing_model
        self.conditioning_embed_aggr_method = conditioning_embed_aggr_method
        self.num_conditioning_embeds = num_conditioning_embeds

        self.visual_ln = nn.LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        if visual_preprocessing_model == "extra_conv":
            self.extra_ln = nn.LayerNorm(d_model)
            self.extra_conv = nn.Conv1d(d_visual_embeds, d_model, kernel_size=9, stride=1, padding=4)
        elif visual_preprocessing_model == "resnet_like":
            self.resnet_block = ResNetLikeBlock(d_visual_embeds)
        elif visual_preprocessing_model == "base":
            self.visual_conv_downsampling = nn.Conv1d(
                d_visual_embeds, d_model, kernel_size=5,
                stride=visual_downsampling_factor, padding=2,
            )

        if conditioning_embed_aggr_method in {"wavg", "softmax_wavg"}:
            self.log_weights = nn.Parameter(torch.ones(num_conditioning_embeds))

    def forward(self, visual_embeds, audio_signal=None):
        is_multispeaker = len(visual_embeds.shape) == 5
        if is_multispeaker:
            B_orig, T_orig, S_orig, C, D = visual_embeds.shape
            visual_embeds = visual_embeds.permute(0, 2, 1, 3, 4).reshape(B_orig * S_orig, T_orig, C, D)

        if self.conditioning_embed_aggr_method == "avg":
            visual_embeds = visual_embeds.mean(dim=2)
        elif self.conditioning_embed_aggr_method == "wavg":
            weights = torch.exp(self.log_weights)
            visual_embeds = (visual_embeds * weights.view(1, 1, -1, 1)).sum(dim=2) / weights.sum()
        elif self.conditioning_embed_aggr_method == "softmax_wavg":
            weights = torch.softmax(self.log_weights, dim=0)
            visual_embeds = (visual_embeds * weights.view(1, 1, -1, 1)).sum(dim=2)
        else:
            raise ValueError(f"Unknown conditioning_embed_aggr_method: {self.conditioning_embed_aggr_method}")

        B_v, T_v, D_v = visual_embeds.shape
        if self.visual_preprocessing_model == "resnet_like":
            downsampled = self.resnet_block(visual_embeds)
        else:
            downsampled = self.visual_conv_downsampling(
                visual_embeds.permute(0, 2, 1).reshape(B_v, D_v, T_v)
            ).reshape(B_v, self.d_model, -1).transpose(-1, -2)

        shape_diff = audio_signal.shape[1] - downsampled.shape[1] if audio_signal is not None else 0
        if shape_diff > 0:
            downsampled = F.pad(downsampled, (0, 0, 0, shape_diff, 0, 0))
        elif shape_diff < 0:
            downsampled = downsampled[:, : audio_signal.shape[1], :]

        if self.visual_preprocessing_model == "extra_conv":
            downsampled = self.extra_ln(downsampled)
            downsampled = self.extra_conv(
                downsampled.permute(0, 2, 1)
            ).reshape(B_v, self.d_model, -1).transpose(-1, -2)

        if is_multispeaker:
            downsampled = downsampled.reshape(B_orig, S_orig, -1, self.d_model).permute(0, 2, 1, 3)

        return self.visual_ln(self.output_linear(downsampled))


class VisualConditioningModule(nn.Module):
    def __init__(self, d_model, visual_conditioning_method, modality_dropout_prob=0.0, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.visual_conditioning_method = visual_conditioning_method
        self.modality_dropout_prob = modality_dropout_prob

        if visual_conditioning_method == "non_lin_cross_fade":
            self.fusion_module = NonLinCrossFadeFusion(d_model)
        elif visual_conditioning_method == "non_lin_cross_fade_film":
            self.fusion_module = NonLinCrossFadeFusion(d_model, use_film_fusion=True)
        elif visual_conditioning_method == "add":
            pass
        elif visual_conditioning_method == "add_project":
            self.proj = nn.Linear(d_model, d_model)
            self.proj.weight.data = torch.eye(d_model) * 0.02
        elif visual_conditioning_method == "add_gate":
            self.gate = nn.Parameter(torch.full((d_model,), -3.0))
        elif visual_conditioning_method == "film":
            self.film_layer = FiLM(d_model)
        else:
            raise ValueError(f"Unsupported visual_conditioning_method: {visual_conditioning_method}")

    def forward(self, audio_signal, visual_embeds, att_mask=None, **kwargs):
        if random.random() < self.modality_dropout_prob and self.training:
            audio_signal = 0 * audio_signal

        if self.visual_conditioning_method == "add":
            return audio_signal + visual_embeds
        elif self.visual_conditioning_method == "add_project":
            return audio_signal + self.proj(visual_embeds)
        elif self.visual_conditioning_method == "add_gate":
            alpha = torch.sigmoid(self.gate).view(1, 1, -1)
            return audio_signal + alpha * visual_embeds
        elif self.visual_conditioning_method in ("non_lin_cross_fade", "non_lin_cross_fade_film"):
            return self.fusion_module(audio_signal, visual_embeds)
        elif self.visual_conditioning_method == "film":
            return self.film_layer(audio_signal, visual_embeds)
        else:
            raise ValueError(f"Unknown method: {self.visual_conditioning_method}")


# ---------------------------------------------------------------------------
# ConformerEncoderSTNOAV  (the main encoder)
# ---------------------------------------------------------------------------

class ConformerEncoderSTNOAV(nn.Module):
    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        causal_downsampling=False,
        subsampling="striding",
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=-1,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model="rel_pos",
        n_heads=4,
        att_context_size=None,
        att_context_probs=None,
        att_context_style="regular",
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type="batch_norm",
        conv_context_size=None,
        use_bias=True,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        stochastic_depth_drop_prob=0.0,
        stochastic_depth_mode="linear",
        stochastic_depth_start_layer=1,
        global_tokens=0,
        global_tokens_spacing=1,
        global_attn_separate=False,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
        sync_max_audio_length=True,
        d_visual_embeds=1024,
        visual_downsampling_factor=2,
        visual_conditioning_method="add",
        use_pre_pe_visual_conditioning=True,
        use_visual_conditioning_on_all_layers=False,
        visual_preprocessing_model="base",
        conditioning_embed_aggr_method="avg",
        num_conditioning_embeds=1,
        modality_dropout_prob=0.0,
        use_visual_adapter_encoder=False,
        share_visual_preprocessing=False,
        use_stno=False,
        visual_conditioning_num_layers=0,
        **kwargs,
    ):
        super().__init__()
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.att_context_style = att_context_style
        self.subsampling_factor = subsampling_factor

        self.self_attention_model = self_attention_model
        self.use_pytorch_sdpa = use_pytorch_sdpa
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends or []

        # Context sizes
        (self.att_context_size_all, self.att_context_size,
         self.att_context_probs, self.conv_context_size) = self._calc_context_sizes(
            att_context_style, att_context_size, att_context_probs, conv_context_size, conv_kernel_size,
        )

        self.xscale = math.sqrt(d_model) if xscaling else None

        # Subsampling
        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            self.pre_encode = ConvSubsampling(
                subsampling=subsampling, subsampling_factor=subsampling_factor,
                feat_in=feat_in, feat_out=d_model, conv_channels=subsampling_conv_channels,
                subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
                activation=nn.ReLU(True), is_causal=causal_downsampling,
            )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        # Positional encoding
        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model, dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len, xscale=self.xscale, dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len, xscale=self.xscale,
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        # Biases
        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        # Conformer layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model, d_ff=d_ff,
                self_attention_model=self_attention_model,
                global_tokens=global_tokens, global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                n_heads=n_heads, conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type, conv_context_size=self.conv_context_size,
                dropout=dropout, dropout_att=dropout_att,
                pos_bias_u=pos_bias_u, pos_bias_v=pos_bias_v,
                att_context_size=self.att_context_size, use_bias=use_bias,
                use_pytorch_sdpa=use_pytorch_sdpa, use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
            self.layers.append(layer)

        # Out projection
        self._feat_out = d_model
        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None

        # Reduction
        self.reduction_subsampling = None
        self.reduction_position = None

        # Max audio length + streaming cfg
        self.max_audio_length = pos_emb_max_len
        self.set_max_audio_length(pos_emb_max_len)
        self.streaming_cfg = _StreamingCfg()

        # Stochastic depth
        self.layer_drop_probs = compute_stochastic_depth_drop_probs(
            len(self.layers), stochastic_depth_drop_prob, stochastic_depth_mode, stochastic_depth_start_layer,
        )

        # ---- AV-specific ----
        self.d_visual_embeds = d_visual_embeds
        self.visual_downsampling_factor = visual_downsampling_factor
        self.visual_conditioning_method = visual_conditioning_method
        self.use_pre_pe_visual_conditioning = use_pre_pe_visual_conditioning
        self.use_visual_conditioning_on_all_layers = use_visual_conditioning_on_all_layers
        self.visual_preprocessing_model = visual_preprocessing_model
        self.conditioning_embed_aggr_method = conditioning_embed_aggr_method
        self.num_conditioning_embeds = num_conditioning_embeds
        self.modality_dropout_prob = modality_dropout_prob
        self.use_visual_adapter_encoder = use_visual_adapter_encoder
        self.share_visual_preprocessing = share_visual_preprocessing
        self.use_stno = use_stno
        self.visual_conditioning_num_layers = visual_conditioning_num_layers

        # Pre-PE visual conditioning
        if use_pre_pe_visual_conditioning:
            if not share_visual_preprocessing:
                self.pre_pe_visual_processing = VisualProcessingModule(
                    d_visual_embeds, d_model, visual_downsampling_factor,
                    visual_preprocessing_model, conditioning_embed_aggr_method, num_conditioning_embeds,
                )
            self.pre_pe_visual_conditioning = VisualConditioningModule(
                d_model, visual_conditioning_method, modality_dropout_prob=modality_dropout_prob,
            )

        # Per-layer visual conditioning
        if use_visual_conditioning_on_all_layers:
            self.visual_conditioning_layer_indices = set(range(n_layers))
            if not share_visual_preprocessing:
                self.processing_modules = nn.ModuleList([
                    VisualProcessingModule(
                        d_visual_embeds, d_model, visual_downsampling_factor,
                        visual_preprocessing_model, conditioning_embed_aggr_method, num_conditioning_embeds,
                    ) for _ in range(n_layers)
                ])
            self.conditioning_modules = nn.ModuleList([
                VisualConditioningModule(d_model, visual_conditioning_method, modality_dropout_prob=modality_dropout_prob)
                for _ in range(n_layers)
            ])
        elif visual_conditioning_num_layers != 0:
            if visual_conditioning_num_layers > 0:
                self.visual_conditioning_layer_indices = set(range(min(visual_conditioning_num_layers, n_layers)))
            else:
                num_to_condition = min(abs(visual_conditioning_num_layers), n_layers)
                self.visual_conditioning_layer_indices = set(range(n_layers - num_to_condition, n_layers))
            if not share_visual_preprocessing:
                self.processing_modules = nn.ModuleList([
                    VisualProcessingModule(
                        d_visual_embeds, d_model, visual_downsampling_factor,
                        visual_preprocessing_model, conditioning_embed_aggr_method, num_conditioning_embeds,
                    ) if i in self.visual_conditioning_layer_indices else None
                    for i in range(n_layers)
                ])
            self.conditioning_modules = nn.ModuleList([
                VisualConditioningModule(d_model, visual_conditioning_method, modality_dropout_prob=modality_dropout_prob)
                if i in self.visual_conditioning_layer_indices else None
                for i in range(n_layers)
            ])
        else:
            self.visual_conditioning_layer_indices = set()

        if share_visual_preprocessing:
            self.shared_visual_processing = VisualProcessingModule(
                d_visual_embeds, d_model, visual_downsampling_factor,
                visual_preprocessing_model, conditioning_embed_aggr_method, num_conditioning_embeds,
            )

    def set_max_audio_length(self, max_audio_length):
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_audio_length, device, dtype)

    def _calc_context_sizes(self, att_context_style, att_context_size, att_context_probs,
                            conv_context_size, conv_kernel_size):
        if att_context_size:
            att_context_size_all = list(att_context_size)
            if isinstance(att_context_size_all[0], int):
                att_context_size_all = [att_context_size_all]
            for i, cs in enumerate(att_context_size_all):
                if isinstance(cs, (list, tuple)):
                    att_context_size_all[i] = list(cs)
        else:
            att_context_size_all = [[-1, -1]]

        if att_context_probs:
            att_context_probs = list(att_context_probs)
        else:
            att_context_probs = [1.0 / len(att_context_size_all)] * len(att_context_size_all)

        if conv_context_size is not None:
            if isinstance(conv_context_size, str) and conv_context_size == "causal":
                conv_context_size = [conv_kernel_size - 1, 0]
            elif isinstance(conv_context_size, list):
                pass
            else:
                conv_context_size = [(conv_kernel_size - 1) // 2, (conv_kernel_size - 1) // 2]
        else:
            conv_context_size = [(conv_kernel_size - 1) // 2, (conv_kernel_size - 1) // 2]

        return att_context_size_all, att_context_size_all[0], att_context_probs, conv_context_size

    def _create_masks(self, att_context_size, padding_length, max_audio_length, offset, device):
        att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)
        if self.att_context_style == "regular":
            if att_context_size[0] >= 0:
                att_mask = att_mask.triu(diagonal=-att_context_size[0])
            if att_context_size[1] >= 0:
                att_mask = att_mask.tril(diagonal=att_context_size[1])
        elif self.att_context_style == "chunked_limited":
            if att_context_size[1] == -1:
                if att_context_size[0] >= 0:
                    att_mask = att_mask.triu(diagonal=-att_context_size[0])
            else:
                chunk_size = att_context_size[1] + 1
                left_chunks_num = att_context_size[0] // chunk_size if att_context_size[0] >= 0 else 10000
                chunk_idx = torch.arange(0, max_audio_length, dtype=torch.int, device=device)
                chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
                diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
                chunked_mask = torch.logical_and(torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0))
                att_mask = torch.logical_and(att_mask, chunked_mask.unsqueeze(0))

        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        if offset is not None:
            pad_mask_off = torch.arange(0, max_audio_length, device=device).expand(
                padding_length.size(0), -1
            ) >= offset.unsqueeze(-1)
            pad_mask = pad_mask_off.logical_and(pad_mask)

        if att_mask is not None:
            pad_mask_for_att = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
            pad_mask_for_att = torch.logical_and(pad_mask_for_att, pad_mask_for_att.transpose(1, 2))
            att_mask = att_mask[:, :max_audio_length, :max_audio_length]
            att_mask = torch.logical_and(pad_mask_for_att, att_mask.to(pad_mask_for_att.device))
            att_mask = ~att_mask

        pad_mask = ~pad_mask
        return pad_mask, att_mask

    def forward(
        self, audio_signal, length, visual_embeds=None, visual_embed_lengths=None,
        cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None,
        bypass_pre_encode=False, num_speakers=None,
        stno_mask=None, stno_mask_length=None,
    ):
        self.update_max_seq_length(audio_signal.size(2), audio_signal.device)
        return self.forward_internal(
            audio_signal, length,
            cache_last_channel=cache_last_channel, cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len, bypass_pre_encode=bypass_pre_encode,
            stno_mask=stno_mask, stno_mask_length=stno_mask_length,
            visual_embeds=visual_embeds, visual_embed_lengths=visual_embed_lengths,
            num_speakers=num_speakers,
        )

    def update_max_seq_length(self, seq_length, device):
        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def forward_internal(
        self, audio_signal, length,
        cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None,
        bypass_pre_encode=False, stno_mask=None, stno_mask_length=None,
        visual_embeds=None, visual_embed_lengths=None, num_speakers=None,
    ):
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64,
            )

        if visual_embeds is not None and len(visual_embeds.shape) == 5:
            visual_embeds = visual_embeds.squeeze(dim=2)

        cur_att_context_size = self.att_context_size

        if not bypass_pre_encode:
            audio_signal = torch.transpose(audio_signal, 1, 2)
            if isinstance(self.pre_encode, nn.Linear):
                audio_signal = self.pre_encode(audio_signal)
            else:
                audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
                length = length.to(torch.int64)

        max_audio_length = audio_signal.size(1)
        padding_length = length
        cache_len = 0
        offset = None

        pad_mask, att_mask = self._create_masks(
            cur_att_context_size, padding_length, max_audio_length, offset, audio_signal.device,
        )

        # Pre-PE visual conditioning
        if self.use_pre_pe_visual_conditioning and visual_embeds is not None:
            if self.share_visual_preprocessing:
                downsampled_vis = self.shared_visual_processing(visual_embeds, audio_signal)
            else:
                downsampled_vis = self.pre_pe_visual_processing(visual_embeds, audio_signal)
            audio_signal = self.pre_pe_visual_conditioning(
                audio_signal=audio_signal, visual_embeds=downsampled_vis, att_mask=att_mask,
            )

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # Shared visual preprocessing
        if self.share_visual_preprocessing and visual_embeds is not None:
            downsampled_vis_shared = self.shared_visual_processing(visual_embeds, audio_signal)
        else:
            downsampled_vis_shared = None

        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = audio_signal

            if lth in self.visual_conditioning_layer_indices and visual_embeds is not None:
                if self.share_visual_preprocessing:
                    vis_for_layer = downsampled_vis_shared
                else:
                    vis_for_layer = self.processing_modules[lth](visual_embeds, audio_signal)
                audio_signal = self.conditioning_modules[lth](
                    audio_signal=audio_signal, visual_embeds=vis_for_layer, att_mask=att_mask,
                )

            audio_signal = layer(
                x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask,
            )

            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                if should_drop:
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)
        return audio_signal, length
