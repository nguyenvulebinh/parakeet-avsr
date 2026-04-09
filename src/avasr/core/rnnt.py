"""RNNT decoder, joint, and supporting modules.

Framework-independent implementation of RNNTDecoder, RNNTJoint,
LSTMDropout, and label_collate.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def label_collate(labels, device=None):
    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor not {type(labels)}")
    batch_size = len(labels)
    max_len = max(len(label) for label in labels)
    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, : len(l)] = l
    labels = torch.tensor(cat_labels, dtype=torch.int64, device=device)
    return labels


class LSTMDropout(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: Optional[float] = 0.0,
        forget_gate_bias: Optional[float] = 1.0,
        t_max: Optional[int] = None,
        weights_init_scale: float = 1.0,
        hidden_hidden_bias_scale: float = 0.0,
        proj_size: int = 0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            proj_size=proj_size,
        )
        if t_max is not None:
            for name, v in self.lstm.named_parameters():
                if "bias" in name:
                    p = getattr(self.lstm, name)
                    n = p.nelement()
                    hs = n // 4
                    p.data.fill_(0)
                    p.data[hs : 2 * hs] = torch.log(
                        nn.init.uniform_(p.data[0:hs], 1, t_max - 1)
                    )
                    p.data[0:hs] = -p.data[hs : 2 * hs]
        elif forget_gate_bias is not None:
            for name, v in self.lstm.named_parameters():
                if "bias_ih" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size].fill_(forget_gate_bias)
                if "bias_hh" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size] *= float(hidden_hidden_bias_scale)
        self.dropout = nn.Dropout(dropout) if dropout else None
        for name, v in self.named_parameters():
            if "weight" in name or "bias" in name:
                v.data *= float(weights_init_scale)

    def forward(
        self, x: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, h = self.lstm(x, h)
        if self.dropout:
            x = self.dropout(x)
        return x, h


class RNNTDecoder(nn.Module):
    """RNN-T decoder / prediction network (LSTM-based)."""

    def __init__(
        self,
        prednet: Dict[str, Any],
        vocab_size: int,
        normalization_mode: Optional[str] = None,
        random_state_sampling: bool = False,
        blank_as_pad: bool = True,
    ):
        super().__init__()
        self.pred_hidden = prednet["pred_hidden"]
        self.pred_rnn_layers = prednet["pred_rnn_layers"]
        self.blank_idx = vocab_size
        self.vocab_size = vocab_size
        self.blank_as_pad = blank_as_pad
        self.random_state_sampling = random_state_sampling

        if normalization_mode is not None:
            raise NotImplementedError(
                f"Only normalization_mode=None is supported in this implementation; got {normalization_mode}"
            )

        forget_gate_bias = prednet.get("forget_gate_bias", 1.0)
        t_max = prednet.get("t_max", None)
        weights_init_scale = prednet.get("weights_init_scale", 1.0)
        hidden_hidden_bias_scale = prednet.get("hidden_hidden_bias_scale", 0.0)
        dropout = prednet.get("dropout", 0.0)
        rnn_hidden_size = prednet.get("rnn_hidden_size", -1)

        if blank_as_pad:
            embed = nn.Embedding(vocab_size + 1, self.pred_hidden, padding_idx=self.blank_idx)
        else:
            embed = nn.Embedding(vocab_size, self.pred_hidden)

        self.prediction = nn.ModuleDict(
            {
                "embed": embed,
                "dec_rnn": LSTMDropout(
                    input_size=self.pred_hidden,
                    hidden_size=rnn_hidden_size if rnn_hidden_size > 0 else self.pred_hidden,
                    num_layers=self.pred_rnn_layers,
                    dropout=dropout,
                    forget_gate_bias=forget_gate_bias,
                    t_max=t_max,
                    weights_init_scale=weights_init_scale,
                    hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                    proj_size=self.pred_hidden if self.pred_hidden < rnn_hidden_size else 0,
                ),
            }
        )

    def forward(self, targets, target_length, states=None):
        y = label_collate(targets)
        g, states = self.predict(y, state=states, add_sos=True)
        g = g.transpose(1, 2)  # (B, D, U)
        return g, target_length, states

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[List[torch.Tensor]] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        if y is not None:
            if y.device != device:
                y = y.to(device)
            y = self.prediction["embed"](y)
        else:
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size
            y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)

        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H), device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()

        if state is None:
            if self.random_state_sampling and self.training:
                state = self.initialize_state(y)

        y = y.transpose(0, 1)  # (U+1, B, H)
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)  # (B, U+1, H)

        return g, hid

    def initialize_state(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = y.size(0)
        if self.random_state_sampling and self.training:
            state = (
                torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
            )
        else:
            state = (
                torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
            )
        return state


class RNNTJoint(nn.Module):
    """RNN-T joint network."""

    def __init__(
        self,
        jointnet: Dict[str, Any],
        num_classes: int,
        num_extra_outputs: int = 0,
        vocabulary: Optional[List] = None,
        log_softmax: Optional[bool] = None,
        preserve_memory: bool = False,
        fuse_loss_wer: bool = False,
        fused_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self._vocab_size = num_classes
        self._num_extra_outputs = num_extra_outputs
        self._num_classes = num_classes + 1 + num_extra_outputs  # +1 for blank

        self.log_softmax = log_softmax
        self.preserve_memory = preserve_memory

        self.encoder_hidden = jointnet["encoder_hidden"]
        self.pred_hidden = jointnet["pred_hidden"]
        self.joint_hidden = jointnet["joint_hidden"]
        self.activation = jointnet["activation"]
        dropout = jointnet.get("dropout", 0.0)

        self.pred, self.enc, self.joint_net = self._joint_net_modules(
            num_classes=self._num_classes,
            pred_n_hidden=self.pred_hidden,
            enc_n_hidden=self.encoder_hidden,
            joint_n_hidden=self.joint_hidden,
            activation=self.activation,
            dropout=dropout,
        )
        self.temperature = 1.0

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_outputs: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)
        if decoder_outputs is not None:
            decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)
        out = self.joint(encoder_outputs, decoder_outputs)
        return out

    def joint(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Full joint: project + combine + output logits."""
        f = self.enc(f)
        g = self.pred(g)
        return self.joint_after_projection(f, g)

    def project_encoder(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.enc(encoder_output)

    def project_prednet(self, prednet_output: torch.Tensor) -> torch.Tensor:
        return self.pred(prednet_output)

    def joint_after_projection(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)
        g = g.unsqueeze(dim=1)  # (B, 1, U, H)
        inp = f + g  # [B, T, U, H]
        del f, g
        res = self.joint_net(inp)  # [B, T, U, V+1]
        del inp

        if self.preserve_memory:
            torch.cuda.empty_cache()

        if self.log_softmax is None:
            if not res.is_cuda:
                if self.temperature != 1.0:
                    res = (res / self.temperature).log_softmax(dim=-1)
                else:
                    res = res.log_softmax(dim=-1)
        else:
            if self.log_softmax:
                if self.temperature != 1.0:
                    res = (res / self.temperature).log_softmax(dim=-1)
                else:
                    res = res.log_softmax(dim=-1)

        return res

    def _joint_net_modules(self, num_classes, pred_n_hidden, enc_n_hidden, joint_n_hidden, activation, dropout):
        pred = nn.Linear(pred_n_hidden, joint_n_hidden)
        enc = nn.Linear(enc_n_hidden, joint_n_hidden)

        if activation not in ["relu", "sigmoid", "tanh"]:
            raise ValueError(f"Unsupported activation: {activation}")

        if activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        elif activation == "tanh":
            act = nn.Tanh()

        layers = (
            [act]
            + ([nn.Dropout(p=dropout)] if dropout else [])
            + [nn.Linear(joint_n_hidden, num_classes)]
        )
        return pred, enc, nn.Sequential(*layers)

    @property
    def num_classes_with_blank(self):
        return self._num_classes

    @property
    def num_extra_outputs(self):
        return self._num_extra_outputs
