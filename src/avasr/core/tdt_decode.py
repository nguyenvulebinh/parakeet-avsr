"""TDT (Token-and-Duration Transducer) greedy decoding.

Framework-independent greedy decoding utilities for TDT outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TDTResult:
    """Result of greedy TDT decoding for a single utterance."""
    tokens: list[int] = field(default_factory=list)
    frame_timestamps: list[int] = field(default_factory=list)
    token_durations: list[int] = field(default_factory=list)


@torch.no_grad()
def greedy_tdt_decode_single(
    encoded: torch.Tensor,
    encoded_len: int,
    decoder,
    joint,
    blank_id: int,
    durations: list[int] | None = None,
    max_symbols: int = 10,
) -> TDTResult:
    """Greedy TDT decode for a single utterance.

    Args:
        encoded: (1, T, D) encoder output for one utterance
        encoded_len: valid length T
        decoder: RNNTDecoder (prediction net)
        joint: RNNTJoint
        blank_id: blank token id
        durations: list of duration values, e.g. [0,1,2,3,4]
        max_symbols: max symbols per frame

    Returns:
        TDTResult with tokens, per-token encoder frame indices, and TDT durations.
    """
    if durations is None:
        durations = [0, 1, 2, 3, 4]

    device = encoded.device
    T = min(encoded_len, encoded.shape[1])
    num_dur = len(durations)
    result = TDTResult()

    hidden = decoder.initialize_state(encoded.unsqueeze(0))  # batch of 1

    last_label = torch.full([1, 1], fill_value=blank_id, dtype=torch.long, device=device)

    time_idx = 0
    while time_idx < T:
        enc_frame = encoded[:, time_idx:time_idx + 1, :]  # (1, 1, D_enc)
        enc_proj = joint.enc(enc_frame)  # (1, 1, D_joint)

        not_blank = True
        sym_count = 0

        while not_blank and sym_count < max_symbols:
            g, hidden_new = decoder.predict(last_label, hidden, add_sos=False, batch_size=1)
            pred_proj = joint.pred(g)  # (1, 1, D_joint)

            logits = joint.joint_after_projection(
                enc_proj, pred_proj
            )  # (1, 1, 1, V + num_dur)

            logp = logits.squeeze()  # (V + num_dur,)
            token_logp = logp[:logp.shape[0] - num_dur]
            dur_logp = logp[logp.shape[0] - num_dur:]

            k = token_logp.argmax().item()
            d_k = dur_logp.argmax().item()
            skip = durations[d_k]

            if k == blank_id:
                not_blank = False
                time_idx += max(skip, 1)
            else:
                result.tokens.append(k)
                result.frame_timestamps.append(time_idx)
                result.token_durations.append(skip)
                hidden = hidden_new
                last_label = torch.tensor([[k]], dtype=torch.long, device=device)
                sym_count += 1

                if skip > 0:
                    not_blank = False
                    time_idx += skip

    return result


@torch.no_grad()
def greedy_tdt_decode_batch(
    encoded: torch.Tensor,
    encoded_len: torch.Tensor,
    decoder,
    joint,
    blank_id: int,
    durations: list[int] | None = None,
    max_symbols: int = 10,
) -> list[TDTResult]:
    """Greedy TDT decode for a batch. Processes each utterance sequentially."""
    B = encoded.shape[0]
    results = []
    for b in range(B):
        enc_b = encoded[b:b + 1]  # (1, D, T)
        enc_b = enc_b.transpose(1, 2)  # (1, T, D) -- expected by greedy loop
        T = int(encoded_len[b].item())
        result = greedy_tdt_decode_single(
            enc_b, T, decoder, joint, blank_id, durations, max_symbols,
        )
        results.append(result)
    return results
