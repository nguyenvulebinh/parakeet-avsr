"""Visual feature extraction utilities for AVASR inference."""
from __future__ import annotations

from pathlib import Path

import torch

from src.avasr.vision.avhubert import AVHubertAVSR


def load_vis_feat_extractor(
    pretrained_path: str | Path,
    finetuned_weights_path: str | Path | None = None,
    device: str = "cpu",
) -> AVHubertAVSR:
    model = AVHubertAVSR.from_pretrained(str(pretrained_path))
    if finetuned_weights_path is not None:
        ft_sd = torch.load(str(finetuned_weights_path), weights_only=True, map_location="cpu")
        _, unexpected = model.load_state_dict(ft_sd, strict=False)
        if unexpected:
            print(f"Warning: unexpected keys in vis_feat weights: {unexpected[:5]}...")
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.avsr.encoder.feature_extractor_video.resnet.eval()
    return model


def make_pad_mask(lengths, maxlen=None):
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = len(lengths)
    if maxlen is None:
        maxlen = int(max(lengths))
    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    return seq_range_expand >= seq_length_expand


def make_non_pad_mask(lengths, maxlen=None):
    return ~make_pad_mask(lengths, maxlen)


def get_visual_feats(
    vis_feat_extractor: AVHubertAVSR,
    video_frames: torch.Tensor,
    video_lengths: torch.Tensor,
    num_speakers: torch.Tensor,
    extract_all_layers: bool = True,
    chunk_length: int = 20,
) -> torch.Tensor:
    B, T, _S = video_frames.shape[:3]
    fps = 25
    chunk_frames = max(1, int(chunk_length * fps))
    n_chunks = (T + chunk_frames - 1) // chunk_frames
    device = video_frames.device
    lengths = video_lengths.to(device)
    num_speakers_prefix = [0] + torch.cumsum(num_speakers, dim=0).cpu().tolist()
    batched_spks = torch.cat(
        [video_frames[b, :, :s, ...].transpose(0, 1) for b, s in enumerate(num_speakers)], dim=0
    )
    B_orig = B
    B = batched_spks.shape[0]
    video_frames = batched_spks
    lengths = lengths.repeat_interleave(num_speakers, dim=0)
    pad_len = n_chunks * chunk_frames
    if pad_len != T:
        pad_shape = (B, pad_len - T, *video_frames.shape[2:])
        video_frames_padded = torch.cat([video_frames, video_frames.new_zeros(pad_shape)], dim=1)
    else:
        video_frames_padded = video_frames
    _, _Tp, C_v, H, W = video_frames_padded.shape
    video_chunks = video_frames_padded.view(B, n_chunks, chunk_frames, C_v, H, W).reshape(
        B * n_chunks, chunk_frames, C_v, H, W
    )
    chunk_lengths = []
    for i in range(B):
        L = int(lengths[i].item())
        for k in range(n_chunks):
            start = k * chunk_frames
            rem = max(0, L - start)
            chunk_lengths.append(min(rem, chunk_frames))
    attn_mask = make_non_pad_mask(chunk_lengths).to(device)
    video_chunks_perm = video_chunks.permute(0, 2, 1, 3, 4)
    encoder_out = vis_feat_extractor.avsr.encoder(
        input_features=None, video=video_chunks_perm, attention_mask=attn_mask
    )
    if extract_all_layers:
        hidden_states = encoder_out.hidden_states
        layer_feats = []
        for hs in hidden_states:
            D = hs.shape[-1]
            hs_reshaped = hs.view(B, n_chunks * chunk_frames, D)[:, :T, :]
            layer_feats.append(hs_reshaped)
        av_feats = torch.stack(layer_feats, dim=2)
    else:
        av_feats_chunks = encoder_out.last_hidden_state
        D = av_feats_chunks.shape[-1]
        av_feats = av_feats_chunks.view(B, n_chunks * chunk_frames, D)[:, :T, :].unsqueeze(2)
    max_speakers = int(num_speakers.max().item())
    per_batch = []
    for i in range(len(num_speakers)):
        s_i = int(num_speakers[i].item())
        cur = av_feats[num_speakers_prefix[i] : num_speakers_prefix[i + 1]]
        if max_speakers - s_i > 0:
            pad = av_feats.new_zeros((max_speakers - s_i, T, av_feats.shape[2], av_feats.shape[3]))
            cur = torch.cat([cur, pad], dim=0)
        per_batch.append(cur)
    return torch.stack(per_batch, dim=0).permute(0, 2, 1, 3, 4)

