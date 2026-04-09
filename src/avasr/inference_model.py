"""Top-level AV-ASR inference model wiring."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.avasr.components import load_decoder, load_encoder, load_joint, load_preprocessor, load_tokenizer
from src.avasr.core.tdt_decode import greedy_tdt_decode_batch
from src.avasr.io.audio_loader import load_audio
from src.avasr.io.video_loader import load_video_frames
from src.avasr.vision.visual_features import get_visual_feats, load_vis_feat_extractor


class AVASRModel:
    def __init__(self, weights_dir: str | Path, visual_encoder_ckpt_path: str | Path, device: str = "cuda", chunk_length: int = 20):
        self.device = device
        self.chunk_length = chunk_length
        weights_dir = Path(weights_dir)
        with open(weights_dir / "tokenizer" / "tokenizer_info.json") as f:
            self.blank_id = json.load(f)["blank_id"]
        with open(weights_dir / "model_config.json") as f:
            self.model_config = json.load(f)
        self.vis_feat_extractor = load_vis_feat_extractor(
            str(visual_encoder_ckpt_path), finetuned_weights_path=weights_dir / "vis_feat_extractor.pt", device=device
        )
        self.extract_all_layers = self.model_config.get("encoder", {}).get("use_visual_conditioning_on_all_layers", True)
        self.preprocessor = load_preprocessor(weights_dir, device=device)
        self.encoder = load_encoder(weights_dir, device=device)
        self.decoder = load_decoder(weights_dir, device=device)
        self.joint = load_joint(weights_dir, device=device)
        self.sp = load_tokenizer(weights_dir)
        self.durations = self.model_config.get("model_defaults", {}).get("tdt_durations", [0, 1, 2, 3, 4])
        self.max_symbols = self.model_config.get("decoding", {}).get("max_symbols", 10)

    @torch.no_grad()
    def transcribe(self, av_path: str | Path) -> str:
        device = self.device
        audio = load_audio(av_path).to(device)
        _, lip_frames, _ = load_video_frames(av_path)
        video_batch = lip_frames.unsqueeze(0).to(device)
        video_lengths = torch.tensor([lip_frames.shape[0]], dtype=torch.int64, device=device)
        num_speakers = torch.tensor([1], dtype=torch.int64, device=device)
        visual_embeds = get_visual_feats(
            self.vis_feat_extractor,
            video_batch,
            video_lengths,
            num_speakers=num_speakers,
            extract_all_layers=self.extract_all_layers,
            chunk_length=self.chunk_length,
        )
        audio_batch = audio.unsqueeze(0)
        audio_lengths = torch.tensor([audio.shape[0]], dtype=torch.int64, device=device)
        mel, mel_len = self.preprocessor(input_signal=audio_batch, length=audio_lengths)
        encoded, encoded_len = self.encoder(
            audio_signal=mel,
            length=mel_len,
            stno_mask=None,
            stno_mask_length=None,
            visual_embeds=visual_embeds,
            visual_embed_lengths=video_lengths,
            num_speakers=num_speakers,
        )
        token_lists = greedy_tdt_decode_batch(
            encoded, encoded_len, self.decoder, self.joint, blank_id=self.blank_id, durations=self.durations, max_symbols=self.max_symbols
        )
        return self.sp.decode(token_lists[0])

    @torch.no_grad()
    def transcribe_to_vtt(self, av_path: str | Path, fps: int = 25) -> str:
        _, _, n_frames = load_video_frames(av_path)
        text = self.transcribe(av_path)
        duration = n_frames / fps
        start = 0.5
        end = max(duration - 0.5, start + 0.1)
        return f"WEBVTT\n\n{_fmt_ts(start)} --> {_fmt_ts(end)}\n{text.strip()}\n"


def _fmt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"