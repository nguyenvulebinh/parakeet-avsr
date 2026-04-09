"""Video loading and transform using local `src.avasr.io.VideoTransform`."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

from src.avasr.io.video_transform import VideoTransform


def load_video_frames(av_path: str | Path):
    dec = VideoDecoder(
        str(av_path), device="cpu", seek_mode="exact", num_ffmpeg_threads=1, dimension_order="NHWC"
    )
    n_frames = len(dec)
    frames_list = [dec[i].numpy() for i in range(n_frames)]
    track_frames = np.stack(frames_list, axis=0) if frames_list else np.zeros((1, 224, 224, 3), dtype=np.uint8)
    raw_frames_tensor = torch.from_numpy(track_frames.copy())
    track_frames_gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in track_frames])
    video_transform = VideoTransform(subset="test")
    lip_frames = video_transform(torch.from_numpy(track_frames_gray).unsqueeze(1)).unsqueeze(2)
    return raw_frames_tensor, lip_frames, n_frames

