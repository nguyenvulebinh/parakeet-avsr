"""Video loading, optional in-memory lip crop, and ``VideoTransform`` for AVSR."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
import torchvision
from torchcodec.decoders import VideoDecoder

from src.avasr.io.video_transform import VideoTransform

LipCropMode = Literal["auto", "force", "skip"]

# Must match ``FPS`` in ``src.avasr.pipeline.single`` and lip-crop assumptions.
DEFAULT_VIDEO_FPS = 25.0


def torchcodec_num_frames(av_path: str | Path) -> int:
    dec = VideoDecoder(
        str(av_path),
        device="cpu",
        seek_mode="exact",
        num_ffmpeg_threads=1,
        dimension_order="NHWC",
    )
    return len(dec)


def resolve_uem_time_range(
    av_path: str | Path,
    uem_start: float | None,
    uem_end: float | None,
    fps: float = DEFAULT_VIDEO_FPS,
) -> tuple[float, float] | None:
    """Return ``(start_sec, end_sec)`` for the UEM window, or ``None`` if no UEM.

    ``end_sec`` matches prior frame slicing: frame indices ``int(start*fps) : int(end*fps)``.
    """
    if uem_start is None and uem_end is None:
        return None
    start_sec = uem_start if uem_start is not None else 0.0
    if uem_end is not None:
        end_sec = uem_end
    else:
        end_sec = torchcodec_num_frames(av_path) / fps
    if end_sec <= start_sec:
        raise ValueError(f"Invalid UEM: end ({end_sec}s) must be greater than start ({start_sec}s).")
    return start_sec, end_sec


def probe_first_frame_is_lip_resolution(av_path: str | Path, frame_index: int = 0) -> bool:
    """True if the given frame is 96×96 RGB (lip-cropped pipeline input)."""
    dec = VideoDecoder(
        str(av_path),
        device="cpu",
        seek_mode="exact",
        num_ffmpeg_threads=1,
        dimension_order="NHWC",
    )
    n = len(dec)
    if n == 0 or frame_index >= n:
        return False
    frame_index = max(0, frame_index)
    f0 = dec[frame_index].numpy()
    return f0.shape[0] == 96 and f0.shape[1] == 96 and f0.ndim == 3 and f0.shape[2] == 3


def _decode_torchcodec_rgb_stack(
    av_path: str | Path,
    time_start_sec: float | None = None,
    time_end_sec: float | None = None,
    fps: float = DEFAULT_VIDEO_FPS,
) -> np.ndarray:
    dec = VideoDecoder(
        str(av_path),
        device="cpu",
        seek_mode="exact",
        num_ffmpeg_threads=1,
        dimension_order="NHWC",
    )
    n_frames = len(dec)
    if n_frames == 0:
        return np.zeros((1, 224, 224, 3), dtype=np.uint8)

    if time_start_sec is None and time_end_sec is None:
        start_i, end_i = 0, n_frames
    else:
        t0 = 0.0 if time_start_sec is None else float(time_start_sec)
        t1 = float(time_end_sec) if time_end_sec is not None else n_frames / fps
        start_i = max(0, min(n_frames, int(t0 * fps)))
        end_i = max(start_i, min(n_frames, int(t1 * fps)))

    if start_i >= end_i:
        return np.zeros((1, 224, 224, 3), dtype=np.uint8)
    frames_list = [dec[i].numpy() for i in range(start_i, end_i)]
    return np.stack(frames_list, axis=0)


def rgb_uint8_nhwc_to_tensors(
    track_frames: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """RGB uint8 (T,H,W,3) -> raw tensor + lip tensor for the encoder (same as legacy path)."""
    n_frames = int(track_frames.shape[0])
    raw_frames_tensor = torch.from_numpy(track_frames.copy())
    track_frames_gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in track_frames])
    video_transform = VideoTransform(subset="test")
    lip_frames = video_transform(torch.from_numpy(track_frames_gray).unsqueeze(1)).unsqueeze(2)
    return raw_frames_tensor, lip_frames, n_frames


def _read_video_and_lip_crop(
    av_path: Path,
    landmarks_detector: Any,
    video_process: Any,
    time_start_sec: float | None = None,
    time_end_sec: float | None = None,
) -> np.ndarray:
    kwargs: dict[str, Any] = {}
    if time_start_sec is not None:
        kwargs["start_pts"] = float(time_start_sec)
    if time_end_sec is not None:
        kwargs["end_pts"] = float(time_end_sec)
    video, _, _ = torchvision.io.read_video(str(av_path), pts_unit="sec", **kwargs)
    video_np = video.numpy()
    landmarks = landmarks_detector(video_np)
    cropped = video_process(video_np, landmarks)
    if cropped is None:
        raise RuntimeError(
            "Lip crop failed: could not obtain valid face landmarks for the sequence."
        )
    return cropped


def load_video_frames(
    av_path: str | Path,
    *,
    lip_crop_mode: LipCropMode = "auto",
    landmarks_detector: Any = None,
    video_process: Any = None,
    time_start_sec: float | None = None,
    time_end_sec: float | None = None,
    fps: float = DEFAULT_VIDEO_FPS,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Load video frames and apply the same grayscale + ``VideoTransform`` as before.

    If ``time_start_sec`` / ``time_end_sec`` are set (UEM), only that time range is decoded and
    lip-processed — face models run on the segment only.

    * ``skip`` — torchcodec only (assume file is already lip-ready).
    * ``force`` — torchvision decode + RetinaFace/FAN lip crop (requires detectors).
    * ``auto`` — 96×96 probe frame → torchcodec; else full lip crop (requires detectors when not 96×96).
    """
    av_path = Path(av_path)

    bounded = time_start_sec is not None or time_end_sec is not None
    if bounded:
        t0 = 0.0 if time_start_sec is None else float(time_start_sec)
        t1 = (
            float(time_end_sec)
            if time_end_sec is not None
            else torchcodec_num_frames(av_path) / fps
        )
        probe_i = max(0, int(t0 * fps))
    else:
        t0, t1 = None, None
        probe_i = 0

    if lip_crop_mode == "skip":
        track = _decode_torchcodec_rgb_stack(av_path, t0, t1, fps)
        return rgb_uint8_nhwc_to_tensors(track)

    if lip_crop_mode == "force":
        if landmarks_detector is None or video_process is None:
            raise ValueError("lip_crop_mode='force' requires landmarks_detector and video_process")
        track = _read_video_and_lip_crop(
            av_path, landmarks_detector, video_process, time_start_sec=t0, time_end_sec=t1
        )
        return rgb_uint8_nhwc_to_tensors(track)

    # auto
    if probe_first_frame_is_lip_resolution(av_path, frame_index=probe_i):
        track = _decode_torchcodec_rgb_stack(av_path, t0, t1, fps)
        return rgb_uint8_nhwc_to_tensors(track)

    if landmarks_detector is None or video_process is None:
        raise ValueError(
            "Full-frame input requires lip crop; pass landmarks_detector and video_process, "
            "or use --force-lip-crop after ensuring face weights are available."
        )
    track = _read_video_and_lip_crop(
        av_path, landmarks_detector, video_process, time_start_sec=t0, time_end_sec=t1
    )
    return rgb_uint8_nhwc_to_tensors(track)
