#!/usr/bin/env python3
"""Decode MCoRec sessions using the src HuggingFace AVASRForTranscription model.

Replicates the data-loading and preprocessing logic from inference_av_dicop.py
(metadata reading, filled-frame stitching, UEM slicing, VideoTransform, chunked
AVHubert visual feature extraction) but replaces the original EncDecRNNTBPEModelSTNOAV
with the src AVASRForTranscription loaded via ``from_pretrained``.

Usage:
    python run_session.py \\
        --model-id nguyenvulebinh/parakeet-avsr \\
        --cache-dir model-bin \\
        --session-dir /path/to/session_40 \\
        --output-dir /path/to/output \\
        --mode full \\
        --timestamps
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers.utils.hub import cached_file

try:
    from torchcodec.decoders import VideoDecoder
except ImportError:
    VideoDecoder = None

SCRIPT_DIR = Path(__file__).resolve().parent
from src.avasr.io.video_transform import VideoTransform
from src.avasr.io.audio_loader import load_audio, AUDIO_SAMPLE_RATE
from src.avasr.vision.visual_features import get_visual_feats
from src.avasr.core.modeling import AVASRForTranscription
from src.avasr.core.tdt_decode import TDTResult

FPS = 25
MS_PER_FRAME = 0.080  # 80ms per encoder frame (12.5 Hz)
DEFAULT_MODEL_ID = "nguyenvulebinh/parakeet-avsr"


# ---------------------------------------------------------------------------
# Session data helpers (ported from inference_av_dicop.py)
# ---------------------------------------------------------------------------

def load_crop_meta(session_dir: Path, track: dict) -> dict:
    meta_path = session_dir / track.get("crop_metadata", "")
    if not meta_path.exists():
        raise FileNotFoundError(f"Crop metadata not found: {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def get_frame_range_from_meta(meta: dict, fps: int = FPS) -> tuple[int, int]:
    if "frame_start" in meta and "frame_end" in meta:
        return int(meta["frame_start"]), int(meta["frame_end"])
    start = int(round(float(meta.get("start_time", 0.0)) * fps))
    end = int(round(float(meta.get("end_time", start / fps)) * fps))
    return start, end


def create_filled_frames(
    session_dir: Path,
    tracks: list[dict],
    video_field: str,
    total_frames: int,
    fps: int = FPS,
) -> Optional[np.ndarray]:
    """Stitch per-track lip crops into a full-length frame array, filling gaps with black."""
    if VideoDecoder is None:
        raise RuntimeError("torchcodec.VideoDecoder is required for frame filling")

    track_descs: list[dict] = []
    target_w: Optional[int] = None
    target_h: Optional[int] = None

    for t in tracks:
        rel = t.get(video_field) or t.get("video") or t.get("lip")
        if rel is None:
            continue
        vid_path = session_dir / rel
        if not vid_path.exists():
            continue
        meta = load_crop_meta(session_dir, t)
        start, end = get_frame_range_from_meta(meta, fps)
        dec = VideoDecoder(str(vid_path), device="cpu", seek_mode="exact",
                           num_ffmpeg_threads=1, dimension_order="NHWC")
        n_frames = len(dec)
        if n_frames > 0:
            frame0 = dec[0]
            h, w = int(frame0.shape[0]), int(frame0.shape[1])
            if target_w is None:
                target_w, target_h = w, h
            else:
                target_w, target_h = max(target_w, w), max(target_h, h)
        track_descs.append({"start": start, "end": end, "dec": dec, "n_frames": n_frames})

    if target_w is None or target_h is None:
        return None

    track_descs.sort(key=lambda x: x["start"])

    frames_list: list[np.ndarray] = []
    for fidx in range(total_frames):
        chosen = None
        for td in track_descs:
            if td["start"] <= fidx < td["end"]:
                local_idx = fidx - td["start"]
                if 0 <= local_idx < td["n_frames"]:
                    try:
                        chosen = td["dec"][local_idx].numpy()
                    except Exception:
                        chosen = None
                break
        if chosen is None:
            frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            frame = chosen
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        frames_list.append(frame)

    return np.stack(frames_list, axis=0)


def load_session_audio(session_dir: Path, video_rel_path: str) -> tuple[torch.Tensor, int]:
    """Load audio from the central video file."""
    central_path = session_dir / video_rel_path
    if not central_path.exists():
        raise FileNotFoundError(f"Central video not found: {central_path}")
    audio = load_audio(central_path, sample_rate=AUDIO_SAMPLE_RATE)
    return audio, AUDIO_SAMPLE_RATE


def count_video_frames(video_path: Path) -> int:
    if VideoDecoder is None:
        raise RuntimeError("torchcodec.VideoDecoder is required")
    dec = VideoDecoder(str(video_path), device="cpu", seek_mode="exact",
                       num_ffmpeg_threads=1, dimension_order="NHWC")
    return len(dec)


def apply_video_transform(frames_rgb: np.ndarray) -> torch.Tensor:
    """Convert RGB frames to grayscale, apply VideoTransform, add speaker dim.

    Args:
        frames_rgb: (T, H, W, 3) uint8 RGB

    Returns:
        (T, 1, 1, 88, 88) float tensor
    """
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb])
    transform = VideoTransform(subset="test")
    lip = transform(torch.from_numpy(gray).unsqueeze(1))  # (T, 1, 88, 88)
    return lip.unsqueeze(2)  # (T, 1, 1, 88, 88) — add speaker dim


# ---------------------------------------------------------------------------
# VTT formatting
# ---------------------------------------------------------------------------

def format_vtt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# Timestamp computation
# ---------------------------------------------------------------------------

def compute_word_timestamps(
    result: TDTResult,
    tokenizer,
) -> list[dict]:
    """Group token-level encoder frame indices into word-level offsets.

    Uses SentencePiece ``\\u2581`` prefix to detect word boundaries.  Each
    returned dict has ``word`` (str), ``start_offset`` and ``end_offset``
    (encoder frame indices, multiply by MS_PER_FRAME to get seconds).
    """
    if not result.tokens:
        return []

    words: list[dict] = []
    cur_ids: list[int] = []
    cur_start: int = 0
    cur_end: int = 0

    for i, tok_id in enumerate(result.tokens):
        piece = tokenizer.id_to_piece(tok_id)
        frame_ts = result.frame_timestamps[i]
        duration = result.token_durations[i]

        if piece.startswith("\u2581") and cur_ids:
            word_text = tokenizer.decode(cur_ids)
            words.append({
                "word": word_text,
                "start_offset": cur_start,
                "end_offset": cur_end,
            })
            cur_ids = [tok_id]
            cur_start = frame_ts
            cur_end = frame_ts + duration
        else:
            if not cur_ids:
                cur_start = frame_ts
            cur_ids.append(tok_id)
            cur_end = frame_ts + duration

    if cur_ids:
        word_text = tokenizer.decode(cur_ids)
        words.append({
            "word": word_text,
            "start_offset": cur_start,
            "end_offset": cur_end,
        })

    return words


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def process_session(
    model: AVASRForTranscription,
    tokenizer,
    session_dir: Path,
    output_dir: Path,
    mode: str = "full",
    timestamps: bool = False,
    device: torch.device = torch.device("cuda"),
):
    metadata_path = session_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"  [skip] metadata.json not found in {session_dir}")
        return

    with open(metadata_path) as f:
        metadata = json.load(f)

    session_name = session_dir.name
    session_out = output_dir / session_name
    session_out.mkdir(parents=True, exist_ok=True)

    for speaker_name, speaker_data in metadata.items():
        central = speaker_data.get("central", {})
        video_rel = central.get("video")
        if video_rel is None:
            print(f"  [skip] No central video for {speaker_name}")
            continue
        crops = central.get("crops", [])
        if not crops:
            print(f"  [skip] No crops for {speaker_name}")
            continue

        audio, sr = load_session_audio(session_dir, video_rel)
        total_frames = count_video_frames(session_dir / video_rel)

        if mode == "per_track":
            track_results = _process_per_track(
                model, tokenizer, session_dir, crops, audio, sr, total_frames,
                timestamps, device,
            )
            _write_vtt_per_track(session_out / f"{speaker_name}.vtt",
                                 track_results, timestamps)
        else:
            text, word_ts = _process_full(
                model, tokenizer, session_dir, speaker_data, crops,
                audio, sr, total_frames, timestamps, device,
            )
            uem = central.get("uem", {})
            uem_start = float(uem.get("start", 0.0))
            uem_end = float(uem.get("end", total_frames / FPS))
            _write_vtt_full(session_out / f"{speaker_name}.vtt",
                            text, uem_start, uem_end, word_ts)

        print(f"  {speaker_name} -> {session_out / f'{speaker_name}.vtt'}")


def _process_full(
    model, tokenizer, session_dir, speaker_data, crops,
    audio, sr, total_frames, timestamps, device,
) -> tuple[str, list[dict] | None]:
    uem = speaker_data.get("central", {}).get("uem", {})
    uem_start = float(uem.get("start", 0.0))
    uem_end = float(uem.get("end", total_frames / FPS))

    start_audio = int(uem_start * sr)
    end_audio = int(uem_end * sr)
    start_video = int(uem_start * FPS)
    end_video = int(uem_end * FPS)

    print(f"    full mode: UEM {uem_start:.1f}s\u2013{uem_end:.1f}s, "
          f"filling {total_frames} frames from {len(crops)} track(s)\u2026")

    lip_rgb = create_filled_frames(session_dir, crops, "lip", total_frames, fps=FPS)
    if lip_rgb is None:
        return "", None

    lip = apply_video_transform(lip_rgb)
    lip_sliced = lip[start_video:end_video]
    audio_sliced = audio[start_audio:end_audio]

    return _transcribe_tensors(model, tokenizer, audio_sliced, lip_sliced,
                               timestamps, device)


def _process_per_track(
    model, tokenizer, session_dir, crops, audio, sr, total_frames,
    timestamps, device,
) -> list[dict]:
    results: list[dict] = []
    for idx, track in enumerate(crops):
        meta = load_crop_meta(session_dir, track)
        t_start, t_end = get_frame_range_from_meta(meta, FPS)
        t_start_sec = t_start / FPS
        t_end_sec = t_end / FPS

        lip_rel = track.get("lip") or track.get("video")
        if lip_rel is None:
            continue
        lip_path = session_dir / lip_rel
        if not lip_path.exists():
            continue

        dec = VideoDecoder(str(lip_path), device="cpu", seek_mode="exact",
                           num_ffmpeg_threads=1, dimension_order="NHWC")
        n = len(dec)
        frames = np.stack([dec[i].numpy() for i in range(n)]) if n > 0 else np.zeros((1, 224, 224, 3), dtype=np.uint8)

        lip = apply_video_transform(frames)
        audio_sliced = audio[int(t_start_sec * sr):int(t_end_sec * sr)]

        print(f"    track {idx}: {t_start_sec:.1f}s\u2013{t_end_sec:.1f}s  ({n} frames)")
        text, word_ts = _transcribe_tensors(model, tokenizer, audio_sliced, lip,
                                            timestamps, device)
        results.append({
            "text": text,
            "word_timestamps": word_ts,
            "track_start_time": t_start_sec,
            "track_end_time": t_end_sec,
            "track_idx": idx,
        })
    return results


@torch.no_grad()
def _transcribe_tensors(
    model: AVASRForTranscription,
    tokenizer,
    audio: torch.Tensor,
    lip_frames: torch.Tensor,
    timestamps: bool,
    device: torch.device,
) -> tuple[str, list[dict] | None]:
    """Run AV-ASR on pre-processed audio waveform and lip-frame tensor.

    Args:
        audio: (samples,) float32 waveform
        lip_frames: (T, 1, 1, H, W) float lip tensor (with speaker dim)
        timestamps: if True, also compute word-level timestamps

    Returns:
        (text, word_timestamps) -- word_timestamps is None when *timestamps* is False.
    """
    audio_batch = audio.unsqueeze(0).to(device)
    audio_lengths = torch.tensor([audio.shape[0]], dtype=torch.int64, device=device)

    video_batch = lip_frames.unsqueeze(0).to(device)  # (1, T, S, C, H, W)
    video_lengths = torch.tensor([lip_frames.shape[0]], dtype=torch.int64, device=device)
    num_speakers = torch.tensor([1], dtype=torch.int64, device=device)

    visual_embeds = get_visual_feats(
        model.vis_feat_extractor,
        video_batch, video_lengths,
        num_speakers=num_speakers,
        extract_all_layers=model.config.extract_visual_features_all_layers,
        chunk_length=model.config.visual_chunk_length_sec,
    )

    encoded, encoded_len = model.forward(
        audio_signal=audio_batch,
        audio_lengths=audio_lengths,
        visual_embeds=visual_embeds,
        visual_embed_lengths=video_lengths,
        num_speakers=num_speakers,
    )

    results = model.decode(encoded, encoded_len)
    result = results[0]
    text = tokenizer.decode(result.tokens)

    word_ts = compute_word_timestamps(result, tokenizer) if timestamps else None
    return text, word_ts


# ---------------------------------------------------------------------------
# VTT writers
# ---------------------------------------------------------------------------

def _write_vtt_full(
    path: Path,
    text: str,
    uem_start: float,
    uem_end: float,
    word_timestamps: list[dict] | None = None,
):
    with open(path, "w") as f:
        f.write("WEBVTT\n\n")
        if word_timestamps:
            for winfo in word_timestamps:
                word = winfo["word"].strip()
                if word and word != "<unk>":
                    start_sec = uem_start + winfo["start_offset"] * MS_PER_FRAME
                    end_sec = uem_start + winfo["end_offset"] * MS_PER_FRAME
                    f.write(f"{format_vtt_ts(start_sec)} --> {format_vtt_ts(end_sec)}\n")
                    f.write(f"{word}\n\n")
        else:
            text = text.strip().replace("<unk>", "").strip()
            if text:
                f.write(f"{format_vtt_ts(uem_start + 0.5)} --> {format_vtt_ts(uem_end - 0.5)}\n")
                f.write(f"{text}\n\n")


def _write_vtt_per_track(
    path: Path,
    track_results: list[dict],
    timestamps: bool = False,
):
    with open(path, "w") as f:
        f.write("WEBVTT\n\n")
        for tr in sorted(track_results, key=lambda x: x["track_start_time"]):
            track_start = tr["track_start_time"]
            word_ts = tr.get("word_timestamps")
            if timestamps and word_ts:
                for winfo in word_ts:
                    word = winfo["word"].strip()
                    if word and word != "<unk>":
                        start_sec = track_start + winfo["start_offset"] * MS_PER_FRAME
                        end_sec = track_start + winfo["end_offset"] * MS_PER_FRAME
                        f.write(f"{format_vtt_ts(start_sec)} --> {format_vtt_ts(end_sec)}\n")
                        f.write(f"{word}\n\n")
            else:
                text = tr["text"].strip().replace("<unk>", "").strip()
                if text:
                    f.write(f"{format_vtt_ts(track_start + 0.5)} --> "
                            f"{format_vtt_ts(tr['track_end_time'] - 0.5)}\n")
                    f.write(f"{text}\n\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decode MCoRec sessions with src AVASRForTranscription (HuggingFace).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local model directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=SCRIPT_DIR / "model-bin",
        help="Cache directory used by Hugging Face model downloads",
    )
    parser.add_argument("--session-dir", type=str, required=True,
                        help="Path to one session directory, or a glob pattern ending with *")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "per_track"], default="full",
                        help="'full' stitches all tracks with filled frames; 'per_track' decodes each track independently")
    parser.add_argument("--timestamps", action="store_true",
                        help="Emit word-level timestamps in VTT format")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import glob as _glob

    if args.session_dir.strip().endswith("*"):
        session_dirs = sorted(Path(p) for p in _glob.glob(args.session_dir))
    else:
        session_dirs = [Path(args.session_dir)]

    if not session_dirs:
        print("No sessions found.", file=sys.stderr)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_id} \u2026")
    model = AVASRForTranscription.from_pretrained(str(args.model_id), cache_dir=str(args.cache_dir))
    model.to(device).eval()

    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor()
    model_path = Path(args.model_id)
    if model_path.exists():
        tok_path = model_path / "tokenizer.model"
    else:
        tok_path = cached_file(
            args.model_id,
            "tokenizer.model",
            cache_dir=str(args.cache_dir),
        )
        if tok_path is None:
            raise FileNotFoundError(f"tokenizer.model not found for {args.model_id}")
    tokenizer.load(str(tok_path))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sd in tqdm(session_dirs, desc="Sessions", unit="session"):
        print(f"\n=== {sd.name} ===")
        process_session(model, tokenizer, sd, args.output_dir,
                        mode=args.mode, timestamps=args.timestamps, device=device)

    print(f"\nDone. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
