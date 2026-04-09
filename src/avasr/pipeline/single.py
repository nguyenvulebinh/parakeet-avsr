from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from transformers.utils.hub import cached_file

from src.avasr.core.modeling import AVASRForTranscription
from src.avasr.io.audio_loader import AUDIO_SAMPLE_RATE, load_audio
from src.avasr.io.face_model_hub import ensure_face_weights
from src.avasr.io.video_loader import (
    LipCropMode,
    load_video_frames,
    probe_first_frame_is_lip_resolution,
    resolve_uem_time_range,
)
from src.model_bin_root import set_model_bin_root
from src.avasr.pipeline.timestamps import MS_PER_FRAME, compute_word_timestamps, format_vtt_ts
from src.avasr.vision.visual_features import get_visual_feats

FPS = 25
DEFAULT_MODEL_ID = "nguyenvulebinh/parakeet-avsr"


def device_str_for_ibug(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    if device.index is not None:
        return f"cuda:{device.index}"
    return "cuda:0"


@torch.no_grad()
def transcribe_file(
    model: AVASRForTranscription,
    tokenizer,
    av_path: str | Path,
    timestamps: bool = False,
    audio_path: Optional[str | Path] = None,
    uem_start: Optional[float] = None,
    uem_end: Optional[float] = None,
    device: torch.device = torch.device("cuda"),
    lip_crop_mode: LipCropMode = "auto",
    landmarks_detector: Any = None,
    video_process: Any = None,
) -> tuple[str, list[dict] | None]:
    uem_span = resolve_uem_time_range(av_path, uem_start, uem_end, FPS)
    if uem_span is not None:
        t0, t1 = uem_span
        _, lip_frames, _ = load_video_frames(
            av_path,
            lip_crop_mode=lip_crop_mode,
            landmarks_detector=landmarks_detector,
            video_process=video_process,
            time_start_sec=t0,
            time_end_sec=t1,
            fps=FPS,
        )
        start_audio = int(t0 * AUDIO_SAMPLE_RATE)
        end_audio = int(t1 * AUDIO_SAMPLE_RATE)
    else:
        _, lip_frames, _ = load_video_frames(
            av_path,
            lip_crop_mode=lip_crop_mode,
            landmarks_detector=landmarks_detector,
            video_process=video_process,
            fps=FPS,
        )
        start_audio = None
        end_audio = None

    sync_samples = max(1, int(round(lip_frames.shape[0] * AUDIO_SAMPLE_RATE / FPS)))
    audio = load_audio(
        audio_path or av_path,
        sample_rate=AUDIO_SAMPLE_RATE,
        fallback_num_samples=sync_samples,
    )

    if start_audio is not None and end_audio is not None:
        # Silence fallback is already segment-length; full files are longer and need UEM slice.
        if audio.shape[0] > sync_samples:
            audio = audio[start_audio:end_audio]

    audio = audio.to(device)

    audio_batch = audio.unsqueeze(0)
    audio_lengths = torch.tensor([audio.shape[0]], dtype=torch.int64, device=device)

    video_batch = lip_frames.unsqueeze(0).to(device)
    video_lengths = torch.tensor([lip_frames.shape[0]], dtype=torch.int64, device=device)
    num_speakers = torch.tensor([1], dtype=torch.int64, device=device)

    visual_embeds = get_visual_feats(
        model.vis_feat_extractor,
        video_batch,
        video_lengths,
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


def write_vtt(f, word_timestamps: list[dict], time_offset: float = 0.0):
    f.write("WEBVTT\n\n")
    for winfo in word_timestamps:
        word = winfo["word"].strip()
        if word and word != "<unk>":
            start_sec = time_offset + winfo["start_offset"] * MS_PER_FRAME
            end_sec = time_offset + winfo["end_offset"] * MS_PER_FRAME
            f.write(f"{format_vtt_ts(start_sec)} --> {format_vtt_ts(end_sec)}\n")
            f.write(f"{word}\n\n")


def main():
    script_dir = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Decode a single AV file with AVASRForTranscription (HuggingFace).",
    )
    parser.add_argument(
        "av_path",
        type=Path,
        help="Video file: lip-cropped 96×96 RGB or full-frame (lip crop runs in memory when needed).",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Separate audio source (e.g. central_video.mp4). If omitted, audio is loaded from av_path.",
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
        default=script_dir / "model-bin",
        help="Directory for Hugging Face AVSR cache and on-disk face (RetinaFace/FAN) weights",
    )
    parser.add_argument("--timestamps", action="store_true", help="Output word-level timestamps (VTT format)")
    parser.add_argument("--uem-start", type=float, default=None, help="UEM start time in seconds")
    parser.add_argument("--uem-end", type=float, default=None, help="UEM end time in seconds")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Write output to file instead of stdout")
    parser.add_argument("--device", default="cuda")
    lip_group = parser.add_mutually_exclusive_group()
    lip_group.add_argument(
        "--force-lip-crop",
        action="store_true",
        help="Always run RetinaFace + FAN lip crop in memory (ignore 96×96 detection).",
    )
    lip_group.add_argument(
        "--skip-lip-crop",
        action="store_true",
        help="Assume input is already lip-cropped; torchcodec decode only (no face models).",
    )
    args = parser.parse_args()

    if not args.av_path.exists():
        print(f"Error: file not found: {args.av_path}", file=sys.stderr)
        sys.exit(1)

    set_model_bin_root(args.cache_dir.resolve())

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.force_lip_crop:
        lip_crop_mode = "force"
    elif args.skip_lip_crop:
        lip_crop_mode = "skip"
    else:
        lip_crop_mode = "auto"

    probe_frame = max(0, int(args.uem_start * FPS)) if args.uem_start is not None else 0
    need_lip_models = lip_crop_mode == "force" or (
        lip_crop_mode == "auto"
        and not probe_first_frame_is_lip_resolution(args.av_path, frame_index=probe_frame)
    )
    model_id_path = Path(args.model_id)
    face_hub_repo = str(args.model_id) if not model_id_path.exists() else DEFAULT_MODEL_ID
    if need_lip_models:
        ensure_face_weights(face_hub_repo, cache_dir=args.cache_dir.resolve())

    landmarks_detector = None
    video_process = None
    if need_lip_models:
        from src.retinaface.detector import LandmarksDetector
        from src.retinaface.video_process import VideoProcess

        landmarks_detector = LandmarksDetector(device=device_str_for_ibug(device))
        video_process = VideoProcess(convert_gray=False)

    model_source = str(args.model_id)
    model = AVASRForTranscription.from_pretrained(model_source, cache_dir=str(args.cache_dir))
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

    text, word_ts = transcribe_file(
        model,
        tokenizer,
        args.av_path,
        timestamps=args.timestamps,
        audio_path=args.audio,
        uem_start=args.uem_start,
        uem_end=args.uem_end,
        device=device,
        lip_crop_mode=lip_crop_mode,
        landmarks_detector=landmarks_detector,
        video_process=video_process,
    )

    time_offset = args.uem_start if args.uem_start is not None else 0.0

    if args.timestamps and word_ts:
        if args.output:
            with open(args.output, "w") as f:
                write_vtt(f, word_ts, time_offset=time_offset)
            print(f"VTT written to {args.output}")
        else:
            import io

            buf = io.StringIO()
            write_vtt(buf, word_ts, time_offset=time_offset)
            print(buf.getvalue(), end="")
    else:
        if args.output:
            args.output.write_text(text)
            print(f"Text written to {args.output}")
        else:
            print(text)

