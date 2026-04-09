from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers.utils.hub import cached_file

from src.avasr.core.modeling import AVASRForTranscription
from src.avasr.io.audio_loader import AUDIO_SAMPLE_RATE, load_audio
from src.avasr.io.video_loader import load_video_frames
from src.avasr.pipeline.timestamps import MS_PER_FRAME, compute_word_timestamps, format_vtt_ts
from src.avasr.vision.visual_features import get_visual_feats

FPS = 25
DEFAULT_MODEL_ID = "nguyenvulebinh/parakeet-avsr"


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
) -> tuple[str, list[dict] | None]:
    audio = load_audio(audio_path or av_path)
    _, lip_frames, n_frames = load_video_frames(av_path)

    if uem_start is not None or uem_end is not None:
        start = uem_start if uem_start is not None else 0.0
        end = uem_end if uem_end is not None else n_frames / FPS

        start_audio = int(start * AUDIO_SAMPLE_RATE)
        end_audio = int(end * AUDIO_SAMPLE_RATE)
        start_video = int(start * FPS)
        end_video = int(end * FPS)

        audio = audio[start_audio:end_audio]
        lip_frames = lip_frames[start_video:end_video]

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
    parser.add_argument("av_path", type=Path, help="Path to the video (lip crop) .av.mp4 file")
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
        help="Cache directory used by Hugging Face model downloads",
    )
    parser.add_argument("--timestamps", action="store_true", help="Output word-level timestamps (VTT format)")
    parser.add_argument("--uem-start", type=float, default=None, help="UEM start time in seconds")
    parser.add_argument("--uem-end", type=float, default=None, help="UEM end time in seconds")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Write output to file instead of stdout")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.av_path.exists():
        print(f"Error: file not found: {args.av_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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

