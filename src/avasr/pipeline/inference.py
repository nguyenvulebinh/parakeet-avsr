from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, TextIO

import torch
from transformers.utils.hub import cached_file

from src.avasr.core.modeling import AVASRForTranscription
from src.avasr.io.audio_loader import AUDIO_SAMPLE_RATE, load_audio
from src.avasr.io.face_model_hub import ensure_face_weights
from src.avasr.io.video_loader import LipCropMode, load_video_frames, resolve_uem_time_range
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


def _word_timestamps_to_sec_rows(
    word_timestamps: list[dict], time_offset: float
) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for winfo in word_timestamps:
        word = winfo["word"].strip()
        if not word or word == "<unk>":
            continue
        t0 = time_offset + winfo["start_offset"] * MS_PER_FRAME
        t1 = time_offset + winfo["end_offset"] * MS_PER_FRAME
        rows.append((word, t0, t1))
    return rows


def _merge_words_into_utterances(
    words: list[tuple[str, float, float]],
    merge_gap_sec: float,
    max_utterance_sec: float,
) -> list[tuple[str, float, float]]:
    """Group consecutive words into one cue if gap ≤ merge_gap_sec and span ≤ max_utterance_sec."""
    if not words:
        return []
    merged: list[tuple[str, float, float]] = []
    group: list[str] = [words[0][0]]
    g_start, g_end = words[0][1], words[0][2]
    for w, t0, t1 in words[1:]:
        gap = t0 - g_end
        span_if_added = t1 - g_start
        if gap > merge_gap_sec or span_if_added > max_utterance_sec:
            merged.append((" ".join(group), g_start, g_end))
            group = [w]
            g_start, g_end = t0, t1
        else:
            group.append(w)
            g_end = t1
    merged.append((" ".join(group), g_start, g_end))
    return merged


def write_vtt(
    word_timestamps: list[dict],
    f: TextIO | None = None,
    time_offset: float = 0.0,
    *,
    merge_utterance: bool = True,
    merge_gap_sec: float = 0.5,
    max_utterance_sec: float = 10.0,
) -> None:
    """Write WebVTT from word-level timestamps (see :func:`compute_word_timestamps`).

    If ``f`` is ``None`` (the default), writes to :data:`sys.stdout`.

    When ``merge_utterance`` is True, adjacent words are merged into one cue if the
    gap between them is at most ``merge_gap_sec`` (default 500 ms) and the cue span
    would not exceed ``max_utterance_sec`` (default 10 s).
    """
    out = sys.stdout if f is None else f
    out.write("WEBVTT\n\n")
    rows = _word_timestamps_to_sec_rows(word_timestamps, time_offset)
    if merge_utterance:
        cues = _merge_words_into_utterances(rows, merge_gap_sec, max_utterance_sec)
    else:
        cues = rows
    for text, start_sec, end_sec in cues:
        out.write(f"{format_vtt_ts(start_sec)} --> {format_vtt_ts(end_sec)}\n")
        out.write(f"{text}\n\n")


class AVSRInference:
    """Load AVSR, tokenizer, and lip (RetinaFace/FAN) pipeline once; call ``transcribe`` per file."""

    def __init__(
        self,
        *,
        model_id: str | Path,
        cache_dir: str | Path,
        device: torch.device,
        face_hub_repo: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.device = device
        self.model_id = str(model_id)
        model_id_path = Path(model_id)

        set_model_bin_root(self.cache_dir)

        hub = (
            face_hub_repo
            if face_hub_repo is not None
            else (str(model_id) if not model_id_path.exists() else DEFAULT_MODEL_ID)
        )
        ensure_face_weights(hub, cache_dir=self.cache_dir)
        from src.retinaface.detector import LandmarksDetector
        from src.retinaface.video_process import VideoProcess

        self.landmarks_detector = LandmarksDetector(device=device_str_for_ibug(device))
        self.video_process = VideoProcess(convert_gray=False)

        self.model = AVASRForTranscription.from_pretrained(
            self.model_id, cache_dir=str(self.cache_dir)
        )
        self.model.to(device).eval()

        import sentencepiece as spm

        tokenizer = spm.SentencePieceProcessor()
        if model_id_path.exists():
            tok_path = model_id_path / "tokenizer.model"
        else:
            tok_path = cached_file(
                self.model_id,
                "tokenizer.model",
                cache_dir=str(self.cache_dir),
            )
            if tok_path is None:
                raise FileNotFoundError(f"tokenizer.model not found for {self.model_id}")
        tokenizer.load(str(tok_path))
        self.tokenizer = tokenizer

    @torch.no_grad()
    def transcribe(
        self,
        av_path: str | Path,
        *,
        audio_path: Optional[str | Path] = None,
        uem_start: Optional[float] = None,
        uem_end: Optional[float] = None,
        timestamps: bool = False,
        lip_crop_mode: LipCropMode = "auto",
    ) -> tuple[str, list[dict] | None]:
        return transcribe_file(
            self.model,
            self.tokenizer,
            av_path,
            timestamps=timestamps,
            audio_path=audio_path,
            uem_start=uem_start,
            uem_end=uem_end,
            device=self.device,
            lip_crop_mode=lip_crop_mode,
            landmarks_detector=self.landmarks_detector,
            video_process=self.video_process,
        )


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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.force_lip_crop:
        lip_crop_mode = "force"
    elif args.skip_lip_crop:
        lip_crop_mode = "skip"
    else:
        lip_crop_mode = "auto"

    inference = AVSRInference(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=device,
    )

    text, word_ts = inference.transcribe(
        args.av_path,
        audio_path=args.audio,
        uem_start=args.uem_start,
        uem_end=args.uem_end,
        timestamps=args.timestamps,
        lip_crop_mode=lip_crop_mode,
    )

    time_offset = args.uem_start if args.uem_start is not None else 0.0

    if args.timestamps and word_ts:
        if args.output:
            with open(args.output, "w") as f:
                write_vtt(word_ts, f, time_offset=time_offset)
            print(f"VTT written to {args.output}")
        else:
            write_vtt(word_ts, time_offset=time_offset)
    else:
        if args.output:
            args.output.write_text(text)
            print(f"Text written to {args.output}")
        else:
            print(text)
