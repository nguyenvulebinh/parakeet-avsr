"""Audio loading from media files with pydub/torchaudio fallback."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

AUDIO_SAMPLE_RATE = 16_000


def _load_audio_pydub(path: str, sample_rate: int) -> torch.Tensor:
    from pydub import AudioSegment

    seg = AudioSegment.from_file(path)
    sr = seg.frame_rate
    num_channels = seg.channels
    samples = np.array(seg.get_array_of_samples())
    if num_channels > 1:
        samples = samples.reshape(-1, num_channels)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)
    bit_depth = seg.sample_width * 8
    max_val = float(2 ** (bit_depth - 1))
    samples = samples / max_val
    if sr != sample_rate:
        import torchaudio

        t = torch.from_numpy(samples).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, sample_rate)
        samples = t.squeeze(0).numpy()
    return torch.tensor(samples, dtype=torch.float32)


def _load_audio_torchaudio(path: str, sample_rate: int) -> torch.Tensor:
    import torchaudio

    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.squeeze(0)


def load_audio(
    path: str | Path,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    *,
    fallback_num_samples: int | None = None,
) -> torch.Tensor:
    """Load mono float audio at ``sample_rate``.

    Tries pydub (ffmpeg), then torchaudio. If both fail (e.g. no audio stream in a
    video file) and ``fallback_num_samples`` is set, returns that many zeros and emits
    a warning so visual-only inference can proceed.
    """
    path = str(path)
    errors: list[BaseException] = []

    try:
        return _load_audio_pydub(path, sample_rate)
    except BaseException as e:
        errors.append(e)

    try:
        return _load_audio_torchaudio(path, sample_rate)
    except BaseException as e:
        errors.append(e)

    if fallback_num_samples is not None and fallback_num_samples > 0:
        pydub_err = errors[0] if errors else None
        ta_err = errors[1] if len(errors) > 1 else None
        warnings.warn(
            f"No audio stream could be read from {path!r}. pydub: {pydub_err!r}; "
            f"torchaudio: {ta_err!r}. Using silence (zeros), length={fallback_num_samples} samples.",
            UserWarning,
            stacklevel=2,
        )
        return torch.zeros(fallback_num_samples, dtype=torch.float32)

    msg = f"Failed to load audio from {path!r}."
    if errors:
        msg += f" pydub: {errors[0]!r}"
    if len(errors) > 1:
        msg += f"; torchaudio: {errors[1]!r}"
    raise RuntimeError(msg) from errors[-1]
