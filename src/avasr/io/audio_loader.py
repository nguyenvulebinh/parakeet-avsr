"""Audio loading from media files with pydub/torchaudio fallback."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


AUDIO_SAMPLE_RATE = 16_000


def load_audio(path: str | Path, sample_rate: int = AUDIO_SAMPLE_RATE) -> torch.Tensor:
    path = str(path)
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path)
        sr = seg.frame_rate
        num_channels = seg.channels
        samples = np.array(seg.get_array_of_samples())
        if num_channels > 1:
            samples = samples.reshape(-1, num_channels)
    except Exception:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        return waveform.squeeze(0)

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

