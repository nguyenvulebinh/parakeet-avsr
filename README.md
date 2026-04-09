# Parakeet AVASR

Simple audio-visual ASR inference with Python scripts.

## 1) Install

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate avasr
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## 2) Quick run (single file)

Run transcription on one video:

```bash
python run_single.py examples/video.mp4
```

Output word timestamps (VTT format) to stdout:

```bash
python run_single.py examples/video.mp4 --timestamps
```

Write output to file:

```bash
python run_single.py examples/video.mp4 --output out.vtt --timestamps
```

Notes:
- Default model id: `nguyenvulebinh/parakeet-avsr`
- Default model cache dir: `model-bin`
- You can override them:

```bash
python run_single.py examples/video.mp4 --model-id nguyenvulebinh/parakeet-avsr --cache-dir model-bin
```

## 3) Python API

For a service or script, construct [`AVSRInference`](src/avasr/pipeline/inference.py) once (loads AVSR, tokenizer, and lip models), then call [`transcribe`](src/avasr/pipeline/inference.py) per video. From another working directory, set `PYTHONPATH` to the repo root (e.g. `PYTHONPATH=/path/to/parakeet-avsr`) so `import src...` resolves.

```python
from pathlib import Path

import torch

from src.avasr.pipeline.inference import AVSRInference, write_vtt

repo_root = Path(__name__).resolve().parent  # or your project root
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

infer = AVSRInference(
    model_id="nguyenvulebinh/parakeet-avsr",
    cache_dir=repo_root / "model-bin",
    device=device,
)

text, word_timestamps = infer.transcribe("examples/video.mp4", timestamps=True)

if word_timestamps:
    with open("out.vtt", "w") as f:
        # Default: merge words into cues when gap ≤ 500 ms and cue ≤ 10 s; use merge_utterance=False for one cue per word.
        write_vtt(word_timestamps, f)  # write_vtt(word_timestamps) prints VTT to stdout
```

`lip_crop_mode` matches the CLI: `"auto"` (default), `"force"` (always RetinaFace + FAN in memory), or `"skip"` (assume video is already lip-cropped 96×96; face weights still load at init).
