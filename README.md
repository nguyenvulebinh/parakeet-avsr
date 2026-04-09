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
