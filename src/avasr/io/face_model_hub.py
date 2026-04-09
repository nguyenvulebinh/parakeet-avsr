"""Resolve face checkpoints via ``hf_hub_download`` (same hub cache layout as Transformers)."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from src.model_bin_root import get_model_bin_root

# (repo path on hub, predictor kind, weight filename for lookup)
FACE_WEIGHT_HUB_ENTRIES: tuple[tuple[str, str, str], ...] = (
    ("face_alignment/fan/weights/2dfan2.pth", "fan", "2dfan2.pth"),
    ("face_detection/retina_face/weights/Resnet50_Final.pth", "retina", "Resnet50_Final.pth"),
)

# Populated by :func:`ensure_face_weights` in the current process.
_hub_weight_paths: dict[tuple[str, str], str] = {}


def get_hf_hub_weight_path(kind: str, filename: str) -> str | None:
    """Path from the last :func:`ensure_face_weights` run, if any (points into the HF cache)."""
    return _hub_weight_paths.get((kind, filename))


def ensure_face_weights(repo_id: str, cache_dir: str | Path | None = None) -> None:
    """Download face weights using the same ``cache_dir`` as ``from_pretrained`` (e.g. ``model-bin/``).

    Files appear under ``<cache_dir>/models--<org>--<name>/`` alongside AVSR checkpoints, not in ``~/.cache/huggingface`` (unless ``cache_dir`` points there).
    """
    cd = str(Path(cache_dir).expanduser().resolve()) if cache_dir is not None else str(get_model_bin_root())
    for hub_rel, kind, fname in FACE_WEIGHT_HUB_ENTRIES:
        _hub_weight_paths[(kind, fname)] = hf_hub_download(repo_id, hub_rel, cache_dir=cd)
