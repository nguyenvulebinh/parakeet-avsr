"""Shared root directory for on-disk checkpoints (AVSR + face alignment / detection).

Set via :func:`set_model_bin_root` from CLI (e.g. ``--cache-dir`` in ``inference``) so
``ibug`` predictors and :mod:`src.avasr.io.face_model_hub` resolve the same paths.
"""
from __future__ import annotations

from pathlib import Path

_override: Path | None = None


def set_model_bin_root(path: str | Path) -> None:
    """Use this directory for ``face_alignment/``, ``face_detection/``, and HF AVSR cache."""
    global _override
    _override = Path(path).expanduser().resolve()


def get_model_bin_root() -> Path:
    """Active model-bin directory (defaults to ``<repo>/model-bin``)."""
    if _override is not None:
        return _override
    # src/model_bin_root.py -> parent = src, parents[1] = repository root
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "model-bin"
