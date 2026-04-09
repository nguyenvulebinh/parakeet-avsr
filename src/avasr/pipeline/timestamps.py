from __future__ import annotations

from src.avasr.core.tdt_decode import TDTResult

MS_PER_FRAME = 0.080  # 80ms per encoder frame (12.5 Hz)


def format_vtt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def compute_word_timestamps(
    result: TDTResult,
    tokenizer,
) -> list[dict]:
    """Group token-level encoder frame indices into word-level offsets."""
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
            words.append(
                {
                    "word": word_text,
                    "start_offset": cur_start,
                    "end_offset": cur_end,
                }
            )
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
        words.append(
            {
                "word": word_text,
                "start_offset": cur_start,
                "end_offset": cur_end,
            }
        )

    return words

